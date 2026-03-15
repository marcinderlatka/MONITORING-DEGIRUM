# Plan poprawy wydajności — instrukcje dla Codex

## Cel
Zoptymalizować aplikację monitoringu PyQt + DeGirum tak, aby:
- zmniejszyć użycie CPU i RAM,
- ograniczyć liczbę kopiowań ramek,
- zmniejszyć lagi GUI przy wielu kamerach,
- ustabilizować nagrywanie przy wolnym dysku,
- nie zmieniać zachowania funkcjonalnego dla użytkownika.

Kod źródłowy do analizy znajduje się głównie w:
- `monitoring/workers.py`
- `monitoring/app.py`
- `monitoring/storage.py`
- `monitoring/config.py`

---

## Najważniejsze problemy znalezione w kodzie

### 1. Za dużo kopiowania ramek w pętli kamery
W `CameraWorker.run()` każda klatka jest kopiowana wielokrotnie:
- `self.prerecord_buffer.append(frame.copy())`
- `display_frame = frame.copy()`
- dodatkowo tworzony jest `alert_frame = frame.copy()`
- potem `display_frame` trafia do GUI i do nagrywania.

To powoduje duże użycie CPU i RAM, szczególnie przy kilku strumieniach RTSP.

### 2. Zbyt ciężkie renderowanie w GUI
W `app.py` metoda `_render_current()` przy każdej klatce wykonuje:
- tworzenie canvasu,
- `cv2.resize`,
- `cv2.cvtColor`,
- `QImage(...).copy()`,
- rysowanie tekstu przez `QPainter`.

To jest wykonywane dla każdej aktualizacji obrazu i obciąża główny wątek UI.

### 3. Brak cache modeli DeGirum
W `start_camera()` model jest ładowany osobno dla każdej kamery przez `dg.load_model(...)`.
Jeśli kilka kamer używa tego samego modelu, to pamięć i czas inicjalizacji są marnowane.

### 4. Operacje I/O JSON wykonywane synchronicznie przy detekcji
`update_recordings_catalog()` w `storage.py` za każdym razem:
- wczytuje cały katalog JSON,
- filtruje listę,
- zapisuje cały plik od nowa.

To samo dotyczy `AlertMemory.save()` — cały JSON jest zapisywany przy każdym dodaniu alertu.
Przy częstych detekcjach to generuje zbędny narzut na dysk i blokuje logikę.

### 5. Nieskończona kolejka do zapisu wideo
`RecordingThread.queue = Queue()` nie ma limitu.
Jeżeli dysk zapisuje wolniej niż przychodzą klatki, kolejka może rosnąć bez końca i zjadać RAM.

### 6. Powtarzane kosztowne operacje w każdej detekcji obiektu
W `workers.py` dla każdego obiektu wykonywane są ponownie:
- `[c.lower() for c in self.visible_classes]`
- `[c.lower() for c in self.record_classes]`

To powinno być znormalizowane raz, a nie dla każdej klatki i każdego obiektu.

### 7. Nagrywanie używa klatek z overlayami
Obecnie do pliku trafia `display_frame`, czyli klatka po rysowaniu overlayów.
Rysowanie prostokątów i napisów zwiększa koszt CPU. Nagranie może zapisywać czystą klatkę, a overlaye zostawić tylko do preview.

### 8. Niebezpieczne zatrzymywanie wątku
`CameraWorker.stop()` używa awaryjnie `self.terminate()`. To może zostawić zasoby w niepewnym stanie.

---

## Priorytety wdrożenia

Wdrożenie zrób etapami. Po każdym etapie kod ma działać i przejść prosty smoke test.

### Etap 1 — szybkie zyski bez zmiany architektury
1. Dodaj cache modeli.
2. Ogranicz kopiowanie ramek.
3. Wprowadź znormalizowane zbiory klas `visible_classes_lower` i `record_classes_lower`.
4. Ogranicz częstotliwość renderowania GUI.
5. Dodaj limit kolejki nagrywania.

### Etap 2 — średni refactor
6. Rozdziel klatkę surową i klatkę podglądową.
7. Przenieś kosztowne zapisy katalogu/alertów do buforowanego writer-a.
8. Ogranicz redraw overlayów, gdy nie było nowej inferencji.

### Etap 3 — większe usprawnienia
9. Dodaj prosty profiler FPS i czasów etapów.
10. Dodaj adaptacyjne pomijanie klatek przy przeciążeniu.
11. Usuń `terminate()` i zamień zatrzymywanie na bezpieczne wygaszanie.

---

## Konkretne instrukcje implementacyjne dla Codex

## Zadanie 1 — cache modeli DeGirum
W `monitoring/app.py`:
- dodaj słownik `self.model_cache: dict[str, Any] = {}` w klasie głównego okna,
- utwórz pomocniczą metodę np. `_get_model(model_name: str) -> Any`,
- jeśli model istnieje w cache, zwróć go,
- jeśli nie, załaduj go raz i zapisz w cache,
- `start_camera()` ma używać `_get_model(...)` zamiast bezpośredniego `dg.load_model(...)`.

### Kryterium akceptacji
- dwie kamery z tym samym `model_name` nie ładują modelu drugi raz,
- log może pokazać „model z cache” vs „model załadowany”.

---

## Zadanie 2 — zredukować kopiowanie ramek w `CameraWorker`
W `monitoring/workers.py`:
- wprowadź dwa byty:
  - `raw_frame` — klatka surowa do prerecord i zapisu,
  - `preview_frame` — klatka do GUI, tworzona tylko gdy naprawdę potrzeba overlayów.
- NIE rób bezwarunkowo `display_frame = frame.copy()`.
- `preview_frame` twórz tylko wtedy, gdy:
  - `self.draw_overlays` jest włączone i są overlaye do narysowania,
  - albo trzeba zbudować klatkę alertu.
- do `self.prerecord_buffer` odkładaj `frame.copy()` tylko wtedy, gdy bufor ma realnie służyć nagrywaniu.
- do nagrywania zapisuj `raw_frame`, nie `preview_frame`.

### Wymaganie
Nagrania mają dalej obejmować 5 sekund przed i 5 sekund po detekcji.

### Kryterium akceptacji
- liczba `copy()` w głównej pętli ma być wyraźnie mniejsza,
- GUI i alerty wizualnie nadal działają,
- plik MP4 zapisuje się poprawnie.

---

## Zadanie 3 — znormalizować klasy tylko raz
W `CameraWorker.__init__()`:
- dodaj:
  - `self.visible_classes_lower = {c.lower() for c in self.visible_classes}`
  - `self.record_classes_lower = {c.lower() for c in self.record_classes}`
- zaktualizuj settery i miejsca, które zmieniają klasy, aby odświeżały te zbiory.
- w pętli detekcji zastąp list comprehensions odwołaniem do gotowych setów.

### Kryterium akceptacji
- brak `[c.lower() for c in ...]` wewnątrz pętli po obiektach.

---

## Zadanie 4 — limit kolejki nagrywania i polityka dropowania
W `RecordingThread`:
- ustaw `Queue(maxsize=...)`, np. na 2–3 sekundy ramek,
- dodaj strategię ochrony RAM:
  - gdy kolejka pełna, odrzuć najstarszą klatkę albo pomiń nową,
  - zaloguj licznik dropów,
- nie blokuj wątku kamery na `queue.put()`.

Proponowane rozwiązanie:
- `max_queue = max(30, int(self.fps * 3))` lub na bazie `stream_fps`,
- helper `safe_put_frame(frame)`.

### Kryterium akceptacji
- przy spowolnionym dysku RAM nie rośnie bez końca,
- aplikacja dalej reaguje.

---

## Zadanie 5 — odchudzenie renderowania GUI
W `monitoring/app.py`:
- nie renderuj pełnego widoku przy każdej odebranej klatce bez limitu,
- dodaj throttling odświeżania UI, np. maks. 10–15 FPS dla podglądu głównego,
- oddziel częstotliwość analizy od częstotliwości wyświetlania,
- cache’uj wynik skalowania, jeśli rozmiar widgetu się nie zmienił,
- nie wywołuj `_compose_letterboxed()` częściej niż trzeba.

### Minimalne wdrożenie
- dodaj w `update_frame()` znacznik czasu i pomijaj render, jeśli od poprzedniego renderu minęło mniej niż np. 66–100 ms,
- nadal zapisuj ostatnią klatkę do `_last_frame[idx]`, ale nie zawsze od razu renderuj ją do `QPixmap`.

### Kryterium akceptacji
- UI pozostaje płynne,
- CPU głównego procesu spada przy kilku kamerach.

---

## Zadanie 6 — nie rysuj overlayów dwa razy bez potrzeby
Obecnie gdy nie ma nowej inferencji, aplikacja potrafi ponownie rysować `last_overlays` na każdej kolejnej klatce preview.

Zmień to tak, aby:
- overlaye były rysowane tylko wtedy, gdy jest nowy wynik inferencji,
- albo gdy naprawdę chcesz utrzymać ostatni wynik przez krótki czas i jest to jawnie kontrolowane.

Najprostsza wersja:
- jeśli `inference_result is None`, wyślij do GUI surową klatkę bez redraw starego overlayu.

### Kryterium akceptacji
- mniej operacji `cv2.rectangle` i `cv2.putText`,
- detekcja nadal działa poprawnie.

---

## Zadanie 7 — buforowany zapis katalogu nagrań i historii alertów
W `monitoring/storage.py`:
- nie zapisuj całego JSON przy każdym pojedynczym evencie,
- dodaj prosty bufor z debounce, np. zapis co 1–2 sekundy albo przy zamknięciu aplikacji,
- najlepiej wydziel `CatalogWriter` i `AlertWriter` albo użyj jednego lekkiego mechanizmu flush.

### Wersja minimalna
- trzymaj dane w pamięci,
- zapisuj dopiero po serii zmian przez timer,
- na wyjściu aplikacji wykonaj `flush()`.

### Kryterium akceptacji
- przy wielu alertach dysk nie jest mielony ciągłym przepisywaniem całego JSON,
- dane nie giną po normalnym zamknięciu aplikacji.

---

## Zadanie 8 — bezpieczne zatrzymywanie workerów
W `CameraWorker.stop()`:
- usuń `terminate()` jako domyślną ścieżkę,
- zatrzymanie ma działać przez flagę `stop_signal`, zamknięcie streamu, stop record thread i `wait()` z sensownym timeoutem,
- tylko w ostateczności loguj problem, ale nie ubijaj wątku brutalnie, jeśli nie jest to absolutnie konieczne.

### Kryterium akceptacji
- zamykanie aplikacji nie zawiesza się,
- nie ma ryzyka uszkodzenia writer-a lub uchwytu RTSP.

---

## Zadanie 9 — prosty profiler runtime
Dodaj lekkie metryki diagnostyczne:
- czas pobrania klatki,
- czas inferencji,
- czas rysowania overlayów,
- czas emitowania do GUI,
- czas zapisu do kolejki nagrywania,
- licznik dropniętych klatek recorder-a.

Loguj agregaty co kilka sekund, nie co klatkę.

### Kryterium akceptacji
- można szybko zobaczyć, gdzie aplikacja traci czas.

---

## Proponowane zmiany architektoniczne

### Docelowy przepływ jednej klatki
1. Odbierz `raw_frame` ze streamu.
2. Dodaj do prerecord buffer w lekkiej formie.
3. Co `detection_interval` uruchom inferencję.
4. Wynik inferencji zapisz jako lekkie metadane overlayów.
5. `preview_frame` twórz tylko dla GUI.
6. Do nagrywania przekazuj `raw_frame`.
7. GUI renderuj z throttlingiem.
8. JSON zapisuj asynchronicznie lub z debounce.

---

## Ważne ograniczenia
Nie wolno zepsuć istniejących funkcji:
- nagrywanie 5 s przed i 5 s po detekcji,
- osobne katalogi nagrań według nazw kamer,
- alert z miniaturą i metadanymi,
- możliwość włączenia/wyłączenia overlayów,
- osobna lista klas widocznych i nagrywanych.

---

## Plan wykonania w commitach

### Commit 1
`perf: cache models and normalize detection class sets`
- cache modeli
- sety `visible_classes_lower`, `record_classes_lower`

### Commit 2
`perf: reduce frame copies and record raw frames`
- rozdzielenie raw/preview
- mniej `copy()`
- zapis raw frame do MP4

### Commit 3
`perf: throttle main preview rendering`
- limit FPS GUI
- mniejsza liczba ciężkich renderów

### Commit 4
`perf: bound recording queue and protect memory`
- `Queue(maxsize=...)`
- drop policy
- liczniki dropów

### Commit 5
`perf: debounce catalog and alert json writes`
- buforowany zapis JSON

### Commit 6
`refactor: make worker shutdown graceful`
- usunięcie `terminate()`
- bezpieczne wygaszanie

### Commit 7
`chore: add lightweight runtime performance metrics`
- logi diagnostyczne

---

## Test plan po wdrożeniu

### Test 1 — jedna kamera RTSP
- uruchom aplikację,
- sprawdź podgląd, alert i zapis nagrania,
- porównaj użycie CPU przed/po.

### Test 2 — 2–4 kamery z tym samym modelem
- sprawdź, że model nie ładuje się wielokrotnie,
- obserwuj RAM i czas startu.

### Test 3 — sztuczne przeciążenie dysku
- sprawdź, że kolejka nagrywania nie rośnie bez limitu,
- aplikacja nie zawiesza się.

### Test 4 — overlaye OFF / detection ON
- nagrywanie i alerty mają działać,
- preview nie powinien robić zbędnych operacji rysowania.

### Test 5 — overlaye ON / recording OFF
- detekcja i rysowanie mają działać,
- brak regresji w GUI.

### Test 6 — zamykanie aplikacji podczas aktywnego nagrania
- plik ma być domknięty poprawnie,
- brak zawieszenia przy shutdown.

---

## Polecenie startowe dla Codex
Użyj poniższego polecenia jako zadania głównego:

> Przeanalizuj projekt monitoringu PyQt + DeGirum i wdroż plan poprawy wydajności. Zacznij od najtańszych zmian o największym wpływie: cache modeli, redukcja kopiowania ramek, normalizacja klas do setów, throttling renderowania GUI, ograniczenie kolejki nagrywania. Następnie rozdziel raw frame od preview frame, przenieś zapis JSON do buforowanego flush, popraw bezpieczne zamykanie workerów i dodaj lekkie metryki wydajności. Zachowaj dotychczasowe funkcje: 5 s przed i 5 s po detekcji, katalogi per kamera, alerty z miniaturą, konfigurację klas i overlayów. Wprowadzaj zmiany etapami w małych commitach i po każdym etapie zostaw działający kod.

---

## Dodatkowe uwagi dla Codex
- Nie rób wielkiej przebudowy całego UI na start.
- Najpierw popraw gorące ścieżki w `workers.py` i `app.py`.
- Preferuj małe, czytelne helpery zamiast jednego dużego refactoru.
- Dodaj komentarze tylko tam, gdzie wyjaśniają decyzję wydajnościową.
- Jeśli trzeba wybrać między „ładniej” a „wydajniej”, wybierz wydajniej bez psucia funkcji.
