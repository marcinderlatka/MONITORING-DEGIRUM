# MONITORING-DEGIRUM

Aplikacja oferuje wielokanałowy monitoring wideo działający na desktopie i zasilany modelami DeGirum. Interfejs graficzny zbudowany na PyQt5 łączy się ze strumieniami RTSP lub lokalnymi kamerami USB, uruchamia inferencję na urządzeniu edge oraz pozwala zarządzać alertami, nagraniami i historią logów w czasie rzeczywistym.

## Najważniejsze funkcje
* Obsługa wielu źródeł jednocześnie (RTSP i USB) oraz szybkie przełączanie widoków pomiędzy listą kamer, głównym podglądem i widokiem siatki.
* Wykrywanie obiektów z użyciem `degirum_tools.predict_stream`, nakładki z ramkami i opisami oraz sterowanie harmonogramem detekcji.
* Buforowanie klatek przed zdarzeniem, automatyczne nagrywanie po wykryciu, generowanie miniaturek i katalogowanie metadanych nagrań.
* Panel alertów z historią, eksportem do CSV, podglądem nagrań oraz sygnalizacją dźwiękową.
* Wbudowany rejestr zdarzeń aplikacji/detekcji/błędów, który jest utrwalany w `log_history.json`.
* Asynchroniczny skaner katalogów nagrań z filtrami po kamerze, klasie, dacie i nazwie pliku oraz z możliwością usuwania wielu pozycji.
* Kreatory do dodawania kamer RTSP/USB, edycji parametrów (model, FPS, progi, harmonogram, klasy) oraz testowania połączenia.

## Struktura projektu
```
.
├── main.py                # Punkt wejścia CLI (`python main.py [--windowed]`)
├── app_01.py              # Alias uruchamiający to samo co main.py
├── monitoring/            # Pakiet z logiką PyQt5
│   ├── app.py             # Główne okno, kreatory, dialogi, przeglądarka nagrań
│   ├── workers.py         # Wątki kamer i nagrywania, integracja z DeGirum
│   ├── storage.py         # Trwała pamięć alertów i katalog nagrań
│   ├── config.py          # Obsługa config.json oraz ścieżek pomocniczych
│   └── widgets/           # Widżety (lista kamer, alerty, logi, siatka)
├── config.json            # Konfiguracja źródeł kamer i globalnych ustawień
├── alerts_history.json    # Trwała historia alertów (generowana automatycznie)
├── log_history.json       # Historia logów aplikacji (generowana automatycznie)
├── icons/                 # Ikony SVG wykorzystywane w UI
└── models/                # Modele DeGirum (podkatalogi z plikami zoo)
```

## Wymagania wstępne
* Python 3.8 lub nowszy.
* Systemowe biblioteki wymagane przez Qt oraz sterowniki kamer/urządzeń wideo.
* Modele DeGirum pobrane lokalnie (np. `models/yolov5nu_silu_coco--640x640_float_tflite_multidevice_1`).

## Instalacja
1. Sklonuj repozytorium i przejdź do katalogu projektu.
2. (Opcjonalnie, zalecane) utwórz i aktywuj wirtualne środowisko:
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Linux/macOS
   .venv\Scripts\activate        # Windows (PowerShell)
   ```
3. Zainstaluj zależności:
   ```bash
   pip install -r requirements.txt
   ```
4. Upewnij się, że katalog `models/` zawiera potrzebne modele DeGirum.

## Modele DeGirum
`monitoring.workers.CameraWorker` ładuje model poprzez `degirum.load_model` z lokalnego katalogu zoo (`models/<nazwa_modelu>`). Nazwę modelu można zmienić w konfiguracji kamery; katalog z modelem musi zawierać artefakty wymagane przez DeGirum (np. plik `manifest.json`).

## Konfiguracja kamer (`config.json`)
Plik konfiguracyjny przechowuje listę kamer oraz opcjonalne parametry globalne (`log_history_path`, `log_retention_hours`). Wszystkie brakujące wartości są uzupełniane domyślnie w `monitoring.config.fill_camera_defaults`.

Minimalny wpis dla kamery RTSP:
```json
{
  "name": "Magazyn",
  "rtsp": "rtsp://admin:haslo@192.168.0.10:554/Streaming/Channels/101",
  "type": "rtsp"
}
```
Najważniejsze pola opcjonalne:

| Klucz | Opis |
| --- | --- |
| `model` | Nazwa katalogu z modelem DeGirum (ładowana przez `degirum.load_model`). |
| `fps` | Docelowa liczba klatek analizowanych na sekundę. |
| `confidence_threshold` | Minimalne prawdopodobieństwo, aby alert/nagranie zostały wyzwolone. |
| `draw_overlays` | Czy rysować ramki i opisy na podglądzie. |
| `enable_detection` | Czy wykonywać inferencję i generować alerty. |
| `enable_recording` | Czy nagrywać fragmenty wideo po detekcji. |
| `detection_hours` | Harmonogram w formacie `HH:MM-HH:MM;...`. |
| `visible_classes` | Lista klas widocznych w nakładkach. |
| `record_classes` | Lista klas uruchamiających nagrywanie/alert. |
| `record_path` | Folder bazowy nagrań (podkatalog o nazwie kamery tworzony automatycznie). |
| `pre_seconds` / `post_seconds` | Liczba sekund bufora przed/po zdarzeniu używana w nagraniach. |
| `lost_seconds` | Tolerancja braku detekcji zanim nagrywanie zostanie zamknięte. |

Zmiany w konfiguracji można wprowadzać z poziomu UI (przycisk „Ustawienia” → dialog kamery) lub ręcznie edytując plik i ponownie uruchamiając aplikację.

## Uruchomienie aplikacji
```bash
python main.py            # tryb pełnoekranowy
python main.py --windowed # tryb okienkowy
```
Uruchomienie tworzy główne okno PyQt5, wczytuje konfigurację i startuje wątki kamer w tle.

## Obsługa interfejsu
### Główne okno
Centralny panel wyświetla bieżącą kamerę z nakładkami, paskiem statusu (nazwa, FPS, ostatni błąd) oraz zestawem przycisków sterujących: lista kamer, katalog nagrań, menu ustawień, zarządzanie kamerami, alerty, dźwięk oraz pełny ekran. Po bokach znajdują się widżety alertów i logów.

### Lista i siatka kamer
* **CameraListWidget** – boczna lista ze zrzutami miniatur, umożliwia przełączanie aktywnej kamery. Miniatury są aktualizowane na bieżąco.
* **CameraGridWidget** – pełnoekranowa siatka (otwierana z przycisku kamery), prezentująca wszystkie źródła jednocześnie i reagująca na kliknięcie, aby przejść do wybranego strumienia.

### Alerty i powiadomienia
Panel boczny **AlertListWidget** prezentuje najnowsze detekcje z miniaturami i metadanymi. Dialog „Alerty” pozwala ukrywać/pokazywać panel, odświeżać historię (`alerts_history.json`), eksportować zdarzenia do CSV i czyścić pamięć. Odtworzenie powiązanego nagrania jest dostępne dwuklikiem.

### Logi operacyjne
Widżet **LogWindow** zapisuje zdarzenia aplikacji, alerty i błędy do pliku `log_history.json`, prezentując je w kolorystycznie odróżnionych sekcjach wraz ze znacznikami czasu i stanem nagrywania/detekcji.

### Przeglądarka nagrań
Dialog „Nagrania” skanuje katalogi nagrań w tle, buduje listę plików MP4 z miniaturami, umożliwia filtrowanie po kamerze, klasie, zakresie dat oraz wyszukiwaniu po nazwie. Z tego miejsca można otwierać nagrania, kasować wiele pozycji jednocześnie (razem z metadanymi `.json` i miniaturami `.jpg`) lub masowo zaznaczać/odznaczać elementy.

**Niezawodne ładowanie miniaturek (widok kafelków):**
* Przeglądarka przebudowuje tylko aktywny widok (kafelki albo lista), a drugi widok odświeża dopiero po przełączeniu — ogranicza to zacięcia podczas otwierania okna.
* Asynchroniczny pipeline miniaturek działa na ograniczonym `QThreadPool` (`maxThreadCount=2`) i blokuje duplikaty zadań dla tego samego pliku.
* Pierwsza fala ładowania próbuje wyłącznie miniatur JPG (metadane `thumb`/`recording_thumb` oraz `*.jpg` obok nagrania); ciężki fallback MP4 uruchamia się warunkowo tylko dla widocznych kafelków.
* Każde żądanie miniatury kończy się stanem końcowym: sukces (miniatura), fallback (klatka MP4) albo porażka z czytelnym placeholderem „Brak miniatury” — bez pozostawiania kart w nieskończonym „Ładowanie miniatury...”.
* Błędy i ostrzeżenia pipeline’u miniaturek są raportowane do panelu **Logi** (`source=recordings-browser`) jako `INFO`/`WARNING`/`ERROR`, co ułatwia diagnostykę brakujących JPG i awarii fallbacku.

### Odtwarzacz nagrań
Podwójne kliknięcie nagrania otwiera odtwarzacz z kontrolkami transportu, przełączaniem między plikami, zrzutem klatki i trybem pełnoekranowym.

### Ustawienia kamery i opisy pól
* Dialog ustawień kamery ma teraz komplet szczegółowych podpowiedzi (hover) dla wszystkich pól konfiguracyjnych.
* Opisy wyjaśniają wpływ parametrów na detekcję, nagrywanie, wydajność GUI/CPU/GPU oraz stabilność logiki zdarzeń.
* Ustawienia runtime są stosowane od razu tam, gdzie to możliwe, bez restartu całej aplikacji.
* Tylko ustawienia krytyczne dla strumienia (np. źródło, typ lub model) powodują automatyczny restart wyłącznie zmienionej kamery.

## Detekcja i nagrywanie
`CameraWorker` uruchamia strumień `degirum_tools.predict_stream`, nakłada ramki na obraz (tylko dla klas z `visible_classes`), generuje alerty/nagrania dla klas z `record_classes` i pilnuje harmonogramu `detection_hours`. Przed rozpoczęciem nagrywania utrzymywany jest bufor klatek z ostatnich `pre_seconds`, a po ustaniu detekcji nagranie trwa jeszcze `post_seconds`. Metadane (`.json`) i miniatury (`.jpg`) są zapisywane obok pliku MP4 oraz dopisywane do katalogu `recordings_catalog.json`.

## Trwałość danych
* `alerts_history.json` – najnowsze alerty (maks. 5000 pozycji).
* `recordings_catalog.json` – indeks nagrań używany przez przeglądarkę i do scalania z historią alertów.
* `log_history.json` – logi aplikacji, czyszczone po przekroczeniu limitu godzin z konfiguracji.

## Rozszerzanie i debugowanie
* Obsługa nowych klas obiektów sprowadza się do aktualizacji list `VISIBLE_CLASSES` i `RECORD_CLASSES` lub konfiguracji pojedynczej kamery.
* W razie problemów z RTSP skorzystaj z kreatora dodawania i wbudowanego testu połączenia (wykorzystuje `cv2.VideoCapture`).
* Błędy łącza/detekcji są raportowane w logach oraz prezentowane w overlayu aktywnej kamery.

## Performance Optimizations

### Model Cache

File: `monitoring/app.py`

Before:
Each camera startup path could trigger an independent DeGirum model load for the same model name.

After:
Application-level model cache (`self.model_cache`) with `_get_model(model_name)` reuses loaded model instances across workers.

Impact:
Reduces repeated model initialization overhead, startup latency for additional cameras, and memory pressure.

### Class Set Optimization

File: `monitoring/workers.py`

Before:
Class filtering in detection paths rebuilt lowercase containers repeatedly during frame processing.

After:
Lowercased sets (`visible_classes_lower`, `record_classes_lower`) are precomputed and refreshed only when class lists change.

Impact:
Removes per-frame list reconstruction and speeds up class membership checks in the hot inference loop.

### Raw/Preview Frame Split

File: `monitoring/workers.py`

Before:
Multiple `frame.copy()` operations could happen in the main processing loop for buffering, overlays, and alert data.

After:
Loop now uses explicit `raw_frame` for recording/buffering and `preview_frame` for overlays/GUI output, eliminating redundant copies.

Impact:
Lowers per-frame memory allocation and reduces CPU overhead from unnecessary array duplication.

### Recording Queue Limit

File: `monitoring/workers.py`

Before:
Recorder queue growth strategy could keep accumulating frames under write pressure, increasing RAM usage risk.

After:
`RecordingThread` uses `Queue(maxsize=120)` and non-blocking enqueue with frame drop + warning when full.

Impact:
Prevents unbounded memory growth during slow disk I/O and keeps capture/detection responsive.

### GUI Render Throttling

File: `monitoring/app.py`

Before:
Main preview rendering could be triggered at full incoming frame rate.

After:
Rendering is throttled using `self.last_render_time` to approximately 15 FPS.

Impact:
Reduces GUI repaint load and CPU usage while preserving smooth operator preview.

### JSON Write Debounce

File: `monitoring/storage.py`

Before:
Alert persistence updates could schedule frequent JSON writes under bursty detections.

After:
`AlertMemory` tracks `last_save_time` and only schedules writes at most once every 2 seconds (with flush on shutdown).

Impact:
Lowers disk I/O amplification and avoids write storms during high alert rates.

### Safe Thread Stop

File: `monitoring/workers.py`

Before:
Recorder shutdown logic relied on stop flags but did not expose explicit running-state semantics.

After:
Recorder thread now uses `self.running = False` and exits naturally after draining the queue, without forceful thread termination.

Impact:
Improves shutdown safety and reduces chance of thread-related instability/crashes.

### Basic Performance Logging

File: `monitoring/workers.py`

Before:
No periodic lightweight timing summary for runtime hot path.

After:
Performance metrics are aggregated and logged in a fixed format (about every 10 seconds) for `capture_fps`, `infer_fps`, `preview_emit_fps`, `ui_render_ms`, `queue_size`, `dropped_frames`, `cpu_percent`, `rss_mb`, with camera identifier and mode (`main/thumb/hidden`, overload `on/off`).

Impact:
Provides low-overhead visibility for profiling and diagnosing bottlenecks in production-like runs.

## Performance baseline

Aby porównać wydajność **przed/po zmianie**, wykonuj testy zawsze w identycznych warunkach:

1. Ta sama liczba kamer.
2. Ta sama rozdzielczość strumienia dla każdej kamery.
3. Ten sam model AI i konfiguracja progów detekcji.
4. Ten sam czas trwania testu (np. 10 minut).

### Procedura

1. Uruchom wersję „przed” i pozwól systemowi pracować przez ustalony czas testu.
2. Zapisz/wyeksportuj logi grupy `performance`.
3. Uruchom wersję „po” w tych samych warunkach.
4. Zbierz analogiczne logi `performance`.
5. Porównaj medianę/średnią oraz p95 dla: `capture_fps`, `infer_fps`, `preview_emit_fps`, `ui_render_ms`, `queue_size`, `dropped_frames`, `cpu_percent`, `rss_mb`.

### Wskazówki interpretacyjne

- Spadek `capture_fps` lub `infer_fps` przy stałym `cpu_percent` może oznaczać wąskie gardło I/O.
- Wzrost `queue_size` i `dropped_frames` wskazuje na problem z nadążaniem zapisu.
- Wzrost `ui_render_ms` przy stabilnym `infer_fps` sugeruje przeciążenie GUI, a nie inferencji.

## Recording and Detection Reliability Fixes

### Natural playback FPS
- **What changed:** Recording writer FPS is now computed with an explicit helper that prefers configured RTSP throttle (`rtsp_fps`), then measured loop cadence, then stream FPS fallback. The computed value is used when creating `RecordingThread`, logged, and persisted as `writer_fps` together with `source_fps` and `detect_fps`.
- **Why:** Previously files were often encoded at stream FPS even when processing was throttled, causing time-compressed playback.
- **Modified file(s):** `monitoring/workers.py`, `monitoring/recordings.py`.
- **Before / After:** Before a 5 FPS processed stream encoded at 25 FPS played too fast; after, encoded FPS matches effective capture cadence and playback is natural.

### Detection-first thumbnails
- **What changed:** A thumbnail JPG is explicitly generated from the first confirmed detection frame (with a visible box), saved near the MP4, and stored in metadata (`thumb`, `thumbnail_ts`, `event_start_ts`, `thumbnail_mode=first_detection`). Browser loading now prioritizes explicit thumbnail metadata.
- **Why:** The old approach could show prerecord / non-event frames as preview.
- **Modified file(s):** `monitoring/workers.py`, `monitoring/recordings.py`, `monitoring/widgets/recordings_browser.py`.
- **Before / After:** Before preview often missed the detected object; after preview is detection-centric and does not depend on extracting frame 0 from MP4.

### Real-time recording timers
- **What changed:** End-of-event timing now uses monotonic timestamps (`detection_last_seen_ts`) instead of frame counters tied to loop FPS.
- **Why:** Frame-count timing stretched/shrank durations whenever actual processing FPS differed from configured FPS.
- **Modified file(s):** `monitoring/workers.py`.
- **Before / After:** Before `lost_seconds` / `post_seconds` were frame-rate dependent; after they represent wall-clock seconds consistently.

### Detection reliability tuning
- **What changed:** Detection and drawing thresholds were split (`confidence_threshold_draw`, `confidence_threshold_record`), inference cadence remains monotonic-time based, and optional trigger stabilization (`required_hits_to_start_recording`) was added with backward-compatible defaults.
- **Why:** Single-threshold, sparse sampling behavior made tuning difficult and could miss events.
- **Modified file(s):** `monitoring/config.py`, `monitoring/app.py`, `monitoring/workers.py`.
- **Before / After:** Before draw/record confidence was coupled and trigger behavior less tunable; after operators can tune visibility and recording trigger sensitivity separately.

### New metadata fields
- **What changed:** Recording sidecar/catalog payloads now include reliability diagnostics: `filepath`, `file`, `time`, `timestamp`, `source_fps`, `writer_fps`, `detect_fps`, `event_start_ts`, `thumbnail_ts`, `frames_written`, `dropped_frames`, `thumbnail_mode`, `inference_count`, `positive_detection_count`.
- **Why:** Rich metadata is required for reliable browsing, diagnostics, and backward-compatible catalog parsing.
- **Modified file(s):** `monitoring/recordings.py`, `monitoring/storage.py`, `monitoring/workers.py`.
- **Before / After:** Before metadata was minimal; after each event has enough context to debug recording cadence and detection behavior while old entries still load.

### Detection reliability tuning (phase 2)
- **Old behavior:** one threshold and frame-counter-based stop logic.
- **New behavior:** independent `confidence_threshold_draw` and `confidence_threshold_record`, start/stop stabilization (`required_hits_to_start_recording`, `required_misses_to_end_detection`), and minimum clip duration (`min_record_seconds`).
- **Practical effect:** fewer false starts/stops and easier tuning for noisy scenes.
- **Files:** `monitoring/config.py`, `monitoring/app.py`, `monitoring/workers.py`.

### Recording start policy
- **Old behavior:** prerecord was always written before the event frame.
- **New behavior:** configurable `record_start_mode`:
  - `detection_first` (default): event frame starts the clip.
  - `include_prerecord_first`: legacy prerecord-first behavior.
- **Practical effect:** browser metadata/poster remains event-first while prerecord is still optionally available.
- **Files:** `monitoring/config.py`, `monitoring/workers.py`, `monitoring/app.py`.

### New camera settings and backward compatibility
New per-camera keys in `config.json`:
- `confidence_threshold_draw`
- `confidence_threshold_record`
- `required_hits_to_start_recording`
- `required_misses_to_end_detection`
- `min_record_seconds`
- `thumbnail_mode` (`first_detection`, `best_detection`, `first_frame`)
- `record_start_mode` (`detection_first`, `include_prerecord_first`)

Backward compatibility:
- legacy `confidence_threshold` is still accepted,
- if split thresholds are missing they are auto-filled from legacy threshold,
- old recording metadata/catalog entries still load.

## Stability and Recording Quality Improvements

Phase 3 rozszerza pipeline o mechanizmy odporności i diagnostyki bez usuwania istniejących optymalizacji.

- **RTSP watchdog i auto-recovery kamery**: worker monitoruje `last_frame_ts`, wykrywa stall strumienia oraz powtarzające się błędy/dekodowanie pustych klatek. Po przekroczeniu progów automatycznie zamyka i ponownie otwiera strumień, dzięki czemu wątek nie kończy się awarią.
- **Lepsza poprawność pipeline FPS**: dodano estymację rzeczywistego `stream_fps` ze sliding window i logowanie metryk `[perf]` co 5 sekund (stream/detect/writer FPS, kolejka, dropy).
- **Heartbeat workerów**: co 10 sekund emitowany jest status (`stream_fps`, `detect_fps`, `writer_fps`, `queue_size`, `dropped_frames`, `recording_active`, `last_detection_seconds`) do GUI.
- **Panel diagnostyczny GUI**: główne okno pokazuje dla wybranej kamery żywe parametry diagnostyczne odświeżane co sekundę.
- **Dokładniejsze metadane nagrań**: do sidecar i katalogu trafiają `event_end_ts`, `recording_duration`/`duration`, `detection_count`, `max_confidence`, `avg_confidence`, `stream_fps`.
- **Lepsze miniatury zdarzeń**: miniatura jest tworzona z obszaru detekcji (crop + margin), skalowana do 320x240 i zapisywana z overlayem detekcji; przy niepoprawnym bbox następuje fallback do pełnej klatki.
- **UX przeglądarki nagrań**: tabela zawiera nowe kolumny (`duration`, `writer_fps`, `dropped_frames`), wspiera sortowanie po czasie/pewności/czasie trwania i używa cache miniaturek w pamięci, by uniknąć wielokrotnego odczytu tych samych JPG.

### Praktyczne korzyści
- Mniej "zamrożonych" kamer po problemach sieciowych RTSP.
- Bardziej naturalna analiza jakości nagrań dzięki metadanym czasu trwania i statystykom zdarzenia.
- Czytelniejsze miniatury dla operatora (obiekt jest centralny i wyraźny).
- Szybsze diagnozowanie przeciążeń (drop frame, kolejka nagrywarki, rozjazdy FPS) bez zaglądania do kodu.

## Scalable Multi-Camera Architecture Improvements

### Preview rate limiting by camera role
- **Old behavior:** all workers emitted preview frames at similar cadence.
- **New behavior:** workers now expose `preview_role` (`main`, `thumb`, `hidden`) with separate limits (`preview_fps_main`, `preview_fps_thumb`) and optional hidden pause.
- **Benefit:** selected camera remains responsive while non-selected cameras reduce GUI pressure.
- **Files changed:** `monitoring/workers.py`, `monitoring/app.py`, `monitoring/config.py`.

### Global overload protection
- **Old behavior:** no centralized graceful degradation when many cameras were active.
- **New behavior:** app-level overload mode monitors active cameras/GUI load and applies staged degradation (thumbnail FPS down first, slight detect-FPS reduction on non-priority cameras, optional nonessential overlay reduction).
- **Benefit:** better stability under high camera count without silently disabling monitoring.
- **Files changed:** `monitoring/app.py`, `monitoring/workers.py`, `monitoring/config.py`.

### Fair inference scheduling
- **Old behavior:** inference cadence was based primarily on last-run timing and could drift.
- **New behavior:** workers keep monotonic `next_inference_due_ts` and advance it deterministically, tracking skipped cycles when behind.
- **Benefit:** more predictable per-camera detection cadence and fairer scheduling.
- **Files changed:** `monitoring/workers.py`.

### Recording priority behavior
- **Old behavior:** idle and recording cameras could be degraded similarly.
- **New behavior:** recording cameras avoid extra detection throttling, remain prioritized, and expose status (`is_recording_active`, `is_overload_degraded`, `preview_role`) in heartbeats.
- **Benefit:** recording correctness and responsiveness are preserved during overload.
- **Files changed:** `monitoring/workers.py`, `monitoring/app.py`.

### Overlay cost reduction
- **Old behavior:** overlays could be drawn even for frames not emitted to preview.
- **New behavior:** overlays are drawn only for preview frames that are emitted, while explicit event thumbnail overlay generation is preserved.
- **Benefit:** lower per-frame CPU cost.
- **Files changed:** `monitoring/workers.py`.

### Staggered camera startup
- **Old behavior:** all cameras started at once from `start_all`.
- **New behavior:** `start_cameras_staggered(camera_names)` starts cameras with short `QTimer.singleShot` delays.
- **Benefit:** reduced startup spikes in CPU/network/model init.
- **Files changed:** `monitoring/app.py`.

### Extended diagnostics metadata
- **Old behavior:** metadata covered core recording values only.
- **New behavior:** sidecar/catalog now include diagnostics fields such as `preview_role_at_start`, `overload_degraded_at_start`, `measured_capture_fps`, `effective_detect_fps`, `preview_frames_dropped`, `skipped_inference_cycles`.
- **Benefit:** richer post-event diagnostics while preserving backward compatibility.
- **Files changed:** `monitoring/workers.py`, `monitoring/recordings.py`, `monitoring/storage.py`, `monitoring/widgets/recordings_browser.py`.

### New configuration keys (defaults)
- `preview_fps_main: 15`
- `preview_fps_thumb: 3`
- `preview_pause_when_hidden: true`
- `overload_protection_enabled: true`
- `overload_camera_count_threshold: 6`
- `overload_reduce_thumb_preview_fps: 1`
- `overload_reduce_detect_fps_factor: 0.75`
- `overload_disable_nonessential_overlays: true`

## Recordings Browser Reliability and Tile View

### Reliable loading from catalog and disk
- **Old behavior:** browser primarily trusted `recordings_catalog.json`; stale or partial catalog could make recordings disappear.
- **New behavior:** loading is now three-stage: catalog read, disk fallback scan, and merge/dedup by absolute filepath with newest-first sorting.
- **Practical effect:** recordings are visible even when catalog is incomplete/corrupted or some catalog paths are stale.
- **Changed files:** `monitoring/recordings.py`, `monitoring/widgets/recordings_browser.py`.

### Default broad date filters
- **Old behavior:** default date range was limited (last day), often hiding valid older recordings.
- **New behavior:** default date bounds are derived from loaded data (`min..max` available date), with fallback to broad last-30-days range when empty.
- **Practical effect:** opening browser shows all available recordings by default.
- **Changed files:** `monitoring/recordings.py`, `monitoring/widgets/recordings_browser.py`.

### Tile/card browser
- **Old behavior:** recordings view defaulted to table/list.
- **New behavior:** default mode is **Kafelki** (tile cards with thumbnail + metadata); optional **Lista** mode remains for diagnostics.
- **Practical effect:** faster visual browsing while preserving sortable tabular diagnostics.
- **Changed files:** `monitoring/widgets/recordings_browser.py`.

### Safe thumbnail loading
- **Old behavior:** thumbnail loading path could construct/use `QPixmap` in worker context and relied on noisy prints.
- **New behavior:** worker decodes thumbnail/video frame to `QImage`; GUI thread converts to `QPixmap` and applies cache by absolute filepath.
- **Practical effect:** thread-safe, non-blocking thumbnail refresh with less UI fragility.
- **Changed files:** `monitoring/widgets/recordings_browser.py`.

### Empty states and self-healing catalog
- **Old behavior:** empty UI gave little guidance, and catalog omissions were not repaired.
- **New behavior:** browser shows explicit empty-state diagnostics (no recordings, filters hide items, missing directories, disk fallback used). Disk-only files are optionally appended to catalog safely.
- **Practical effect:** clearer operator feedback and automatic recovery of missing catalog entries.
- **Changed files:** `monitoring/recordings.py`, `monitoring/widgets/recordings_browser.py`, `monitoring/storage.py`.

### Recording start mode semantics
- **Old behavior:** `include_prerecord_first` name did not match write order in runtime.
- **New behavior:** `include_prerecord_first` now truly writes prerecord buffer frames before detection frame; `detection_first` remains default.
- **Practical effect:** metadata/config semantics match actual clip ordering.
- **Changed files:** `monitoring/workers.py`, `monitoring/tests/test_workers.py`.

Additional notes:
- Tile view is now the default browser mode.
- List view remains available as a secondary mode.
- Browser can recover and show recordings directly from disk when catalog data is stale.

## Live Camera Settings and Detailed Tooltips

- Po zapisaniu ustawień pojedynczej kamery aplikacja natychmiast zapisuje je do konfiguracji i próbuje zastosować „na żywo” bez restartu, kiedy dotyczy to pól bezpiecznych runtime (np. progi confidence, klasy, harmonogram, nagrywanie, pre/post, preview FPS).
- Dla zmian krytycznych dla strumienia (np. RTSP/source, typ źródła, model) wykonywany jest automatyczny restart **tylko tej jednej kamery** — bez restartu całej aplikacji i bez wpływu na pozostałe kamery.
- Po zapisie użytkownik dostaje czytelne potwierdzenie (status + komunikat), czy zmiany:
  - zastosowano live bez restartu,
  - wymagały i wykonały automatyczny restart tej kamery,
  - zostały tylko zapisane (kamera była zatrzymana).
- Dialog ustawień kamery zawiera teraz szczegółowe opisy (tooltip/what's this) dla wszystkich pól konfiguracyjnych; opisy pojawiają się zarówno po najechaniu etykiety, jak i kontrolki.
- Zachowana jest kompatybilność ze starszym `config.json`, w tym polem legacy `confidence_threshold` mapowanym do nowych progów rysowania/nagrywania.

## UX refinement pass (alerty, nagrania, ustawienia, HUD, stop kamery)

### 1) Szerokie miniatury alertów
- **Stare zachowanie:** panel alertów był wąski, miniatury wyglądały na małe i niemal kwadratowe.
- **Nowe zachowanie:** panel alertów został poszerzony, a miniatura alertu skaluje się proporcjonalnie do szerokości karty (16:9), prawie na pełną szerokość.
- **Efekt praktyczny:** podgląd detekcji jest czytelniejszy i szybciej można ocenić kontekst zdarzenia.
- **Pliki:** `monitoring/widgets/alerts.py`.

### 2) Przeglądarka nagrań – kafelki z działającymi JPG
- **Stare zachowanie:** nagrania ładowały się poprawnie, ale miniatury na kafelkach bywały puste.
- **Nowe zachowanie:** domyślny widok kafelków ma większe karty; loader miniatur najpierw próbuje bezpośrednio wczytać JPG (QImage), potem fallback do OpenCV i dopiero potem klatkę z MP4.
- **Efekt praktyczny:** jeśli plik JPG istnieje, miniatura jest widoczna na karcie; użytkownik nie widzi pustych pól.
- **Pliki:** `monitoring/widgets/recordings_browser.py`, `monitoring/recordings.py`.

### 3) Duże, poziome okno ustawień kamery
- **Stare zachowanie:** pionowy, stosunkowo wąski dialog utrudniał wygodną edycję wielu pól.
- **Nowe zachowanie:** dialog otwiera się na ~90% szerokości i ~86% wysokości ekranu; pola są rozłożone w trzech kolumnach z zachowaniem wszystkich dotychczasowych opcji.
- **Efekt praktyczny:** szybsza konfiguracja i lepsza czytelność przy dużej liczbie parametrów.
- **Pliki:** `monitoring/app.py`.

### 4) Scalony HUD informacji o kamerze na obrazie
- **Stare zachowanie:** status był rozdzielony między panel pod obrazem i górny overlay tekstowy.
- **Nowe zachowanie:** informacje są scalone do jednego HUD na obrazie (prawy dolny róg), biały tekst na półprzezroczystym (50%) czarnym tle.
- **Efekt praktyczny:** kluczowe dane (status, FPS, kolejka, dropy, flagi REC/DET/ERR/DEG) są w jednym miejscu bez dublowania.
- **Pliki:** `monitoring/app.py`.

### 5) Ustawienie ukrywania HUD
- **Stare zachowanie:** brak dedykowanego przełącznika dla scalonego okna informacji.
- **Nowe zachowanie:** dodano `show_camera_info_overlay` (domyślnie `true`) jako ustawienie kamery z opisem tooltip.
- **Efekt praktyczny:** użytkownik może ukryć overlay bez wyłączania detekcji i nagrywania.
- **Pliki:** `monitoring/config.py`, `monitoring/app.py`.

### 6) Stabilniejsze zatrzymywanie pojedynczej kamery
- **Stare zachowanie:** stop bywał niepewny i zostawiał niespójny stan UI.
- **Nowe zachowanie:** worker zatrzymuje się kooperacyjnie, kończy zapis nagrania, próbuje zwolnić strumień i zwraca wynik stop; UI czyści stan klatki/FPS/statusu i odświeża wskaźniki.
- **Efekt praktyczny:** kamera zatrzymuje się bez potrzeby restartu całej aplikacji.
- **Pliki:** `monitoring/workers.py`, `monitoring/app.py`.

### 7) Zapis ustawień – natychmiastowe zastosowanie + potwierdzenie
- **Stare zachowanie:** część zmian była stosowana, ale UX po zapisie był niespójny.
- **Nowe zachowanie:** zachowano podział na zmiany live/restart i komunikaty potwierdzające po zapisie.
- **Efekt praktyczny:** użytkownik otrzymuje jasny komunikat czy zmiany weszły bez restartu, po auto-restarcie kamery, czy tylko zapisano konfigurację.
- **Pliki:** `monitoring/app.py`, `monitoring/workers.py`.

## Aktualizacje UI/miniatur (regresje)

### Alerty: miniatury sceny zamiast cropów obiektu
- Lista alertów została rozdzielona semantycznie od miniatur posterów nagrań.
- Alerty preferują miniatury sceny (`alert_thumb` / `*.alert.jpg` / `*.scene.jpg`), a dopiero na końcu starsze pole `thumb`.
- Widżet alertu zachowuje format prostokątny (16:9), więc po restarcie aplikacji miniatury nie zapadają się do kwadratu i nie pokazują wyłącznie przybliżonego obiektu.

### Overlay informacji kamery bezpośrednio na obrazie
- HUD kamery (status, FPS, queue/drop i flagi REC/DET/ERR/DEG) jest rysowany bezpośrednio na podglądzie.
- Pozycjonowanie overlayu liczone jest względem faktycznie widocznego prostokąta obrazu po letterboxie, nie względem całego widgetu.
- Overlay jest osadzony w lewym-dolnym rogu obrazu, z półprzezroczystym tłem dla czytelności.

### Przeglądarka nagrań: kafelki miniatur JPG
- Pipeline miniatur kafelków został dopięty do końca: worker → sygnał → mapowanie po absolutnej ścieżce → render w głównym wątku.
- Wczytywanie preferuje jawny `thumb_path`, potem pliki sidecar JPG/JPEG, a na końcu fallback z pierwszej klatki MP4.
- Kafelki zawsze kończą w stanie końcowym: miniatura albo placeholder „brak miniatury” (bez wiecznego „trwa wczytywanie”).

### Rozdział ról miniatur
- `alert_thumb` reprezentuje podgląd sceny dla listy alertów.
- `thumb` pozostaje posterem nagrania używanym przez przeglądarkę nagrań.
- Zachowana kompatybilność ze starszymi metadanymi i historią.

## Stability Logging and Error Diagnostics

Aplikacja używa teraz centralnego mostu logowania (`AppLogBridge`), który bezpiecznie przekazuje wpisy z wątków workerów i kodu UI do panelu **Logi**.

Najważniejsze usprawnienia:
- panel Logi obsługuje rozszerzone grupy (m.in. `warning`, `worker`, `ui`, `browser`, `performance`) oraz bogatsze pola wpisów (`source`, `level`, `details`, `traceback`),
- ostrzeżenia i błędy z aplikacji, workerów i przeglądarki nagrań trafiają do panelu zamiast znikać w konsoli,
- globalne hooki wyjątków (`sys.excepthook`, `threading.excepthook`) logują traceback przed zakończeniem procesu,
- handler komunikatów Qt (`qInstallMessageHandler`) zapisuje ostrzeżenia/błędy frameworka do panelu,
- heartbeat watchdog wykrywa potencjalne zawieszenia workerów (brak heartbeat),
- GUI watchdog wykrywa podejrzane stany (np. brak odświeżania klatek dla aktywnej kamery),
- logowanie cyklu życia kamer (start/stop/restart po zmianie ustawień) poprawia diagnostykę freeze’ów i restartów.

Korzyść praktyczna: gdy pojawia się problem runtime (błąd modelu, timeout stopu workera, błąd miniatury lub wyjątek), użytkownik widzi kontekst od razu w panelu **Logi**, razem z kluczowymi szczegółami diagnostycznymi.

## Runtime stability diagnostics (Logi panel)

Ostatnie zmiany wzmacniają diagnostykę runtime bez cichego zawieszania:

- **Overload anti-thrashing**: overload nie aktywuje się poniżej minimalnej liczby kamer (`overload_min_camera_count`, domyślnie 2) oraz używa debounce/histerezy (`overload_enter_debounce_seconds`, `overload_exit_debounce_seconds`), więc tryb nie „klapie” dla 1 kamery.
- **Lifecycle stop/start/restart**: logowane są teraz sekwencje `camera start requested`, `camera stop requested`, `stop_camera completed`, `worker stop begin/success/timeout`, `automatic camera restart due to settings change`, `camera restart success/failure`.
- **Heartbeat watchdog**: timeout heartbeatu workerów (domyślnie 15s) generuje ostrzeżenie `worker heartbeat timeout` z wiekiem ostatniego heartbeat; po powrocie heartbeat logowana jest pojedyncza informacja o recovery.
- **Browser/thumbnail diagnostics**: logi obejmują brakujące JPG, użycie fallbacku z pierwszej klatki MP4, błąd fallbacku, wyjątek w workerze miniatur oraz błąd aplikacji miniatury do widżetu.
- **Metrics explainability**: okresowe logi metryk workerów zawierają `source_fps`, `stream_fps`, `detect_fps_target`, rolę podglądu, liczbę emitowanych/odrzucanych preview i `skipped_inference_cycles` wraz z powodem (`overload`, `config-throttle`, `camera/runtime`).
- **Detection/recording transitions**: logowane są przejścia stanu (`detection became active`, `detection ended`, `recording session started/finalized`) oraz ostrzeżenia o pełnej kolejce i zrzucanych klatkach.
- **Wyjątki z traceback**: ścieżki krytyczne (worker stream exception, browser refresh, thumbnail pipeline, restart kamery, launcher shutdown) logują traceback zamiast cichego tłumienia błędów.

Wszystkie powyższe wpisy są publikowane przez istniejący mechanizm logów i są widoczne w panelu **Logi**.

## Poprawki: miniatury nagrań i HUD kamery (2026)

### Miniatury w widoku kafelków nagrań
- **Stare zachowanie:** kafelki potrafiły pozostać w stanie „Ładowanie miniatury...”, mimo że metadane i lista nagrań były poprawnie wczytane.
- **Nowe zachowanie:** pipeline miniatur zawsze kończy się jednym z trzech stanów: sukces (JPG), fallback (klatka z MP4), albo porażka („Brak miniatury”).
- **Priorytet źródeł miniatur:**
  1. jawna miniatura z metadanych (`thumb` / `recording_thumb`),
  2. `*.jpg` obok nagrania,
  3. dodatkowe znane warianty nazw JPG,
  4. fallback: pierwsza klatka z pliku MP4.
- **Efekt praktyczny:** kafelki nie pozostają „zawieszone” na ładowaniu, a przy braku JPG użytkownik dostaje czytelną miniaturę zapasową lub komunikat o braku miniatury.

### HUD informacji kamery na obrazie
- **Stare zachowanie:** główne informacje o kamerze bywały rozproszone i nie odświeżały się od razu po zmianie ustawień.
- **Nowe zachowanie:** główny HUD jest rysowany bezpośrednio na obrazie kamery, w **prawym dolnym rogu rzeczywistego widocznego obszaru obrazu** (z uwzględnieniem letterbox), jako pionowa kolumna linii.
- **Język HUD:** etykiety i treści są po polsku (m.in. Kamera, Status, FPS, Kolejka, Pominięte klatki, Tryb, Połączenie, Błąd).
- **Ustawienia runtime:** po zapisaniu zmian HUD odświeża się natychmiast; ustawienia „live-safe” są stosowane bez restartu, a ustawienia wymagające restartu uruchamiają automatyczny restart tylko tej kamery.
- **Widoczność HUD:** flaga `show_camera_info_overlay` (domyślnie `true`) działa od razu po zmianie.

### Diagnostyka (panel Logi)
- Dodano bardziej jednoznaczne logi dla ładowania miniaturek (`source=recordings-browser`) oraz odświeżania HUD i zastosowania ustawień (`source=camera-hud` / `source=settings`).
- Logowane są przejścia i błędy, bez spamowania każdej klatki.
