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

### Odtwarzacz nagrań
Podwójne kliknięcie nagrania otwiera odtwarzacz z kontrolkami transportu, przełączaniem między plikami, zrzutem klatki i trybem pełnoekranowym.

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
