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
Performance metrics are aggregated and logged roughly every 5 seconds for capture, inference, overlay draw, emit/enqueue, and recorder queue size.

Impact:
Provides low-overhead visibility for profiling and diagnosing bottlenecks in production-like runs.

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
