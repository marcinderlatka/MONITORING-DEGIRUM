# MONITORING-RTSP

Aplikacja PyQt5 do monitoringu strumieni RTSP z wykorzystaniem modeli DeGirum. 
Program umożliwia podgląd obrazu z wielu kamer, wykrywanie obiektów oraz nagrywanie fragmentów wideo z pre/post‑recordingiem.

## Wymagania
* Python 3.8+
* PyQt5
* OpenCV
* biblioteki DeGirum (`degirum`, `degirum_tools`)

## Uruchomienie
```
python app_01.py --windowed
```
Opcja `--windowed` uruchamia aplikację w oknie, brak opcji – w trybie pełnoekranowym.

## Plik `config.json`
Plik konfiguracyjny zawiera wyłącznie listę kamer. Każda kamera przechowuje pełny zestaw parametrów:

* `name` – nazwa kamery
* `rtsp` – adres strumienia RTSP
* `model` – nazwa modelu DeGirum
* `record_path` – katalog nagrań
* `confidence_threshold` – próg pewności wykrycia
* `fps` – częstotliwość analizy
* `draw_overlays` – rysowanie nakładek
* `enable_detection` – włączanie detekcji
* `enable_recording` – włączanie nagrywania
* `detection_hours` – przedziały godzinowe `HH:MM-HH:MM` oddzielone `;`
* `visible_classes` – lista klas rysowanych na podglądzie
* `record_classes` – lista klas wyzwalających nagrywanie
* `pre_seconds`, `post_seconds` – długość bufora przed i po zdarzeniu

Przykładowa kamera:
```json
{
  "name": "Kamera 1",
  "rtsp": "rtsp://admin:IBLTSQ@192.168.8.165:554",
  "model": "yolov5nu_silu_coco--640x640_float_tflite_multidevice_1",
  "fps": 1,
  "confidence_threshold": 0.5,
  "draw_overlays": true,
  "enable_detection": true,
  "enable_recording": true,
  "detection_hours": "00:00-23:59",
  "visible_classes": ["person", "car", "dog"],
  "record_classes": ["person", "car"],
  "record_path": "./nagrania",
  "pre_seconds": 5,
  "post_seconds": 5
}
```

## Ustawienia kamery
Z listy kamer wybierz pozycję prawym przyciskiem i wybierz **Ustawienia…**. W oknie dialogowym można:
* zmienić wszystkie parametry kamery,
* przetestować połączenie RTSP,
* wskazać katalog nagrań,
* ustawić czasy pre/post‑recordingu.
Po zatwierdzeniu ustawienia są zapisywane w `config.json`, a wątek kamery zostaje ponownie uruchomiony.

## Nagrania
Nagrania zapisywane są w podkatalogach `record_path/<nazwa_kamery>`. Przeglądanie oraz odtwarzanie plików umożliwia pozycja **Nagrania → Przeglądaj nagrania** w menu głównym.

## Widżet alertów
Lista alertów wyposażona jest w przyciski:

* **Wczytaj ponownie** – odświeża historię z pliku,
* **Eksport do CSV** – zapisuje historię alertów do pliku CSV,
* **Wyczyść pamięć** – usuwa wszystkie zapamiętane alerty.
