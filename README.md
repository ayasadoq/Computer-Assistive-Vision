# Voice-guided vision assistant

A Python/OpenCV application for exploring a live camera feed with YOLOv8 object
detection, approximate spatial cues, and optional speech output. It includes a
desktop CLI and a browser-based Streamlit demo.


## Features

- YOLOv8 pretrained object detection from a webcam or video file.
- Each detection reports class, confidence, left/center/right position, near/far
  range, and a rough distance estimate.
- Announcements prioritize the nearest detected object and throttle repeats.
- Optional offline TTS via `pyttsx3` and voice scene queries via
  `SpeechRecognition`.
- Streamlit/WebRTC live demo with bounding boxes, current scene narration, and a
  detection table.
- YAML configuration for model, confidence, camera calibration, TTS language,
  recognition language, and cooldown.

## Setup

Python 3.10 or later is recommended. Create and activate a virtual environment,
then install dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The first run downloads the Ultralytics `yolov8n.pt` weights if they are not
already available. The pretrained weights are not bundled in this project.
SpeechRecognition's microphone extra installs PyAudio; on Windows, installation
may require a compatible PyAudio wheel. The CLI can still run without speech
features using `--no-voice`.

## CLI usage

Run with the default webcam:

```powershell
python main.py
```

Choose another camera or a video file:

```powershell
python main.py --video 1
python main.py --video .\sample.mp4 --no-voice
```

Press **Q** in the video window to exit. With voice enabled, ask “what's in
front of me?” or “what do you see?” for a scene summary. Say “stop” or “quit”
to exit. Speech recognition currently uses Google's recognition service and
therefore needs an internet connection; TTS uses the installed system voices.
The configured language is used for generated phrases and the closest matching
installed speech voice, if one is available.

## Streamlit demo

```powershell
streamlit run streamlit_app.py
```

Allow camera access in the browser, then start the video component. WebRTC may
need STUN/TURN connectivity depending on the browser and network. The browser
demo displays narration as text and does not play audio.

## Distance calibration

The estimator uses the pinhole-camera approximation
`distance = object_height_m * focal_length_px / bounding_box_height_px`.
`focal_length_px` and class-specific heights in [config.yaml](config.yaml) are
examples, not universal constants.

To calibrate, place a known object at a measured distance, measure its pixel
height in the camera image, and set
`focal_length_px = distance_m * box_height_px / object_height_m`. Repeat with
the camera resolution and zoom you intend to use. Add real-world heights for
frequently detected classes under `spatial.object_heights_m`; unmatched classes
use `default_object_height_m`. The `near`/`far` label boundary is controlled by
`near_distance_m`.

## Configuration

Edit [config.yaml](config.yaml):

- `detector.model_path`: Ultralytics model name/weights path (default
  `yolov8n.pt`).
- `detector.confidence`: minimum detection confidence, from 0 to 1.
- `spatial.*`: distance calibration and near/far boundary.
- `speech.language`: announcement language (`en` or `fr`).
- `speech.recognition_language`: SpeechRecognition locale, e.g. `en-US` or
  `fr-FR`.
- `speech.announcement_cooldown_seconds`: minimum time before repeating the
  same class/position/range announcement.

The detector also validates camera frame availability and reports model,
microphone, and speech-service setup errors rather than silently treating them
as successful operation.

## Project layout

```text
detector.py       YOLO inference and detection data model
spatial.py        left/center/right and calibrated distance estimates
narrator.py       concise narration and announcement cooldown
voice.py          background TTS and voice-command listener
main.py           webcam/video CLI pipeline
streamlit_app.py  browser-based live demonstration
config.yaml       model, calibration, and speech settings
tests/            focused spatial and narration unit tests
```

Run the unit tests with:

```powershell
pytest
```
