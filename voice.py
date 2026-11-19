"""Background text-to-speech and voice-command input."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable


class SpeechOutput:
    """Run offline pyttsx3 speech in a worker to keep video capture responsive."""

    def __init__(self, language: str = "en") -> None:
        self.language = language
        self._messages: queue.Queue[str | None] = queue.Queue(maxsize=1)
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def speak(self, text: str) -> None:
        if not text:
            return
        try:
            self._messages.put_nowait(text)
        except queue.Full:
            try:
                self._messages.get_nowait()
            except queue.Empty:
                pass
            self._messages.put_nowait(text)

    def close(self) -> None:
        try:
            self._messages.put_nowait(None)
        except queue.Full:
            try:
                self._messages.get_nowait()
            except queue.Empty:
                pass
            self._messages.put_nowait(None)
        self._worker.join(timeout=3)

    def _run(self) -> None:
        try:
            import pyttsx3
        except ImportError as exc:
            print(f"TTS unavailable: install pyttsx3 ({exc})")
            return
        engine = pyttsx3.init()
        language_prefix = self.language.lower().split("-")[0]
        for voice in engine.getProperty("voices"):
            voice_languages = [
                value.decode("utf-8", errors="ignore").lower()
                if isinstance(value, bytes)
                else str(value).lower()
                for value in getattr(voice, "languages", [])
            ]
            if any(language_prefix in value for value in voice_languages) or (
                language_prefix in voice.id.lower()
            ):
                engine.setProperty("voice", voice.id)
                break
        while True:
            message = self._messages.get()
            if message is None:
                break
            engine.say(message)
            engine.runAndWait()
        engine.stop()


class VoiceCommandListener:
    """Listen for scene queries and exit commands on a background thread."""

    def __init__(
        self,
        on_query: Callable[[], None],
        on_quit: Callable[[], None],
        language: str = "en-US",
    ) -> None:
        self.on_query = on_query
        self.on_quit = on_quit
        self.language = language
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._listen, daemon=True)

    def start(self) -> None:
        self._worker.start()

    def close(self) -> None:
        self._stop.set()
        self._worker.join(timeout=2)

    def _listen(self) -> None:
        try:
            import speech_recognition as sr
        except ImportError as exc:
            print(f"Voice input unavailable: install SpeechRecognition ({exc})")
            return

        recognizer = sr.Recognizer()
        try:
            microphone = sr.Microphone()
        except (OSError, AttributeError) as exc:
            print(f"Voice input unavailable: microphone initialization failed ({exc})")
            return

        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
        while not self._stop.is_set():
            try:
                with microphone as source:
                    audio = recognizer.listen(source, timeout=1, phrase_time_limit=4)
                phrase = recognizer.recognize_google(
                    audio, language=self.language
                ).lower()
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                continue
            except sr.RequestError as exc:
                print(f"Speech recognition service error: {exc}")
                continue
            except OSError as exc:
                print(f"Voice input error: {exc}")
                return

            if any(word in phrase for word in ("quit", "exit", "stop", "quitter", "arrêter")):
                self.on_quit()
                return
            if any(
                phrase_part in phrase
                for phrase_part in (
                    "what do you see",
                    "what's in front",
                    "describe the scene",
                    "what is in front",
                    "que vois-tu",
                    "qu'y a-t-il devant",
                    "décris la scène",
                )
            ):
                self.on_query()
