import logging
import os
import signal
from enum import StrEnum
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread

from Cocoa import (
    NSApplication,
    NSBezierPath,
    NSColor,
    NSImage,
    NSImageOnly,
    NSStatusBar,
    NSVariableStatusItemLength,
    NSPasteboard,
)
from Foundation import NSObject
from PyObjCTools import AppHelper
from parakeet_mlx import from_pretrained
import pyaudio
import wave


fmt = "%(levelname)s:%(asctime).19s: %(message)s"
logging.basicConfig(format=fmt)
logger = logging.getLogger("rosella")
logger.setLevel("INFO")
if os.environ.get("ROSELLA_DEBUG"):
    logger.setLevel("DEBUG")
    logger.debug("Log level set to debug")


CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1  # if sys.platform == 'darwin' else 2
RATE = 44100


class State(StrEnum):
    ready = "ready"
    processing = "processing"
    recording = "recording"


class StatusBarDelegate(NSObject):
    def applicationDidFinishLaunching_(self, notification):
        status_bar = NSStatusBar.systemStatusBar()
        self.status_item = status_bar.statusItemWithLength_(NSVariableStatusItemLength)

        button = self.status_item.button()
        if button is None:
            raise RuntimeError("Unable to create status bar button")

        # Green
        self.ready_icon = self._create_circle_icon(
            diameter=10,
            color=NSColor.colorWithCalibratedRed_green_blue_alpha_(0.15, 0.9, 0.2, 1.0),
        )
        # Orange
        self.processing_icon = self._create_circle_icon(
            diameter=10,
            color=NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.55, 0.0, 1.0),
        )
        # Red
        self.recording_icon = self._create_circle_icon(
            diameter=10,
            color=NSColor.colorWithCalibratedRed_green_blue_alpha_(0.9, 0.15, 0.2, 1.0),
        )

        button.setImage_(self.ready_icon)
        button.setImagePosition_(NSImageOnly)  # Avoid text next to the icon

        # Init scribe object
        self.scribe = Scribe()

        # init state and bind button
        self.state = State.ready
        self.processing_timer = None
        button.setTarget_(self)
        button.setAction_("statusItemClicked:")

    def _create_circle_icon(self, diameter, color):
        image = NSImage.alloc().initWithSize_((diameter, diameter))
        image.lockFocus()
        try:
            color.setFill()
            path = NSBezierPath.bezierPathWithOvalInRect_(
                ((0, 0), (diameter, diameter)),
            )
            path.fill()
        finally:
            image.unlockFocus()

        image.setTemplate_(False)
        return image

    def statusItemClicked_(self, sender):
        self._update_state()

    def _update_state(self):
        button = self.status_item.button()
        if button is None:
            return

        if self.state == State.ready:
            self.state = State.processing
            button.setImage_(self.processing_icon)
            AppHelper.callAfter(self._start_recording_flow)
        elif self.state == State.recording:
            self.state = State.processing
            button.setImage_(self.processing_icon)
            AppHelper.callAfter(self._finish_recording_flow)

    def _start_recording_flow(self):
        button = self.status_item.button()
        if button is None:
            return

        if self.scribe.model is None:
            self.scribe.init_model()

        button.setImage_(self.recording_icon)
        self.state = State.recording
        self.scribe.start()

    def _finish_recording_flow(self):
        button = self.status_item.button()
        if button is None:
            return

        result = self.scribe.stop()
        if result is not None:
            self._paste_to_clipboard(result)

        self.state = State.ready
        button.setImage_(self.ready_icon)

    def _paste_to_clipboard(self, text):
        pasteboard = NSPasteboard.generalPasteboard()
        pasteboard.clearContents()
        pasteboard.setString_forType_(text, "public.utf8-plain-text")


class Scribe:

    def __init__(self):
        self.model = None
        self.record_thread = None
        self.result = None

    def init_model(self):
        logger.debug("Loading model")
        self.model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")

    def start(self):
        self.capture = True
        self.record_thread = Thread(target=self.record)
        self.record_thread.start()

    def record(self):
        logger.debug("Recording")
        p = pyaudio.PyAudio()
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "output.wav"
            with wave.open(str(temp_path), 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True)
                while self.capture:
                    wf.writeframes(stream.read(CHUNK))
                stream.close()
                p.terminate()

            if temp_path.stat().st_size > CHUNK:
                logger.debug("Transcribing")
                result = self.model.transcribe(temp_path)
                self.result = result.text
                logger.info("Result: %s", self.result)
            else:
                logger.info("Sample too short")
                self.result = None

    def stop(self):
        self.capture = False
        self.record_thread.join()
        return self.result


def handle_sigint(signum, frame):
    AppHelper.stopEventLoop()


def main():
    # handle signal interrupt
    signal.signal(signal.SIGINT, handle_sigint)
    # Instantiate and start app
    app = NSApplication.sharedApplication()
    delegate = StatusBarDelegate.alloc().init()
    app.setDelegate_(delegate)
    AppHelper.installMachInterrupt()
    AppHelper.runEventLoop()


if __name__ == "__main__":
    main()
