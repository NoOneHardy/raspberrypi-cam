import os
import threading
import time
from collections.abc import Callable

import cv2
import mediapipe as mp
import spotipy
from dotenv import load_dotenv
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.gesture_recognizer import GestureRecognizerOptions, GestureRecognizer
from mediapipe.tasks.python.vision.gesture_recognizer_result import GestureRecognizerResult
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections
from spotipy import SpotifyOAuth

load_dotenv()


def handle_closed():
    print('Closed')

    client_id = os.environ['CLIENT_ID']
    client_secret = os.environ['CLIENT_SECRET']

    spotify = spotipy.Spotify(
        auth_manager=SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri="https://wm-dev.no1hardy.ch",
            scope="user-modify-playback-state user-read-playback-state",
        )
    )

    device_id: str
    playback = spotify.current_playback()
    if playback is None:
        devices = spotify.devices()['devices']
        if len(devices) == 0:
            raise Exception('No devices found')
        device_id = devices[0]['id']
    else:
        device_id = playback['device']['id']

    if spotify.currently_playing() is not None and spotify.currently_playing()['is_playing']:
        spotify.pause_playback()
    else:
        spotify.start_playback(device_id)

    cam_state.close()


class CamState:
    _instance = None

    _ready = False

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
        return cls._instance

    def start_timer(self):
        time.sleep(5)
        if self.ready():
            print('Timed out')
            self.close()

    def ready(self):
        return self._ready

    def open(self):
        self._ready = True
        threading.Thread(target=self.start_timer).start()
        print('Listening...')

    def close(self):
        self._ready = False


cam_state = CamState()


def listen(result: GestureRecognizerResult):
    if len(result.gestures) < 1:
        return

    gesture = result.gestures[0]
    category = gesture[0]

    if category.category_name == 'Victory' and not cam_state.ready():
        cam_state.open()
        return

    if category.category_name == 'Closed_Fist' and cam_state.ready():
        handle_closed()


def handle_image(result: GestureRecognizerResult, image: mp.Image, cap: cv2.VideoCapture):
    bgr_image = cv2.cvtColor(image.numpy_view(), cv2.COLOR_RGB2BGR)
    for hand in result.hand_landmarks:
        drawing_utils.draw_landmarks(bgr_image, hand, [
            HandLandmarksConnections.Connection(4, 8),
            HandLandmarksConnections.Connection(8, 12),
            HandLandmarksConnections.Connection(12, 16),
            HandLandmarksConnections.Connection(16, 20)
        ])

    listen(result)

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    cv2.imshow('Result', rgb_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()


def init(cap: cv2.VideoCapture) -> GestureRecognizer:
    base_options = BaseOptions(model_asset_path='gesture_recognizer.task')
    rec_options = GestureRecognizerOptions(
        base_options=base_options,
        num_hands=1,
        running_mode=VisionTaskRunningMode.LIVE_STREAM,
        result_callback=lambda result, image, _: handle_image(result, image, cap))
    return GestureRecognizer.create_from_options(rec_options)


def handle_stream(
        rec: GestureRecognizer,
        cap: cv2.VideoCapture,
        frame: cv2.Mat
):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    try:
        rec.recognize_async(image=mp_image, timestamp_ms=int(cap.get(cv2.CAP_PROP_POS_MSEC)))
    except ValueError as e:
        if 'timestamp_ms' in str(e):
            pass


def main():
    capture_video_stream(lambda cap: init(cap), handle_stream)


def capture_video_stream(
        init_cb: Callable[[cv2.VideoCapture], GestureRecognizer],
        handle_stream_cb: Callable[[GestureRecognizer, cv2.VideoCapture, cv2.Mat], None]
):
    cap: cv2.VideoCapture = cv2.VideoCapture(0)
    rec: GestureRecognizer = init_cb(cap)

    while cap.isOpened():
        frame: cv2.Mat
        is_success, frame = cap.read()
        if not is_success:
            print('Ignoring empty camera frame.')
            continue

        handle_stream_cb(rec, cap, frame)
        # time.sleep(.5)


if __name__ == '__main__':
    main()
