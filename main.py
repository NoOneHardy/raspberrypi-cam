import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker, HandLandmarkerOptions, HandLandmarkerResult, \
    HandLandmarksConnections

cap: cv2.VideoCapture = cv2.VideoCapture(0)

def handle_image(result: HandLandmarkerResult, image: mp.Image, _):
    bgr_image = cv2.cvtColor(image.numpy_view(), cv2.COLOR_RGB2BGR)
    for hand in result.hand_landmarks:
        drawing_utils.draw_landmarks(bgr_image, hand, [
            HandLandmarksConnections.Connection(4, 8),
            HandLandmarksConnections.Connection(8, 12),
            HandLandmarksConnections.Connection(12, 16),
            HandLandmarksConnections.Connection(16, 20)
        ])

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    cv2.imshow('Result', rgb_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()


def main():
    lm_options = HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path='hand_landmarker.task'),
        num_hands=1,
        running_mode=VisionTaskRunningMode.LIVE_STREAM,
        result_callback=handle_image)
    lm = HandLandmarker.create_from_options(lm_options)

    while cap.isOpened():
        frame: cv2.Mat
        isSuccess, frame = cap.read()
        if not isSuccess:
            print('Ignoring empty camera frame.')
            continue

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        try:
            lm.detect_async(image=mp_image, timestamp_ms=int(cap.get(cv2.CAP_PROP_POS_MSEC)))
        except ValueError as e:
            if 'timestamp_ms' in str(e):
                pass

if __name__ == '__main__':
    main()