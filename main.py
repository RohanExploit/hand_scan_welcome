import os
import cv2
import mediapipe as mp
import numpy as np
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, 'hand_template.png')
print("Mencoba membuka:", file_path)
template = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
if template is None:
    print("Gagal membuka file:", file_path)
    exit(1)
else:
    print("Berhasil membuka file:", file_path)
template_h, template_w = template.shape[:2]

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

# Mediapipe untuk deteksi tangan
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)

# Fungsi overlay gambar transparan
def overlay_image_alpha(img, img_overlay, pos):
    x, y = pos
    h, w = img_overlay.shape[0], img_overlay.shape[1]

    # Batas frame
    if x < 0 or y < 0 or x + w > img.shape[1] or y + h > img.shape[0]:
        return

    if img_overlay.shape[2] == 4:
        alpha_overlay = img_overlay[:, :, 3] / 255.0
        alpha_background = 1.0 - alpha_overlay
        for c in range(0, 3):
            img[y:y+h, x:x+w, c] = (alpha_overlay * img_overlay[:, :, c] +
                                    alpha_background * img[y:y+h, x:x+w, c])
    else:
        img[y:y+h, x:x+w] = img_overlay

video_playing = False
hand_in_template_since = None
required_hold_time = 4  # detik tangan harus stay di template

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    center_x, center_y = w // 2 - template_w // 2, h // 2 - template_h // 2

    overlay_image_alpha(frame, template, (center_x, center_y))

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    play_video = False
    hand_in_template = False

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
            y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)

            if (center_x < x_min < center_x + template_w and
                center_y < y_min < center_y + template_h and
                center_x < x_max < center_x + template_w and
                center_y < y_max < center_y + template_h):
                hand_in_template = True

    # Timer logic
    if hand_in_template:
        if hand_in_template_since is None:
            hand_in_template_since = time.time()
        elif time.time() - hand_in_template_since >= required_hold_time:
            play_video = True
    else:
        hand_in_template_since = None

    cv2.imshow('Hand Scanner', frame)

    if play_video and not video_playing:
        video_playing = True
        hand_in_template_since = None  # reset timer
        video_path = os.path.join(BASE_DIR, 'vid.mp4')
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            print("Gagal membuka video:", video_path)
            video_playing = False
        else:
            while video.isOpened():
                ret_vid, frame_vid = video.read()
                if not ret_vid:
                    break
                cv2.imshow('Hand Scanner', frame_vid)
                if cv2.waitKey(30) & 0xFF == 27:
                    break
            video.release()
            video_playing = False

    if cv2.waitKey(30) & 0xFF == 27:
        cap.release()
        cv2.destroyAllWindows()
        exit(0)

cap.release()
cv2.destroyAllWindows()

