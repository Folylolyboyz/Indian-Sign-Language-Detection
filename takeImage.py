import cv2
import os
import mediapipe as mp
import numpy as np

SAVE_FOLDER = "Dataset/Z"
MARGIN = 10
FRAMES_TO_SAVE = 10
BUFFER_SIZE = 5
FRAME_BUFFERS = {}
SQUARE_RATIO = 1.25

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def setup_camera(width=1280, height=720):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera Resolution: {frame_width}x{frame_height}")
    return cap, frame_width, frame_height

def get_square_position(frame_width, frame_height):
    square_size = int(min(frame_width, frame_height) / SQUARE_RATIO)
    return frame_width - square_size - MARGIN, MARGIN, square_size

def draw_square(frame, square_x, square_y, square_size):
    cv2.rectangle(frame, (square_x, square_y), 
                  (square_x + square_size, square_y + square_size), 
                  (0, 255, 0), 2)

def smooth_landmarks(hand_id, current_landmarks):
    global FRAME_BUFFERS
    if hand_id not in FRAME_BUFFERS:
        FRAME_BUFFERS[hand_id] = []
    
    buffer = FRAME_BUFFERS[hand_id]
    if len(buffer) >= BUFFER_SIZE:
        buffer.pop(0)
    buffer.append(np.array(current_landmarks))
    
    if len(buffer) < BUFFER_SIZE:
        return np.array(current_landmarks)
    
    avg_landmarks = np.mean(buffer, axis=0)
    return avg_landmarks.tolist()

def detect_hands(frame, hands, square_x, square_y, square_size):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_id, hand_landmarks in enumerate(results.multi_hand_landmarks):
            landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
            smoothed_landmarks = smooth_landmarks(hand_id, landmarks)
            
            for i, (x, y) in enumerate(smoothed_landmarks):
                h, w, _ = frame.shape
                cx, cy = int(x * w), int(y * h)
                hand_landmarks.landmark[i].x = x
                hand_landmarks.landmark[i].y = y
                
                if square_x <= cx <= square_x + square_size and square_y <= cy <= square_y + square_size:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    break

def capture_frames(cap, square_x, square_y, square_size):
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
    
    for _ in range(FRAMES_TO_SAVE):
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        frame = cv2.flip(frame, 1)
        cropped_image = frame[square_y:square_y + square_size, square_x:square_x + square_size]
        image_name = os.path.join(SAVE_FOLDER, f"captured_{len(os.listdir(SAVE_FOLDER)) + 1}.png")
        cv2.imwrite(image_name, cropped_image)
        print(f"Saved: {image_name}")

def process_frame(cap, hands, square_x, square_y, square_size):
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        return None
    frame = cv2.flip(frame, 1)
    draw_square(frame, square_x, square_y, square_size)
    detect_hands(frame, hands, square_x, square_y, square_size)
    return frame

def main():
    cap, frame_width, frame_height = setup_camera()
    square_x, square_y, square_size = get_square_position(frame_width, frame_height)
    
    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8, max_num_hands=2) as hands:
        while True:
            frame = process_frame(cap, hands, square_x, square_y, square_size)
            if frame is None:
                break
            cv2.imshow("Camera Feed", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                capture_frames(cap, square_x, square_y, square_size)
            elif key == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()