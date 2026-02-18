import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import mediapipe as mp
import winsound
from statistics import mean, stdev
import logging
import time
import threading

# Logging sozlamalari
logging.basicConfig(
    filename="sleep_detection.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Dlib yuz aniqlagich va prediktor (bosh burchagi uchun)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/User/Documents/models/shape_predictor_68_face_landmarks.dat")

# MediaPipe Face Mesh sozlamalari
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Kamera ichki parametrlari (taxminiy)
camera_matrix = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]], dtype="double")
dist_coeffs = np.zeros((4, 1))

# MediaPipe ko'z nuqtalari indekslari (chap va o'ng ko'z uchun)
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]  # Chap ko'z
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]  # O'ng ko'z

# EAR hisoblash funksiyasi (MediaPipe nuqtalari uchun)
def eye_aspect_ratio(eye_points, landmarks, image_shape):
    h, w = image_shape[:2]
    eye = []
    for idx in eye_points:
        lm = landmarks[idx]
        eye.append([lm.x * w, lm.y * h])
    eye = np.array(eye)
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# SOS signalini chiqarish funksiyasi (alohida oqimda)
def play_sos_signal_async(duration_seconds):
    threading.Thread(target=play_sos_signal, args=(duration_seconds,)).start()

def play_sos_signal(duration_seconds):
    dot_duration = 200  # ms
    dash_duration = 600  # ms
    pause_duration = 200  # ms
    letter_pause = 600  # ms

    sos_pattern = [
        (dot_duration, 1000), (pause_duration, 0),  # .
        (dot_duration, 1000), (pause_duration, 0),  # .
        (dot_duration, 1000), (letter_pause, 0),  # .
        (dash_duration, 1000), (pause_duration, 0),  # -
        (dash_duration, 1000), (pause_duration, 0),  # -
        (dash_duration, 1000), (letter_pause, 0),  # -
        (dot_duration, 1000), (pause_duration, 0),  # .
        (dot_duration, 1000), (pause_duration, 0),  # .
        (dot_duration, 1000), (letter_pause, 0)  # .
    ]

    total_duration_ms = duration_seconds * 1000
    start_time = time.time()

    while time.time() - start_time < duration_seconds:
        for duration, freq in sos_pattern:
            if time.time() - start_time >= duration_seconds:
                break
            if freq > 0:
                try:
                    winsound.Beep(freq, duration)
                except Exception as e:
                    print(f"Ovozli signalni chiqarishda xatolik: {e}")
                    logging.error(f"Ovozli signalni chiqarishda xatolik: {e}")
            else:
                time.sleep(duration / 1000)
        print("SOS signalining bir sikli tugadi")  # Tekshirish uchun
        logging.info("SOS signalining bir sikli tugadi")  # Tekshirish uchun

# Ovozli signalni boshqa oqimda ishlatish funksiyasi
def play_sound_async(freq, duration):
    try:
        threading.Thread(target=winsound.Beep, args=(freq, duration)).start()
    except Exception as e:
        print(f"Ovozli signalni chiqarishda xatolik: {e}")
        logging.error(f"Ovozli signalni chiqarishda xatolik: {e}")

# Chegaralar
EAR_THRESHOLD = 0.25
CONSECUTIVE_FRAMES = 120  # 4 soniya (30 kadr/sekund)
BLINK_FRAMES = 5
CALIBRATION_FRAMES = 90  # 3 soniya (30 kadr/sekund)
UPDATE_INTERVAL = 300

# O'zgaruvchilar
blink_counter = 0
alert = False
alert_triggered = False
calibration_stage = 0  # Kalibrlash bosqichi (0: boshlang'ich, 1: ochiq ko'zlar, 2: yopiq ko'zlar, 3: ishga tushirildi)
frame_counter = 0
ear_values_open = []
ear_values_closed = []
ear_history = []
blink_durations = []
calibration_start_time = 0
last_calibration_sound_time = 0

# SOS signalining davomiyligi (sobit qiymat)
sos_duration = 5  # sekund

# Kamerani ochish
cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # MediaPipe uchun RGB formatga o'tkazish
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        # Dlib uchun kulrang tasvir
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        ear = None  # Initialize ear here

        # MediaPipe yordamida ko'z holatini aniqlash
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_ear = eye_aspect_ratio(LEFT_EYE_INDICES, face_landmarks.landmark, frame.shape)
                right_ear = eye_aspect_ratio(RIGHT_EYE_INDICES, face_landmarks.landmark, frame.shape)
                ear = (left_ear + right_ear) / 2.0

                for idx in LEFT_EYE_INDICES + RIGHT_EYE_INDICES:
                    lm = face_landmarks.landmark[idx]
                    x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -3)

        # Kalibrlash bosqichlari
        if calibration_stage == 0:
            cv2.putText(frame, "Kalibrlash boshlanmoqda...", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (13, 255, 255), 2)  # Kalibrlash boshlanmoqda...
            calibration_stage = 1
            calibration_start_time = time.time()
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 255), 10)  # Sariq hoshiya
            play_sound_async(440, 300)
            time.sleep(1)
            play_sound_async(440, 300)  # Sariq hoshiya paytida 2 ta qisqa signal berish

        elif calibration_stage == 1:
            if time.time() - calibration_start_time < 3:
                if ear is not None:
                    ear_values_open.append(ear)
                cv2.putText(frame, "Ochiq ko'zlar yozilmoqda...", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)  # Ochiq ko'zlar yozilmoqda...
            else:
                 # 3 soniya sanash va 100 Hz signal
                for _ in range(3):
                    play_sound_async(400, 1000)  # 100 Hz signal, 1 soniya davomida
                    time.sleep(0.4)  # Har bir signal orasida 1 soniya kutish
                               
                calibration_stage = 2
                calibration_start_time = time.time()
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (255, 0, 0), 10)  # Ko'k hoshiya
                cv2.putText(frame, "Yopiq ko'zlar yozilmoqda...", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)  # Yopiq ko'zlar yozilmoqda...
              

        elif calibration_stage == 2:
            if time.time() - calibration_start_time < 3:
                if ear is not None:
                    ear_values_closed.append(ear)
            else:
                 # 3 soniya sanash va 100 Hz signal
                for _ in range(3):
                    play_sound_async(800, 1000)  # 100 Hz signal, 1 soniya davomida
                    time.sleep(1)  # Har bir signal orasida 1 soniya kutish
                               
                calibration_stage = 3
                avg_ear_open = mean(ear_values_open) if ear_values_open else 0
                std_ear_open = stdev(ear_values_open) if len(ear_values_open) > 1 else 0
                EAR_THRESHOLD = avg_ear_open - 1.5 * std_ear_open
                print(f"Kalibrlangan EAR chegarasi: {EAR_THRESHOLD}")
                logging.info(f"Kalibrlangan EAR chegarasi: {EAR_THRESHOLD}")
                play_sound_async(220, 800)  # Ishga tushirish signali
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 10)  # Yashil hoshiya
                # 3 soniya sanash va 100 Hz signal
                for _ in range(3):
                    play_sound_async(100, 1000)
                    time.sleep(1)

        elif calibration_stage == 3:
            # Ko'z holatini aniqlash
            if ear is not None:  # ear qiymati mavjudligini tekshirish
                if ear < EAR_THRESHOLD:
                    blink_counter += 1
                    ear_values_closed.append(ear)
                    if len(ear_values_closed) > 50:
                        ear_values_closed.pop(0)
                else:
                    if blink_counter > 0:
                        blink_durations.append(blink_counter)
                        if len(blink_durations) > 50:
                            blink_durations.pop(0)
                    ear_values_open.append(ear)
                    if len(ear_values_open) > 50:
                        ear_values_open.pop(0)
                    blink_counter = 0
                    alert_triggered = False  # Qo'shildi

                if blink_counter >= BLINK_FRAMES and blink_counter < CONSECUTIVE_FRAMES:
                    cv2.putText(frame, "Blink aniqlandi", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)  # Blink aniqlandi
                elif blink_counter >= CONSECUTIVE_FRAMES:
                    alert = True
                    cv2.putText(frame, "OGOHLANTIRISH: Ko'zlar yopiq!", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # OGOHLANTIRISH: Ko'zlar yopiq!
                    if not alert_triggered:
                        play_sos_signal_async(sos_duration)
                        alert_triggered = True
                        logging.info("OGOHLANTIRISH: Ko'zlar yopiq!")  # OGOHLANTIRISH: Ko'zlar yopiq!

            # Qizil hoshiya chizish
            if alert:
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 10)  # Qizil hoshiya

        # Ekranda ma'lumotlarni ko'rsatish
        cv2.putText(frame, f"EAR: {ear:.2f}" if ear is not None else "EAR: Mavjud emas", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Chegara: {EAR_THRESHOLD:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Log qilish (faqat ogohlantirishlar)
        if alert:
            logging.info(f"EAR: {ear}, Threshold: {EAR_THRESHOLD}, Alert: {alert}")

        # Har 300 kadrda threshold va chegaralarni yangilash
        if frame_counter % UPDATE_INTERVAL == 0 and frame_counter > CALIBRATION_FRAMES:
            if ear_values_open and ear_values_closed:
                avg_open = mean(ear_values_open)
                avg_closed = mean(ear_values_closed)
                EAR_THRESHOLD = (avg_open + avg_closed) / 2
                print(f"Yangilangan EAR chegarasi: {EAR_THRESHOLD}")
                logging.info(f"Yangilangan EAR chegarasi: {EAR_THRESHOLD}")

            if blink_durations:
                avg_blink_duration = mean(blink_durations)
                BLINK_FRAMES = max(3, int(avg_blink_duration * 1.5))
                CONSECUTIVE_FRAMES = max(10, int(avg_blink_duration * 4))
                print(f"Yangilangan BLINK_FRAMES: {BLINK_FRAMES}, CONSECUTIVE_FRAMES: {CONSECUTIVE_FRAMES}")
                logging.info(f"Yangilangan BLINK_FRAMES: {BLINK_FRAMES}, CONSECUTIVE_FRAMES: {CONSECUTIVE_FRAMES}")

        frame_counter += 1
        cv2.imshow("Wake focus", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Dastur foydalanuvchi tomonidan to'xtatildi.")
    logging.info("Dastur foydalanuvchi tomonidan to'xtatildi.")
finally:
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
