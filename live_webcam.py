import argparse
import math
import time
import cv2
import mediapipe as mp
import numpy as np

try:
    import winsound
except ImportError:
    winsound = None


def distance(a, b):
    return math.dist(a, b)


def point(landmarks, index, width, height):
    p = landmarks[index]
    return int(p.x * width), int(p.y * height)


def eye_aspect_ratio(landmarks, ids, width, height):
    p = [point(landmarks, i, width, height) for i in ids]
    return (distance(p[1], p[5]) + distance(p[2], p[4])) / (2 * distance(p[0], p[3]) + 1e-6)


def head_yaw(landmarks, width, height):
    image_points = np.array([
        point(landmarks, 1, width, height), point(landmarks, 152, width, height),
        point(landmarks, 33, width, height), point(landmarks, 263, width, height),
        point(landmarks, 61, width, height), point(landmarks, 291, width, height)
    ], dtype=np.float64)
    model_points = np.array([
        (0.0, 0.0, 0.0), (0.0, -63.6, -12.5), (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0), (-28.9, -28.9, -24.1), (28.9, -28.9, -24.1)
    ], dtype=np.float64)
    focal = float(width)
    camera = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float64)
    ok, rotation, _ = cv2.solvePnP(model_points, image_points, camera, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0
    matrix, _ = cv2.Rodrigues(rotation)
    return float(cv2.RQDecomp3x3(matrix)[0][1])


def beep():
    if winsound:
        try:
            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        except RuntimeError:
            pass
    else:
        print("\a", end="", flush=True)


def class_name(names, class_id):
    if isinstance(names, dict):
        return str(names.get(class_id, class_id)).lower().replace("_", " ")
    if isinstance(names, (list, tuple)) and class_id < len(names):
        return str(names[class_id]).lower().replace("_", " ")
    return str(class_id)


def main():
    parser = argparse.ArgumentParser(description="SafeDrive AI live driver monitor")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--ear", type=float, default=0.21)
    parser.add_argument("--eye-seconds", type=float, default=1.5)
    parser.add_argument("--mar", type=float, default=0.55)
    parser.add_argument("--yawn-seconds", type=float, default=1.0)
    parser.add_argument("--yaw", type=float, default=25.0)
    parser.add_argument("--yolo", default="yolov8n.pt")
    parser.add_argument("--no-yolo", action="store_true")
    parser.add_argument("--confidence", type=float, default=0.30)
    parser.add_argument("--sound-cooldown", type=float, default=3.0)
    parser.add_argument("--no-sound", action="store_true")
    args = parser.parse_args()

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        face_mesh.close()
        raise SystemExit("Could not open webcam. Try --camera 1 or check camera permissions.")

    detector = None
    if not args.no_yolo:
        try:
            from ultralytics import YOLO
            print("Loading YOLO:", args.yolo)
            detector = YOLO(args.yolo)
            print("YOLO loaded. First run may download the model weights.")
        except Exception as exc:
            print("YOLO load failed:", exc)
            print("Face monitoring will continue, but object detection is OFF.")

    # General YOLOv8n COCO weights include cell phone, bottle and cup.
    # Cigarette is not a COCO class; use custom-trained weights for it.
    aliases = {
        "phone": "MOBILE PHONE DETECTED",
        "cell phone": "MOBILE PHONE DETECTED",
        "mobile phone": "MOBILE PHONE DETECTED",
        "cigarette": "CIGARETTE DETECTED",
        "cig": "CIGARETTE DETECTED",
        "bottle": "DRINK CONTAINER DETECTED",
        "cup": "DRINK CONTAINER DETECTED",
        "drink container": "DRINK CONTAINER DETECTED",
        "water bottle": "DRINK CONTAINER DETECTED"
    }

    left_eye = [362, 385, 387, 263, 373, 380]
    right_eye = [33, 160, 158, 133, 153, 144]
    eye_since = yawn_since = distraction_since = None
    last_yolo = 0.0
    last_sound = 0.0
    cached_object_alerts = []
    cached_boxes = []

    print("SafeDrive AI started. Press Q in the webcam window to quit.")
    if detector:
        print("General model: phone/bottle/cup. Cigarette needs custom-trained weights.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read webcam frame.")
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            now = time.time()
            alerts = []
            ear = mar = yaw = 0.0

            result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if result.multi_face_landmarks:
                lm = result.multi_face_landmarks[0].landmark
                ear = (eye_aspect_ratio(lm, left_eye, w, h) + eye_aspect_ratio(lm, right_eye, w, h)) / 2
                mar = distance(point(lm, 13, w, h), point(lm, 14, w, h)) / max(distance(point(lm, 78, w, h), point(lm, 308, w, h)), 1)
                yaw = head_yaw(lm, w, h)

                if ear < args.ear:
                    eye_since = eye_since or now
                    if now - eye_since >= args.eye_seconds:
                        alerts.append("DROWSINESS: EYES CLOSED")
                else:
                    eye_since = None

                if mar > args.mar:
                    yawn_since = yawn_since or now
                    if now - yawn_since >= args.yawn_seconds:
                        alerts.append("YAWNING DETECTED")
                else:
                    yawn_since = None

                if abs(yaw) > args.yaw:
                    distraction_since = distraction_since or now
                    if now - distraction_since >= 0.8:
                        alerts.append("DISTRACTION: LOOKING AWAY")
                else:
                    distraction_since = None

                for idx in (33, 263, 1, 13, 14, 152):
                    cv2.circle(frame, point(lm, idx, w, h), 2, (0, 255, 255), -1)
            else:
                eye_since = yawn_since = distraction_since = None
                alerts.append("FACE NOT DETECTED")

            if detector and now - last_yolo >= 0.25:
                try:
                    results = detector.predict(frame, conf=args.confidence, verbose=False)
                    cached_object_alerts = []
                    cached_boxes = []
                    if results:
                        names = results[0].names
                        for box in results[0].boxes:
                            cid = int(box.cls[0])
                            label = class_name(names, cid)
                            confidence = float(box.conf[0])
                            coords = tuple(map(int, box.xyxy[0].tolist()))
                            cached_boxes.append((coords, label, confidence))
                            warning = aliases.get(label)
                            if warning:
                                cached_object_alerts.append(warning)
                    last_yolo = now
                except Exception as exc:
                    print("YOLO inference error:", exc)
                    detector = None

            for coords, label, confidence in cached_boxes:
                x1, y1, x2, y2 = coords
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 0), 2)

            alerts.extend(cached_object_alerts)
            alerts = list(dict.fromkeys(alerts))
            risk = min(100, len(alerts) * 25)
            status = "HIGH RISK" if risk >= 50 else "CAUTION" if risk else "MONITORING"

            if alerts and not args.no_sound and now - last_sound >= args.sound_cooldown:
                beep()
                last_sound = now

            cv2.rectangle(frame, (0, 0), (w, 145), (25, 25, 25), -1)
            cv2.putText(frame, f"SafeDrive AI | {status} | Risk: {risk}/100", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 220, 255), 2)
            cv2.putText(frame, f"EAR {ear:.2f}  MAR {mar:.2f}  Head yaw {yaw:.1f} deg", (12, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1)
            detection_state = "OBJECT DETECTION: ON" if detector else "OBJECT DETECTION: OFF"
            cv2.putText(frame, detection_state, (12, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.50, (80, 230, 80) if detector else (0, 165, 255), 1)
            if detector:
                cv2.putText(frame, "COCO model: phone / bottle / cup; cigarette needs custom weights",
                            (12, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1)

            if alerts:
                for i, alert in enumerate(alerts[:2]):
                    cv2.putText(frame, alert, (12, 126 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                                0.52, (0, 70, 255), 2)
            else:
                cv2.putText(frame, "No risk signs detected", (12, 126),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 230, 80), 2)

            cv2.imshow("SafeDrive AI - Live Webcam (Q to quit)", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        cap.release()
        face_mesh.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
