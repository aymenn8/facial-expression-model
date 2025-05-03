import cv2
import numpy as np
from tensorflow.keras.models import load_model
from ultralytics import YOLO

# Modeles
try:
    # YOLOv8 pour la détection de visages 
    face_detector = YOLO('yolov8n.pt') 

    # notre model 
    emotion_model = load_model("goatv1.keras")
    emotion_labels =  ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
except Exception as e:
    print(f"Erreur lors du chargement des modèles : {e}")
    exit()

#  Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la webcam.")
    exit()

target_size = (48, 48) 
gray_scale = True

def preprocess_face(face_roi):
    if gray_scale:
        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        resized_face = cv2.resize(face_gray, target_size)
        normalized_face = resized_face / 255.0
        expanded_face = np.expand_dims(normalized_face, axis=0)
        final_face = np.expand_dims(expanded_face, axis=-1)
    else:
        resized_face = cv2.resize(face_roi, target_size)
        normalized_face = resized_face / 255.0
        final_face = np.expand_dims(normalized_face, axis=0)
    return final_face

while True:
    ret, frame = cap.read()
    if not ret:
        print("bug webcam.")
        break

    # Détection visages avec YOLOv8
    results = face_detector(frame, conf=0.5) 

    detected_faces = []
    face_boxes = []
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                # Vérifier class = "person" (class_id = 0)
                if int(box.cls[0]) == 0 and box.conf[0] > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    face_roi = frame[y1:y2, x1:x2]
                    if not face_roi.size == 0:
                      detected_faces.append(face_roi)
                      face_boxes.append((x1, y1, x2 - x1, y2 - y1)) 

    predicted_emotions = []
    for face_roi in detected_faces:
        processed_face = preprocess_face(face_roi)
        emotion_prediction = emotion_model.predict(processed_face)
        emotion_index = np.argmax(emotion_prediction)
        predicted_emotion = emotion_labels[emotion_index]
        predicted_emotions.append(predicted_emotion)

    # Dessiner les box
    for i, (x, y, w, h) in enumerate(face_boxes):
        emotion_label = predicted_emotions[i]
        color = (0, 255, 0) # Couleur box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Émotions", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()