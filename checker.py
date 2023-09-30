import cv2
import dlib
import numpy as np
import tensorflow as tf

# モデルのロード
model = tf.keras.models.load_model('emotion_model.h5')

# 顔の検出に関連する部分
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 表情のラベル
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        
        face_crop = gray[y:y+h, x:x+w]
        face_crop = cv2.resize(face_crop, (48, 48)) # このサイズはモデルに合わせて調整する
        face_crop = np.expand_dims(face_crop, axis=0)
        face_crop = np.expand_dims(face_crop, axis=-1)
        
        # ここで表情解析を実施
        predictions = model.predict(face_crop)
        max_index = int(np.argmax(predictions))
        emotion = emotion_labels[max_index]
        
        cv2.putText(frame, emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()