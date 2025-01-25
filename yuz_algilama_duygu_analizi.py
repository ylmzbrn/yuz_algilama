import cv2
import os
from deepface import DeepFace

# Yüz tanıma için Haar Cascade dosyasını yükle
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Kamerayı başlat
cap = cv2.VideoCapture(0)

# Yüzler için bir sayaç
face_count = 0

# Kaydedilecek yüz görüntüleri için klasör oluştur
output_folder = "detected_faces"
os.makedirs(output_folder, exist_ok=True)

while True:
    # Kameradan görüntü al
    ret, frame = cap.read()

    if not ret:
        print("Kamera erişiminde sorun oluştu.")
        break

    # Görüntüyü gri tonlamaya çevir
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Yüzleri algıla
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Algılanan yüzlerin etrafına dikdörtgen çiz
    for (x, y, w, h) in faces:
        # Yüzü algıla
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Yüz görüntüsünü kaydet
        face = frame[y:y + h, x:x + w]
        face_path = os.path.join(output_folder, f"face_{face_count}.jpg")
        cv2.imwrite(face_path, face)
        face_count += 1

        # Duygu analizi yap
        try:
            result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
            emotion = max(result['emotion'], key=result['emotion'].get)
            cv2.putText(frame, f"Duygu: {emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        except Exception as e:
            print("Duygu analizi başarısız:", e)

    # Yüz sayısını ekrana yaz
    cv2.putText(frame, f"Yuz Sayisi: {len(faces)}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Görüntüyü göster
    cv2.imshow('Yuz Algilama', frame)

    # Çıkmak için 'q' tuşuna bas
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamera ve pencereyi kapat
cap.release()
cv2.destroyAllWindows()
