import cv2
import numpy as np
from tensorflow import load_model

# Загрузка обученной модели
model = load_model('object_detection_model.h5')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Предобработка кадра
    resized_frame = cv2.resize(frame, (224, 224))
    normalized_frame = resized_frame / 255.0
    input_frame = np.expand_dims(normalized_frame, axis=0)

    # Распознавание объектов
    predictions = model.predict(input_frame)
    class_id = np.argmax(predictions)

    # Отобразите результат
    labels = ['Кубик', 'Трубка', 'Цилиндр', 'Площадка']
    label = labels[class_id]
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Detections", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()