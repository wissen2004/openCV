import cv2
import face_recognition

# To capture video from webcam.
cap = cv2.VideoCapture(0)
# To use a video file as input
# cap = cv2.VideoCapture('filename.mp4')

while True:
    # Read the frame
    _, img = cap.read()

    # Находим координаты лиц с использованием face_recognition
    faces_locations = face_recognition.face_locations(img)

    # Draw the rectangle around each face
    for (top, right, bottom, left) in faces_locations:
        cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 2)

    # Если обнаружено два лица, сравниваем их
    if len(faces_locations) == 2:
        # Преобразуем кортежи в списки
        face_encodings = [face_recognition.face_encodings(img, [face_location])[0] for face_location in faces_locations]

        # Сравниваем лица
        results = face_recognition.compare_faces(face_encodings, face_encodings[1])

        # Выводим результат
        if all(results):
            print("Yes, лица похожи!")

            # Сохраняем изображения только если лица совпадают
            for i, (top, right, bottom, left) in enumerate(faces_locations):
                cv2.imwrite(f'images/face_{i + 1}.png', img[top:bottom, left:right])

            # Прерываем цикл, чтобы не продолжать поиск
            break
        else:
            print("No, лица различны.")

    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release the VideoCapture object
cap.release()
# Close all OpenCV windows
cv2.destroyAllWindows()
