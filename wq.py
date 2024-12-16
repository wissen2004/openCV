import cv2
import numpy as np
import pytesseract
from matplotlib import pyplot as plt
import shutil
import os
import re

# Укажите путь к исполняемому файлу tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\ASUS\Desktop\fishrungame\treadBibliotek\tesseract.exe'

# Папка с конечными изображениями
end_img_dir = 'end_img/'

# Создание папки, если она не существует
if not os.path.exists(end_img_dir):
    os.makedirs(end_img_dir)

# Папка с конечными текстовыми файлами
end_txt_dir = 'end_txt/'

# Создание папки, если она не существует
if not os.path.exists(end_txt_dir):
    os.makedirs(end_txt_dir)


def draw_boxes(image, boxes, texts):
    for i, box in enumerate(boxes):
        x, y, x_w, y_h = box
        cv2.rectangle(image, (x, y), (x + x_w, y + y_h), (0, 255, 0), 2)
        cv2.putText(image, texts[i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def process_image(image_path, percentage_coordinates_list):
    # Загрузка изображения и изменение размера
    original_image = cv2.imread(image_path)
    resized_image = cv2.resize(original_image, (662, 429))

    # Преобразование в оттенки серого
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Применение OCR
    custom_config = r'--oem 3 --psm 6 -l kir+eng+rus'
    text = pytesseract.image_to_string(gray, config=custom_config)

    # Получение размеров изображения
    h, w, _ = resized_image.shape

    # Преобразование процентных координат в абсолютные
    coordinates_list = [
        (
            int(x * w / 100),
            int(y * h / 100),
            int(x_w * w / 100),
            int(y_h * h / 100)
        )
        for x, y, x_w, y_h in percentage_coordinates_list
    ]

    # Фильтрация квадратов по проценту (если нужно)
    percentage_threshold = 1
    filtered_boxes = [box for box in coordinates_list if (box[2] * box[3]) / (w * h) * 100 >= percentage_threshold]

    # Извлечение текста из области каждого квадрата
    texts = []
    for box in filtered_boxes:
        x, y, x_w, y_h = box
        crop_img = gray[y:y + y_h, x:x + x_w]
        cropped_text = pytesseract.image_to_string(crop_img, config=custom_config)
        texts.append(cropped_text)
        print(f"Text inside the box at ({x}, {y}): {cropped_text}")

    # Если текст в первом квадрате соответствует "ИДЕНТИФИКАЦИОННАЯ КАРТА"
    if texts[0].strip().lower() == "идентификационная карта":
        # Рисование квадратов и вывод текста
        draw_boxes(resized_image, filtered_boxes, texts)

        # Отображение изображения с квадратами с использованием matplotlib
        plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
        plt.title('Image with Boxes and Text')
        plt.show()

        # Перемещение и переименование изображения
        new_image_name = move_and_rename_image(image_path)
        print(f"Изображение скопировано и переименовано как '{new_image_name}'")

        # Сохранение текста в файл
        text_to_save = "\n".join(texts)
        file_path_to_save = os.path.join(end_txt_dir, f'{os.path.splitext(new_image_name)[0]}.txt')
        save_text_to_file(text_to_save, file_path_to_save)
        print(f"Текст сохранен в файл '{file_path_to_save}'")

    else:
        print("Текст в первом квадрате не соответствует 'ИДЕНТИФИКАЦИОННАЯ КАРТА'. Продолжаем поиск и сохранение.")


def move_and_rename_image(image_path):
    # Создание нового имени файла
    new_image_name = generate_unique_name(image_path)
    # Путь к новому изображению
    new_image_path = os.path.join(end_img_dir, new_image_name)
    # Перемещение и переименование файла
    shutil.copyfile(image_path, new_image_path)
    return new_image_name


def generate_unique_name(image_path):
    base_name, extension = os.path.splitext(os.path.basename(image_path))
    counter = 1
    new_name = f"{base_name}_{counter}{extension}"
    while os.path.exists(os.path.join(end_img_dir, new_name)):
        counter += 1
        new_name = f"{base_name}_{counter}{extension}"
    return new_name


def save_text_to_file(text, file_path):
    # Производим необходимые изменения в тексте
    modified_text = modify_text(text)
    # Сохраняем текст в файл с явной указанием кодировки UTF-8
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(modified_text)




def modify_text(text):
    # Разбиваем текст на строки
    lines = text.split('\n')
    # Обрабатываем каждую строку независимо
    modified_lines = []
    for line in lines:
        # Пропускаем пустые строки
        if not line.strip():
            modified_lines.append(line)
            continue
        # Убираем нежелательные символы и добавляем пробелы между словами
        modified_line = re.sub(r'[^a-zA-Zа-яА-Я0-9\.]', ' ', line)
        # Проверяем, содержит ли строка число из 9 цифр
        if re.search(r'\b\d{9}\b', modified_line):
            # Если да, заменяем первые две цифры на "ID"
            modified_line = re.sub(r'\b(\d{2})', r'ID', modified_line)
        modified_lines.append(modified_line.strip())  # Убираем лишние пробелы в начале и в конце строки
    # Объединяем обработанные строки обратно в текст
    modified_text = '\n'.join(modified_lines)
    return modified_text








def detect_rectangle(frame, template, threshold):
    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    # Check if the maximum correlation coefficient is above the threshold
    print(max_val)

    if max_val >= threshold:
        # Define the rectangle area
        h, w = template.shape[:2]
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        # Draw rectangle on the frame
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        # Crop the region within the rectangle
        cropped_region = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        # Save the cropped region as an image
        cv2.imwrite('end_img.png', cropped_region)

        # Process the saved image
        image_path = 'end_img.png'
        # Задайте координаты для каждого квадрата в процентном соотношении (x, y, x_w, y_h)
        percentage_coordinates_list = [(42, 10.2, 29, 7), (38, 22, 30, 7), (38, 35, 30, 7), (38, 48, 30, 7),
                                       (42.6, 56, 30, 7),
                                       (38, 65, 45, 5.5), (38, 73, 20, 8), (68, 75.1, 20, 7), (68, 86, 20, 8)]
        process_image(image_path, percentage_coordinates_list)

    return frame


def main():
    # Load the template image (ID card template)
    template = cv2.imread('img/mp.png', 0)  # Make sure to replace with the actual template image

    # Start capturing video from the default camera (you can change the index if using an external camera)
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame")
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect rectangles in the frame with a threshold
        frame_with_rectangle = detect_rectangle(gray_frame, template, threshold=0.57)  # 0.57

        # Display the result
        cv2.imshow('Object Detection', frame_with_rectangle)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
