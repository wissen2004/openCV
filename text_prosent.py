# import cv2
# import pytesseract
# from matplotlib import pyplot as plt
#
# # Укажите путь к исполняемому файлу tesseract
# pytesseract.pytesseract.tesseract_cmd = r'C:\Users\ASUS\Desktop\fishrungame\treadBibliotek\tesseract.exe'
#
#
# def draw_boxes(image, boxes, texts):
#     for i, box in enumerate(boxes):
#         x, y, x_w, y_h = box
#         cv2.rectangle(image, (x, y), (x + x_w, y + y_h), (0, 255, 0), 2)
#         cv2.putText(image, texts[i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#
#
# def process_image(image_path, percentage_coordinates_list):
#     image = cv2.imread(image_path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # Применение OCR
#     custom_config = r'--oem 3 --psm 6 -l kir+eng+rus'
#     text = pytesseract.image_to_string(gray, config=custom_config)
#
#     # Получение размеров изображения
#     h, w, _ = image.shape
#
#     # Преобразование процентных координат в абсолютные
#     coordinates_list = [
#         (
#             int(x * w / 100),
#             int(y * h / 100),
#             int(x_w * w / 100),
#             int(y_h * h / 100)
#         )
#         for x, y, x_w, y_h in percentage_coordinates_list
#     ]
#
#     # Фильтрация квадратов по проценту (если нужно)
#     percentage_threshold = 1
#     filtered_boxes = [box for box in coordinates_list if (box[2] * box[3]) / (w * h) * 100 >= percentage_threshold]
#
#     # Извлечение текста из области каждого квадрата
#     texts = []
#     for box in filtered_boxes:
#         x, y, x_w, y_h = box
#         crop_img = gray[y:y + y_h, x:x + x_w]
#         cropped_text = pytesseract.image_to_string(crop_img, config=custom_config)
#         texts.append(cropped_text)
#         print(f"Text inside the box at ({x}, {y}): {cropped_text}")
#
#     # Рисование квадратов и вывод текста
#     draw_boxes(image, filtered_boxes, texts)
#
#     # Отображение изображения с квадратами с использованием matplotlib
#     plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     plt.title('Image with Boxes and Text')
#     plt.show()
#
#
# if __name__ == "__main__":
#     image_path = 'img/img.png'
#
#     # Задайте координаты для каждого квадрата в процентном соотношении (x, y, x_w, y_h)
#     percentage_coordinates_list = [(38, 23, 30, 11), (38, 35.5, 30, 10.5), (38, 48, 30, 7), (38, 56.4, 30, 5.5), (38, 65, 45, 5.5), (38, 73, 20, 6), (68, 75.1, 20, 6.3), (68, 87, 20, 6)]
#
#     process_image(image_path, percentage_coordinates_list)
import cv2
import pytesseract
from matplotlib import pyplot as plt

# Укажите путь к исполняемому файлу tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\ASUS\Desktop\fishrungame\treadBibliotek\tesseract.exe'


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

        # Рисование квадратов и вывод текста
        draw_boxes(resized_image, filtered_boxes, texts)

        # Отображение изображения с квадратами с использованием matplotlib
        plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
        plt.title('Image with Boxes and Text')
        plt.show()


if __name__ == "__main__":
        image_path = 'output_vlad.png'

        # Задайте координаты для каждого квадрата в процентном соотношении (x, y, x_w, y_h)
        percentage_coordinates_list = [(38, 23, 30, 11), (38, 35.5, 30, 10.5), (38, 48, 30, 7), (38, 56.4, 30, 5.5),
                                       (38, 65, 45, 5.5), (38, 73, 20, 6), (68, 75.1, 20, 6.3), (68, 87, 20, 6)]

        process_image(image_path, percentage_coordinates_list)

