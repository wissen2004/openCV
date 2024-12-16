import cv2
import numpy as np
import pytesseract
# попробывать сделать поис и вырезание квадрата если в вырезаном есть в нужных координитак инден карта то рисуем остальные квадраты и считываем с них если нет то делаем все заново
# Укажите путь к исполняемому файлу tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\ASUS\Desktop\fishrungame\treadBibliotek\tesseract.exe'

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

        return True, top_left, bottom_right

    return False, None, None

def process_image(image, top_left, bottom_right):
    # Cut part with ИДЕНТИФИКАЦИОННАЯ КАРТА
    card_part = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    cv2.imwrite('identification_card_part.png', card_part)
    # Adjust the coordinates as per your requirement

    # Применение OCR
    custom_config = r'--oem 3 --psm 6 -l kir+eng+rus'
    text = pytesseract.image_to_string(card_part, config=custom_config)

    return text, card_part

def display_percentage_squares(image, top_left, bottom_right, percentage_coordinates_list):
    # Create a copy of the image to draw squares
    image_with_squares = image.copy()

    for percentage_coordinates in percentage_coordinates_list:
        x, y, x_w, y_h = [int(coord) for coord in percentage_coordinates]

        # Convert percentages to pixel coordinates
        x_pixel = int(x * (bottom_right[0] - top_left[0]) / 100)
        y_pixel = int(y * (bottom_right[1] - top_left[1]) / 100)
        x_w_pixel = int(x_w * (bottom_right[0] - top_left[0]) / 100)
        y_h_pixel = int(y_h * (bottom_right[1] - top_left[1]) / 100)

        top_left_square = (x_pixel, y_pixel)
        bottom_right_square = (x_pixel + x_w_pixel, y_pixel + y_h_pixel)

        cv2.rectangle(image_with_squares, top_left_square, bottom_right_square, (0, 255, 0), 2)

    # Display the image with squares
    cv2.imshow('Percentage Squares', image_with_squares)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # Load the template image (ID card template)
    template = cv2.imread('img/img.jpg', 0)  # Make sure to replace with the actual template image

    # Start capturing video from the default camera (you can change the index if using an external camera)
    cap = cv2.VideoCapture(0)

    # Define custom configuration for pytesseract
    custom_config = r'--oem 3 --psm 6 -l kir+eng+rus'

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame")
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect rectangles in the frame with a threshold
        success, top_left, bottom_right = detect_rectangle(gray_frame, template, threshold=0.35)

        if success:
            # Задайте координаты для каждого квадрата в процентном соотношении (x, y, x_w, y_h)
            percentage_coordinates_list = [
                (38, 23.5, 30, 11), (38, 36.5, 30, 10.5), (38, 48.5, 30, 7), (38, 57, 30, 5.5),
                (38, 65, 45, 5.5), (38, 73.3, 20, 6), (70, 75.1, 20, 6.3), (70, 87, 20, 6)
            ]

            # Process the cropped region
            text, card_part = process_image(frame, top_left, bottom_right)
            print("Text in the first box:", text)

            # Check if the text in the first box contains the desired word
            #if text == "ИДЕНТИФИКАЦИОННАЯ КАРТА":
            if "ИДЕНТИФИКАЦИОННАЯ КАРТА" in text:
                print("Text inside the first box matches the desired word.")

                # Save the cropped region as an image
                cv2.imwrite('cropped_region.png', card_part)

                # Display squares on the original frame
                display_percentage_squares(card_part, top_left, bottom_right, percentage_coordinates_list)

        # Display the result
        cv2.imshow('Object Detection', gray_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
