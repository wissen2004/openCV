import cv2
import numpy as np

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
        cv2.imwrite('output_vlad.png', cropped_region)
        exit(0)

    return frame

def main():
    # Load the template image (ID card template)
    template = cv2.imread('img/img.jpg', 0)  # Make sure to replace with the actual template image

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
        frame_with_rectangle = detect_rectangle(gray_frame, template, threshold=0.47)

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
