import cv2
from ultralytics import YOLO
import pyautogui
import os

def detect_objects_and_plot(video_path=None):
    # Load the YOLO model
    yolo_model = YOLO('./runs/detect/Normal_Compressed/weights/best.pt')

    # Open the video capture (webcam or recorded video)
    if video_path is None:
        video_capture = cv2.VideoCapture(0)
    else:
        video_capture = cv2.VideoCapture(video_path)

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        if not ret:
            print("Failed to capture frame or end of video")
            break

        # Detect objects in the frame
        results = yolo_model(frame)

        # Process the detection results
        for result in results:
            classes = result.names
            cls = result.boxes.cls
            conf = result.boxes.conf
            detections = result.boxes.xyxy

            for pos, detection in enumerate(detections):
                if conf[pos] >= 0.5:
                    xmin, ymin, xmax, ymax = detection
                    label = f"{classes[int(cls[pos])]} {conf[pos]:.2f}"
                    color = (0, int(cls[pos]), 255)
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                    cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # Display the resulting frame
        frame = cv2.resize(frame, (1080, 720))
        cv2.imshow('Object Detection', frame)

        # Check for the 'q' key to quit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Check if the cursor position exceeds the threshold (top-right corner)
        screen_width, screen_height = pyautogui.size()
        cursor_x, cursor_y = pyautogui.position()
        if cursor_x >= screen_width - 1 and cursor_y <= 1:
            break

    # Release the video capture and close all OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()

# Show the GUI first
cv2.namedWindow('Object Detection')
cv2.moveWindow('Object Detection', 250, 250)

# Video path variable (set to None to use the webcam)
video_path = './video/1.mp4'

# Call the function to start object detection using webcam or video
detect_objects_and_plot(video_path)

# Focus the Object Detection window (Windows-specific)
os.system("start /min cmd /c \"timeout /t 1 && nircmd win activate title 'Object Detection'\"")
