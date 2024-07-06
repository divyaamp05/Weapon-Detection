import cv2
from darknet import load_network, perform_detection
import requests

# Load YOLO model
net = load_network("cfg/yolov3.cfg", "weights/yolov3.weights", "cfg/coco.data")

# Function to perform object detection on a frame
def detect_objects(frame):
    detections = perform_detection(net, frame)
    return detections

# Function to send detection information to the control center
def send_detection_info(label, confidence, control_center_url):
    data = {
        'label': label,
        'confidence': confidence
    }
    try:
        response = requests.post(control_center_url, json=data)
        response.raise_for_status()
        print("Detection information sent successfully!")
    except Exception as e:
        print("Error sending detection information:", e)

# Main function
def main():
    # Open webcam
    cap = cv2.VideoCapture(0)

    # Control center URL (replace with actual URL)
    control_center_url = "http://example.com/control_center"

    while True:
        # Capture frame from webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection
        detections = detect_objects(frame)

        # Process detection results
        for detection in detections:
            x, y, w, h = detection['box']
            label = detection['label']
            confidence = detection['confidence']

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Send detection information to control center
            if label == 'weapon' and confidence >= 0.5:
                send_detection_info(label, confidence, control_center_url)

        # Display frame
        cv2.imshow('Object Detection', frame)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if _name_ == "_main_":
    main()