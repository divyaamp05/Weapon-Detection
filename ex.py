import cv2
import requests
from darknet import load_network, perform_detection

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

# Function to capture frame from CCTV camera feed
def capture_cctv_frame():
    # Implement code to capture frame from CCTV camera here
    # For demonstration, let's assume we capture a frame from a local webcam
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    return frame

# Main function
def main():
    # Control center URL (replace with actual URL)
    control_center_url = "http://example.com/control_center"

    while True:
        # Capture frame from CCTV camera
        frame = capture_cctv_frame()

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

            # Send detection information to control center if weapon detected
            if label == 'weapon' and confidence >= 0.5:
                send_detection_info(label, confidence, control_center_url)

        # Display frame
        cv2.imshow('Object Detection', frame)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cv2.destroyAllWindows()

if _name_ == "_main_":
    main()