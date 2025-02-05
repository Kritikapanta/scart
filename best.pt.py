from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO("runs/detect/keyboard_mouse_detection/weights/best.pt")

# Capture video from webcam
cap = cv2.VideoCapture(0)  # Use 0 for webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model(frame)

    # Display results
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv8 Inference", annotated_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()