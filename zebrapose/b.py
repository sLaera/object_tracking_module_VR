import cv2
from ultralytics import YOLO

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8n.pt')

# Train the model
# results = model.train(data='Objects365.yaml', epochs=100, imgsz=640)

'''model = model.train(
    data='datasets/Expo_marker_test/data.yaml',
    imgsz=640,
    epochs=500,
    batch=100,
    name='Expo_marker_test',
    degrees=180,
    translate=0.1,
    shear=1
)'''

model = YOLO('runs/detect/Expo_marker_test7/weights/best.pt')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
while True:
    ret, frame = cap.read()

    # Perform object detection on an image using the model
    results = model.predict(source=frame, show=True)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
