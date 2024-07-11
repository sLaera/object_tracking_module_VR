from ultralytics import YOLO

RESUME_PT = "laryngoscope/train2/weights/best.pt"

if RESUME_PT:
    model = YOLO(RESUME_PT)
else:
# Load a model
    model = YOLO("yolov8n.yaml").load("yolov8n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="datasets/laryngoscope.yaml", 
                      epochs=100, 
                      imgsz=600, 
                      project="laryngoscope",
                      hsv_h = 0.05,
                      degrees = 180,
                      shear = 5,
                      perspective = 0.0001,
                      flipud = 0.5,
                      fliplr = 0.5,
                      )

print(results)
