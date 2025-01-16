from ultralytics import YOLO

# Load the YOLOv8 model (e.g., "yolov8n.pt" for a small model, or use "yolov8s.pt", "yolov8m.pt", etc.)
model = YOLO("yolov8n.pt")  # you can choose any variant of YOLOv8

# Train the model
model.train(
    data="Lab_2-1/data.yaml",  # Path to the data.yaml file in your Roboflow dataset
    epochs=50,  # Number of epochs (adjust as needed)
    imgsz=640  # Image size (adjust as needed)
)
