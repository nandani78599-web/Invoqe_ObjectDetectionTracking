from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")

# Run object detection on a sample image
results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    save=True,
    conf=0.5
)

print("Detection completed successfully!")