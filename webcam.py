from ultralytics import YOLO
import cv2
import time

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam open nahi ho raha!")
    exit()

print("Object Detection + Tracking started.")
print("Press Q to exit.")

# FPS variables
prev_time = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame read nahi ho raha!")
        break

    # YOLO detection + ByteTrack
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.5,
        verbose=False
    )

    # Annotated frame
    annotated_frame = results[0].plot()

    # Count detected objects
    object_count = len(results[0].boxes)

    # Count persons
    person_count = 0

    if results[0].boxes.cls is not None:
        class_ids = results[0].boxes.cls.cpu().numpy()

        for class_id in class_ids:
            if int(class_id) == 0:
                person_count += 1

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    # Display information
    cv2.putText(
        annotated_frame,
        f"Objects: {object_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Persons: {person_count}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.1f}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # Show result
    cv2.imshow("Invoqe Object Detection & Tracking", annotated_frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()