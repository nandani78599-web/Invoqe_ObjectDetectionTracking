from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolo11n.pt")

# Change this to your video filename
video_path = "test_video.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Video open nahi ho raha!")
    exit()

print("Video tracking started.")
print("Press Q to exit.")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Detection + ByteTrack tracking
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.5,
        verbose=False
    )

    # Draw boxes + labels + tracking IDs
    annotated_frame = results[0].plot()

    cv2.imshow(
        "Invoqe Video Object Detection & Tracking",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Video processing completed.")