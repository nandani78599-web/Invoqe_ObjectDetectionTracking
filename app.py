import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os

st.set_page_config(
    page_title="Invoqe Object Detection",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Invoqe Object Detection & Tracking")
st.write("YOLO + ByteTrack Computer Vision System")

st.sidebar.header("Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    0.1,
    1.0,
    0.5,
    0.1
)

source = st.sidebar.radio(
    "Select Input",
    ["Image", "Video"]
)

# Load YOLO model
model = YOLO("yolo11n.pt")


# =========================
# IMAGE DETECTION
# =========================

if source == "Image":

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        import numpy as np

        file_bytes = np.asarray(
            bytearray(uploaded_file.read()),
            dtype=np.uint8
        )

        image = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        results = model(
            image,
            conf=confidence
        )

        output = results[0].plot()

        st.image(
            cv2.cvtColor(output, cv2.COLOR_BGR2RGB),
            caption="Detection Result",
            use_container_width=True
        )

        object_count = len(results[0].boxes)

        st.success(
            f"Objects Detected: {object_count}"
        )


# =========================
# VIDEO DETECTION + TRACKING
# =========================

else:

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:

        # Save uploaded video temporarily
        input_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        input_file.write(
            uploaded_video.read()
        )

        input_file.close()

        cap = cv2.VideoCapture(
            input_file.name
        )

        if not cap.isOpened():
            st.error("Video open nahi ho raha.")
            os.remove(input_file.name)
            st.stop()

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        output_path = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ).name

        fourcc = cv2.VideoWriter_fourcc(*"avc1")

        
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            fps if fps > 0 else 25,
            (width, height)
        )

        st.info(
            "Processing video... Please wait."
        )

        progress = st.progress(0)

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        frame_number = 0

        total_objects = 0
        total_persons = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            # YOLO + ByteTrack
            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=confidence,
                verbose=False
            )

            annotated_frame = results[0].plot()

            # Count objects
            object_count = len(
                results[0].boxes
            )

            # Count persons
            person_count = 0

            if results[0].boxes.cls is not None:

                class_ids = (
                    results[0]
                    .boxes
                    .cls
                    .cpu()
                    .numpy()
                )

                for class_id in class_ids:

                    if int(class_id) == 0:
                        person_count += 1

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

            out.write(
                annotated_frame
            )

            frame_number += 1

            if total_frames > 0:

                progress.progress(
                    min(
                        frame_number / total_frames,
                        1.0
                    )
                )

        cap.release()
        out.release()

        progress.progress(1.0)

        st.success(
            "Video processing completed!"
        )

        # Display processed video
        with open(
            output_path,
            "rb"
        ) as video_file:

            video_bytes = video_file.read()

        st.video(
            video_bytes
        )

        # Download button
        st.download_button(
            label="⬇️ Download Processed Video",
            data=video_bytes,
            file_name="detected_video.mp4",
            mime="video/mp4"
        )

        # Cleanup
        os.remove(
            input_file.name
        )

        os.remove(
            output_path
        )


st.markdown("---")

st.caption(
    "Invoqe_ObjectDetectionTracking | "
    "YOLO + ByteTrack"
)