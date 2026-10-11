# Invoqe Object Detection & Tracking

## Project Overview

Invoqe Object Detection & Tracking is a Computer Vision project developed as part of the AI & ML Development Internship.

The system uses YOLO for real-time object detection and ByteTrack for object tracking. It can process webcam input, images, and uploaded videos.

The system detects objects, draws bounding boxes, identifies object classes, assigns tracking IDs, counts objects and persons, and displays processed video results.

---

## Objective

The main objectives of this project are:

- Detect objects using a pretrained YOLO model.
- Track detected objects using ByteTrack.
- Display bounding boxes and class labels.
- Assign unique tracking IDs to objects.
- Count detected objects and persons.
- Display real-time FPS.
- Process webcam and video input.
- Provide a simple Streamlit-based interface.

---

## Features

- YOLO-based object detection
- ByteTrack object tracking
- Unique tracking IDs
- Bounding boxes
- Class labels
- Person counting
- Object counting
- FPS monitoring
- Webcam detection
- Image detection
- Video detection
- Processed video preview
- Processed video download
- Streamlit dashboard

---

## Technologies Used

- Python
- YOLO
- Ultralytics
- OpenCV
- ByteTrack
- NumPy
- Streamlit

---

## System Workflow

```text
Image / Webcam / Video
          ↓
     YOLO Detection
          ↓
 Bounding Boxes + Classes
          ↓
   ByteTrack Tracking
          ↓
     Tracking IDs
          ↓
 Object / Person Counting
          ↓
    FPS Calculation
          ↓
 Detection & Tracking Output