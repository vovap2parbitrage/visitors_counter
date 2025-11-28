# YOLO + ReID Detection with FastAPI

This project provides a web interface for **object detection** and **person re-identification (ReID)** using **YOLOv8** and **ResNet-18 embeddings**. Users can upload images or videos and view detection results directly in the browser. Progress of video processing is displayed in real-time.

---

<img width="1280" height="627" alt="image" src="https://github.com/user-attachments/assets/1b41f4bd-7c4e-41c2-abe2-c333641379a0" />


## Features

- **Image Detection**: Upload an image and get YOLO-based object detection results.
- **Video Detection with ReID**: Upload a video to detect people, assign consistent IDs across frames using ReID embeddings, and track them.
- **Real-time Progress Tracking**: Video processing progress is reported to the frontend.
- **FastAPI Backend**: Asynchronous backend for handling uploads and background video processing.
- **Lightweight UI**: Simple web interface to display detection results.

---

