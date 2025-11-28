import os
import shutil
import asyncio
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import Dict
from ultralytics import YOLO 
import cv2
import torch
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from scipy.spatial.distance import cosine
import numpy as np

app = FastAPI(title="YOLO + ReID Detection")

UPLOAD_DIR = "uploads"
IMAGE_OUTPUT_DIR = "detection_results"
VIDEO_OUTPUT_DIR = "detection_results_video"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="."), name="static")

progress: Dict[str, float] = {}

class IdentityManager:
    def __init__(self):
        self.known_identities = {}
        self.active_ids = set()
        self.encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.encoder = torch.nn.Sequential(*(list(self.encoder.children())[:-1]))
        self.encoder.eval()
        
        self.preprocess = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_embedding(self, img_crop):
        if len(img_crop.shape) == 3 and img_crop.shape[2] == 3:
            img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
            
        input_tensor = self.preprocess(img_crop).unsqueeze(0)
        with torch.no_grad():
            embedding = self.encoder(input_tensor).flatten().numpy()
        return embedding / np.linalg.norm(embedding) 

    def resolve_identity(self, img_crop, current_tracker_id, match_threshold=0.2, alpha=0.4):
        new_embedding = self.get_embedding(img_crop)
        best_match_id = None
        min_dist = float('inf')

        for known_id, known_embedding in self.known_identities.items():
            dist = cosine(new_embedding, known_embedding)
            if dist < min_dist and dist < match_threshold:
                min_dist = dist
                best_match_id = known_id

        if best_match_id is not None:
            old_emb = self.known_identities[best_match_id]
            self.known_identities[best_match_id] = (1 - alpha) * old_emb + alpha * new_embedding
            self.known_identities[best_match_id] /= np.linalg.norm(self.known_identities[best_match_id])
            return best_match_id
        else:
            self.known_identities[current_tracker_id] = new_embedding
            return current_tracker_id

MODEL_PATH = 'runs/detect/train9/weights/best.pt' 
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Помилка завантаження моделі YOLO за шляхом '{MODEL_PATH}': {e}")

id_manager = IdentityManager()

def detect_image(image_path, output_dir=IMAGE_OUTPUT_DIR):
    results = model(image_path)
    
    base_name = Path(image_path).stem
    suffix = Path(image_path).suffix
    output_filename = f"{base_name}_detected{suffix}"
    output_path = os.path.join(output_dir, output_filename)
    
    im_bgr = results[0].plot() 
    cv2.imwrite(output_path, im_bgr)
    
    return output_path

def process_video_with_progress(video_path: str, output_path: str, job_id: str,
                                skip_frames=3, min_frames=5, conf=0.6):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        progress[job_id] = -1.0 
        return

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    
    if fps == 0:
        fps = 30 
        
    effective_fps = fps / skip_frames
    
    fourcc = cv2.VideoWriter_fourcc(*'X264') 

    out = cv2.VideoWriter(output_path, fourcc, effective_fps, (frame_width, frame_height))

    track_history = {}
    i = 0

    while video.isOpened():
        ret = video.grab() 
        if not ret:
            break

        i += 1
        if (i - 1) % skip_frames != 0:
            continue

        success, frame = video.retrieve() 
        if not success:
            break

        results = model.track(frame, persist=True, conf=conf, verbose=False, tracker="bytetrack.yaml")
        id_manager.active_ids = set()

        annotated_frame = frame.copy()

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            classes = results[0].boxes.cls.int().cpu().numpy()
            
            active_now = set(track_ids)
            id_manager.active_ids = active_now

            for box, track_id, cls in zip(boxes, track_ids, classes):
                if cls != 0:
                    continue
                
                track_history[track_id] = track_history.get(track_id, 0) + 1
                
                if track_history[track_id] <= min_frames:
                    continue

                x1, y1, x2, y2 = map(int, box)
                
                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size > 0 and person_crop.shape[0] > 10 and person_crop.shape[1] > 10:
                    final_id = id_manager.resolve_identity(person_crop, track_id)
                    
                    color = (0, 255, 0) 
                    
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, f"ID: {final_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out.write(annotated_frame)

        if frame_count > 0:
            progress[job_id] = (i / frame_count) * 100
        else:
             progress[job_id] = 100.0
             
        track_history = {k: v for k, v in track_history.items() if k in active_now}

    video.release()
    out.release()
    progress[job_id] = 100.0

@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse("templates/index.html")

@app.post("/upload_image")
async def upload_image(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    result_path = detect_image(file_path)
    
    return HTMLResponse(f'<h3>Detection result:</h3><img src="/static/{result_path}" width="100%">')

@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    base_name = Path(file.filename).stem
    output_filename = f"{base_name}_reid.mp4"
    output_path = os.path.join(VIDEO_OUTPUT_DIR, output_filename)
    
    job_id = file.filename + "_job"
    progress[job_id] = 0.0
    
    background_tasks.add_task(process_video_with_progress, file_path, output_path, job_id)
    
    return JSONResponse({"job_id": job_id, "result_path": "/static/" + output_path})

@app.get("/progress/{job_id}")
async def get_progress(job_id: str):
    return JSONResponse({"progress": progress.get(job_id, 0.0)})