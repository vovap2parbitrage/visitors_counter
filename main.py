import os
import shutil
import time
import uuid
from typing import Dict

from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.background import BackgroundTask
from ultralytics import YOLO

import cv2
import numpy as np

app = FastAPI()

UPLOAD_DIR = "uploads"
PROCESSED_DIR = "detection_results_video"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

app.mount("/detection_results_video", StaticFiles(directory=PROCESSED_DIR), name="detection_results_video")

templates = Jinja2Templates(directory="templates")

job_status: Dict[str, dict] = {}

LINE_START = (0, 0)
LINE_END = (0, 0)
LINE_COLOR = (0, 0, 255)
TEXT_COLOR = (255, 255, 255)

def remove_file(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)

def orientation(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - \
          (q[0] - p[0]) * (r[1] - q[1])
    return val

def run_counter_async(input_path: str, output_filename: str, job_id: str, conf_threshold: float, skip_frames: int,
                      line_p1_x: int, line_p1_y: int, line_p2_x: int, line_p2_y: int):
    job_status[job_id]["status"] = "Processing"
    is_closed = False

    try:
        model = YOLO("runs/detect/train10/best.pt")

        in_count = 0
        out_count = 0
        counted_ids = set()

        video = cv2.VideoCapture(input_path)
        if not video.isOpened():
            raise Exception("Не вдалося відкрити відеофайл")

        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fps = int(video.get(cv2.CAP_PROP_FPS) / skip_frames) if skip_frames > 0 else int(video.get(cv2.CAP_PROP_FPS))
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_number = 0

        l1 = (line_p1_x, line_p1_y)
        l2 = (line_p2_x, line_p2_y)

        output_path = os.path.join(PROCESSED_DIR, output_filename)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'X264'), fps, (frame_width, frame_height))

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            frame_number += 1

            if skip_frames > 0 and frame_number % skip_frames != 0:
                continue

            results = model.track(
                frame,
                conf=conf_threshold,
                persist=True,
                verbose=False
            )

            if results and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                confidence = results[0].boxes.conf.cpu()

                for box, track_id, conf in zip(boxes, ids, confidence):
                    x1, y1, x2, y2 = box

                    center = (x1 + x2) // 2, (y1 + y2) // 2

                    if track_id not in counted_ids:
                        o1 = orientation(l1, l2, center)

                        if abs(center[0] - line_p1_x) < 50:
                            if o1 > 0:
                                out_count += 1
                            else:
                                in_count += 1

                            counted_ids.add(track_id)
                            cv2.circle(frame, center, 20, (0, 255, 0), -1)

                        conf_label = f"Person:{conf:.2f}"
                        cv2.circle(frame, center, 5, (255, 0, 255), -1)
                        cv2.rectangle(frame, (x1, y1),(x2, y2),(0, 255, 0), 2)
                        cv2.putText(frame, conf_label, (x1, y1-4), 0, 0.6, (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

            cv2.line(frame, l1, l2, LINE_COLOR, 2)

            current_visitors = max(0, in_count - out_count)

            info_text = [
                f"In: {in_count}",
                f"Out: {out_count}",
                f"Total: {current_visitors}"
            ]

            cv2.rectangle(frame, (0, 0), (250, 120), (0,0,0), -1)
            for idx, text in enumerate(info_text):
                cv2.putText(frame, text, (10, 35 + (idx * 35)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2)

            if not is_closed:
                cv2.imshow("Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                is_closed = True
            out.write(frame)

            progress_percent = int((frame_number / total_frames) * 100)
            job_status[job_id]["progress"] = min(100, progress_percent)

        video.release()
        out.release()

        cv2.destroyAllWindows()

        job_status[job_id]["progress"] = 100
        job_status[job_id]["status"] = "Completed"
        job_status[job_id]["result_file"] = output_filename

    except Exception as e:
        print(f"Помилка обробки: {e}")
        job_status[job_id]["progress"] = -1
        job_status[job_id]["status"] = f"Failed: {str(e)}"

        cv2.destroyAllWindows()


@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload_video")
async def upload_video(
    file: UploadFile = File(...),
    skip_frames: int = Form(3),
    conf: float = Form(0.6),
    line_p1_x: int = Form(...),
    line_p1_y: int = Form(...),
    line_p2_x: int = Form(...),
    line_p2_y: int = Form(...)
):
    if not file.content_type.startswith("video/"):
        return JSONResponse(status_code=400, content={"message": "Тільки відеофайли дозволено."})

    job_id = str(uuid.uuid4())
    input_filename = f"{job_id}_{file.filename}"
    output_filename = f"processed_{job_id}.mp4"
    input_path = os.path.join(UPLOAD_DIR, input_filename)

    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Не вдалося зберегти файл: {e}"})

    job_status[job_id] = {
        "progress": 0,
        "status": "Queued",
        "result_file": output_filename,
        "input_file": input_path
    }

    task = BackgroundTask(
        run_counter_async,
        input_path,
        output_filename,
        job_id,
        conf,
        skip_frames,
        line_p1_x,
        line_p1_y,
        line_p2_x,
        line_p2_y
    )

    return JSONResponse(
        content={
            "job_id": job_id,
            "message": "Обробка розпочата",
            "result_path": f"/detection_results_video/{output_filename}"
        },
        background=task
    )


@app.get("/progress/{job_id}")
async def get_progress(job_id: str):
    status = job_status.get(job_id)
    if not status:
        return JSONResponse(status_code=404, content={"message": "Завдання не знайдено."})

    if status["progress"] >= 100 or status["progress"] < 0:
        if "input_file" in status and os.path.exists(status["input_file"]):
            os.remove(status["input_file"])
            del status["input_file"]

    return JSONResponse(content={"progress": status["progress"], "status": status["status"]})
