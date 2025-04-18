import cv2
import tempfile
import numpy as np
import streamlit as st
from ultralytics import YOLO
import os
import easyocr
from typing import List, Tuple

reader = easyocr.Reader(['en'], gpu=True)
vehicles = [2]  # Vehicle class IDs

license_plate_model = YOLO('best.pt')
vehicle_model = YOLO('yolov11m.pt')  # Replace with your YOLOv11-medium model

def get_car(license_plate, vehicle_track_ids):
    x1, y1, x2, y2, *_ = license_plate
    for vehicle in vehicle_track_ids:
        vx1, vy1, vx2, vy2, vid = vehicle
        if x1 > vx1 and y1 > vy1 and x2 < vx2 and y2 < vy2:
            return vx1, vy1, vx2, vy2, vid
    return -1, -1, -1, -1, -1

def apply_basicocr_with_zoom(images: List[np.ndarray]) -> List[Tuple[np.ndarray, List[str]]]:
    results = []
    for image in images:
        # Track all objects in the frame using built-in YOLOv11 tracking
        track_results = vehicle_model.track(image, persist=True, classes=vehicles, verbose=False)[0]
        vehicle_boxes = []
        if track_results.boxes.id is not None:
            for box, track_id in zip(track_results.boxes.xyxy, track_results.boxes.id):
                x1, y1, x2, y2 = map(int, box.tolist())
                vehicle_boxes.append([x1, y1, x2, y2, int(track_id)])

        # Detect license plates
        license_plates = license_plate_model(image)[0]
        license_plate_boxes = np.array(license_plates.boxes.data.tolist()) if license_plates.boxes.data is not None else []

        detected_texts = []

        for license_plate in license_plate_boxes:
            x1, y1, x2, y2, score, class_id = license_plate
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, vehicle_boxes)

            if car_id != -1:
                # Crop and OCR
                crop = image[int(y1):int(y2), int(x1):int(x2)]
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY_INV)
                result = reader.readtext(thresh)

                full_text = ''
                for bbox, text, conf in result:
                    if conf > 0.3:
                        detected_texts.append(text.upper())
                        top_left = (int(bbox[0][0] + x1), int(bbox[0][1] + y1))
                        bottom_right = (int(bbox[2][0] + x1), int(bbox[2][1] + y1))
                        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
                        cv2.putText(image, text.upper(), (top_left[0], top_left[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        full_text += text.upper() + ' '
                if full_text.strip():
                    detected_texts.append(full_text.strip())

        results.append((image, detected_texts))
    return results

def process_frames(frames: List[np.ndarray], frame_counts: List[int], total_frames: int, message_placeholder) -> List[np.ndarray]:
    processed_results = []
    results = apply_basicocr_with_zoom(frames)
    for (processed_frame, detected_texts), frame_count in zip(results, frame_counts):
        if detected_texts:
            for text in detected_texts:
                message_placeholder.success(f"Detected License Plate: {text}")
        processed_results.append(processed_frame)
    return processed_results

def process_video(video_file, csv_file=None):
    temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video_file.write(video_file.read())
    temp_video_file.close()

    video_capture = cv2.VideoCapture(temp_video_file.name, cv2.CAP_FFMPEG)
    frames_directory = "processed_frames"
    os.makedirs(frames_directory, exist_ok=True)

    message_placeholder = st.empty()
    progress_bar = st.progress(0)

    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video_path = os.path.join(frames_directory, "processed_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(output_video_path, fourcc, video_capture.get(cv2.CAP_PROP_FPS), (frame_width, frame_height))

    batch_size = 20
    skip_frames = 0
    frames = []
    frame_counts = []
    frame_count = 0

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        if frame_count % (skip_frames + 1) == 0:
            frames.append(frame)
            frame_counts.append(frame_count)
        if len(frames) == batch_size:
            processed_frames = process_frames(frames, frame_counts, total_frames, message_placeholder)
            for f in processed_frames:
                out.write(f)
            frames, frame_counts = [], []

        frame_count += 1
        progress_bar.progress(frame_count / total_frames)

    if frames:
        processed_frames = process_frames(frames, frame_counts, total_frames, message_placeholder)
        for f in processed_frames:
            out.write(f)

    video_capture.release()
    out.release()

    st.session_state.video_processed = True
    st.session_state.output_video_path = output_video_path
    progress_bar.progress(1.0)
