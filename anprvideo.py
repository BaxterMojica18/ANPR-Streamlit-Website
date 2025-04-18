import os
import cv2
import numpy as np
from ByteTrack.yolox.tracker.byte_tracker import STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass
from typing import List, Tuple
import supervision as sv
from supervision import Color
import streamlit as st
from carcounter import model
from multiplelicense import detect_multiple_plates
import easyocr
from ultralytics import YOLO
from streamlit.runtime.scriptrunner.script_runner import ScriptRunner
from prediction_functions import get_device, is_valid_license_plate
from ffmpegcv import VideoCaptureNV, VideoWriterNV
from storage_functions import get_next_detection_number, save_detection
import datetime


output_image_path = "countedcars/trackedimage.jpg"  # Path for saving processed video
csv_file = "detections.csv"

# Vehicle classes to detect
CLASS_NAMES_DICT = model.names
SELECTED_CLASS_NAMES = ['car', 'bus', 'truck', 'motorcycle']
SELECTED_CLASS_IDS = [
    {value: key for key, value in CLASS_NAMES_DICT.items()}[class_name]
    for class_name in SELECTED_CLASS_NAMES
]

# Define BYTETrackerArgs class
@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

# Helper functions for detection-tracking matching
def detections2boxes(detections: sv.Detections) -> np.ndarray:
    return np.hstack((detections.xyxy, detections.confidence[:, np.newaxis]))

def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([track.tlbr for track in tracks], dtype=float)

# Define the model and the predict function
model = YOLO('best.pt').to(get_device())  # Load model at startup

def anpr_video(input_video_path: str, output_video_path: str, confidence_threshold=0.47, batch_size=30):
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Input video not found: {input_video_path}")

    output_dir = os.path.dirname(output_video_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_capture = cv2.VideoCapture(input_video_path)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    progress_bar = st.progress(0)
    frame_count = 0
    frames = []
    all_detected_plates = []

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        frames.append(frame)
        if len(frames) == batch_size:
            batch_annotated_frames, detected_plates_batch = process_anpr_batch(frames, confidence_threshold)
            all_detected_plates.extend(detected_plates_batch)
            for annotated_frame in batch_annotated_frames:
                writer.write(annotated_frame)
            frames = []

        frame_count += 1
        progress_bar.progress(min(frame_count / total_frames, 1.0))

    if frames:
        batch_annotated_frames, detected_plates_batch = process_anpr_batch(frames, confidence_threshold)
        all_detected_plates.extend(detected_plates_batch)
        for annotated_frame in batch_annotated_frames:
            writer.write(annotated_frame)

    video_capture.release()
    writer.release()
    progress_bar.progress(1.0)

    st.success("Video processing complete!")
    return output_video_path, all_detected_plates


def process_anpr_batch(frames, confidence_threshold=0.47):
    batch_annotated_frames = []
    detection_number = get_next_detection_number("detections.csv")
    detected_plates = []
    saved_plates = set()  # Set to track unique plates saved to the CSV file

    for frame in frames:
        try:
            detections = detect_multiple_plates(frame)
            if not detections:
                batch_annotated_frames.append(frame)
                continue

            for cropped_image, detected_text in detections:
                if not detected_text:
                    continue

                # Use OCR confidence threshold and bounding box for cropping
                x1, y1, x2, y2 = map(int, cv2.boundingRect(cropped_image[:, :, 0]))
                h, w = frame.shape[:2]
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

                zoomed_plate = frame[y1:y2, x1:x2]
                if zoomed_plate.size == 0:
                    continue

                # Annotate frame with detected text
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, detected_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Validate license plates
                if is_valid_license_plate(detected_text) and detected_text not in saved_plates:
                    save_detection(detection_number, "A License Plate is Detected!", datetime.datetime.now().strftime("%Y-%m-%d"),
                                   datetime.datetime.now().strftime("%H:%M:%S"), detected_text, "detections.csv")
                    detection_number += 1
                    saved_plates.add(detected_text)  # Add plate to the set of saved plates
                    detected_plates.append(detected_text)

            batch_annotated_frames.append(frame)

        except Exception as e:
            print(f"Error processing frame: {e}")

    return batch_annotated_frames, detected_plates