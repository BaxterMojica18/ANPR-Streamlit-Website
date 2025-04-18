import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ByteTrack.yolox.tracker.byte_tracker import STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass
from typing import List
import supervision as sv
from supervision import Point
from supervision import Color
import streamlit as st
from ffmpegcv import VideoCaptureNV, VideoWriterNV  # GPU-enabled FFmpeg for video processing


# Set device to GPU if available
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")

device = get_device()
print(device)

# Load the YOLO model
MODEL = "yolo11x.pt"
model = YOLO(MODEL).to(device)
model.fuse()

output_video_path = "countedcars/trackedvideo.mp4"  # Path for saving processed video

# Vehicle classes to detect
CLASS_NAMES_DICT = model.model.names
SELECTED_CLASS_NAMES = ['car', 'truck']
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

def match_detections_with_tracks(detections: sv.Detections, tracks: List[STrack]) -> sv.Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))
    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)
    tracker_ids = [None] * len(detections)
    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id
    return tracker_ids

# Settings for line counting
LINE_START = Point(50, 1500)
LINE_END = Point(3840 - 50, 1500)

def process_batch(frames, byte_tracker, box_annotator, label_annotator, trace_annotator, line_zone_annotator, line_zone, batch_size):
  batch_detections = []
  batch_annotated_frames = []

  for frame in frames:
    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]
    detections = byte_tracker.update_with_detections(detections)
    batch_detections.append(detections)

  for i, frame in enumerate(frames):
    detections = batch_detections[i]

    # Annotate frame
    labels = np.array([
      f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
      for confidence, class_id, tracker_id
      in zip(detections.confidence, detections.class_id, detections.tracker_id)
    ])
    annotated_frame = frame.copy()
    annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    # Update line counter
    line_zone.trigger(detections)
    annotated_frame = line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)

    # Convert the color back to RGB (added line)
    #annotated_frame = cv2.cvtColor(annotated_frame)

    batch_annotated_frames.append(annotated_frame)

  return batch_annotated_frames

def count_vehicles(input_video_path: str, output_video_path: str, batch_size=15):
    # Validate the input video path
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Input video not found: {input_video_path}")

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_video_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Selected classes
    SELECTED_CLASS_NAMES = ['car', 'truck']
    CLASS_NAMES_DICT = model.model.names
    SELECTED_CLASS_IDS = np.array([
        {value: key for key, value in CLASS_NAMES_DICT.items()}[class_name]
        for class_name in SELECTED_CLASS_NAMES
    ])

    # Initialize BYTETracker and video processing settings
    byte_tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=30,
        minimum_consecutive_frames=3
    )
    byte_tracker.reset()

    # Open the video file with ffmpegcv for GPU-accelerated decoding
    video_capture = VideoCaptureNV(input_video_path)
    fps = video_capture.fps
    frame_width = video_capture.width
    frame_height = video_capture.height
    duration_in_seconds = video_capture.duration  # Total video duration
    total_frames = int(fps * duration_in_seconds)  # Approximate total frame count

    # Set up the output video writer with ffmpegcv for GPU-accelerated encoding
    writer = VideoWriterNV(output_video_path, codec="h264_nvenc", fps=fps)

    # Annotators setup
    box_annotator = sv.BoxAnnotator(thickness=4)
    label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1.5, text_color=sv.Color.BLACK)
    trace_annotator = sv.TraceAnnotator(thickness=4, trace_length=50)
    line_zone_annotator = sv.LineZoneAnnotator(thickness=4, text_thickness=4, text_scale=2)
    line_zone = sv.LineZone(start=sv.Point(50, 1500), end=sv.Point(3840 - 50, 1500))

    # Initialize Streamlit progress bar
    progress_bar = st.progress(0)

    frame_count = 0
    frames = []

    # Process the video frames
    for frame in video_capture:
        frames.append(frame)
        if len(frames) == batch_size:
            batch_annotated_frames = process_batch(
                frames, byte_tracker, box_annotator, label_annotator, trace_annotator, line_zone_annotator, line_zone, batch_size
            )
            for annotated_frame in batch_annotated_frames:
                writer.write(annotated_frame)  # Write annotated frames to the output video
            frames = []  # Clear the batch

        frame_count += 1
        progress_bar.progress(min(frame_count / total_frames, 1.0))  # Ensure progress doesn't exceed 100%

    # Process the last batch of frames (if any)
    if frames:
        batch_annotated_frames = process_batch(
            frames, byte_tracker, box_annotator, label_annotator, trace_annotator, line_zone_annotator, line_zone, batch_size
        )
        for annotated_frame in batch_annotated_frames:
            writer.write(annotated_frame)

    # Release resources
    video_capture.release()
    writer.release()

    # Update session state
    st.session_state.video_processed = True
    st.session_state.output_video_path = output_video_path

    progress_bar.progress(1.0)

    return output_video_path
