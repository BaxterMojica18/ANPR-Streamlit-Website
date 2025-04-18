import cv2
import re
import csv
import os
import torch
import easyocr
import numpy as np
import streamlit as st
from ultralytics import YOLO
from typing import List, Tuple
from collections import defaultdict
from storage_functions import save_detection
from ffmpegcv import VideoReaderNV, VideoWriterNV
from multiplelicense import detect_multiple_plates
from prediction_functions2 import correct_misread_characters, is_valid_license_plate, preliminary_filter_fragments, extract_license_plate, flip_license_plate
from ffmpegcv import VideoCaptureNV

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_vehicle_model():
    return YOLO('yolo11m.pt').to(get_device())

def load_license_plate_model():
    return YOLO('best.pt').to(get_device())

print(torch.cuda.is_available())  # Should print True if CUDA is available

# Define the classes to detect, track, and count
class_names = ['car', 'truck']

def count_vehicles(frames, batch_detections, model_names):
    batch_counts = [defaultdict(int) for _ in range(len(frames))]
    for i, detections in enumerate(batch_detections):
        for detection in detections:
            cls = detection.cls.item()  # Access the class index
            label = model_names[int(cls)]
            if label in class_names:
                batch_counts[i][label] += 1
    return frames, batch_counts

def process_batch(frames, vehicle_model, lpm):
    results = vehicle_model(frames)
    batch_detections = [result.boxes for result in results]
    frames, vehicle_counts = count_vehicles(frames, batch_detections, vehicle_model.names)
    frames, detected_texts = detect_license_plates(frames, batch_detections, vehicle_model.names, lpm)
    return frames, detected_texts




def detect_license_plates(frames, batch_detections, vehicle_model_names, lpm):
    detected_texts = []
    batch_counts = [defaultdict(int) for _ in range(len(frames))]
    for i, detections in enumerate(batch_detections):
        frame = np.copy(frames[i])  # Ensure frame is writable
        for detection in detections:
            box = detection.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, box)
            cls = detection.cls.item()  # Access the class index
            label = vehicle_model_names[int(cls)]
            if label in class_names:
                batch_counts[i][label] += 1
                car_number = batch_counts[i][label]
                # Crop the vehicle area
                cropped_vehicle = frame[y1:y2, x1:x2]
                plate_detections = detect_multiple_plates(cropped_vehicle, lpm)
                license_plate_text = ""
                if plate_detections:
                    for cropped_plate, detected_text in plate_detections:
                        if detected_text:
                            detected_texts.append(detected_text)
                            license_plate_text = detected_text
                            #x_plate1, y_plate1, x_plate2, y_plate2 = cv2.boundingRect(cropped_plate[:, :, 0])
                # Draw bounding box for vehicle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Annotate bounding box with car number, vehicle type, and license plate
                annotation = f"{label} #{car_number}"
                if license_plate_text:
                    annotation += f", Plate: {license_plate_text}"
                cv2.putText(frame, annotation, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 2)  # Increased font scale for vehicle text
        frames[i] = frame
    return frames, detected_texts

# Function to write detected plates to CSV
def save_to_csv(detected_texts, output_csv='detections.csv'):
    # Ensure the output CSV file exists with the correct headers
    if not os.path.exists(output_csv):
        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Detection Number', "A License Plate is Detected!", 'Date', 'Time', 'License Plate'])  # Add relevant headers

    # Write detected license plates to CSV
    with open(output_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        for i, plate in enumerate(detected_texts, start=1):
            from datetime import datetime
            current_time = datetime.now()
            date = current_time.strftime('%Y-%m-%d')
            time = current_time.strftime('%H:%M:%S')
            writer.writerow([i, plate, date, time])

# Modify `process_video` to include progress bar
def process_video(input_video, output_video, output_csv='detections.csv', batch_size=30):
    from datetime import datetime
    import os

    # Load models
    vehicle_model = load_vehicle_model()
    lpm = load_license_plate_model()

    # Initialize video capture and writer
    cap = VideoReaderNV(input_video)
    video_capture = VideoCaptureNV(input_video)
    fps = video_capture.fps
    duration_in_seconds = video_capture.duration  # Total video duration
    total_frames = int(fps * duration_in_seconds)  # Approximate total frame count
    bitrate_option = '10M'
    out = VideoWriterNV(output_video, fps=30, codec='h264_nvenc', preset='slow', bitrate=bitrate_option)

    frames = []
    frame_count = 0
    detection_number = 1  # Initialize detection number

    # Create a progress bar
    progress_bar = st.progress(0)

    for frame in cap:
        if frame is None:
            break

        frames.append(frame)
        frame_count += 1

        # Process frames in batches
        if frame_count % batch_size == 0:
            frames, detected_texts = process_batch(frames, vehicle_model, lpm)

            # Write each detected plate to the CSV
            for plate in detected_texts:
                current_time = datetime.now()
                date_of_detection = current_time.strftime('%Y-%m-%d')
                time_of_detection = current_time.strftime('%H:%M:%S')
                detection_name = "A License Plate is Detected!"
                save_detection(detection_number, detection_name, date_of_detection, time_of_detection, plate, output_csv)
                detection_number += 1

            # Write processed frames to output video
            for frame in frames:
                out.write(frame)
            frames = []

        # Update progress bar
        progress_bar.progress(min(frame_count / total_frames, 1.0))

    # Process any remaining frames that didn't fit into a full batch
    if frames:
        frames, detected_texts = process_batch(frames, vehicle_model, lpm)

        # Write each detected plate to the CSV
        for plate in detected_texts:
            current_time = datetime.now()
            date_of_detection = current_time.strftime('%Y-%m-%d')
            time_of_detection = current_time.strftime('%H:%M:%S')
            detection_name = "A License Plate is Detected!"
            save_detection(detection_number, detection_name, date_of_detection, time_of_detection, plate, output_csv)
            detection_number += 1

        # Write processed frames to output video
        for frame in frames:
            out.write(frame)

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Finalize progress bar
    progress_bar.progress(1.0)

    st.success("Video processing complete!")
    print("Files saved on the backend side")
    return output_video


def predict_multiple(image, lpm, multiple=True, confidence_threshold=0.3, target_class="NumberPlate", batch_size=30):
    results = lpm(image)[0]
    detections = []
    for detection in results.boxes:
        class_id = int(detection.cls[0].item())
        score = detection.conf[0].item()
        detected_name = lpm.names[class_id]
        if detected_name == target_class and score >= confidence_threshold:
            x1, y1, x2, y2 = map(int, detection.xyxy[0].tolist())
            bbox = (x1, y1, x2, y2)
            cropped_plate = image[y1:y2, x1:x2]
            detections.append((detected_name, cropped_plate, score, bbox))
            if not multiple:
                return detected_name, cropped_plate, score, bbox
    if multiple:
        for i in range(0, len(detections), batch_size):
            yield detections[i:i + batch_size]
    else:
        return detections if multiple else (None, None, None, None)

def detect_multiple_plates(image: np.ndarray, lpm) -> List[Tuple[np.ndarray, str]]:
    batch_size = 30
    results = []
    cropped_plates = []
    for detection_batch in predict_multiple(image, lpm, multiple=True, batch_size=batch_size):
        for detection in detection_batch:
            _, cropped_plate, _, (x1, y1, x2, y2) = detection
            cropped_plate = image[y1:y2, x1:x2]
            cropped_plates.append(cropped_plate)
        batch_results = apply_easyocrmultiple(cropped_plates, batch_size=batch_size)
        for img, detected_texts in batch_results:
            if detected_texts:
                detected_plate = ''.join(detected_texts).upper()
                results.append((img, detected_plate))
    return results


def apply_easyocrmultiple(images: List[np.ndarray], batch_size=30) -> List[Tuple[np.ndarray, List[str]]]:
    reader = easyocr.Reader(['en'], gpu=True)
    results = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        batch_results = []
        for image in batch:  
            ocr_results = reader.readtext(image)
            threshold = 0.3
            detected_texts = []
            bounding_boxes = []
            for bbox, text, score in ocr_results:
                if score > threshold:
                    detected_texts.append(text.upper())
                    bounding_boxes.append(bbox)
            for bbox, text in zip(bounding_boxes, detected_texts):
                top_left = tuple(map(int, bbox[0]))
                bottom_right = tuple(map(int, bbox[2]))
                cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 3)
                cv2.putText(image, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            filtered_fragments = preliminary_filter_fragments(detected_texts)
            concatenated_text = " ".join(filtered_fragments)
            corrected_text = correct_misread_characters(concatenated_text)
            cleaned_text = re.sub(r"[^\w\s]", "", corrected_text.strip())
            filtered_text = None
            if cleaned_text:
                filtered_text = extract_license_plate(cleaned_text)
                if not filtered_text:
                    flipped_text = flip_license_plate(cleaned_text)
                    if flipped_text:
                        filtered_text = extract_license_plate(flipped_text)
                if filtered_text and is_valid_license_plate(filtered_text):
                    detected_texts = [filtered_text]
            batch_results.append((image, detected_texts))
        results.extend(batch_results)
    return results

if __name__ == '__main__':
    input_video = 'E:/Downloads/ImageIdentification/SOURCES/ANPRVideo/VID2.mp4'  # Replace with your video file path
    output_folder = 'E:/Downloads/ImageIdentification/outputvideo/'  # Replace with your output folder path
    output_video = output_folder + 'output_video.mp4'

    # Ensure the output folder exists
    import os
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    process_video(input_video, output_video)
