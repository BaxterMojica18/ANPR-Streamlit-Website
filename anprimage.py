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
from prediction_functions2 import get_device


output_image_path = "countedcars/trackedimage.jpg"  # Path for saving processed video

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

st.markdown(
    """
    <style>
    div.stAlert {
        color: black; /* Change font color */
        background-color: #dff0d8; /* Keep the green success box background */
        border: 1px solid #3c763d; /* Optional: Keep the green border */
    }
    </style>
    """,
    unsafe_allow_html=True
)

def anpr_image(input_image, output_image_path: str, confidence_threshold=0.47, batch_size=15):
    # Check if the input is a string (file path) or ndarray (image data)
    if isinstance(input_image, str):
        if not os.path.exists(input_image):
            raise FileNotFoundError(f"Input image not found: {input_image}")
        image = cv2.imread(input_image)
        if image is None:
            raise FileNotFoundError(f"Failed to load the image: {input_image}")
    elif isinstance(input_image, np.ndarray):
        image = input_image
    else:
        raise TypeError(f"Input should be a string path or an ndarray, but got {type(input_image).__name__}")

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_image_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process the image for vehicle detection
    results = model(image, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]

    # Filter detections by confidence threshold
    high_confidence_indices = np.where(detections.confidence > confidence_threshold)[0]
    detections = detections[high_confidence_indices]

    if detections.xyxy.shape[0] == 0:
        st.warning("No vehicles detected in the image.")
        return None

    vehicle_count = detections.xyxy.shape[0]
    ocr_results = []

    # Process each detected vehicle
    for i in range(vehicle_count):
        # Extract bounding box for the vehicle
        x1, y1, x2, y2 = map(int, detections.xyxy[i])
        cropped_vehicle = image[y1:y2, x1:x2]

        # Detect multiple license plates in the cropped vehicle image
        plate_detections = detect_multiple_plates(cropped_vehicle)

        if plate_detections:
            # Collect OCR results for each detected plate
            detected_texts = [detected_text for _, detected_text in plate_detections]
            detected_text = detected_texts[0] if detected_texts else "No Text"

            # Get the first detected license plate's cropped frame
            zoomed_plate = plate_detections[0][0]

            # Resize the zoomed plate to be 30% smaller
            zoomed_plate_resized = cv2.resize(
                zoomed_plate,
                (int((x2 - x1) * 0.7), int(((x2 - x1) * 0.5) * 0.7))
            )

            # Determine where to overlay the zoomed plate
            overlay_y1 = max(0, y1 - zoomed_plate_resized.shape[0])
            overlay_y2 = overlay_y1 + zoomed_plate_resized.shape[0]
            overlay_x1 = x1
            overlay_x2 = x1 + zoomed_plate_resized.shape[1]

            # Embed the zoomed plate above the bounding box
            image[overlay_y1:overlay_y2, overlay_x1:overlay_x2] = zoomed_plate_resized
        else:
            detected_text = "License Plate Not Visible"

        ocr_results.append(detected_text)

        # Annotate the vehicle ROI with OCR results
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, detected_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0), 2)

    # Annotate the image with bounding boxes and labels
    box_annotator = sv.BoxAnnotator(thickness=4)
    label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1.5, text_color=Color.BLACK)

    labels = [
        f"{i+1}. {CLASS_NAMES_DICT[detections.class_id[i]]} ({ocr_results[i]})"
        for i in range(vehicle_count)
    ]

    annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

    # Convert the color back to RGB for display
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    # Save the annotated image
    cv2.imwrite(output_image_path, annotated_image)

    # Display the annotated image in Streamlit
    st.image(annotated_image, caption=f'Processed Image - {vehicle_count} vehicles detected', use_container_width=True)

    return output_image_path



