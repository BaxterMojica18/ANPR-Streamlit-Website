import os
import cv2
import numpy as np
from ByteTrack.yolox.tracker.byte_tracker import STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass
from typing import List
import supervision as sv
from supervision import Color
import streamlit as st
from carcounter import model

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

def count_vehicles_in_image(input_image, output_image_path: str, confidence_threshold=0.47, batch_size=5):
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

    # Process the image
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

    # Create batches of detections
    for i in range(0, len(detections), batch_size):
        batch_detections = detections[i:i + batch_size]

        # Format custom labels with vehicle number
        labels = np.array([
            f"{j+1}. {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
            for j, (confidence, class_id) in enumerate(zip(batch_detections.confidence, batch_detections.class_id))
        ])

        # Annotate image
        box_annotator = sv.BoxAnnotator(thickness=4)
        label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1.5, text_color=Color.BLACK)
        annotated_image = image.copy()
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=batch_detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=batch_detections, labels=labels)

        # Convert the color back to RGB
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    # Save the annotated image
    cv2.imwrite(output_image_path, annotated_image)

    # Display the vehicle count and progress in Streamlit
    st.image(annotated_image, caption=f'Processed Image - {vehicle_count} vehicles detected', use_container_width=True)

    return output_image_path
