from typing import List, Tuple
import numpy as np
from prediction_functions import apply_easyocrmultiple, model  # Assuming predict uses YOLO for object detection

def predict_multiple(image, multiple=True, confidence_threshold=0.3, target_class="NumberPlate", batch_size=30):
    # Perform inference using the YOLO model
    results = model(image)[0]
    detections = []

    # Extract detections from YOLO results
    for detection in results.boxes:
        class_id = int(detection.cls[0].item())  # Extract class ID
        score = detection.conf[0].item()  # Extract confidence score
        detected_name = model.names[class_id]  # Convert class ID to name

        # Check if the detected object matches the target class with high enough confidence
        if detected_name == target_class and score >= confidence_threshold:
            print(f"{target_class} detected with score: {score}")

            # Get the bounding box coordinates
            x1, y1, x2, y2 = map(int, detection.xyxy[0].tolist())  # Extract bounding box
            bbox = (x1, y1, x2, y2)  # Store the bounding box as a tuple

            # Crop the detected license plate area
            cropped_plate = image[y1:y2, x1:x2]
            detections.append((detected_name, cropped_plate, score, bbox))

            # Stop after first detection if not detecting multiple plates
            if not multiple:
                return detected_name, cropped_plate, score, bbox

    # Return all detections if `multiple=True`, otherwise return None
    if multiple:
        # Process detections in batches
        for i in range(0, len(detections), batch_size):
            yield detections[i:i + batch_size]
    else:
        return detections if multiple else (None, None, None, None)

def detect_multiple_plates(image: np.ndarray) -> List[Tuple[np.ndarray, str]]:
    batch_size = 30
    results = []
    cropped_plates = []

    # Detect license plates using the YOLO model with batch processing
    for detection_batch in predict_multiple(image, multiple=True, batch_size=batch_size):
        for detection in detection_batch:
            # Unpack the bounding box coordinates
            _, cropped_plate, _, (x1, y1, x2, y2) = detection  # Adjust unpacking based on `predict_multiple` output

            # Crop the detected license plate area using NumPy slicing
            cropped_plate = image[y1:y2, x1:x2]
            cropped_plates.append(cropped_plate)

        # Apply OCR to each batch of cropped plates
        batch_results = apply_easyocrmultiple(cropped_plates, batch_size=batch_size)
        for img, detected_texts in batch_results:
            if detected_texts:
                detected_plate = ''.join(detected_texts).upper()
                results.append((img, detected_plate))

    return results
