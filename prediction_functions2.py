import cv2
from ultralytics import YOLO
import numpy as np
import torch
import datetime
import easyocr
import re
import os
import easyocr

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")

model = YOLO('best.pt').to(get_device())  # Load model at startup

import numpy as np

def predict(image):
    # Perform inference using YOLO model
    result = model(image)[0]
    
    # Extract detection data in bulk using NumPy arrays
    boxes = np.array([detection.xyxy[0].tolist() for detection in result.boxes])
    class_ids = np.array([int(detection.cls[0].item()) for detection in result.boxes])
    confidences = np.array([detection.conf[0].item() for detection in result.boxes])

    # Filter for License Plate detections with confidence > 0.3
    license_plate_indices = np.where(
        (np.array([model.names[class_id] for class_id in class_ids]) == "NumberPlate")
        & (confidences > 0.3)
    )[0]

    if license_plate_indices.size > 0:
        # Process the first valid License Plate detection
        idx = license_plate_indices[0]
        x1, y1, x2, y2 = boxes[idx].astype(int)
        
        print(f"LicensePlate detected with score: {confidences[idx]}")

        # Crop the detected license plate area
        cropped_plate = image[y1:y2, x1:x2]
        return "A LicensePlate is detected!", cropped_plate

    # Return None if no LicensePlate is detected
    return None, None

def is_valid_license_plate(plate_text):
    """Checks if the license plate format is valid and performs basic cleaning."""

    # Convert text to uppercase before validating
    plate_text = plate_text.upper()

    # Define the expected format patterns
    patterns = np.array([
        r'^[A-Z]{3}\s\d{3,4}$',  # Standard plates: ABC 123 or ABC 1234
        r'^[A-Z]{3}\d{3,4}$',
        r'^[A-Z]{3}\s\d{2}$',    # Custom plates: ABC 12
        r'^[A-Z]{3}\d{2}$'
    ])

    # Check the plate text against all patterns using vectorized operations
    match = np.array([re.match(pattern, plate_text) is not None for pattern in patterns])

    if np.any(match):  # If any pattern matches
        print(f"Valid license plate: {plate_text}")
        return plate_text

    # Perform basic cleaning if no pattern matched initially
    cleaned_text = plate_text.strip()
    cleaned_text = re.sub(r"[^\w\s]+$", "", cleaned_text)  # Remove extraneous characters

    # Recheck the cleaned text
    match_cleaned = np.array([re.match(pattern, cleaned_text) is not None for pattern in patterns])

    if np.any(match_cleaned):  # If any pattern matches after cleaning
        print(f"Cleaned license plate: {cleaned_text}")
        return cleaned_text

    # No match found, return None
    print(f"Invalid license plate: {plate_text}")
    return None

def correct_misread_characters(text):
    """
    Corrects misread characters in the detected text fragments based on predefined replacements.
    """
    replacements = {'O': '0', 'I': '1', 'Z': '2', 'S': '5'}
    replacements_np = np.array(list(replacements.keys()))
    replacements_values = np.array(list(replacements.values()))

    def correct_part(part):
        # Ensure the input is a numpy array
        char_array = np.array(list(part), dtype=str)

        # Create mask for characters needing replacement
        mask = np.isin(char_array, replacements_np)

        # Debugging shapes
        #print("char_array:", char_array, "Shape:", char_array.shape)
        #print("mask:", mask, "Shape:", mask.shape)

        if np.any(mask):
            # Get replacement values
            replacement_indices = np.searchsorted(replacements_np, char_array[mask])
            replacements = replacements_values[replacement_indices]

            # Debugging replacements
            #mentsprint("Replacements:", replacements, "Shape:", replacements.shape)

            # Replace only the masked elements
            char_array[mask] = replacements

        return ''.join(char_array)


    try:
        corrected_text = ' '.join([correct_part(part) for part in text.split()])
    except IndexError as e:
        print(f"Error during character correction: {e}")
        return text
    
    
    #print("replacements_np shape:", replacements_np.shape)
    #print("replacements_values shape:", replacements_values.shape)


    return corrected_text

def extract_license_plate(text):
    """
    Attempts to extract a valid license plate directly from the OCR text, optimized with NumPy.
    """
    if text is None or not text.strip():
        print("No text provided for extraction.")
        return None

    formatted_text = correct_misread_characters(text)
    char_array = np.array(list(formatted_text))
    formatted_text = ''.join(char_array)

    pattern = r"[A-Z]{3}\s?\d{2,4}"

    match = re.search(pattern, formatted_text)
    if match:
        main_plate = match.group().strip()
        print(f"Extracted license plate: '{main_plate}'")
        return main_plate
    else:
        print("No match found")
        return None

def is_mostly_letters(part):
    """
    Check if a part is mostly letters using NumPy for optimization.
    """
    char_array = np.array(list(part))  # Convert string to a NumPy array of characters
    is_alpha = np.char.isalpha(char_array)  # Check which characters are alphabetic
    is_digit = np.char.isdigit(char_array)  # Check which characters are numeric
    return np.sum(is_alpha) > np.sum(is_digit)  # Compare counts of alphabetic and numeric characters

def is_mostly_numbers(part):
    """
    Check if a part is mostly numbers using NumPy for optimization.
    """
    char_array = np.array(list(part))  # Convert string to a NumPy array of characters
    is_alpha = np.char.isalpha(char_array)  # Check which characters are alphabetic
    is_digit = np.char.isdigit(char_array)  # Check which characters are numeric
    return np.sum(is_digit) > np.sum(is_alpha)  # Compare counts of numeric and alphabetic characters

def flip_license_plate(plate_text):
    """
    Reorders license plate text fragments to match the valid format (letters followed by numbers).
    Removes any extraneous characters and ensures only one space between parts.

    Args:
        plate_text: The license plate text fragment from OCR.

    Returns:
        The corrected license plate if it can be flipped, or None if it's invalid.
    """
    if not plate_text or not plate_text.strip():
        print("Invalid input: empty or None.")
        return None

    plate_text = re.sub(r"[^\w\s]", "", plate_text).strip()
    parts = np.array(plate_text.split())
    if len(parts) == 2:
        is_alpha = np.char.isalpha(parts[0])
        is_digit = np.char.isdigit(parts[1])

        if is_digit and is_alpha:
            flipped_plate = f"{parts[1]} {parts[0]}"
        elif is_alpha and is_digit:
            flipped_plate = f"{parts[0]} {parts[1]}"
        else:
            print(f"License plate not flipped: {plate_text}")
            return None

        print(f"Formatted license plate after flip: {flipped_plate}")
        return flipped_plate

    print(f"License plate not flipped: {plate_text}")
    return None


def preliminary_filter_fragments(detected_texts):
    """
    Filters the detected text fragments to retain only those likely to be part of a valid license plate.
    Optimized with NumPy for better performance.
    """
    if not detected_texts or not isinstance(detected_texts, list):
        print("Invalid input: detected_texts must be a list of strings.")
        return []

    # Preprocessing: Clean and sanitize fragments
    sanitized_fragments = np.array([re.sub(r"[^\w\s]", "", text).strip() for text in detected_texts])
    sanitized_fragments = np.char.replace(sanitized_fragments, '  ', ' ')  # Remove extra spaces
    
    def process_fragment(fragment):
        # Remove extra spaces and sanitize further
        fragment = re.sub(r'\s+', ' ', fragment).strip()
        
        # Remove unwanted "I" or "1" at the beginning or specific positions
        if len(fragment) > 3:
            if fragment[0] == 'I':
                fragment = fragment[1:]
            if len(fragment) > 3 and (fragment[3] == 'I' or fragment[3] == '1'):
                fragment = fragment[:3] + fragment[4:]
        
        # Apply pattern checks
        if re.match(r'^[A-Z]{3} \w{5}$', fragment):
            corrected_fragment = fragment[:8]  # Remove the last digit
            print(f"Fragment corrected by removing last digit: {corrected_fragment}")
            return corrected_fragment
        elif re.match(r'^\w{3} \w{2,4}$', fragment):
            print(f"Fragment passed filter (whole plate pattern): {fragment}")
            return fragment
        elif re.match(r'^[A-Z]{3}\d{2,4}$', fragment):
            print(f"Fragment passed filter (letters and digits): {fragment}")
            return fragment
        elif re.match(r'^[A-Z]{3}$', fragment):
            print(f"Fragment passed filter (3 letters): {fragment}")
            return fragment
        elif re.match(r'^\d{2,4}$', fragment):
            print(f"Fragment passed filter (2-4 digits): {fragment}")
            return fragment
        elif len(fragment) in [6, 7] and re.match(r'^\w{6,7}$', fragment):
            corrected_fragment = fragment[:3] + ' ' + fragment[3:]
            print(f"Fragment passed filter (corrected long fragment): {corrected_fragment}")
            return corrected_fragment
        else:
            print(f"Fragment did not pass filter: {fragment}")
            return None

    # Process each fragment using NumPy vectorized operations
    processed_fragments = np.array([process_fragment(fragment) for fragment in sanitized_fragments])
    filtered_fragments = processed_fragments[processed_fragments != np.array(None)]  # Remove `None` entries
    
    # If only one fragment is detected, assume it could be the whole license plate number
    if len(filtered_fragments) == 1:
        fragment = filtered_fragments[0]
        if re.match(r'^\w{3} \w{2,4}$', fragment) or re.match(r'^[A-Z]{3}\d{2,4}$', fragment):
            print(f"Single fragment assumed as whole plate: {fragment}")
        else:
            print(f"Single fragment does not match whole plate pattern: {fragment}")
    
    return filtered_fragments.tolist()


def apply_easyocr(image):
    """Process the cropped license plate image using EasyOCR with higher confidence threshold."""
    reader = easyocr.Reader(['en'], gpu=True)
    ocr_results = reader.readtext(image)  # Process image with EasyOCR

    threshold = 0.3  # Set the OCR confidence threshold
    detected_texts = []
    bounding_boxes = []

    # Extract OCR results with confidence scores above the threshold
    for bbox, text, score in ocr_results:
        if score > threshold:
            detected_texts.append(text.upper())  # Convert to uppercase
            bounding_boxes.append(bbox)  # Save bounding boxes for annotations

    # Annotate image with bounding boxes and detected text
    for bbox, text in zip(bounding_boxes, detected_texts):
        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 3)
        cv2.putText(image, text, top_left, cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 3)

    print(f"Initial Detected Texts: {detected_texts}")

    # Apply the preliminary filter to detected texts
    filtered_fragments = preliminary_filter_fragments(detected_texts)
    concatenated_text = " ".join(filtered_fragments)
    print(f"Concatenated Text (filtered): {concatenated_text}")

    # Correct misread characters after the preliminary filter
    corrected_text = correct_misread_characters(concatenated_text)

    # Proceed with cleaning of the corrected text
    cleaned_text = re.sub(r"[^\w\s]", "", corrected_text.strip())

    filtered_text = None
    if cleaned_text:
        filtered_text = extract_license_plate(cleaned_text)

        # If no license plate is extracted, try flipping the text and re-extracting
        if not filtered_text:
            flipped_text = flip_license_plate(cleaned_text)
            if flipped_text:
                filtered_text = extract_license_plate(flipped_text)

        # Validate the final filtered text
        if filtered_text and is_valid_license_plate(filtered_text):
            detected_texts = [filtered_text]
            print(f"Final valid license plate: {filtered_text}")
            print(f"Final detected text: {detected_texts}")
        else:
            print("No valid license plate detected after processing.")
    else:
        print("No cleaned text found; skipping license plate extraction.")

    return image, detected_texts

import easyocr
import cv2
import re
from typing import List, Tuple

def apply_easyocrmultiple(images: List[np.ndarray], batch_size=5) -> List[Tuple[np.ndarray, List[str]]]:
    """Process the cropped license plate images using EasyOCR with higher confidence threshold in batches."""
    reader = easyocr.Reader(['en'], gpu=True)
    results = []

    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        batch_results = []

        for image in batch:
            ocr_results = reader.readtext(image)  # Process image with EasyOCR

            threshold = 0.3  # Set the OCR confidence threshold
            detected_texts = []
            bounding_boxes = []

            # Extract OCR results with confidence scores above the threshold
            for bbox, text, score in ocr_results:
                if score > threshold:
                    detected_texts.append(text.upper())  # Convert to uppercase
                    bounding_boxes.append(bbox)  # Save bounding boxes for annotations

            # Annotate image with bounding boxes and detected text
            for bbox, text in zip(bounding_boxes, detected_texts):
                top_left = tuple(map(int, bbox[0]))
                bottom_right = tuple(map(int, bbox[2]))
                cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 3)
                cv2.putText(image, text, top_left, cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 3)

            print(f"Initial Detected Texts: {detected_texts}")

            # Apply the preliminary filter to detected texts
            filtered_fragments = preliminary_filter_fragments(detected_texts)
            concatenated_text = " ".join(filtered_fragments)
            print(f"Concatenated Text (filtered): {concatenated_text}")

            # Correct misread characters after the preliminary filter
            corrected_text = correct_misread_characters(concatenated_text)

            # Proceed with cleaning of the corrected text
            cleaned_text = re.sub(r"[^\w\s]", "", corrected_text.strip())

            filtered_text = None
            if cleaned_text:
                filtered_text = extract_license_plate(cleaned_text)

                # If no license plate is extracted, try flipping the text and re-extracting
                if not filtered_text:
                    flipped_text = flip_license_plate(cleaned_text)
                    if flipped_text:
                        filtered_text = extract_license_plate(flipped_text)

                # Validate the final filtered text
                if filtered_text and is_valid_license_plate(filtered_text):
                    detected_texts = [filtered_text]
                    print(f"Final valid license plate: {filtered_text}")
                else:
                    print("No valid license plate detected after processing.")
            else:
                print("No cleaned text found; skipping license plate extraction.")

            batch_results.append((image, detected_texts))
        
        results.extend(batch_results)

    return results
