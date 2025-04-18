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
    replacements = {
    'O': '0', 'I': '1', 'Z': '2', 'S': '5', 
    ']': '1', '|': '1', 'B': '8', 'G': '6', 'Q': '0'
    }
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

def correct_numeric_fragment(fragment):
    """Fix common OCR mistakes in numeric-heavy parts only."""
    replacements = {
        ']': '1', '[': '1', '(': '1', ')': '1', '|': '1',
        'O': '0', 'D': '0', 'B': '8', 'S': '5', '$': '5'
    }
    for wrong, right in replacements.items():
        fragment = fragment.replace(wrong, right)
    return fragment


def is_mostly_numbers(part):
    """
    Check if a part is mostly numbers using NumPy for optimization.
    """
    char_array = np.array(list(part))  # Convert string to a NumPy array of characters
    is_alpha = np.char.isalpha(char_array)  # Check which characters are alphabetic
    is_digit = np.char.isdigit(char_array)  # Check which characters are numeric
    return np.sum(is_digit) > np.sum(is_alpha)  # Compare counts of numeric and alphabetic characters


def preliminary_filter_fragments(detected_texts):
    """
    Filters the detected text fragments to retain only those likely to be part of a valid license plate.
    Optimized with NumPy for better performance. Also removes duplicates and joins fragments like 'ABC' + '1234'.
    """
    if not detected_texts or not isinstance(detected_texts, list):
        print("Invalid input: detected_texts must be a list of strings.")
        return []

    # Preprocessing: Clean and sanitize fragments
    sanitized_fragments = np.array([
        normalize_ocr_fragment(re.sub(r"[^\w\s]", "", text).strip())
        for text in detected_texts
    ])
    sanitized_fragments = np.char.replace(sanitized_fragments, '  ', ' ')  # Remove extra spaces

    def process_fragment(fragment):
        fragment = re.sub(r'\s+', ' ', fragment).strip()

        # Apply numeric correction if mostly digits
        if is_mostly_numbers(fragment):
            original = fragment
            fragment = correct_numeric_fragment(fragment)
            print(f"Corrected numeric fragment: '{original}' → '{fragment}'")

        # Further fix common structure issues
        if len(fragment) > 3:
            if fragment[0] == 'I':
                fragment = fragment[1:]
            if len(fragment) > 3 and (fragment[3] == 'I' or fragment[3] == '1'):
                fragment = fragment[:3] + fragment[4:]

        # Apply known license plate format patterns
        if re.match(r'^[A-Z]{3} \w{5}$', fragment):
            corrected_fragment = fragment[:8]
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

    # Process each fragment
    processed_fragments = [process_fragment(fragment) for fragment in sanitized_fragments]
    filtered_fragments = list(filter(None, processed_fragments))

    # Deduplicate
    unique_fragments = list(set(filtered_fragments))

    # Try to join two-part fragments
    if len(unique_fragments) == 2:
        part1, part2 = unique_fragments
        if re.match(r'^[A-Z]{3}$', part1) and re.match(r'^\d{2,4}$', part2):
            combined = f"{part1} {part2}"
            print(f"Joined fragments into one plate: {combined}")
            return [combined]
        elif re.match(r'^[A-Z]{3}$', part2) and re.match(r'^\d{2,4}$', part1):
            combined = f"{part2} {part1}"
            print(f"Joined fragments into one plate: {combined}")
            return [combined]

    return unique_fragments


def preprocess_plate_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize to enlarge small characters
    scaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    # Apply adaptive thresholding for better contrast
    thresh = cv2.adaptiveThreshold(scaled, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 41, 15)
    # Invert to keep text black on white if needed
    thresh = cv2.bitwise_not(thresh)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

def normalize_ocr_fragment(text):
    replacements = {
        ']': '1', '[': '1', '(': '1', ')': '1', '|': '1',
        'D': '0', 'O': '0', 'I': '1', 'S': '5', '$': '5'
    }
    for wrong, right in replacements.items():
        text = text.replace(wrong, right)
    return text


def apply_easyocr(image):
    """Process the cropped license plate image using EasyOCR with higher confidence threshold."""
    reader = easyocr.Reader(['en'], gpu=True)
    preprocessed_image = preprocess_plate_image(image)
    ocr_results = reader.readtext(preprocessed_image)  # Process image with EasyOCR

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
