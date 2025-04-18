import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import os
import csv
from prediction_functions import apply_easyocr

csv_file = "detections.csv"

def store_detected_image(image):
    processed_image, detected_texts = apply_easyocr(image)
    st.session_state.detected_image = processed_image
    st.session_state.detected_texts = detected_texts  # Store detected_texts directly
    success_image = st.success("Detected image with text has been stored successfully.")
    success_image.empty()  # Clear the success message after 5 seconds

def show_detected_image():
    if "detected_image" in st.session_state and st.session_state.detected_image is not None:
        st.image(st.session_state.detected_image, channels="BGR")
        if "detected_texts" in st.session_state and st.session_state.detected_texts:
            for text in st.session_state.detected_texts:
                st.success(f"Your License Plate Number is: {text}")
    else:
        st.warning("No detected image to display.")

def get_current_detection_count(csv_file="E:/Downloads/ImageIdentification/detections.csv"):
    """Get the current detection count from the CSV file."""
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file, on_bad_lines='skip')  # Skip bad lines
            if len(df) > 0:
                return int(df["Detection Number"].max())
        except pd.errors.EmptyDataError:
            return 0  # Handle case where file exists but is empty
    return 0

def correct_detection_numbers(csv_file="E:/Downloads/ImageIdentification/detections.csv"):
    """Check and correct duplicate detection numbers in the CSV file."""
    if os.path.exists(csv_file):
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)

            # Sort by detection number and reset the index
            df = df.sort_values(by="Detection Number").reset_index(drop=True)

            # Correct duplicate detection numbers using NumPy for efficiency
            detection_numbers = df["Detection Number"].to_numpy()
            unique_numbers = np.maximum.accumulate(detection_numbers)
            df["Detection Number"] = unique_numbers

            # Save the updated DataFrame back to the CSV file
            df.to_csv(csv_file, index=False)
            success_correct = st.success("Detection numbers have been corrected.")
            success_correct.empty()
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.warning("CSV file does not exist.")

# Define the column order
COLUMN_ORDER = ["Detection Number", "Detection Name", "Date of Detection", "Time of Detection", "Plate Number"]

def get_next_detection_number(csv_file="E:/Downloads/ImageIdentification/detections.csv"):
    """
    Get the next detection number from the CSV file.
    If the file is empty or doesn't exist, start with 1.
    """
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file, on_bad_lines='skip')
            if not df.empty and "Detection Number" in df.columns:
                return int(df["Detection Number"].max()) + 1
        except pd.errors.EmptyDataError:
            return 1  # Handle case where file exists but is empty
    return 1  # Start with 1 if file doesn't exist

def rearrange_csv(csv_file="E:/Downloads/ImageIdentification/detections.csv"):
    """
    Rearrange the CSV file by sorting it and ensuring consistent detection numbers.
    """
    if not os.path.exists(csv_file):
        st.warning("CSV file does not exist.")
        return

    try:
        df = pd.read_csv(csv_file)
        if df.empty:
            st.warning("CSV file is empty.")
            return

        # Ensure all required columns exist
        for col in COLUMN_ORDER:
            if col not in df.columns:
                df[col] = ""  # Add missing columns with empty values

        # Convert date and time columns to a single datetime column for sorting
        df['Datetime'] = pd.to_datetime(
            df['Date of Detection'] + ' ' + df['Time of Detection'],
            format='%m/%d/%Y %H:%M:%S', errors='coerce'
        )

        # Sort by the new datetime column
        df = df.sort_values(by='Datetime').reset_index(drop=True)

        # Reassign detection numbers sequentially
        df['Detection Number'] = np.arange(1, len(df) + 1)

        # Save the cleaned DataFrame back to the CSV
        df.to_csv(csv_file, index=False, columns=COLUMN_ORDER)
        arrangesuccess = st.success("CSV file rearranged and detection numbers corrected.")
        arrangesuccess.empty()
    except Exception as e:
        st.error(f"Error rearranging CSV file: {e}")

# Define the column order
COLUMN_ORDER = ["Detection Number", "Detection Name", "Date of Detection", "Time of Detection", "Plate Number"]

def save_detection(detection_number, detection_name, date_of_detection, time_of_detection, plate_number, csv_file):

    # Construct data in the correct column order
    data = {
        "Detection Number": [detection_number],
        "Detection Name": [detection_name],
        "Date of Detection": [date_of_detection],
        "Time of Detection": [time_of_detection],
        "Plate Number": [plate_number]
    }

    # Convert to NumPy array and then to DataFrame
    np_data = np.array([detection_number, detection_name, date_of_detection, time_of_detection, plate_number])
    df = pd.DataFrame([np_data], columns=COLUMN_ORDER)

    # Append to CSV, ensuring consistent column order and append mode
    df.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False, columns=COLUMN_ORDER)

    # Display success message
    #st.success(f"Detection {detection_number} saved successfully!")

def clean_csv(file_path):
    """
    Clean the CSV file by ensuring required fields, parsing dates and times, 
    sorting records, and removing duplicates.

    Parameters:
    - file_path (str): Path to the CSV file to clean.
    """
    # Step 1: Read the existing data from the CSV file
    if not os.path.exists(file_path):
        print(f"CSV file not found: {file_path}")
        return

    records = []

    with open(file_path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            # Ensure each record has the required fields
            try:
                detection_number = int(row['Detection Number'].strip())
                detection_name = row['Detection Name'].strip()
                date_of_detection = row['Date of Detection'].strip()
                time_of_detection = row['Time of Detection'].strip()
                plate_number = row.get('Plate Number', '').strip()
                
                # Parse date and time to ensure consistency
                datetime.strptime(date_of_detection, "%m/%d/%Y")  # MM/DD/YYYY format
                datetime.strptime(time_of_detection, "%H:%M:%S")  # HH:MM:SS format
                
                # Add record to the list
                records.append([
                    detection_number, detection_name, date_of_detection, 
                    time_of_detection, plate_number
                ])
                
            except ValueError:
                # Skip rows with missing or invalid data
                print(f"Skipping invalid row: {row}")

    # Convert to NumPy array for efficient processing
    records = np.array(records)

    # Step 2: Sort records by Detection Number (or by Date/Time if needed)
    detection_numbers = records[:, 0].astype(int)
    sorted_indices = np.argsort(detection_numbers)
    records = records[sorted_indices]

    # Step 3: Remove duplicates (based on Detection Number and Plate Number)
    unique_records = []
    seen = set()
    for record in records:
        key = (record[0], record[4])  # Detection Number and Plate Number
        if key not in seen:
            seen.add(key)
            unique_records.append(record)

    # Convert back to DataFrame for writing to CSV
    df = pd.DataFrame(unique_records, columns=[
        'Detection Number', 'Detection Name', 'Date of Detection', 
        'Time of Detection', 'Plate Number'
    ])

    # Step 4: Write the cleaned data back to the CSV file
    df.to_csv(file_path, index=False)
    print("CSV file cleaned and saved successfully!")

