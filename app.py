import streamlit as st

# Set page configuration at the beginning
st.set_page_config(page_title="Automatic License Plate Identifier", page_icon=":camera:")

import cv2
import numpy as np
import datetime
import os
import tempfile
import base64
from camera_functions import select_camera_0, select_camera_1, select_camera_2
from prediction_functions2 import predict, apply_easyocr, is_valid_license_plate
from storage_functions import show_detected_image, get_current_detection_count, save_detection, correct_detection_numbers, rearrange_csv

from live_functions import live_camera_inference
from carcounter import count_vehicles
from carcounterimage import count_vehicles_in_image
from multiplelicense import detect_multiple_plates
from anprimage import anpr_image
from anprvideo import anpr_video
from yolov11 import process_video
import streamlit.components.v1 as components
import torch
import streamlit as st
import time
import pandas as pd

def clear_gpu_memory():
    """
    Clears the GPU memory to offload resources.
    Displays a success message in Streamlit and logs a message in the terminal.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("GPU memory has been cleared.")
        cleargpu = st.success("GPU Memory Offloaded")
        cleargpu.empty()
    else:
        print("No GPU detected. Skipping GPU memory clearance.")
        nogpu = st.warning("No GPU detected. Skipping GPU memory clearance.")
        nogpu.empty()


# Path to the CSV file
csv_file = "detections.csv"
csv_file_path = csv_file
output_video_path = "countedcars/trackedvideo.mp4"  # Path for saving processed video
output_image_path = "countedcars/trackedimage.jpg"  # Path for saving processed video

def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Apply the custom CSS
load_css("styles.css")

# Function to load images as base64 for custom background
def load_bg_as_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Initialize sidebar state in session state
if "sidebar_open" not in st.session_state:
    st.session_state.sidebar_open = True  # Default sidebar state is open

# Load the external JavaScript file for sidebar detection
with open("functions.js", "r") as js_file:
    sidebar_js = f"<script>{js_file.read()}</script>"

# Inject the JavaScript into Streamlit
components.html(sidebar_js, height=0)

# Placeholder for the logo
logo_placeholder = st.empty()

st.sidebar.header("Automatic License Plate Identifier")    
st.sidebar.markdown("---")  # Separator line

# Path for CSV file where detections will be saved
csv_file_path = "detections.csv"



# Initialize session states for camera control
if "selected_camera" not in st.session_state:
    st.session_state.selected_camera = 0  # Default to Camera 0
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False  # Camera is initially inactive
if "detection_count" not in st.session_state:
    st.session_state.detection_count = 0  # Start detection count at 0
    


# Source selection for different functionalities
source_selection = st.sidebar.radio("Choose source:", ("Read License Plate Photo", "Read Multiple Licenses in Image", "Open Camera", "Read License Plate in Video", "Live Detection", "Track and Count Vehicles in Image", "Track and Count Vehicles in Video", "ANPR in Video", "ANPR in Image"))



st.sidebar.markdown("---")  # Separator line

# Define button classes for styling
button_class = "alpi-button"  # Custom class for buttons
active_button_class = "alpi-button-active"  # Class for active button

# Buttons to select the camera, calling specific functions
if st.sidebar.button("Camera 0"):
    select_camera_0()
if st.sidebar.button("Camera 1"):
    select_camera_1()
if st.sidebar.button("Camera 2"):
    select_camera_2()
    
st.sidebar.markdown("---")  # Separator line




if source_selection == "Read License Plate Photo": #or st.sidebar.button("Upload Photo"):
    st.title("Upload Photo")
    st.markdown(  '''
    <span style="color: black;">
    Please upload an image file containing a license plate. You can upload the image by clicking the button below. It accepts common image file types which is JPEG, JPG, and PNG.
    </span>
    ''',
    unsafe_allow_html=True,
    )
    #st.markdown(','':black[Please upload an image file containing a license plate. You can upload the image by clicking the button below. It accepts common image file types which is JPEG, JPG, and PNG,]''', unsafe_allow_html=True)
    st.markdown(" ")
    uploaded_file = st.file_uploader("Choose an image:", type=["jpg", "png"])
    if uploaded_file is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        st.image(image, channels="BGR")

        if st.button("Predict", key="predict_image"):
            detected_name, detected_image = predict(image)
            if detected_name:
                st.success("License Plate detected in the image!")

                # Store the detected image temporarily
                temp_image_path = os.path.join(tempfile.gettempdir(), 'temp_image.jpg')
                cv2.imwrite(temp_image_path, detected_image)

                # Apply OCR and get detected text
                detected_image, detected_texts = apply_easyocr(detected_image)

                if detected_texts:
                    detected_plate = ''.join(detected_texts).upper()
                    
                    if is_valid_license_plate(detected_plate):
                        # Increment detection count
                        st.session_state.detection_count += 1
                        now = datetime.datetime.now()
                        date_of_detection = now.strftime("%Y-%m-%d")
                        time_of_detection = now.strftime("%H:%M:%S")

                        # Save detection with the CSV file path
                        save_detection(st.session_state.detection_count, detected_name, date_of_detection, time_of_detection, detected_plate, csv_file)
                        
                        # Display detected plate
                        st.success(f"Your License Plate Number is: {detected_plate}")
                        rgb_detected_image = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)
                        st.image(rgb_detected_image)
                    else:
                        st.error("Invalid license plate format detected. Image has been deleted.")
                        st.error(detected_plate)
                else:
                    st.warning("No License Plate detected in the image.")
            else:
                st.warning("No License Plate detected in the image.")
                
if source_selection == "Read Multiple Licenses in Image":
    st.title("Upload Photo with Multiple Licenses")
    st.markdown('''
    <span style="color: black;">
    Please upload an image file containing multiple license plates. You can upload the image by clicking the button below. It accepts common image file types: JPEG, JPG, and PNG.
    </span>
    ''', unsafe_allow_html=True)

    st.markdown(" ")
    uploaded_file = st.file_uploader("Choose an image:", type=["jpg", "png"])
    
    if uploaded_file is not None:
        # Read and display the uploaded image
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        st.image(image, channels="BGR")

        if st.button("Predict", key="predict_image"):
            # Detect multiple license plates
            detection_results = detect_multiple_plates(image)
            
            if detection_results:
                st.success(f"Detected {len(detection_results)} license plate(s) in the image!")
                
                for idx, (cropped_plate, detected_text) in enumerate(detection_results, start=1):
                    st.markdown(f"### License Plate {idx}")
                    
                    # Display the cropped license plate image
                    rgb_cropped_plate = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2RGB)
                    st.image(rgb_cropped_plate, caption=f"License Plate {idx}")
                    
                    # Validate the detected license plate text
                    if is_valid_license_plate(detected_text):
                        # Increment detection count
                        st.session_state.detection_count += 1
                        now = datetime.datetime.now()
                        date_of_detection = now.strftime("%Y-%m-%d")
                        time_of_detection = now.strftime("%H:%M:%S")

                        # Save detection to the CSV file
                        save_detection(
                            st.session_state.detection_count,
                            "A LicensePlate is detected!",
                            date_of_detection,
                            time_of_detection,
                            detected_text,
                            csv_file
                        )

                        # Display the detected text
                        st.success(f"Detected License Plate Number: {detected_text}")
                    else:
                        st.error(f"Invalid license plate format for Plate {idx}: {detected_text}")
            else:
                st.warning("No license plates detected in the image.")


if source_selection == "Track and Count Vehicles in Video":
    st.title("Track and Count Vehicles in Video")
    st.markdown("")
    st.markdown(  '''
    <span style="color: black;">
    Please upload a video file containing vehicles that are moving. They will be counted and read and then displayed below as a playable video of all the detected vehicles It accepts common video file types which is MP4, AVI, MOV, and MPEG4.
    </span>
    ''',
    unsafe_allow_html=True,
    )
    st.markdown(" ")
    uploaded_video = st.file_uploader("Choose a video file:", type=["mp4", "avi", "mov"], key="trackvideo")
    
    if uploaded_video is not None:
        # Display the uploaded video
        st.video(uploaded_video)

        # Process video when "Process Video" button is clicked
        if st.button("Process Video"):
            # Save uploaded video to a temporary location
            temp_input_path = f"temp_uploaded_{uploaded_video.name}"
            with open(temp_input_path, "wb") as f:
                f.write(uploaded_video.read())
            
            # Process the video with count_vehicles function
            processed_video_path = count_vehicles(temp_input_path, output_video_path)
            
            # Set session state for processed video
            st.session_state['video_processed'] = True
            st.session_state['output_video_path'] = processed_video_path
            st.success("Video processed successfully!")

            # Clean up the temporary file
            os.remove(temp_input_path)

    # Display the "Show Video" button if video has been processed
    if st.session_state.get('video_processed', False):
        output_video_path = st.session_state.get('output_video_path', None)
        if output_video_path and os.path.exists(output_video_path):
            if st.button("Show Video"):
                st.video(output_video_path, start_time=0)

if source_selection == "Track and Count Vehicles in Image":
    st.title("Track and Count Vehicles in Image")
    st.markdown(  '''
    <span style="color: black;">
    Please upload an image file containing vehicles and a clear license plate. The vehicles in the image will be counted and then a processed image is displayed below as an image of all the detected vehicles after pressing the 'Show' button.It accepts common image file types: JPEG, JPG, and PNG.
    </span>
    ''',
    unsafe_allow_html=True,
    )
    st.markdown(" ")
    uploaded_file = st.file_uploader("Choose an image:", type=["jpg", "png"])
    if uploaded_file is not None:
        car_image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        st.image(car_image, channels="BGR")
        if st.button("Process Image"):
            results = count_vehicles_in_image(car_image, csv_file_path)  # Process the video and display processed frames
            
            #if results:
                #st.success(f"Detected {len(results)} license plate(s) in the image!")
            
if source_selection == "ANPR in Image":
        st.title("Automatic Number Plate Recognition in Image")
        st.markdown(  '''
        <span style="color: black;">
        Please upload an image file containing vehicles with a clear license plate. The vehicles in the image will be counted and then a processed image is displayed below as an image of all the detected vehicles after pressing the 'Show' button. It accepts common image file types: JPEG, JPG, and PNG.
        </span>
        ''',
        unsafe_allow_html=True,
        )
        st.markdown(" ")
        uploaded_file = st.file_uploader("Choose an image:", type=["jpg", "png"])
        if uploaded_file is not None:
            car_image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
            st.image(car_image, channels="BGR")
            if st.button("Process Image"):
                anpr_results = anpr_image(car_image, csv_file_path)  # Process the video and display processed frames         
            
if source_selection == "Read License Plate in Video":
    st.title("Automatic Number Plate Recognition in Video")
    st.markdown("")
    st.markdown('''
    <span style="color: black;">
    Please upload a video file containing vehicles that are moving. They will be counted and read and then displayed below as a playable video of all the detected vehicles. It will count, track, and read the license plate of all the detected vehicles. It accepts common video file types which is MP4, AVI, MOV, and MPEG4.
    </span>
    ''',
    unsafe_allow_html=True,
    )
    st.markdown(" ")
    uploaded_video = st.file_uploader("Choose a video file:", type=["mp4", "avi", "mov"], key="trackvideo2")

    if uploaded_video is not None:
        # Display the uploaded video
        st.video(uploaded_video)

        # Process video when "Process Video" button is clicked
        if st.button("Process Video"):
            # Save uploaded video to a temporary location
            temp_input_path = f"temp_uploaded_{uploaded_video.name}"
            with open(temp_input_path, "wb") as f:
                f.write(uploaded_video.read())
            
            # Process the video with the anpr_video function
            try:
                processed_video_path, detected_plates = anpr_video(temp_input_path, "output_video.mp4")
                
                # Set session state for processed video
                st.session_state['video_processed'] = True
                st.session_state['output_video_path'] = processed_video_path
                st.success("Video processed successfully!")

                # Check for detected license plates
                if detected_plates:
                    st.markdown("### Detected License Plates:")
                    for idx, plate in enumerate(detected_plates, start=1):
                        st.success(f"Detected License Plate: {plate}")

                        # Save each detected license plate to the CSV file
                        detection_number = idx
                        detection_name = "A License Plate is Detected!"
                        date_of_detection = pd.Timestamp.now().strftime("%Y-%m-%d")
                        time_of_detection = pd.Timestamp.now().strftime("%H:%M:%S")
                        save_detection(
                            detection_number,
                            detection_name,
                            date_of_detection,
                            time_of_detection,
                            plate,
                            csv_file="detections.csv"
                        )
                else:
                    st.warning("No license plates were detected in the video.")
                
            except Exception as e:
                st.error(f"Error during processing: {e}")
            finally:
                #Clean up the temporary file
                os.remove(temp_input_path)

    # Display the "Show Video" button if video has been processed
    if st.session_state.get('video_processed', False):
        output_video_path = st.session_state.get('output_video_path', None)
        if output_video_path and os.path.exists(output_video_path):
            if st.button("Show Video"):
                st.video(output_video_path, start_time=0)



        
if source_selection == "ANPR in Video":
    st.title("Upload A Video")
    st.markdown('''
    <span style="color: black;">
    Please upload a video file containing a license plate. It accepts common video file types: MP4, AVI, MOV, and MPEG4.
    </span>
    ''', unsafe_allow_html=True)
    
    st.markdown(" ")
    uploaded_video = st.file_uploader("Choose a video file:", type=["mp4", "avi", "mov"], key="uploadvideo")
    
    if uploaded_video is not None:
        st.video(uploaded_video)  # Display the original video
        
        if st.button("Process Video"):
            import tempfile
            import os

            # Save uploaded video to a temporary file
            temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_video_file.write(uploaded_video.read())
            temp_video_file.close()

            # Define output paths
            os.makedirs("processed_frames", exist_ok=True)
            output_video_path = os.path.join("processed_frames", "processed_video.mp4")

            # Process the video using the saved file path
            process_video(temp_video_file.name, output_video_path, csv_file_path)

            # Save session state to indicate processing is complete
            st.session_state.video_processed = True
            st.session_state.output_video_path = output_video_path

        # Show processed video
        if st.session_state.get('video_processed', False):
            output_video_path = st.session_state.get('output_video_path', None)
            if output_video_path and os.path.exists(output_video_path):
                if st.button("Show Video"):
                    with st.empty():
                        st.video(output_video_path, start_time=0)





if source_selection == "Live Detection": # or st.sidebar.button("Live Detection"):
    st.title("Live Detection")
    st.markdown(  '''
    <span style="color: black;">
   In this option, the camera will turn on and will start detecting if there are license plates available in the frame realtime.
    </span>
    ''',
    unsafe_allow_html=True,
    )
    st.markdown(" ")
    # Add Stop Camera button
    if st.button("Stop Camera", key="stop_live_camera"):
        st.session_state.camera_active = False  # Deactivate the camera
        st.write("Camera stopped.")
        if "cap" in st.session_state and st.session_state.cap is not None:
            st.session_state.cap.release()  # Release the camera when stopped
            st.session_state.cap = None  # Clear the cap variable
            
      # Camera settings sliders
    #st.sidebar.header("Camera Settings")
    #brightness = st.sidebar.slider("Brightness", min_value=0, max_value=100, value=50, step=1)
    #exposure = st.sidebar.slider("Exposure", min_value=-10, max_value=10, value=0, step=1)
    
    # Add Start Camera button
    if st.button("Start Camera", key="start_live_camera"):
        st.session_state.camera_active = True  # Set the camera to active
        live_camera_inference(fps=30)  # Start live inference when button is clicked

    

if source_selection == "Open Camera": # or st.sidebar.button("Open Camera"):
    st.title("Open Camera and Take Photo")
    st.markdown(  '''
    <span style="color: black;">
    In this option, the selected camera will open and once the license plate is in frame, press the 'Take a Photo' button and it will process the image based on that selected frame and display an output.
    </span>
    ''',
    unsafe_allow_html=True,
    )
    st.markdown(" ")
    # Show the Start Camera button if the camera is not active
    if not st.session_state.camera_active:
        if st.button("Start Camera", key="start_camera"):
            st.session_state.camera_active = True  # Activate the camera
            st.session_state.cap = cv2.VideoCapture(st.session_state.selected_camera)  # Open camera
            st.write("Camera started.")
            
    # Show the Take Photo button while the camera is active
    if st.session_state.camera_active:
        # Outside the loop: Add a button to take a photo
        if st.button("Take Photo"):
            if st.session_state.cap is not None:
                ret, frame = st.session_state.cap.read()
                if ret:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Display the captured image
                    st.image(rgb_frame, channels="RGB", caption="Captured Image")

                    detected_name, detected_image = predict(rgb_frame)
                    if detected_name:
                        st.success(f"Detected object: {detected_name}")

                        # Store the detected image temporarily
                        temp_image_path = os.path.join(tempfile.gettempdir(), 'temp_image.jpg')
                        cv2.imwrite(temp_image_path, detected_image)

                        # Apply OCR and get detected text
                        detected_image, detected_texts = apply_easyocr(detected_image)

                        if detected_texts:
                            detected_plate = ''.join(detected_texts).upper()
                            
                            if is_valid_license_plate(detected_plate):
                                # Increment detection count
                                st.session_state.detection_count += 1
                                now = datetime.datetime.now()
                                date_of_detection = now.strftime("%Y-%m-%d")
                                time_of_detection = now.strftime("%H:%M:%S")

                                # Save detection with the CSV file path
                                save_detection(st.session_state.detection_count, detected_name, date_of_detection, time_of_detection, detected_plate, csv_file_path)
                                
                                # Display detected plate
                                st.success(f"Your License Plate Number is: {detected_plate}")
                                rgb_detected_image = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)
                                st.image(rgb_detected_image)
                            else:
                                st.error("Invalid license plate format detected. Image has been deleted.")
                        else:
                            st.warning("No License Plate detected.")
                else:
                    st.warning("No frame captured.")

        # Add a button to show the stored image if it exists
        if st.button("Show Detected Image", key="show_image"):
            show_detected_image()

    # Show the Stop Camera button if the camera is active
    if st.session_state.camera_active:
        if st.button("Stop Camera", key="stop_camera"):
            st.session_state.camera_active = False  # Deactivate the camera
            st.write("Camera input has been stopped.")
            if st.session_state.cap is not None:
                st.session_state.cap.release()  # Release the camera when stopped
                st.session_state.cap = None  # Clear the cap variable

        # Create a placeholder for the camera feed
        frame_placeholder = st.empty()

        # Capture frames while the camera is active
        while st.session_state.camera_active:
            if st.session_state.cap is not None:
                ret, frame = st.session_state.cap.read()
                if ret:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB for displaying
                    frame_placeholder.image(rgb_frame, channels="RGB", caption="Camera Feed", use_container_width=True)  # Update the image

                # Control the frame rate (24 FPS)
                time.sleep(1 / 24)

        # Release the camera if it's not active
        if not st.session_state.camera_active and st.session_state.cap is not None:
            st.session_state.cap.release()
            
# State management for option change detection
if "last_selection" not in st.session_state:
    st.session_state.last_selection = None

# Check if the selected option has changed
if source_selection != st.session_state.last_selection:
    clear_gpu_memory()  # Clear GPU memory
    st.session_state.last_selection = source_selection  # Update the last selected option

# Run data cleaning steps in the correct order
#clean_csv(csv_file)                               # Step 1: Clean the CSV data
#rearrange_csv(csv_file)                           # Step 2: Rearrange columns, remove duplicates, and sort
#correct_detection_numbers(csv_file)               # Step 3: Correct detection numbers sequentially
#current_detection_count = get_current_detection_count(csv_file)  # Step 4: Get the latest detection count
