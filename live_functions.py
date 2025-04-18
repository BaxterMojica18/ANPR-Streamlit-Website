import streamlit as st
import cv2
import numpy as np
from camera_functions import initialize_window_and_capture, display_camera_feed, initialize_camera
from prediction_functions2 import predict, preliminary_filter_fragments
import easyocr

def apply_basicocr2(image):
    """
    Process the cropped license plate image using EasyOCR with a high confidence threshold.
    """
    reader = easyocr.Reader(['en'], gpu=True)
    text_ = reader.readtext(image)

    threshold = 0.2  # Set the OCR confidence threshold
    detected_texts = []

    for bbox, text, score in text_:
        # Process only high-confidence OCR results
        if score > threshold:
            detected_texts.append(text.upper())
            print(f"Detected text fragment: {text.upper()}, Confidence score: {score}")

            top_left = tuple(np.int32(bbox[0]))
            bottom_right = tuple(np.int32(bbox[2]))
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 3)
            cv2.putText(image, text.upper(), top_left, cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 3)

    # Apply the preliminary filter to detected texts
    concatenated_text = " ".join(detected_texts)
    print(f"Concatenated Text (filtered): {concatenated_text}")

    return image, concatenated_text

def live_camera_inference(fps=50):
    frame_width = 640
    frame_height = 480

    # Use the selected camera from session state
    selected_camera = st.session_state.get("selected_camera", st.session_state.selected_camera)

    # Initialize camera capture only if `camera_active` is True
    if "cap" not in st.session_state or st.session_state.cap is None:
        st.session_state.cap = initialize_window_and_capture(frame_width, frame_height, selected_camera)

    # Check if camera is active before capturing frames
    if st.session_state.camera_active:
        cap = st.session_state.cap
        ###cap.set(cv2.CAP_PROP_EXPOSURE, 0.1)
        

        # Create placeholders for the camera feed and success messages
        camera_feed_placeholder = st.empty()
        success_placeholder = st.empty()

        while st.session_state.camera_active:
            ret, frame = cap.read()
            if not ret:
                break

            # Perform prediction on the BGR frame
            detected_name, detected_image = predict(frame)
            processed_frame = frame  # Default to original frame in case of no detections

            if detected_name:
                frame, detected_texts = apply_basicocr2(detected_image)

                if detected_texts:
                    concatenated_text = " ".join(detected_texts)
                    # Update the success message without breaking the loop
                    success_placeholder.success(f"Detected License Plate: {concatenated_text}")

            # Display live camera feed with bounding boxes and text
            camera_feed_placeholder.image(frame, channels="BGR", caption="Camera Feed", use_container_width=True)

        # Stop camera and clean up if deactivated
        cap.release()
        cv2.destroyAllWindows()

