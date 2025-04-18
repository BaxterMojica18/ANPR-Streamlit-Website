import cv2
import streamlit as st

def initialize_window_and_capture(window_width, window_height, camera_index):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1 if camera_index == 0 else 0)  # Try the other camera if the selected one fails
    cap.set(3, window_width)
    cap.set(4, window_height)
    cap.set(cv2.CAP_PROP_EXPOSURE, -3)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap

def initialize_camera(camera_index):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        st.error(f"Error: Unable to open Camera {camera_index + 1}.")
    else:
        st.success(f"Camera {camera_index} is active.")
    return cap

def display_camera_feed(cap):
    stframe = st.empty()
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video. Stopping.")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame)

# New functions for selecting cameras
def select_camera_0():
    st.session_state.selected_camera = 0
    st.session_state.cap = initialize_camera(st.session_state.selected_camera)
    st.success("Camera 0 selected and initialized.")

def select_camera_1():
    st.session_state.selected_camera = 1
    st.session_state.cap = initialize_camera(st.session_state.selected_camera)
    st.success("Camera 1 selected and initialized.")
    
def select_camera_2():
    st.session_state.selected_camera = 2
    st.session_state.cap = initialize_camera(st.session_state.selected_camera)
    st.success("Camera 2 selected and initialized.")