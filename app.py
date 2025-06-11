import streamlit as st
import cv2
import mediapipe as mp
import tempfile
import numpy as np

mp_pose = mp.solutions.pose

def process_video(video_path):
    stride_count = 0
    total_distance = 0
    frame_count = 0

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        prev_left_ankle_x = None
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
                left_ankle_x = left_ankle.x

                if prev_left_ankle_x is not None:
                    movement = abs(left_ankle_x - prev_left_ankle_x)
                    if movement > 0.1:  # tune this threshold
                        stride_count += 1
                        total_distance += movement
                prev_left_ankle_x = left_ankle_x

            frame_count += 1

    cap.release()

    normalized_stride = total_distance / stride_count if stride_count > 0 else 0
    duration = frame_count / fps
    return stride_count, normalized_stride, duration

# Streamlit UI
st.title("Stride Analyzer")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov"])

if uploaded_file is not None:
    # Save uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    stride_count, stride_length, duration = process_video(tfile.name)
    st.success("Processing complete!")

    st.write(f"**Stride Count:** {stride_count}")
    st.write(f"**Normalized Stride Length (unitless):** {stride_length:.4f}")
    st.write(f"**Video Duration:** {duration:.2f} seconds")
