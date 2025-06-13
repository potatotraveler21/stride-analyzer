import streamlit as st
import tempfile
import cv2
import mediapipe as mp
import numpy as np
import os

st.title("Stride Analyzer with MediaPipe")

uploaded_file = st.file_uploader("Upload a side profile view of a video of your stride", type=["mp4", "mov", "avi"])
st.write("Example video: (courtesy of iTrain Hockey on YouTube)")
st.video("https://www.youtube.com/watch?v=F9Q4Ha01M-A")
if uploaded_file:
    # Save uploaded file to temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.close()

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)

    ret, frame = cap.read()
    if not ret:
        st.error("Couldn't read video.")
        cap.release()
        os.unlink(tfile.name)
    else:
        h, w, _ = frame.shape
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind

        output_tempfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        out = cv2.VideoWriter(output_tempfile.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            prev_left_ankle_x_px = None
            stride_count = 0
            frame_count = 0
            stride_frames = []
            hip_widths_2d = []
            stride_positions_2d = []
            bentKneeSum = 0
            extendedKneeSum = 0
            workingFrameCount = 0
            first_crossover_skipped = False

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True

                annotated_frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                    )

                    landmarks = results.pose_landmarks.landmark

                    def get_px_coords(part):
                        lm = landmarks[part.value]
                        return np.array([int(lm.x * w), int(lm.y * h)])

                    left_ankle_px = get_px_coords(mp_pose.PoseLandmark.LEFT_ANKLE)
                    right_ankle_px = get_px_coords(mp_pose.PoseLandmark.RIGHT_ANKLE)
                    left_hip_px = get_px_coords(mp_pose.PoseLandmark.LEFT_HIP)
                    right_hip_px = get_px_coords(mp_pose.PoseLandmark.RIGHT_HIP)

                    hip_width_px = np.linalg.norm(left_hip_px - right_hip_px)
                    hip_widths_2d.append(hip_width_px)

                    if prev_left_ankle_x_px is not None and abs(left_ankle_px[0] - prev_left_ankle_x_px) > 20:
                        stride_count += 1
                        stride_frames.append(frame_count)
                        stride_positions_2d.append(left_ankle_px)

                    prev_left_ankle_x_px = left_ankle_px[0]

                    def angle_between(a, b, c):
                        ba = a - b
                        bc = c - b
                        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

                    coords = {
                        'l_hip': left_hip_px,
                        'r_hip': right_hip_px,
                        'l_knee': get_px_coords(mp_pose.PoseLandmark.LEFT_KNEE),
                        'r_knee': get_px_coords(mp_pose.PoseLandmark.RIGHT_KNEE),
                        'l_ankle': left_ankle_px,
                        'r_ankle': right_ankle_px
                    }

                    l_knee_ang = angle_between(coords['l_hip'], coords['l_knee'], coords['l_ankle'])
                    r_knee_ang = angle_between(coords['r_hip'], coords['r_knee'], coords['r_ankle'])

                    if r_knee_ang > l_knee_ang:
                        if not first_crossover_skipped:
                            first_crossover_skipped = True
                        else:
                            bentKneeSum += l_knee_ang
                            extendedKneeSum += r_knee_ang
                            workingFrameCount += 1
                    else:
                        if not first_crossover_skipped:
                            first_crossover_skipped = True
                        else:
                            bentKneeSum += r_knee_ang
                            extendedKneeSum += l_knee_ang
                            workingFrameCount += 1

                out.write(annotated_frame)

        cap.release()
        out.release()
        os.unlink(tfile.name)

        st.subheader("Pose-Landmarked Video")
        st.write(f"Path: {output_tempfile.name}")
        st.write(f"Size: {os.path.getsize(output_tempfile.name)} bytes")

        st.video(output_tempfile.name)

        st.subheader("Results:")
        st.write(f"Total Strides Detected: {stride_count}")

        if workingFrameCount > 0:
            st.write("Average Bent Knee Angle:", round(bentKneeSum / workingFrameCount, 2))
            st.write("Average Extended Knee Angle:", round(extendedKneeSum / workingFrameCount, 2))
        else:
            st.write("No valid frames to compute knee angles.")

        if len(stride_frames) > 1:
            stride_duration = (stride_frames[-1] - stride_frames[0]) / fps
            stride_freq = (len(stride_frames) - 1) / stride_duration
            st.write(f"Stride Frequency: {stride_freq:.2f} strides/sec")

        stride_lengths_2d = []
        for i in range(1, len(stride_positions_2d)):
            p1 = stride_positions_2d[i - 1]
            p2 = stride_positions_2d[i]
            stride_lengths_2d.append(np.linalg.norm(p2 - p1))

        if stride_lengths_2d and hip_widths_2d:
            avg_stride_length_2d = np.mean(stride_lengths_2d)
            avg_hip_width_2d = np.mean(hip_widths_2d)
            normalized_stride_length_2d = avg_stride_length_2d / avg_hip_width_2d
            st.write(f"Average 2D Stride Length normalized to hip width (stride lengths per hip width) : {normalized_stride_length_2d:.2f}")
        else:
            st.write("Not enough data to compute 2D stride length.")
