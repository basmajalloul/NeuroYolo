
# processor.py — Analysis Core for NeuroKinematic Analyzer (NKA)

import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial.distance import cdist
from dtw import dtw
import os
import json
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans
import time

# Load YOLOv11 Pose Model
model = YOLO('best.pt') 

def extract_keypoints(video_path, model, st_placeholder=None):
    cap = cv2.VideoCapture(video_path)
    keypoints_sequence = []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    progress_bar = st.progress(0) if st_placeholder else None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        if hasattr(results, 'keypoints') and results.keypoints is not None:
            keypoints = results.keypoints.xy[0].cpu().numpy()
            if keypoints.size == 0:
                print("⚠️ No keypoints detected in this frame.")
            else:
                print("✅ Keypoints detected:", keypoints)
            keypoints_sequence.append(keypoints)
        else:
            print("⚠️ No keypoints attribute found in results.")

        current_frame += 1
        if progress_bar:
            progress_percent = int((current_frame / frame_count) * 100)
            progress_bar.progress(min(progress_percent, 100))

    cap.release()

    if progress_bar:
        progress_bar.empty()
        st_placeholder.success("Keypoints extracted successfully!")

    return keypoints_sequence

def load_or_extract_keypoints(video_path, original_filename, model, st_placeholder=None):
    base_name = os.path.splitext(original_filename)[0]
    json_path = os.path.join("videos", f"{base_name}_keypoints.json")

    if st_placeholder:
        st_placeholder.info(f"Looking for JSON at: {json_path}")
    else:
        print(f"Looking for JSON at: {json_path}")

    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)

        keypoints_sequence = []
        total_frames = len(data)

        # Initialize progress bar if using Streamlit
        progress_bar = st.progress(0) if st_placeholder else None

        for idx, frame in enumerate(sorted(data.keys())):
            keypoints = [
                kp["coordinates"] for kp in data[frame]["keypoints"] if "coordinates" in kp
            ]
            keypoints_sequence.append(np.array(keypoints))

            if progress_bar:
                progress_percent = int(((idx + 1) / total_frames) * 100)
                progress_bar.progress(min(progress_percent, 100))

        if progress_bar:
            progress_bar.empty()
            st_placeholder.success("Keypoints loaded successfully!")

        return keypoints_sequence

    else:
        not_found_msg = f"❌ JSON not found. Extracting keypoints for: {video_path}"
        if st_placeholder:
            st_placeholder.warning(not_found_msg)
        else:
            print(not_found_msg)

        # Extract and Save JSON
        keypoints_sequence = extract_keypoints(video_path, model, st_placeholder)
        os.makedirs(os.path.dirname(json_path), exist_ok=True)

        with open(json_path, 'w') as f:
            json.dump({
                f"frame_{i}": {"keypoints": kp.tolist()} 
                for i, kp in enumerate(keypoints_sequence)
            }, f)

        return keypoints_sequence

def calculate_angle(a, b, c):
    """Calculate angle between three points (a, b, c)."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def compute_joint_angles_for_all_joints(keypoints_sequence):
    """Compute joint angles for all possible joints in each frame."""
    # Define all triplets of joints for which you want to compute angles
    limbs = [
        (5, 7, 9),   # Left arm: shoulder, elbow, wrist
        (6, 8, 10),  # Right arm
        (11, 13, 15),# Left leg: hip, knee, ankle
        (12, 14, 16) # Right leg
    ]

    all_angles = []
    for keypoints in keypoints_sequence:
        frame_angles = []
        for limb_indices in limbs:
            try:
                p1, p2, p3 = keypoints[limb_indices[0]], keypoints[limb_indices[1]], keypoints[limb_indices[2]]
                angle = calculate_angle(p1, p2, p3)
            except:
                angle = 0  # Handle missing keypoints
            frame_angles.append(angle)
        all_angles.append(frame_angles)
    return all_angles

def calculate_similarity(seq1, seq2):
    """Calculate similarity score using DTW on joint angle sequences."""
    seq1 = np.array(seq1).reshape(-1, 1)
    seq2 = np.array(seq2).reshape(-1, 1)

    distance, _, _, _ = dtw(seq1, seq2, dist=lambda x, y: np.linalg.norm(x - y))
    normalized_distance = distance / max(len(seq1), len(seq2))
    similarity_score = max(0, 100 - normalized_distance)  # Scale to 0-100%
    return similarity_score

def get_similarity_label(score):
    if score >= 90:
        return "🟢 Excellent"
    elif score >= 75:
        return "🟡 Good"
    elif score >= 50:
        return "🟠 Needs Improvement"
    else:
        return "🔴 Critical"
    
def calculate_max_angle_deviation(coach_angles, patient_angles):
    """Calculate the maximum deviation between joint angles directly."""
    max_deviation = 0
    max_deviation_frame = -1
    max_deviation_joint = -1

    for frame_idx, (angles1, angles2) in enumerate(zip(coach_angles, patient_angles)):
        for joint_idx in range(len(angles1)):
            try:
                diff = abs(angles1[joint_idx] - angles2[joint_idx])
                if diff > max_deviation:
                    max_deviation = diff
                    max_deviation_frame = frame_idx
                    max_deviation_joint = joint_idx
            except:
                continue

    return round(max_deviation, 2), max_deviation_frame, max_deviation_joint

def select_keyframes_kmeans(keypoints_sequence, num_keyframes=10):
    """
    Selects representative keyframes using K-Means clustering on pose keypoints.
    """
    # Ensure all keypoints arrays have the same shape before flattening
    max_joints = max((len(kp) for kp in keypoints_sequence), default=0)
    flattened_kps = []

    for kp in keypoints_sequence:
        if len(kp) == 0:
            continue  # Skip empty keypoints
        elif len(kp) < max_joints:
            padding = np.zeros((max_joints - len(kp), 2))
            padded_kp = np.vstack((kp, padding))
        else:
            padded_kp = kp
        flattened_kps.append(padded_kp.flatten())

    if len(flattened_kps) == 0:
        print("⚠️ No valid keypoints found. Falling back to uniform keyframe selection.")
        total_frames = len(keypoints_sequence)
        return list(np.linspace(0, total_frames - 1, min(num_keyframes, total_frames), dtype=int))

    flattened_kps = np.array(flattened_kps)

    # Safe normalization
    if flattened_kps.size > 0:
        min_val = flattened_kps.min()
        max_val = flattened_kps.max()
        if max_val - min_val != 0:
            flattened_kps = (flattened_kps - min_val) / (max_val - min_val + 1e-6)
        else:
            flattened_kps = np.zeros_like(flattened_kps)  # All points are identical

    # KMeans clustering
    kmeans = KMeans(n_clusters=min(num_keyframes, len(flattened_kps)), random_state=42).fit(flattened_kps)
    centers = kmeans.cluster_centers_

    # Find closest frames to each center
    selected_indices = []
    for center in centers:
        distances = [np.linalg.norm(kp - center) for kp in flattened_kps]
        selected_indices.append(np.argmin(distances))

    with open("processed/last_session_keyframes.json", "w") as f:
        json.dump([int(x) for x in sorted(selected_indices)], f)

    return sorted(selected_indices)

def determine_optimal_keyframes(total_frames):
    """
    Automatically determines a suitable number of keyframes based on video length.
    """
    if total_frames <= 100:
        return 5
    elif total_frames <= 500:
        return 10
    elif total_frames <= 1000:
        return 15
    else:
        return 20  # Cap at 20 keyframes for very long videos

def get_gradient_color(value, max_value=100):
    """Return BGR color based on value using a gradient from green (low) to red (high)."""
    cmap = plt.get_cmap('RdYlGn_r')  # Reversed colormap
    normalized = np.clip(value / max_value, 0, 1)
    color = cmap(normalized)
    return (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))  # BGR for OpenCV

def fancy_highlight_limb(frame, keypoints, limb_indices, angle_diff, alert_threshold=30):
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np

    def get_gradient_color(value, max_value=100):
        cmap = plt.get_cmap('RdYlGn_r')
        normalized = np.clip(value / max_value, 0, 1)
        color = cmap(normalized)
        return (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))  # BGR

    color = get_gradient_color(angle_diff, max_value=alert_threshold * 2)
    thickness = 6  # Thicker for better visibility

    # First draw the highlighted limb on top of YOLO skeleton
    for start, end in zip(limb_indices[:-1], limb_indices[1:]):
        start_point = tuple(map(int, keypoints[start][:2]))
        end_point = tuple(map(int, keypoints[end][:2]))
        cv2.line(frame, start_point, end_point, color, thickness, lineType=cv2.LINE_AA)

    # Then draw the circle with the angle value *after* everything else
    mid_x = int((keypoints[limb_indices[0]][0] + keypoints[limb_indices[-1]][0]) / 2)
    mid_y = int((keypoints[limb_indices[0]][1] + keypoints[limb_indices[-1]][1]) / 2)

    # Circle background with larger radius for better readability
    cv2.circle(frame, (mid_x, mid_y), 20, color, -1)

    # White angle text over the circle
    cv2.putText(frame, f"{angle_diff:.1f}", (mid_x - 15, mid_y + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

def calculate_spatial_displacement(keypoints1, keypoints2):
    """Compute average displacement between corresponding keypoints."""
    valid_joints = min(len(keypoints1), len(keypoints2))
    total_disp = 0
    for i in range(valid_joints):
        total_disp += np.linalg.norm(keypoints1[i] - keypoints2[i])
    return total_disp / valid_joints if valid_joints > 0 else 0
    
def estimate_pose_two_videos_resized(frame1, frame2, keypoints1, keypoints2, ALERT_THRESHOLD=5, ALPHA=0.7, BETA=0.3):
    """Compare two frames using joint angles and spatial displacement, highlight limbs with deviations, and compute similarity."""

    limbs = {
        'left_elbow_angle': (5, 7, 9),        # Left Shoulder - Left Elbow - Left Wrist
        'right_elbow_angle': (6, 8, 10),      # Right Shoulder - Right Elbow - Right Wrist
        'left_knee_angle': (11, 13, 15),      # Left Hip - Left Knee - Left Ankle
        'right_knee_angle': (12, 14, 16),     # Right Hip - Right Knee - Right Ankle
        'left_hip_angle': (5, 11, 13),        # Left Shoulder - Left Hip - Left Knee
        'right_hip_angle': (6, 12, 14),       # Right Shoulder - Right Hip - Right Knee
        'left_armpit_angle': (11, 5, 7),      # Left Hip - Left Shoulder - Left Elbow
        'right_armpit_angle': (12, 6, 8)      # Right Hip - Right Shoulder - Right Elbow
    }

    frame_angle_diff = {}
    angle_diffs = []

    try:
        for limb_name, limb_indices in limbs.items():
            if all(idx < len(keypoints1) for idx in limb_indices) and all(idx < len(keypoints2) for idx in limb_indices):
                p1_a, p1_b, p1_c = keypoints1[limb_indices[0]], keypoints1[limb_indices[1]], keypoints1[limb_indices[2]]
                p2_a, p2_b, p2_c = keypoints2[limb_indices[0]], keypoints2[limb_indices[1]], keypoints2[limb_indices[2]]

                angle1 = calculate_angle(p1_a, p1_b, p1_c)
                angle2 = calculate_angle(p2_a, p2_b, p2_c)
                angle_difference = abs(angle1 - angle2)

                frame_angle_diff[limb_name] = angle_difference
                angle_diffs.append(angle_difference)

                if angle_difference > ALERT_THRESHOLD:
                    fancy_highlight_limb(frame2, keypoints2, limb_indices, angle_difference, ALERT_THRESHOLD)

        # --- Spatial Displacement Component ---
        spatial_disp = 0
        valid_joints = min(len(keypoints1), len(keypoints2))
        if valid_joints > 0:
            for i in range(valid_joints):
                spatial_disp += np.linalg.norm(keypoints1[i] - keypoints2[i])
            spatial_disp /= valid_joints

        # --- Final Similarity Score ---
        avg_angle_diff = np.mean(angle_diffs) if angle_diffs else 0
        final_pose_score = ALPHA * avg_angle_diff + BETA * spatial_disp
        frame_angle_diff["final_pose_score"] = round(final_pose_score, 2)

    except Exception as e:
        print(f"[ERROR] Failed to process frame: {e}")

    os.makedirs("processed", exist_ok=True)
    with open("processed/last_frame_angle_diff.json", "w") as f:
        # Ensure all values are standard floats for JSON compatibility
        serializable_diff = {k: float(v) for k, v in frame_angle_diff.items()}
        json.dump(serializable_diff, f)

    return frame_angle_diff

def generate_comparison_video_live_inference(coach_video_path, patient_video_path, 
                                              keyframe_indices, output_path="processed/processed_video.mp4"):
    import cv2
    import numpy as np
    import json
    import os
    import time
    from ultralytics import YOLO

    # Load YOLO model
    model = YOLO('best.pt')

    coach_cap = cv2.VideoCapture(coach_video_path)
    patient_cap = cv2.VideoCapture(patient_video_path)

    frame_width = int(coach_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(coach_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(coach_cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(coach_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    adjusted_fps = int(fps * (len(keyframe_indices) / total_frames))
    adjusted_fps = max(1, adjusted_fps)

    with open("processed/last_session_fps.json", "w") as f:
        json.dump({"fps": int(fps)}, f)

    os.makedirs("processed", exist_ok=True)
    out = cv2.VideoWriter(output_path, 
                          cv2.VideoWriter_fourcc(*'avc1'), 
                          adjusted_fps, 
                          (frame_width * 2, frame_height))

    event_log = []
    similarity_trend = []

    for idx, original_frame_idx in enumerate(keyframe_indices):
        coach_cap.set(cv2.CAP_PROP_POS_FRAMES, original_frame_idx)
        patient_cap.set(cv2.CAP_PROP_POS_FRAMES, original_frame_idx)

        ret1, frame1 = coach_cap.read()
        ret2, frame2 = patient_cap.read()

        if not ret1 or not ret2:
            continue

        result_coach = model(frame1)[0]
        result_patient = model(frame2)[0]

        keypoints_coach = result_coach.keypoints.xy[0].cpu().numpy() if result_coach.keypoints is not None else []
        keypoints_patient = result_patient.keypoints.xy[0].cpu().numpy() if result_patient.keypoints is not None else []

        # Analyze frame for deviations
        frame_metrics = estimate_pose_two_videos_resized(
            result_coach.orig_img,
            result_patient.orig_img,
            keypoints_coach,
            keypoints_patient,
            ALERT_THRESHOLD=5
        )

        similarity_trend.append(frame_metrics.get("final_pose_score", 0))

        # Compute timestamp
        timestamp_sec = idx * (1 / adjusted_fps if adjusted_fps > 0 else 1 / 25)
        timestamp = time.strftime('%M:%S', time.gmtime(timestamp_sec))

        # Add significant deviations to event log
        for joint, deviation in frame_metrics.items():
            if joint != "final_pose_score" and deviation > 5:
                event_log.append(f"- **{timestamp}** — {joint.replace('_', ' ').title()} deviation: {deviation:.1f}°")

        # Draw skeletons
        annotated_frame1 = result_coach.plot()  # Keep skeleton for coach

        # For patient, draw the raw frame and let fancy_highlight_limb handle the deviation highlights directly
        annotated_frame2 = result_patient.orig_img.copy()

        combined_frame = np.hstack((annotated_frame1, annotated_frame2))
        out.write(combined_frame)

    coach_cap.release()
    patient_cap.release()
    out.release()

    # After processing all frames, save similarity_trend to JSON
    with open("processed/pose_similarity_trend.json", "w") as f:
        json.dump([float(val) for val in similarity_trend], f)

    # Save event log
    with open("processed/event_log.json", "w") as f:
        json.dump(event_log, f)

    print(f"✅ Comparison video and event log generated at: {output_path}")
    
@st.cache_data(show_spinner=True)
def compute_dtw_alignment_cached(coach_keypoints_seq, patient_keypoints_seq):
    from dtw import dtw
    import numpy as np

    coach_flattened = [kp.flatten() for kp in coach_keypoints_seq if kp is not None and len(kp) > 0]
    patient_flattened = [kp.flatten() for kp in patient_keypoints_seq if kp is not None and len(kp) > 0]

    coach_array = np.array(coach_flattened)
    patient_array = np.array(patient_flattened)

    # DTW using default metric (Euclidean)
    alignment = dtw(coach_array, patient_array, keep_internals=True)
    dist = alignment.distance
    path = (alignment.index1, alignment.index2)

    return coach_array, patient_array, path


def visualize_dtw_alignment(coach_keypoints_seq, patient_keypoints_seq):
    # Flatten keypoints for each frame (if not already arrays)
    coach_flattened = [kp.flatten() for kp in coach_keypoints_seq if len(kp) > 0]
    patient_flattened = [kp.flatten() for kp in patient_keypoints_seq if len(kp) > 0]

    # Compute DTW alignment (use your existing cached function)
    coach_array, patient_array, path = compute_dtw_alignment_cached(coach_keypoints_seq, patient_keypoints_seq)

    # Compute "movement magnitude" as norm per frame (assume both arrays are aligned in length)
    coach_magnitude = [np.linalg.norm(arr) for arr in coach_array]
    patient_magnitude = [np.linalg.norm(arr) for arr in patient_array]

    # X-axis is frame index
    frames = np.arange(len(coach_magnitude))

    # Plotly static line chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frames, y=coach_magnitude, mode='lines', name="Coach", line=dict(color='#2797FF', width=3)))
    fig.add_trace(go.Scatter(x=frames, y=patient_magnitude, mode='lines', name="Participant", line=dict(color='#10D6D3', width=3)))

    fig.update_layout(
        title="DTW Alignment Trend: Coach vs. Participant",
        xaxis_title="Frame",
        yaxis_title="Movement Magnitude",
        template="plotly_white",
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(0,0,0,0)')
    )

    # Show in Streamlit
    import streamlit as st
    st.plotly_chart(fig, use_container_width=True)

def plot_joint_deviation_heatmap(coach_keypoints_seq, patient_keypoints_seq):
    # --- Extract Deviation Per Joint ---
    joint_names = [
        'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear',
        'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
        'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip',
        'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle'
    ]

    num_joints = len(joint_names)
    deviations = np.zeros(num_joints)

    for coach_kp, patient_kp in zip(coach_keypoints_seq, patient_keypoints_seq):
        if len(coach_kp) == num_joints and len(patient_kp) == num_joints:
            for i in range(num_joints):
                deviations[i] += np.linalg.norm(coach_kp[i] - patient_kp[i])

    # Average over frames
    deviations /= len(coach_keypoints_seq)

    # --- Create Heatmap ---
    df = pd.DataFrame({'Joint': joint_names, 'Deviation': deviations})
    df = df.pivot_table(index='Joint', values='Deviation')

    plt.figure(figsize=(8, 6))
    sns.heatmap(df, annot=True, cmap="Reds", linewidths=0.5, cbar_kws={'label': 'Average Deviation (px)'})
    plt.title("Joint-Wise Average Deviation Heatmap")
    st.pyplot(plt)

def save_keypoints_to_json(coach_kps, patient_kps):
    os.makedirs("processed", exist_ok=True)

    with open("processed/last_session_coach.json", "w") as f:
        json.dump([kp.tolist() for kp in coach_kps], f)

    with open("processed/last_session_participant.json", "w") as f:
        json.dump([kp.tolist() for kp in patient_kps], f)    

def plot_joint_angle_deviation_bar(frame_angle_diff):
    # Filter out the final_pose_score if present
    filtered_diff = {k: v for k, v in frame_angle_diff.items() if k != "final_pose_score"}

    joint_names = [name.replace("_angle", "").replace("_", " ").title() for name in filtered_diff.keys()]
    deviations = list(filtered_diff.values())

    df = pd.DataFrame({"Joint": joint_names, "Avg Deviation (°)": deviations})

    fig = px.bar(
        df,
        x="Avg Deviation (°)",
        y="Joint",
        orientation='h',
        color="Avg Deviation (°)",
        color_continuous_scale=px.colors.sequential.Bluered,
        text="Avg Deviation (°)"
    )

    fig.update_traces(texttemplate='%{text:.1f}°', textposition='inside')
    fig.update_layout(
        title="Joint-Wise Average Angle Deviation (Degrees)",
        xaxis_title="Average Deviation (°)",
        yaxis_title="",
        coloraxis_showscale=False,
        plot_bgcolor="#fff",  # Match dashboard background
        paper_bgcolor="#fff",
        font=dict(family="Poppins, sans-serif", color="#000"),
        margin=dict(l=40, r=40, t=40, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

def plot_joint_angle_deviation_heatmap(frame_angle_diff):
    # Remove final_pose_score if present
    filtered_diff = {k: v for k, v in frame_angle_diff.items() if k != "final_pose_score"}

    joint_names = [name.replace("_angle", "").replace("_", " ").title() for name in filtered_diff.keys()]
    deviations = list(filtered_diff.values())

    df = pd.DataFrame({"Joint": joint_names, "Avg Deviation (°)": deviations})

    fig = px.imshow(
        [deviations],
        labels=dict(x="Joint", color="Avg Deviation (°)"),
        x=joint_names,
        y=["Deviation"],
        color_continuous_scale=px.colors.sequential.Reds
    )

    fig.update_layout(
        title="Joint-Wise Average Angle Deviation Heatmap",
        xaxis_title="Joint",
        yaxis_title="",
        plot_bgcolor="#fff",  # Match dashboard background
        paper_bgcolor="#fff",
        font=dict(family="Poppins, sans-serif", color="#000"),
        margin=dict(l=40, r=40, t=40, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)    

def plot_pose_similarity_trend(similarity_scores, keyframe_indices, original_fps):
    time_axis = np.array(keyframe_indices) / original_fps

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=time_axis,
        y=similarity_scores,
        mode='lines+markers',
        line=dict(color='deepskyblue', width=3),
        marker=dict(size=6, color='deepskyblue'),
        name='Similarity Score'
    ))

    # Add thresholds
    fig.add_hline(y=30, line_dash="dash", line_color="orange", 
                  annotation_text="Warning Zone", annotation_position="top left")
    fig.add_hline(y=60, line_dash="dash", line_color="red", 
                  annotation_text="High Deviation", annotation_position="top left")

    fig.update_layout(
        title="Pose Similarity Trend Over Time",
        xaxis_title="Time (seconds)",
        yaxis_title="Pose Similarity Score",
        template="plotly_dark",
        plot_bgcolor="#fff",
        paper_bgcolor="#fff",
        font=dict(family="Poppins, sans-serif", color="#000"),
        margin=dict(l=40, r=40, t=40, b=40),        
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)


def calculate_movement_smoothness_index(joint_angle_sequence):
    """
    Calculate the Movement Smoothness Index (MSI) for a given joint angle sequence.
    :param joint_angle_sequence: List or numpy array of joint angles over time.
    :return: MSI value (lower is smoother).
    """
    diffs = np.diff(joint_angle_sequence, axis=0)
    abs_diffs = np.abs(diffs)
    msi = np.mean(abs_diffs)
    return round(msi, 2)

def save_msi(coach_msi, patient_msi):
    os.makedirs("processed", exist_ok=True)
    with open("processed/last_session_msi.json", "w") as f:
        json.dump({
            "coach_msi": round(float(coach_msi), 2),
            "patient_msi": round(float(patient_msi), 2)
        }, f)

def plot_cumulative_error(similarity_scores, keyframe_indices, original_fps):

    """Plots cumulative error over time using actual keyframe timestamps."""
    errors = [100 - s for s in similarity_scores]
    cumulative_errors = np.cumsum(errors)

    # Compute time axis using real keyframe indices and original FPS
    time_axis = np.array(keyframe_indices) / original_fps

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=cumulative_errors,
        mode='lines+markers',
        line=dict(color='#FF2D6F', width=3),
        marker=dict(size=6),
        name='Cumulative Error'
    ))

    # Fatigue Threshold Example (Configurable)
    fatigue_threshold = 1000
    fig.add_hline(
        y=fatigue_threshold,
        line_dash="dash",
        line_color="red",
        annotation_text="Fatigue Threshold",
        annotation_position="top right"
    )

    fig.update_layout(
        title="📈 Cumulative Error Over Time",
        xaxis_title="Time (seconds)",
        yaxis_title="Cumulative Error",
        template="plotly_dark",
        plot_bgcolor="#fff",
        paper_bgcolor="#fff",
        font=dict(family="Poppins, sans-serif", color="#000"),
        margin=dict(l=40, r=40, t=40, b=40),
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

def save_frame_drop_stats(total_frames, selected_indices):
    dropped = total_frames - len(selected_indices)
    drop_rate = round((dropped / total_frames) * 100, 2) if total_frames > 0 else 0.0
    stats = {
        "total_frames": total_frames,
        "selected_keyframes": len(selected_indices),
        "dropped_frames": dropped,
        "frame_drop_rate_percent": drop_rate,
        "skipped_indices": sorted(list(set(range(total_frames)) - set(selected_indices)))
    }
    with open("processed/frame_drop_stats.json", "w") as f:
        json.dump(stats, f, indent=2)


def get_model_size(model_path):
    size_bytes = os.path.getsize(model_path)
    size_mb = size_bytes / (1024 * 1024)
    return round(size_mb, 2)

def benchmark_yolo_inference_speed(model_path, video_path, device, n_frames=20):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    processed = 0
    t0 = time.time()
    while cap.isOpened() and processed < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        _ = model(frame, device=device)[0]  # may need to adjust for your model/device
        processed += 1
    cap.release()
    avg_time = (time.time() - t0) / processed if processed > 0 else None
    return round(avg_time, 3)

def save_cpu_gpu_benchmarks(model_path, video_path):
    cpu = benchmark_yolo_inference_speed(model_path, video_path, device="cpu", n_frames=20)
    gpu = benchmark_yolo_inference_speed(model_path, video_path, device=0, n_frames=20)
    with open("processed/cpu_gpu_benchmark.json", "w") as f:
        json.dump({"CPU_avg_inference_s": cpu, "GPU_avg_inference_s": gpu}, f, indent=2)