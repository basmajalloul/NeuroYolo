# app.py — NeuroKinematic Analyzer (NeuroYolo)
import streamlit as st
from pathlib import Path
from processor import determine_optimal_keyframes, select_keyframes_kmeans, compute_joint_angles_for_all_joints, calculate_similarity, load_or_extract_keypoints, get_similarity_label, calculate_max_angle_deviation
from ultralytics import YOLO
import json, os
from processor import plot_cumulative_error, calculate_movement_smoothness_index, plot_pose_similarity_trend, plot_joint_angle_deviation_heatmap, generate_comparison_video_live_inference, visualize_dtw_alignment, plot_joint_deviation_heatmap, save_keypoints_to_json, plot_joint_angle_deviation_bar
from processor import save_frame_drop_stats, get_model_size, benchmark_yolo_inference_speed, save_cpu_gpu_benchmarks
import plotly.graph_objects as go
import numpy as np
from dtw import dtw
import time
import pandas as pd 
import re 

model = YOLO('best.pt') 

# Usage example:
fine_tuned_size = get_model_size("best.pt")
default_size = get_model_size("yolo11n-pose.pt")

with open("processed/model_sizes.json", "w") as f:
    json.dump({
        "fine_tuned_model_MB": fine_tuned_size,
        "default_model_MB": default_size
    }, f, indent=2)


st.set_page_config(page_title="NeuroYolo Dashboard", layout="wide")

# --- Sidebar Navigation ---
st.sidebar.title("NeuroYolo Dashboard")
page = st.sidebar.radio("Navigation", ["Home", "Upload & Analyze", "Results Dashboard", "Reports"])

# --- Page 1: Home ---
if page == "Home":
    # Centered logo as the main header
    svg_logo = open("assets/file.svg").read()
    st.markdown(f"""
    <div style='text-align: center; margin-bottom: 0;'>
        <div style='text-align: center; width:120px; display:inline-block;'>{svg_logo}</div>
        <div style='font-size:1.3em; color:#10D6D3; margin-top: 0.2em; font-weight: 700;'>
            Precision Movement Benchmarking for Neuro-Motor Health
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Main features and how-to section as before
    st.markdown(
        """
        <div style="background-color: #f1f5fa; border-radius: 8px; padding: 20px; margin: 2em 0 1em 0;">
        <b>What is NeuroYolo?</b><br>
        <ul>
            <li>🔬 <b>Real-time pose analytics</b> for rehabilitation and movement science</li>
            <li>💡 <b>Automated, per-joint anomaly detection</b> with interactive event logs</li>
            <li>📈 <b>Exportable clinical reports</b> and time-series analytics</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <b>How it works:</b>
        <ol>
            <li>Upload coach and participant videos in <b>Upload & Analyze</b> 🔼</li>
            <li>Run the automated analysis to benchmark motion quality ⚡</li>
            <li>Review <span style="color: #A34FFF;"><b>joint anomalies, trend plots, and event logs</b></span> in the dashboard</li>
            <li>Export results for your clinical workflow 💾</li>
        </ol>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="background-color: #e1f7ef; border-radius: 8px; padding: 16px; margin-bottom: 1em;">
        <b>💡 Demo Data Preloaded:</b> For first-time visitors, sample results are already preloaded in <b>Results Dashboard</b> and <b>Reports</b>.
        <br>
        Explore these sections without uploading data to quickly preview the platform's capabilities!
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown("---")
    st.success("Ready to get started? Use the sidebar to upload your videos and run your first analysis!")


# --- Page 2: Upload & Analyze ---
elif page == "Upload & Analyze":
    st.title("Upload & Analyze")
    st.markdown("Upload two videos to begin movement analysis.")

    video1 = st.file_uploader("Upload Coach Video", type=["mp4"], key="vid1")
    video2 = st.file_uploader("Upload Patient Video", type=["mp4"], key="vid2")

    if st.button("Start Analysis") and video1 and video2:
        import tempfile
        import time
        timing_report = {}

        # --- 1. Save uploaded videos to temp files ---
        temp_video1 = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{video1.name}")
        temp_video1.write(video1.read())
        temp_video1.flush()
        temp_video1.close()
        video1_path = temp_video1.name
        video1_filename = video1.name

        temp_video2 = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{video2.name}")
        temp_video2.write(video2.read())
        temp_video2.flush()
        temp_video2.close()
        video2_path = temp_video2.name
        video2_filename = video2.name

        progress_bar = st.progress(0)
        status_placeholder = st.empty()
        total_steps = 3
        current_step = 0

        # --- Step 1: Keypoint Extraction ---
        t0 = time.time()
        coach_kps = load_or_extract_keypoints(video1_path, video1_filename, model, status_placeholder)
        patient_kps = load_or_extract_keypoints(video2_path, video2_filename, model, status_placeholder)
        timing_report["Keypoint Extraction (s)"] = round(time.time() - t0, 2)

        save_cpu_gpu_benchmarks("yolo11n-pose.pt", video1_path)

        st.session_state["coach_kps"] = coach_kps
        st.session_state["patient_kps"] = patient_kps

        total_frames = len(coach_kps)  # Assuming both videos have similar length

        # --- Step 2: Keyframe Selection ---
        t1 = time.time()
        num_keyframes = determine_optimal_keyframes(total_frames)
        selected_indices = select_keyframes_kmeans(coach_kps, num_keyframes)
        timing_report["Keyframe Selection (s)"] = round(time.time() - t1, 2)

        save_frame_drop_stats(total_frames, selected_indices)

        # Apply keyframe filtering
        coach_kps = [coach_kps[i] for i in selected_indices]
        patient_kps = [patient_kps[i] for i in selected_indices]

        save_keypoints_to_json(coach_kps, patient_kps)

        # --- Step 3: Joint Angle & Metrics ---
        t2 = time.time()
        coach_angles = compute_joint_angles_for_all_joints(coach_kps)
        patient_angles = compute_joint_angles_for_all_joints(patient_kps)
        timing_report["Angle Computation (s)"] = round(time.time() - t2, 2)

        current_step += 1
        progress_bar.progress(int((current_step / total_steps) * 100))
        status_placeholder.text("Computed joint angles...")

        coach_msi = calculate_movement_smoothness_index(np.array(coach_angles))
        patient_msi = calculate_movement_smoothness_index(np.array(patient_angles))

        with open("processed/msi_results.json", "w") as f:
            json.dump({
                "coach_msi": float(coach_msi),
                "patient_msi": float(patient_msi)
            }, f)

        # Generate deviation per joint per frame for visualization
        deviation_per_frame = []
        for angles1, angles2 in zip(coach_angles, patient_angles):
            frame_deviation = [abs(a1 - a2) for a1, a2 in zip(angles1, angles2)]
            deviation_per_frame.append(frame_deviation)

        # --- Step 4: Similarity & Max Deviation (Not separately timed) ---
        similarity = calculate_similarity(coach_angles, patient_angles)
        max_deviation, max_deviation_frame, max_deviation_joint = calculate_max_angle_deviation(coach_angles, patient_angles)

        # Save these into the results JSON
        os.makedirs("processed", exist_ok=True)
        with open("processed/last_results.json", "w") as f:
            json.dump({
                "similarity": float(similarity),
                "max_deviation": float(max_deviation),
                "max_deviation_frame": int(max_deviation_frame),
                "max_deviation_joint": int(max_deviation_joint)
            }, f)
            f.flush()
            os.fsync(f.fileno())

        current_step += 1
        progress_bar.progress(int((current_step / total_steps) * 100))
        status_placeholder.text("Similarity calculation complete...")

        # --- Step 5: Video Generation ---
        t3 = time.time()
        with st.spinner("Generating comparison video..."):
            generate_comparison_video_live_inference(
                video1_path,
                video2_path,
                selected_indices,
                output_path="processed/processed_video.mp4"
            )
        timing_report["Video Generation (s)"] = round(time.time() - t3, 2)

        # --- Save timing_report as JSON ---
        with open("processed/timing_report.json", "w") as f:
            json.dump(timing_report, f, indent=2)

        # Only show success after the video is fully generated
        progress_bar.progress(100)
        status_placeholder.success("Analysis complete! Proceed to Results Dashboard.")
        st.session_state["video_ready"] = True



# --- Page 3: Results Dashboard ---
elif page == "Results Dashboard":
    st.title("Results Dashboard")
    st.markdown("Visual summary of detected anomalies and movement quality metrics.")

    # Load results JSON
    if os.path.exists("processed/last_results.json") and os.path.getsize("processed/last_results.json") > 0:
        with open("processed/last_results.json", "r") as f:
            results = json.load(f)
    else:
        st.warning("No valid analysis results found. Please run the analysis first.")
        results = {}


    similarity = round(results.get("similarity", 0), 2)
    max_deviation = round(results.get("max_deviation", 0), 2)
    frame = results.get("max_deviation_frame", "N/A")
    joint_idx = results.get("max_deviation_joint", "N/A")

    # Joint names mapping (adjust if needed based on your model)
    joint_names = ["Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear", 
                    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow", 
                    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip", 
                    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"]

    joint_name = joint_names[joint_idx] if isinstance(joint_idx, int) and joint_idx < len(joint_names) else "Unknown"

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div style='text-align:center; background: #F5F5F5; padding: 30px 0px;'>
            <span style='font-size:1.1em; color:#666;'>Avg Similarity</span><br>
            <span style='font-size:2.4em; font-weight:600; color:#22223B;'>{similarity:.2f}%</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style='text-align:center; background: #F5F5F5; padding: 30px 0px;'>
            <span style='font-size:1.1em; color:#666;'>
                Max Deviation 
                <span title="Occurred at joint: {joint_name}, frame: {frame}">&#9432;</span>
            </span><br>
            <span style='font-size:2.4em; font-weight:600; color:#22223B;'>{max_deviation:.2f}&deg;</span>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        # Modern gradient badge for assessment
        def get_assessment_badge(score):
            if score >= 90:
                color = "linear-gradient(135deg, #2fe478 0%, #10d6d3 100%)"  # Green/teal
                label = "Excellent"
            elif score >= 75:
                color = "linear-gradient(135deg, #ffe266 0%, #ffb54d 100%)"   # Yellow/orange
                label = "Good"
            elif score >= 50:
                color = "linear-gradient(135deg, #ff8c42 0%, #ff2d6f 100%)"  # Orange/pink
                label = "Needs Improvement"
            else:
                color = "linear-gradient(135deg, #e94d4d 0%, #8f1537 100%)"  # Red/dark
                label = "Critical"
            return color, label

        grad, assess_label = get_assessment_badge(similarity)
        st.markdown(f"""
        <div style='text-align:center; background: #F5F5F5; padding: 30px 0px;'>
            <span style='font-size:1.1em; color:#666;'>Assessment</span><br>
            <span style="font-size:2.4em; font-weight:600;">
                <span style="
                    display:inline-block;
                    width:1.2em; height:1.2em;
                    vertical-align:middle;
                    margin-right:0.3em;
                    border-radius:50%;
                    background: {grad};
                    box-shadow: 0 2px 8px #bbb2;">
                </span>
                <span style='color:#22223B;'>{assess_label}</span>
            </span>
        </div>

        """, unsafe_allow_html=True)

    st.markdown(f"""<div style="background:#fff; height: 30px; width: 100%; display: block;"></div>""", unsafe_allow_html=True)

    #st.write("Available files in processed/:", os.listdir("processed/"))

    st.subheader("Pose Comparison Playback")

    if st.session_state.get("video_ready") or os.path.exists("processed/processed_video.mp4"):
        # Display the video
        with open("processed/processed_video.mp4", "rb") as video_file:
            st.video(video_file.read())

        with open("processed/event_log.json", "r") as f:
            event_log = json.load(f)

        def parse_event(event):
            # "- **00:15** — Left Elbow Angle deviation: 87.5°"
            time = re.search(r"\*\*(\d{2}:\d{2})\*\*", event)
            joint = re.search(r"— ([A-Za-z ]+) deviation", event)
            deviation = re.search(r"deviation: ([\d\.]+)", event)
            return {
                "Time": time.group(1) if time else "",
                "Joint": joint.group(1) if joint else "",
                "Deviation (°)": float(deviation.group(1)) if deviation else 0
            }

        events_df = pd.DataFrame([parse_event(e) for e in event_log])
        # Show only top 5 by deviation
        top_events_df = events_df.sort_values("Deviation (°)", ascending=False).head(5)

        def get_severity(val):
            if val > 80:
                return '<span style="color: #fff; background: linear-gradient(90deg, #FF2D6F, #A34FFF); border-radius: 12px; padding:2px 10px;">Critical</span>'
            elif val > 60:
                return '<span style="color: #fff; background: linear-gradient(90deg, #FFB54D, #FF2D6F); border-radius: 12px; padding:2px 10px;">Warning</span>'
            elif val > 30:
                return '<span style="color: #fff; background: linear-gradient(90deg, #10D6D3, #2797FF); border-radius: 12px; padding:2px 10px;">Moderate</span>'
            else:
                return '<span style="color: #fff; background: linear-gradient(90deg, #79F2A1, #10D6D3); border-radius: 12px; padding:2px 10px;">Low</span>'

        top_events_df["Severity"] = top_events_df["Deviation (°)"].apply(get_severity)

        def highlight_deviation(val):
            if val > 80:
                color = '#FF2D6F'
            elif val > 60:
                color = '#FFB54D'
            elif val > 30:
                color = '#10D6D3'
            else:
                color = '#2797FF'
            return f'color: {color}; font-weight:700'


        st.markdown("""<div style="background:#F5F5F5; padding: 30px;"><div style='font-size:1.4em; font-weight:700; margin-bottom:0.4em;'>
                <span style="font-size:1.1em;">&#9888;&#65039;</span> Top 5 <span style="color:#FF2D6F;">Critical Deviations</span>
            </div><table style='width:100%; border-collapse: collapse;'>
            <tr style='background:#fff;'><th style="text-align:center">Time</th><th style="text-align:center">Joint</th><th style="text-align:center">Deviation (°)</th><th style="text-align:center">Severity</th></tr>"""
            + "".join(
                f"<tr>"
                f"<td style='background:#fff;text-align:center'>{row['Time']}</td>"
                f"<td style='background:#fff;text-align:center'>{row['Joint']}</td>"
                f"<td style='background:#fff;text-align:center; color:{'#FF2D6F' if row['Deviation (°)']>80 else '#FFB54D' if row['Deviation (°)']>60 else '#10D6D3' if row['Deviation (°)']>30 else '#2797FF'}; font-weight:700'>{row['Deviation (°)']:.1f}</td>"
                f"<td style='background:#fff;text-align:center; font-size: 13px;'>{get_severity(row['Deviation (°)'])}</td>"
                f"</tr>"
                for idx, row in top_events_df.iterrows()
            )
            + "</table></div>",
            unsafe_allow_html=True)




    else:
        st.warning("Processed video not found. Please run the analysis first.")

    st.session_state["video_ready"] = False

# --- Page 4: Reports ---
elif page == "Reports":
    st.title("Session Reports")

    def load_last_session_keypoints():
        try:
            with open("processed/last_session_coach.json", "r") as f:
                coach_kps = [np.array(kp) for kp in json.load(f)]
            with open("processed/last_session_participant.json", "r") as f:
                patient_kps = [np.array(kp) for kp in json.load(f)]
            return coach_kps, patient_kps
        except FileNotFoundError:
            return None, None

    coach_kps, patient_kps = load_last_session_keypoints()

    if coach_kps and patient_kps:
        visualize_dtw_alignment(coach_kps, patient_kps)

        # Assuming you saved the last frame's angle deviations as JSON:
        with open("processed/last_frame_angle_diff.json", "r") as f:
            frame_angle_diff = json.load(f)

        plot_joint_angle_deviation_bar(frame_angle_diff)
        plot_joint_angle_deviation_heatmap(frame_angle_diff)  # New heatmap plot

        if os.path.exists("processed/msi_results.json"):
            with open("processed/msi_results.json", "r") as f:
                msi_results = json.load(f)
            
            st.subheader("Movement Smoothness Index (MSI)")
            col1, col2 = st.columns(2)
            col1.metric("Coach MSI", f"{msi_results['coach_msi']:.2f}")
            col2.metric("Patient MSI", f"{msi_results['patient_msi']:.2f}")

            # Optional: Visualize as a bar plot
            fig = go.Figure(data=[go.Bar(
                x=["Coach", "Patient"],
                y=[msi_results["coach_msi"], msi_results["patient_msi"]],
                marker_color=["#1f77b4", "#ff7f0e"]
            )])

            fig.update_layout(
                title="Movement Smoothness Index Comparison",
                yaxis_title="MSI (Lower is Smoother)",
                template="plotly_dark"
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("MSI results not found. Please run the analysis first.")

        
        if os.path.exists("processed/pose_similarity_trend.json") and \
        os.path.exists("processed/last_session_keyframes.json") and \
        os.path.exists("processed/last_session_fps.json"):

            with open("processed/pose_similarity_trend.json", "r") as f:
                similarity_scores = json.load(f)
            with open("processed/last_session_keyframes.json", "r") as f:
                keyframe_indices = json.load(f)
            with open("processed/last_session_fps.json", "r") as f:
                original_fps = json.load(f)["fps"]

            plot_cumulative_error(similarity_scores, keyframe_indices, original_fps)
        else:
            st.warning("Required data for cumulative error plot is missing. Please run the analysis first.")

        if os.path.exists("processed/pose_similarity_trend.json") and os.path.exists("processed/last_session_fps.json"):
            with open("processed/pose_similarity_trend.json", "r") as f:
                similarity_scores = json.load(f)
            with open("processed/last_session_fps.json", "r") as f:
                dynamic_fps = json.load(f).get("fps", 25)

            plot_pose_similarity_trend(similarity_scores, keyframe_indices, original_fps)

        else:
            st.warning("Required data for pose similarity trend plot is missing. Please run the analysis first.")
    
    else:
        st.warning("No previous session keypoints found. Please run an analysis first.")

