#main.py
import streamlit as st
import os
import pandas as pd
from datetime import datetime
import cv2
from collections import Counter
from config import CONFIDENCE_THRESHOLD

from utils.database_utils import (
    check_plate_in_database,
    add_visitor_entry,
    add_registered_vehicle,
    delete_registered_vehicle,
    get_all_registered_vehicles,
    get_all_visitor_logs,
)

from utils.detection_utils import (
    process_frame,
    expand_box

)

from utils.ocr_utils import (
    smart_correct_ocr_text,
    try_ocr_with_retries,
    preprocess_for_ocr,
    model,
    ocr_model
)

plate_pattern = r"^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$"

# --- Load custom CSS ---
css_path = os.path.join("static", "style.css")
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="TATA Steel UISL",
    page_icon="static/TATA.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Header ---
st.markdown("""
<div class="header">
    <h1>TATA STEEL UISL</h1>
    <p>Vehicle Recognition System - Automatic Number Plate Recognition</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div class="sidebar-content">
        <h2 style="color: #00d4ff; text-align: center;">🚗 Vehicle Management</h2>
    </div>
    """, unsafe_allow_html=True)
    selected_option = st.radio(
        "Select Operation:",
        ["🎥 Live Video Processing",
          "📝 Register New Vehicle", 
          "🗑️ Remove Vehicle", 
          "📋 Registered Vehicle List", 
          "👥 Visitor"],
        index=0
    )


# Main content based on selection
if selected_option == "🎥 Live Video Processing":
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div class="info-card">
            <h3 style="color: #00d4ff;">📹 Live Video Feed</h3>
        </div>
        """, unsafe_allow_html=True)
        input_mode = st.radio("Select Input Mode:", ("Webcam", "Browse Files"), horizontal=True)
        stframe = st.empty()
        progress_bar = st.progress(0)

    with col2:
        st.markdown("""
        <div class="vehicle-details">
            <h3 style="color: #00d4ff;">Vehicle Details</h3>
        </div>
        """, unsafe_allow_html=True)
        details_box = st.empty()

    # Webcam mode
    if input_mode == "Webcam":
        run_webcam = st.button("Start Webcam")
        if run_webcam:
            cap = cv2.VideoCapture(0)
            plate_history = []
            most_common_plate = None
            last_box = None
            is_employee, pass_no = None, None
            visitor_logged = False
            frame_count = 0
            max_no_detection = 15
            frames_since_seen = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                processed, detected, box, found = process_frame(frame, model, ocr_model, plate_pattern, CONFIDENCE_THRESHOLD, smart_correct_ocr_text, try_ocr_with_retries, target_plate=most_common_plate, last_box=last_box)
                if detected:
                    plate_history.extend(detected)
                    most_common_plate, _ = Counter(plate_history).most_common(1)[0]
                    is_employee, pass_no = check_plate_in_database(most_common_plate)
                    visitor_logged = False
                    if not is_employee and not visitor_logged:
                        visit_date = datetime.now().date()
                        visit_time = datetime.now().time().strftime('%H:%M:%S')
                        add_visitor_entry(most_common_plate, visit_date, visit_time)
                        visitor_logged = True
                    st.session_state.current_plate = most_common_plate
                    st.session_state.vehicle_status = (is_employee, pass_no)
                    st.session_state.detection_time = datetime.now()
                if box and found:
                    last_box = box
                    frames_since_seen = 0
                else:
                    frames_since_seen += 1
                if last_box and frames_since_seen < max_no_detection:
                    x1, y1, x2, y2 = last_box
                    color = (0, 255, 0) if is_employee else (0, 0, 255)
                    thickness = 3
                    cv2.rectangle(processed, (x1, y1), (x2, y2), color, thickness)
                now = datetime.now()
                
                stframe.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                # Vehicle details
                if st.session_state.current_plate:
                    details_box.markdown(f"""
                    <div class="status-card {'status-with-pass' if is_employee else 'status-without-pass'}">
                        {'✅ Recognition Status: With PASS' if is_employee else '❌ Recognition Status: Without PASS'}
                    </div>
                    <div class="info-card">
                        <p><strong>PASS No.:</strong> {pass_no if pass_no else 'N/A'}</p>
                        <p><strong>Plate Number:</strong> {st.session_state.current_plate}</p>
                        <p><strong>Date:</strong> {st.session_state.detection_time.strftime('%d/%m/%Y') if st.session_state.detection_time else 'N/A'}</p>
                        <p><strong>Time:</strong> {st.session_state.detection_time.strftime('%H:%M:%S') if st.session_state.detection_time else 'N/A'}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    details_box.markdown("""
                    <div class="info-card">
                        <p style="text-align: center; color: rgba(255,255,255,0.7);">
                            No vehicle detected yet.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                frame_count += 1
                progress_bar.progress((frame_count % 100) / 100)
            cap.release()

    # File Upload mode
    else:
        uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"], key="video_upload")
        if uploaded_video:
            temp_video_path = r"./data/temp_video.mp4"
            with open(temp_video_path, "wb") as f:
                f.write(uploaded_video.read())
            cap = cv2.VideoCapture(temp_video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter("./data/output_detected.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            batch_size = 40
            plate_history = []
            # First pass to detect plates
            for i in range(0, total_frames, batch_size):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                for _ in range(batch_size):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    _, detected, _, _ = process_frame(frame, model, ocr_model, plate_pattern, CONFIDENCE_THRESHOLD, smart_correct_ocr_text, try_ocr_with_retries)
                    plate_history.extend(detected)
                    progress_bar.progress(min(i / total_frames, 1.0))
                if plate_history:
                    break
            cap.release()
            if not plate_history:
                st.markdown("""
                <div class="alert-error">
                    ❌ No license plates detected in the video.
                </div>
                """, unsafe_allow_html=True)
            else:
                most_common_plate, _ = Counter(plate_history).most_common(1)[0]
                is_employee, pass_no = check_plate_in_database(most_common_plate)
                visitor_logged = False
                if not is_employee and not visitor_logged:
                    visit_date = datetime.now().date()
                    visit_time = datetime.now().time().strftime('%H:%M:%S')
                    add_visitor_entry(most_common_plate, visit_date, visit_time)
                    visitor_logged = True
                st.session_state.current_plate = most_common_plate
                st.session_state.vehicle_status = (is_employee, pass_no)
                st.session_state.detection_time = datetime.now()
                cap = cv2.VideoCapture(temp_video_path)
                last_box = None
                frames_since_seen = 0
                max_no_detection = 15
                visitor_logged = False
                frame_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    processed, _, box, found = process_frame(frame, model, ocr_model, plate_pattern, CONFIDENCE_THRESHOLD, smart_correct_ocr_text, try_ocr_with_retries, target_plate=most_common_plate, last_box=last_box)
                    if box and found:
                        last_box = box
                        frames_since_seen = 0
                    else:
                        frames_since_seen += 1
                    if last_box and frames_since_seen < max_no_detection:
                        x1, y1, x2, y2 = last_box
                        color = (0, 255, 0) if is_employee else (0, 0, 255)
                        thickness = 3
                        cv2.rectangle(processed, (x1, y1), (x2, y2), color, thickness)
                    now = datetime.now()
                    
                    out.write(processed)
                    if frame_count % 10 == 0:
                        stframe.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB),
                                      caption=f"Plate Detected: {most_common_plate}", channels="RGB", use_container_width=True)
                        # Vehicle details
                        if st.session_state.current_plate:
                            if is_employee:
                                details_box.markdown(f"""
                                <div class="status-card status-with-pass">
                                    ✅ Recognition Status: With PASS
                                </div>
                                <div class="info-card">
                                    <p><strong>PASS No.:</strong> {pass_no if pass_no else 'N/A'}</p>
                                    <p><strong>Plate Number:</strong> {st.session_state.current_plate}</p>
                                    <p><strong>Date:</strong> {st.session_state.detection_time.strftime('%d/%m/%Y') if st.session_state.detection_time else 'N/A'}</p>
                                    <p><strong>Time:</strong> {st.session_state.detection_time.strftime('%H:%M:%S') if st.session_state.detection_time else 'N/A'}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                details_box.markdown(f"""
                                <div class="status-card status-without-pass">
                                    ❌ Recognition Status: Without PASS
                                </div>
                                <div class="info-card">
                                    <p><strong>Plate Number:</strong> {st.session_state.current_plate}</p>
                                    <p><strong>Date:</strong> {st.session_state.detection_time.strftime('%d/%m/%Y') if st.session_state.detection_time else 'N/A'}</p>
                                    <p><strong>Time:</strong> {st.session_state.detection_time.strftime('%H:%M:%S') if st.session_state.detection_time else 'N/A'}</p>
                                </div>
                                """, unsafe_allow_html=True)

                        else:
                            details_box.markdown("""
                            <div class="info-card">
                                <p style="text-align: center; color: rgba(255,255,255,0.7);">
                                    No vehicle detected yet.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                    frame_count += 1
                    progress_bar.progress(min(frame_count / total_frames, 1.0))
                cap.release()
                out.release()

elif selected_option == "📝 Register New Vehicle":
    st.markdown("""
    <div class="info-card">
        <h3 style="color: #00d4ff; margin-bottom: 20px;">📝 Register New Vehicle</h3>
    </div>
    """, unsafe_allow_html=True)
    with st.form("add_vehicle_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("👤 Employee Name", placeholder="Enter full name")
            personal_no = st.text_input("🆔 Personal Number", placeholder="Enter personal number")
        with col2:
            pass_no = st.text_input("🎫 Pass Number", placeholder="Enter pass number")
            vehicle_no = st.text_input("🚗 Vehicle Number", placeholder="e.g., MH12AB1234")
        submitted = st.form_submit_button("✅ Register Vehicle")
        if submitted:
            if name and personal_no and pass_no and vehicle_no:
                result = add_registered_vehicle(name, personal_no, pass_no, vehicle_no)
                if result:
                    st.markdown("""
                    <div class="alert-success">
                        ✅ Vehicle registered successfully!
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="alert-error">
                        ❌ This vehicle is already registered. Please check the vehicle number or try a different one.
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="alert-error">
                    ❌ Please fill all fields.
                </div>
                """, unsafe_allow_html=True)

elif selected_option == "🗑️ Remove Vehicle":
    st.markdown("""
    <div class="info-card">
        <h3 style="color: #00d4ff; margin-bottom: 20px;">🗑️ Remove Registered Vehicle</h3>
    </div>
    """, unsafe_allow_html=True)
    with st.form("delete_vehicle_form"):
        del_vehicle_no = st.text_input("🚗 Vehicle Number to Delete", placeholder="Enter vehicle number")
        del_submitted = st.form_submit_button("🗑️ Delete Vehicle")
        if del_submitted:
            if del_vehicle_no:
                delete_registered_vehicle(del_vehicle_no)
                st.markdown("""
                <div class="alert-success">
                    ✅ Vehicle deleted successfully!
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="alert-error">
                    ❌ Please enter a vehicle number.
                </div>
                """, unsafe_allow_html=True)

elif selected_option == "📋 Registered Vehicle List":
    st.markdown("""
    <div class="info-card">
        <h3 style="color: #00d4ff; margin-bottom: 20px;">📋 All Registered Vehicles</h3>
    </div>
    """, unsafe_allow_html=True)
    registered_vehicles = get_all_registered_vehicles()
    print(registered_vehicles)

    if registered_vehicles:
        df = pd.DataFrame([list(row) for row in registered_vehicles], columns=["Name", "Personal No", "Pass No", "Vehicle No"])
        st.dataframe(df, use_container_width=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Vehicles", len(registered_vehicles))
        with col2:
            st.metric("Active Passes", len(registered_vehicles))
        with col3:
            st.metric("System Status", "Online", delta="Active")
    else:
        st.markdown("""
        <div class="info-card">
            <p style="text-align: center; color: rgba(255,255,255,0.7);">
                No registered vehicles found. Use the "Register New Vehicle" option to add vehicles.
            </p>
        </div>
        """, unsafe_allow_html=True)

elif selected_option == "👥 Visitor":
    st.markdown("""
    <div class="info-card">
        <h3 style="color: #00d4ff; margin-bottom: 20px;">👥 Visitor</h3>
    </div>
    """, unsafe_allow_html=True)

    visitor_logs = get_all_visitor_logs()
    if visitor_logs:
        import pandas as pd
        df = pd.DataFrame(visitor_logs, columns=["Timestamp", "Detected Plate"])
        df = df.sort_values(by="Timestamp", ascending=False).reset_index(drop=True)
        st.dataframe(df, use_container_width=True)
    else:
        st.markdown("""
        <div class="info-card">
            <p style="text-align: center; color: rgba(255,255,255,0.7);">
                No visitor records found yet.
            </p>
        </div>
        """, unsafe_allow_html=True)

    if st.button("🔄 Refresh Log"):
        st.rerun()


# --- Footer ---
st.markdown("""
<div style="margin-top: 50px; text-align: center; padding: 20px; border-top: 1px solid rgba(255,255,255,0.1);">
    <p style="color: rgba(255,255,255,0.7); font-size: 0.9em;">
        An initiative by TATA Steel to transform urban infrastructure. <br>© 2025 TATA Steel UISL. All rights reserved.               
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="position: fixed; bottom: 20px; right: 20px; background: linear-gradient(45deg, #28a745, #20c997); color: white; padding: 10px 15px; border-radius: 20px; font-size: 0.8em; font-weight: bold; z-index: 1000; box-shadow: 0 4px 15px rgba(40,167,69,0.3);">
    🟢 System Online
</div>
""", unsafe_allow_html=True)
