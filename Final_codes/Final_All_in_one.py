import streamlit as st
import cv2
import easyocr
import numpy as np
import os
import hashlib
from datetime import datetime
import tempfile
from PIL import Image

# Create the 'Detected_plates' folder
output_folder = "Detected_plates"
os.makedirs(output_folder, exist_ok=True)

# Initialize EasyOCR
reader = easyocr.Reader(['en'])

# Load Haar cascade
vehicle_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

# Function to compute hash of an image
def compute_image_hash(image):
    return hashlib.md5(image.tobytes()).hexdigest()

# 👉 Function to detect plates in images
def detect_plate_image(uploaded_image, output_folder, vehicle_cascade):
    st.markdown("## 🚗 Vehicle License Number Plate Detection for **Images**")
    st.divider()

    new_folder = os.path.join(output_folder, "From_Images")
    os.makedirs(new_folder, exist_ok=True)

    image = np.array(Image.open(uploaded_image))
    st.image(image, caption="📸 Uploaded Image", use_column_width=True)

    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    plates = vehicle_cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=6)

    st.markdown(f"### 🛡️ Plates Detected: **{len(plates)}**")

    if len(plates) == 0:
        st.warning("⚠️ No number plates were detected in the image.")
        return

    c = 1
    for (x, y, w, h) in plates:
        plate_area = image[y+7:y+h-7, x+7:x+w-7]

        if plate_area.size == 0:
            continue

        results = reader.readtext(plate_area)
        plate_text = ''
        for (_, text, _) in results:
            plate_text = text.strip()
        # if results:
        #     plate_text = results[0][1]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plate_filename = os.path.join(new_folder, f"plate_{c}_{timestamp}.jpg")
        cv2.imwrite(plate_filename, plate_area)

        # st.image(plate_area, caption=f"Detected Plate {c}: {plate_text}", use_column_width=True)
        st.image(plate_area,use_column_width=True)
        st.write(f"**Detected Plate {c}:** `{plate_text}`")
        
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(image, plate_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)

        c += 1

    st.markdown("### 🖼️ Processed Image:")
    st.image(image, caption="Image with Detected Plates", use_column_width=True)
    st.divider()

# 👉 Function to detect plates in videos
def detect_plate_video(video_file, output_folder, vehicle_cascade):
    st.markdown("## 🎬 Vehicle License Number Plate Detection for **Videos**")
    st.divider()

    st.subheader('📹 Provided Video:')
    st.video(video_file)
    st.subheader('⚙️ Processing Video...')

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(video_file.read())
        video_path = temp_file.name

    cap = cv2.VideoCapture(video_path)
    new_folder = os.path.join(output_folder, "From_Videos")
    os.makedirs(new_folder, exist_ok=True)

    frame_count = 0
    plate_count = 0
    plate_hashes = set()
    last_detected_text = ""

    video_placeholder = st.empty()
    stop_button = st.button("🛑 Stop Processing")

    while cap.isOpened():
        frame_count += 1
        has_frame, frame = cap.read()

        if not has_frame:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        plates = vehicle_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in plates:
            if frame_count % 20 != 0 or (w * h >= 7000 or w * h <= 2000):
                continue

            plate_area = frame[y+7:y+h-7, x+7:x+w-7]

            if plate_area.size == 0:
                continue

            plate_hash = compute_image_hash(plate_area)
            if plate_hash in plate_hashes:
                continue

            results = reader.readtext(plate_area)
            plate_text = ''
            for (_, text, _) in results:
                plate_text = text.strip()

            if plate_text == last_detected_text:
                continue

            last_detected_text = plate_text
            plate_hashes.add(plate_hash)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plate_filename = os.path.join(new_folder, f"plate_{plate_count}_{timestamp}.jpg")
            cv2.imwrite(plate_filename, plate_area)

            plate_count += 1

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, plate_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        video_placeholder.image(frame, channels="BGR", use_column_width=True)

        if stop_button:
            st.warning("🛑 Video processing stopped.")
            break

    cap.release()

# 👉 Function to detect plates in live feed
def detect_plate_live(output_folder, vehicle_cascade):
    st.markdown("## 📹 Real-Time License Plate Detection")
    st.divider()

    cap = cv2.VideoCapture(0)
    new_folder = os.path.join(output_folder, "From_Live")
    os.makedirs(new_folder, exist_ok=True)

    frame_count = 0
    plate_count = 0
    plate_hashes = set()
    last_detected_text = ""

    video_placeholder = st.empty()
    stop_button = st.button("🛑 Stop Live Detection")

    while cap.isOpened():
        frame_count += 1
        has_frame, frame = cap.read()

        if not has_frame:
            st.error("❌ Failed to capture video frame. Please check your camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        plates = vehicle_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in plates:
            if frame_count % 20 != 0 :    #or (w * h >= 7000 or w * h <= 2000)
                continue

            plate_area = frame[y+7:y+h-7, x+7:x+w-7]

            if plate_area.size == 0:
                continue

            plate_hash = compute_image_hash(plate_area)
            if plate_hash in plate_hashes:
                continue

            results = reader.readtext(plate_area)
            plate_text = ''
            for (_, text, _) in results:
                plate_text = text.strip()

            if plate_text == last_detected_text:
                continue

            last_detected_text = plate_text
            plate_hashes.add(plate_hash)

            # Save unique plate image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plate_filename = os.path.join(new_folder, f"plate_{plate_count}_{timestamp}.jpg")
            cv2.imwrite(plate_filename, plate_area)

            plate_count += 1

            # Draw rectangle and add text
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, plate_text, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display frame in Streamlit
        video_placeholder.image(frame, channels="BGR", use_column_width=True)

        if stop_button:
            st.warning("🛑 Live detection stopped.")
            break

    cap.release()


# 👉 Function to handle option selection
def selected_option(option):
    if option == "Image":
        uploaded_image = st.file_uploader("📸 Upload an Image", type=["jpg", "jpeg", "png"])
        if uploaded_image:
            detect_plate_image(uploaded_image, output_folder, vehicle_cascade)

    elif option == "Video":
        video_file = st.file_uploader("🎬 Upload a Video", type=["mp4", "avi", "mov", "mkv"])
        if video_file:
            detect_plate_video(video_file, output_folder, vehicle_cascade)

    elif option == "Real-Time(Live)":
        if st.button("🚦 Start Live Detection"):
            st.warning("📡 Turn on your camera or connect an input device.")
            detect_plate_live(output_folder, vehicle_cascade)
            st.success("✅ Live detection started!")

# 👉 Streamlit UI setup
st.markdown("<h1 style='text-align: center; color: #FF5733;'> Vehicle License Number Plate Detection 🚘</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: left; color: #444;'>Choose the Input Type you want to provide:</h4>", unsafe_allow_html=True)
# Select input type: Image or Video
option = st.selectbox('',["Image", "Video","Real-Time(Live)"])


selected_option(option)

st.markdown("<br><hr><p style='text-align: center;'>🚀 Built with ❤️ using Streamlit, OpenCV, and EasyOCR</p>", unsafe_allow_html=True)
