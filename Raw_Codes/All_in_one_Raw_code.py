################################################ Process-1 ###############################################################
# import streamlit as st
# import cv2
# import easyocr
# import numpy as np
# import os
# import hashlib
# from datetime import datetime
# import tempfile
# from PIL import Image

# # Step 1: Create the 'Detected_plates' folder
# output_folder = "Detected_plates"
# os.makedirs(output_folder, exist_ok=True)
# # Initialize EasyOCR
# reader = easyocr.Reader(['en'])

# # Load Haar cascade
# vehicle_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

# # Function to compute hash of an image
# def compute_image_hash(image):
#     return hashlib.md5(image.tobytes()).hexdigest()


# def detect_plate_image(image,output_folder,vehicle_cascade):
#     # Step 2: Create the 'from_images' folder inside 'Detected_plates'for saving Image number plates
#     new_folder = os.path.join(output_folder, "From_Images")
#     os.makedirs(new_folder, exist_ok=True)

#     # Streamlit App Interface
#     st.subheader("Vehicle License Number Plate Detection for Images")
#     if uploaded_image:
#         # Load image using PIL and convert to OpenCV format
#         image = Image.open(uploaded_image)
#         image = np.array(image)
#         st.image(image, caption="Uploaded Image", use_column_width=True)
#         # Convert to BGR (for OpenCV) if needed
#         # if len(image.shape) == 3 and image.shape[2] == 4:  # If image has alpha channel (RGBA)
#         #     image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
#         # Convert image to grayscale
#         image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

#         # Detect plates
#         plates = vehicle_cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=6)


#         st.write(f"### Number of Plates Detected: **{len(plates)}**")

#         # If no plates detected, notify the user
#         if len(plates) == 0:
#             st.warning("No number plates were detected in the image.")

#         # Process detected plates
#         detected_plate_texts = []
#         c=1
#         for (x,y,w,h) in plates:
#             # Draw rectangle around the detected plate
#             cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)

#             # Ensure valid cropping
#             # y1, y2 = max(0, y+7), min(y+h-7, image.shape[0])
#             # x1, x2 = max(0, x+7), min(x+w-7, image.shape[1])
#             # plate_area = image[y1:y2, x1:x2]
#             plate_area=image[y+7:y+h-7,x+7:x+w-7]

#             # Perform OCR
#             results = reader.readtext(plate_area)

#             # Extract text (handling case where OCR detects nothing)
#             plate_text = "Unknown"
#             if results:
#                 plate_text = results[0][1]  # Extract text from first result
            
#             detected_plate_texts.append(plate_text)

#             # Display detected plate
#             st.image(plate_area, caption=f"Detected Plate {c}", use_column_width=True)
#             # plate_filename = os.path.join(output_folder, f"plate_{c}.jpg")
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
#             plate_filename = os.path.join(new_folder, f"plate_{c}_{timestamp}.jpg")
#             cv2.imwrite(plate_filename, plate_area)
#             cv2.putText(image, plate_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)
#             st.write(f"**Detected Plate {c}:** `{plate_text}`")
#             c+=1

#         # Display final image with detected plates
#         st.image(image, caption="Image with Detected Number Plates", use_column_width=True)
        
# def detect_plate_video(video_file,output_folder,vehicle_cascade):
#     # Create the 'from_Videos' folder inside 'Detected_plates'for saving Video number plates
#     new_folder = os.path.join(output_folder, "From_Videos")
#     os.makedirs(new_folder, exist_ok=True)

#     # Initialize variables
#     frame_count =-1
#     plate_count = 0
#     plate_hashes = set()
#     last_detected_text = ""

#     # Streamlit UI
#     st.subheader("Vehicle License Number Plate Detection for Videos")


#     # If video is uploaded
#     if video_file is not None:
#         st.video(video_file)

#         # Create a temporary file to store video data
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
#             temp_file.write(video_file.read())
#             video_path = temp_file.name

#         # Open the video file
#         cap = cv2.VideoCapture(video_path)

#         if not cap.isOpened():
#             st.error("Error: Couldn't open video file.")
#             st.stop()

#         # Streamlit placeholder for video
#         video_placeholder = st.empty()

#         # Stop button
#         stop_button = st.button("Stop Video Processing")

#         while cap.isOpened():
#             has_frame, frame = cap.read()
#             if not has_frame:
#                 break

#             # Convert frame to grayscale for better plate detection
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#             # Detect plates using Haar cascade
#             plates = vehicle_cascade.detectMultiScale(gray, 1.1, 4)

#             for (x, y, w, h) in plates:
#                 #frame_count += 1
#                 if frame_count % 30 != 0:  # Process every 5th frame
#                     continue
#                 if (w*h>=7000 or w*h<=2000):
#                     continue
#                 # st.write(w*h)
#                 # st.write(frame_count)
#                 # Draw rectangle around detected plate
#                 cv2.rectangle(frame,(x, y), (x + w, y + h), (255, 0, 0), 3)

#                 # Extract number plate area
#                 plate_area = frame[max(0, y + 7): min(y + h - 7, frame.shape[0]),
#                                 max(0, x + 7): min(x + w - 7, frame.shape[1])]

#                 if plate_area.size == 0:
#                     continue
                
#                 # Convert plate to grayscale
#                 plate_gray = cv2.cvtColor(plate_area, cv2.COLOR_BGR2GRAY)
#                 plate_hash = compute_image_hash(plate_gray)

#                 # Avoid duplicate plate detection using hash
#                 if plate_hash in plate_hashes:
#                     continue

#                 # OCR Processing
#                 results = reader.readtext(plate_area)
#                 plate_text = " "
#                 for (_, text, _) in results:
#                     plate_text = text.strip()

#                 # Skip if the same text was detected previously
#                 if plate_text == last_detected_text:
#                     continue

#                 last_detected_text = plate_text

#                 # Save unique plate image every 10 frames or if a new plate is detected
#                 if frame_count % 30 == 0 or plate_text != last_detected_text:
#                     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#                     plate_filename = os.path.join(new_folder, f"plate_{plate_count}_{timestamp}.jpg")
#                     cv2.imwrite(plate_filename, plate_area)
#                     plate_count += 1
#                     plate_hashes.add(plate_hash)  # Store hash to avoid duplicates

#                 # Display detected text on frame
#                 cv2.putText(frame, plate_text, (x, y - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#             # Display the frame in Streamlit
#             video_placeholder.image(frame, channels="BGR", use_column_width=True)

#             # Stop processing if the stop button is clicked
#             if stop_button:
#                 #st.write(frame_count)
#                 st.warning("Video processing stopped.")
#                 break

#         cap.release()

#     else:
#         st.info("Please upload a video file to start processing.")

# def detect_plate_live(output_folder,vehicle_cascade):
#     # Create the 'from_Videos' folder inside 'Detected_plates'for saving Video number plates
#     new_folder = os.path.join(output_folder, "From_Live")
#     os.makedirs(new_folder, exist_ok=True)
#     frame_count=-1
#     plate_hashes = set()
#     plate_count=0
#     last_detected_text = ""
#     # Streamlit UI
#     st.subheader("Real-Time Vehicle License Number Plate Detection")

#     # Start video capture (Webcam)
#     cap = cv2.VideoCapture(0)

#     # Streamlit placeholders for video
#     video_placeholder = st.empty()

#     # Stop button handling
#     if "stop" not in st.session_state:
#         st.session_state.stop = False

#     if st.button("Stop Video Processing"):
#         st.session_state.stop = True

#     while cap.isOpened() and not st.session_state.stop:
#         has_frame, frame = cap.read()
#         if not has_frame:
#             break

#         # Detect number plates
#         plates = vehicle_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=4)

#         for (x, y, w, h) in plates:
#             frame_count += 1
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

#             # Extract number plate area
#             plate_area = frame[max(0, y+7): min(y+h-7, frame.shape[0]), 
#                     max(0, x+7): min(x+w-7, frame.shape[1])]
            
#             if plate_area.size == 0:
#                 continue

#             plate_gray = cv2.cvtColor(plate_area, cv2.COLOR_BGR2GRAY)
#             plate_hash = compute_image_hash(plate_gray)

#             # Check for duplicate image
#             if plate_hash in plate_hashes:
#                 continue
                
#             # OCR Processing
#             results = reader.readtext(plate_area)
#             plate_text = "Unknown"
#             for (_, text, _) in results:
#                 plate_text = text.strip()

#             # Skip redundant detections
#             if plate_text == last_detected_text:
#                 continue  

#             # Using set avoids duplicates
#             last_detected_text = plate_text  # Update last detected text

#                 # Save unique plate image every 10 frames or if a new plate is detected
#             if frame_count % 20 == 0 or plate_text != last_detected_text:
#                 timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
#                 plate_filename = os.path.join(new_folder, f"plate_{plate_count}_{timestamp}.jpg")
#                 cv2.imwrite(plate_filename, plate_area)
#                 plate_count += 1
#                 plate_hashes.add(plate_hash)  # Store hash to avoid duplicates

#                 # Display detected text on frame
#             cv2.putText(frame, plate_text, (x, y - 10), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#             # Show processed video in Streamlit
        
#         video_placeholder.image(frame, channels="BGR", use_column_width=True)

#     cap.release()

# def selected_option(option):
#     if option == "Image":
#             # Upload image for plate detection
#             uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
#             if uploaded_image:
#                 image = Image.open(uploaded_image)
#                 # Perform license plate detection on the uploaded image
#                 processed_image = detect_plate_image(image,output_folder,vehicle_cascade)

#     elif option == "Video":
#             # Upload video for plate detection
#             video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])
#             if video_file:
#                 # Perform license plate detection on the uploaded video
#                 detect_plate_video(video_file,output_folder,vehicle_cascade)

#     elif option == "Real-Time(Live)":
#         # Only show live detection button if "Real-Time" is selected
#         if st.button("Start Live Detection"):
#             st.warning("Turn on your Camera or Connect any input device ")
#             detect_plate_live(output_folder,vehicle_cascade)
#             st.success("Live detection started!")





# st.title("Vehicle License Number Plate Detection")
# # Select input type: Image or Video
# option = st.selectbox("Choose the Input Type you want to provide", ["Image", "Video","Real-Time(Live)"])

# selected_option(option)

################################################ Process-2 ###############################################################

# import streamlit as st
# import cv2
# import easyocr
# import numpy as np
# import os
# import hashlib
# from datetime import datetime
# import tempfile
# from PIL import Image

# # Create the 'Detected_plates' folder
# output_folder = "Detected_plates"
# os.makedirs(output_folder, exist_ok=True)

# # Initialize EasyOCR
# reader = easyocr.Reader(['en'])

# # Load Haar cascade
# vehicle_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

# # Function to compute hash of an image
# def compute_image_hash(image):
#     return hashlib.md5(image.tobytes()).hexdigest()

# # 👉 Function to detect plates in images
# def detect_plate_image(uploaded_image, output_folder, vehicle_cascade):
#     st.header("🚗 Vehicle License Number Plate Detection for Images")

#     # Create the 'From_Images' folder inside 'Detected_plates'
#     new_folder = os.path.join(output_folder, "From_Images")
#     os.makedirs(new_folder, exist_ok=True)

#     # Load image using PIL and convert to OpenCV format
#     image = np.array(Image.open(uploaded_image))
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # Convert image to grayscale
#     image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

#     # Detect plates using Haar cascade
#     plates = vehicle_cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=6)

#     st.write(f"### 🛡️ Plates Detected: **{len(plates)}**")

#     if len(plates) == 0:
#         st.warning("No number plates were detected in the image.")
#         return

#     c = 1
#     for (x, y, w, h) in plates:
#         plate_area = image[y+7:y+h-7, x+7:x+w-7]

#         if plate_area.size == 0:
#             continue

#         # OCR for plate text
#         results = reader.readtext(plate_area)
#         plate_text=''
#         for (_, text, _) in results:
#             plate_text = text.strip()
#         # Save detected plate image
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         plate_filename = os.path.join(new_folder, f"plate_{c}_{timestamp}.jpg")
#         cv2.imwrite(plate_filename, plate_area)

#         # Display detected plate and text
#         st.image(plate_area, caption=f"Detected Plate {c}", use_column_width=True)
#         st.write(f"**Detected Plate {c}:** `{plate_text}`")

#         # Draw rectangle and add text to the image
#         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
#         cv2.putText(image, plate_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)

#         c += 1

#     st.image(image, caption="Image with Detected Plates", use_column_width=True)

# # 👉 Function to detect plates in video streams (both video files and live)


# # 👉 Function to detect plates in videos
# def detect_plate_video(video_file, output_folder, vehicle_cascade):
#     st.header("🎬 Vehicle License Number Plate Detection for Videos")
#     st.subheader('Provided Video')
#     st.video(video_file)
#     st.subheader('Processing Video')

#     with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
#         temp_file.write(video_file.read())
#         video_path = temp_file.name

#     cap = cv2.VideoCapture(video_path)
#     new_folder = os.path.join(output_folder, "From_Videos")
#     os.makedirs(new_folder, exist_ok=True)

#     frame_count = 0
#     plate_count = 0
#     plate_hashes = set()
#     last_detected_text = ""

#     video_placeholder = st.empty()
#     stop_button = st.button("Stop Processing")

#     while cap.isOpened():
#         frame_count += 1
#         has_frame, frame = cap.read()

#         if not has_frame:
#             break

#         # Convert frame to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Detect plates using Haar cascade
#         plates = vehicle_cascade.detectMultiScale(gray, 1.1, 4)

#         for (x, y, w, h) in plates:
#             if frame_count % 20 != 0 or (w * h >= 7000 or w * h <= 2000):
#                 continue

#             plate_area = frame[y+7:y+h-7, x+7:x+w-7]

#             if plate_area.size == 0:
#                 continue

#             plate_hash = compute_image_hash(plate_area)
#             if plate_hash in plate_hashes:
#                 continue

#             # OCR Processing
#             results = reader.readtext(plate_area)
#             plate_text=''
#             for (_, text, _) in results:
#                 plate_text = text.strip()

#             # Skip if the same text was detected previously
#             if plate_text == last_detected_text:
#                 continue

#             last_detected_text = plate_text
#             plate_hashes.add(plate_hash)

#             # Save unique plate image
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             plate_filename = os.path.join(new_folder, f"plate_{plate_count}_{timestamp}.jpg")
#             cv2.imwrite(plate_filename, plate_area)

#             plate_count += 1

#             # Draw rectangle and add text
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.putText(frame, plate_text, (x, y - 10), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#         # Display frame in Streamlit
#         video_placeholder.image(frame, channels="BGR", use_column_width=True)

#         if stop_button:
#             st.warning("Video processing stopped.")
#             break

#     cap.release()

# # 👉 Function to detect plates in live feed
# def detect_plate_live(output_folder, vehicle_cascade):
#     st.header("📹 Real-Time License Plate Detection")

#     cap = cv2.VideoCapture(0)
#     new_folder = os.path.join(output_folder, "From_Live")
#     os.makedirs(new_folder, exist_ok=True)

#     frame_count = 0
#     plate_count = 0
#     plate_hashes = set()
#     last_detected_text = ""

#     video_placeholder = st.empty()
#     stop_button = st.button("Stop Processing")

#     while cap.isOpened():
#         frame_count += 1
#         has_frame, frame = cap.read()

#         if not has_frame:
#             break

#         # Convert frame to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Detect plates using Haar cascade
#         plates = vehicle_cascade.detectMultiScale(gray, 1.1, 4)

#         for (x, y, w, h) in plates:
#             if frame_count % 20 != 0 or (w * h >= 7000 or w * h <= 2000):
#                 continue

#             plate_area = frame[y+7:y+h-7, x+7:x+w-7]

#             if plate_area.size == 0:
#                 continue

#             plate_hash = compute_image_hash(plate_area)
#             if plate_hash in plate_hashes:
#                 continue

#             # OCR Processing
#             results = reader.readtext(plate_area)
#             plate_text=''
#             for (_, text, _) in results:
#                 plate_text = text.strip()

#             # Skip if the same text was detected previously
#             if plate_text == last_detected_text:
#                 continue

#             last_detected_text = plate_text
#             plate_hashes.add(plate_hash)

#             # Save unique plate image
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             plate_filename = os.path.join(new_folder, f"plate_{plate_count}_{timestamp}.jpg")
#             cv2.imwrite(plate_filename, plate_area)

#             plate_count += 1

#             # Draw rectangle and add text
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.putText(frame, plate_text, (x, y - 10), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#         # Display frame in Streamlit
#         video_placeholder.image(frame, channels="BGR", use_column_width=True)

#         if stop_button:
#             st.warning("Video processing stopped.")
#             break

#     cap.release()

# # 👉 Function to handle option selection
# def selected_option(option):
#     if option == "Image":
#         uploaded_image = st.file_uploader("📸 Upload an Image", type=["jpg", "jpeg", "png"])
#         if uploaded_image:
#             detect_plate_image(uploaded_image, output_folder, vehicle_cascade)

#     elif option == "Video":
#         video_file = st.file_uploader("🎬 Upload a Video", type=["mp4", "avi", "mov", "mkv"])
#         if video_file:
#             detect_plate_video(video_file, output_folder, vehicle_cascade)

#     elif option == "Real-Time(Live)":
#         if st.button("🚦 Start Live Detection"):
#             st.warning("Turn on your camera or connect an input device.")
#             detect_plate_live(output_folder, vehicle_cascade)
#             st.success("Live detection started!")

# # 👉 Streamlit UI setup
# st.title("🚘 Vehicle License Number Plate Detection")
# st.markdown("## Choose the input type you want to provide:")

# # Align UI elements using columns
# col1, col2, col3 = st.columns(3)
# with col1:
#     option = st.radio("", ["Image", "Video", "Real-Time(Live)"])

# selected_option(option)