################################################ Process-1 ###############################################################
# import streamlit as st
# import cv2
# import easyocr
# import numpy as np
# import os
# import hashlib

# # Initialize OCR reader
# reader = easyocr.Reader(['en'])

# # Load Haar cascade for number plate detection
# vehicle_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

# # Create folder for saving plates
# output_folder = "plates"
# os.makedirs(output_folder, exist_ok=True)
# frame_count=-1
# # Lists to store detected plates & hashes for unique images
# detected_plates = []
# plate_hashes = set()  # Stores unique hashes
# plate_count = 0  # Counter for saved images
# last_saved_plate = None  # Store last saved image for comparison
# last_detected_text = ""
# # Streamlit UI
# st.title("Real-Time License Plate Detection")

# # Start video capture (Webcam)
# cap = cv2.VideoCapture(0)

# # Streamlit placeholders for video
# video_placeholder = st.empty()

# # Stop button handling
# if "stop" not in st.session_state:
#     st.session_state.stop = False

# if st.button("Stop Video Processing"):
#     st.session_state.stop = True

# # Function to compute hash of an image
# def compute_image_hash(image):
#     """Compute a unique hash for an image to prevent duplicates."""
#     return hashlib.md5(image.tobytes()).hexdigest()

# # # Process video frames
# # while cap.isOpened() and not st.session_state.stop:
# #     has_frame, frame = cap.read()
# #     if not has_frame:
# #         break  # Exit loop if no frame is captured

# #     # Detect number plates
# #     plates = vehicle_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=4)
    
# #     for (x, y, w, h) in plates:
# #         frame_count+=1
# #         # Draw red rectangle around detected plate
# #         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

# #         # Extract plate region safely
# #         # plate_area = frame[max(0, y+7): min(y+h-7, frame.shape[0]), 
# #         #                    max(0, x+7): min(x+w-7, frame.shape[1])]
# #         plate_area = frame[y:y+h, x:x+w]

# #         # Ensure valid cropping (avoid empty images)
# #         if plate_area.size == 0:
# #             continue

# #         # Convert plate to grayscale for better OCR
# #         plate_gray = cv2.cvtColor(plate_area, cv2.COLOR_BGR2GRAY)

# #         # Compute hash for uniqueness check
# #         plate_hash = compute_image_hash(plate_gray)

# #         # **Check for duplicate image** (Avoid redundant storage)
# #         if plate_hash in plate_hashes:
# #             continue  # Skip saving duplicate plate image

# #         #**Check if new plate is too similar to the last saved plate**
# #         # Check if new plate is too similar to the last saved plate
# #         if last_saved_plate is not None:
# #             # Resize current plate to match the last saved plate
# #             plate_gray_resized = cv2.resize(plate_gray, (last_saved_plate.shape[1], last_saved_plate.shape[0]))

# #             # Ensure both images are grayscale (1 channel)
# #             if len(plate_gray_resized.shape) == 3:
# #                 plate_gray_resized = cv2.cvtColor(plate_gray_resized, cv2.COLOR_BGR2GRAY)

# #             # Compute difference
# #             diff = cv2.absdiff(last_saved_plate, plate_gray_resized)
            
# #             # If the difference is too small, skip saving
# #             if np.mean(diff) < 5:  
# #                 continue  

# #             # Update last saved plate
# #             last_saved_plate = plate_gray_resized.copy()
# #         else:
# #             # First image, store it directly
# #             last_saved_plate = plate_gray.copy()

# #         # Save plate hash & update last saved plate
# #         plate_hashes.add(plate_hash)
# #         last_saved_plate = plate_gray.copy()

# #         # Perform OCR on the extracted plate
# #         results = reader.readtext(plate_area)

# #         # Extract text from OCR results
# #         plate_text = "Unknown"
# #         for (bbox, text, prob) in results:
# #             plate_text = text.strip()
# #             detected_plates.append(plate_text)
# #         # st.write(text+'\n')
# #         if plate_text == last_detected_text:
# #             continue  

# #         # Save unique plate image
# #         if frame_count%10==0 or plate_text != last_detected_text:
# #             plate_filename = os.path.join(output_folder, f"plate_{plate_count}.jpg")
# #             cv2.imwrite(plate_filename, plate_area)
# #             plate_count += 1
# #             last_detected_text = plate_text  # Update last detected text

# #         # Draw detected text on the frame
# #         cv2.putText(frame, plate_text, (x, y - 10), 
# #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# #     # Display the processed frame in Streamlit
# #     video_placeholder.image(frame, channels="BGR", use_column_width=True)

# # # Release the camera after stopping
# # cap.release()

# # # Remove duplicate detected plates
# # detected_plates = list(set(detected_plates))


# # Start video capture


# while cap.isOpened() and not st.session_state.stop:
#     has_frame, frame = cap.read()
#     if not has_frame:
#         break

#     # Detect number plates
#     plates = vehicle_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=4)

#     for (x, y, w, h) in plates:
#         frame_count += 1
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

#         # Extract number plate area
#         plate_area = frame[y+7:y+h-7,x+7:x+w-7]
#         if plate_area.size == 0:
#             continue

#         plate_gray = cv2.cvtColor(plate_area, cv2.COLOR_BGR2GRAY)
#         plate_hash = compute_image_hash(plate_gray)

#         # Check for duplicate image
#         if plate_hash in plate_hashes:
#             continue
            
#         # OCR Processing
#         results = reader.readtext(plate_area)
#         plate_text = "Unknown"
#         for (_, text, _) in results:
#             plate_text = text.strip()
#             detected_plates.append(plate_text)

#         # Skip redundant detections
#         if plate_text == last_detected_text:
#             continue  

#         # Using set avoids duplicates
#         last_detected_text = plate_text  # Update last detected text

#             # Save unique plate image every 10 frames or if a new plate is detected
#         if frame_count % 5 == 0 or plate_text != last_detected_text:
#             plate_filename = os.path.join(output_folder, f"plate_{plate_count}.jpg")
#             cv2.imwrite(plate_filename, plate_area)
#             plate_count += 1
#             plate_hashes.add(plate_hash)  # Store hash to avoid duplicates

#             # Display detected text on frame
#         cv2.putText(frame, plate_text, (x, y - 10), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#         # Show processed video in Streamlit
#     video_placeholder.image(frame, channels="BGR", use_column_width=True)

# cap.release()

#     # Remove duplicate detected plates
# detected_plates = list(set(detected_plates))


# if st.session_state.stop:
#     # # Display detected number plates
#     if detected_plates:
#         st.success("✅ **Detected Number Plates:**")
#         for plate in detected_plates:
#             st.write(plate)
#     else:
#         st.warning("❌ No number plates detected.")

#     # # Display saved image filenames
#     st.subheader("📁 Saved License Plate Images:")
#     if plate_count > 0:
#         for i in range(plate_count):
#             st.write(f"- `plate_{i}.jpg` saved in `{output_folder}/`")
#     else:
#        st.warning("❌ No license plate images were saved.")


################################################ Process-2 ###############################################################

# # import streamlit as st
# # import cv2
# # import easyocr
# # import numpy as np
# # import os
# # import hashlib

# # # Initialize OCR reader
# # reader = easyocr.Reader(['en'])

# # # Load Haar cascade for number plate detection
# # vehicle_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

# # # Create folder for saving plates
# # output_folder = "plates"
# # os.makedirs(output_folder, exist_ok=True)

# # # Initialize variables
# # frame_count = 0
# # detected_plates = []  # Using set to prevent duplicates immediately
# # plate_hashes = set()  # Stores unique image hashes
# # plate_count = 0
# # last_saved_plate = None
# # last_detected_text = ""

# # def compute_image_hash(image):
# #     return hashlib.md5(image.tobytes()).hexdigest()

# # def start():
# #     # Start video capture
# #     cap = cv2.VideoCapture(0)
# #     video_placeholder = st.empty()

# #     while cap.isOpened() and not st.session_state.stop:
# #         has_frame, frame = cap.read()
# #         if not has_frame:
# #             break

# #         # Detect number plates
# #         plates = vehicle_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=4)

# #         for (x, y, w, h) in plates:
# #             frame_count += 1
# #             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

# #             # Extract number plate area
# #             plate_area = frame[y+7:y+h-7,x+7:x+w-7]
# #             if plate_area.size == 0:
# #                 continue

# #             plate_gray = cv2.cvtColor(plate_area, cv2.COLOR_BGR2GRAY)
# #             plate_hash = compute_image_hash(plate_gray)

# #             # Check for duplicate image
# #             if plate_hash in plate_hashes:
# #                 continue
            
# #             # OCR Processing
# #             results = reader.readtext(plate_area)
# #             plate_text = "Unknown"
# #             for (_, text, _) in results:
# #                 plate_text = text.strip()
# #                 detected_plates.append(plate_text)

# #             # Skip redundant detections
# #             if plate_text == last_detected_text:
# #                 continue  

# #             # Using set avoids duplicates
# #             last_detected_text = plate_text  # Update last detected text

# #             # Save unique plate image every 10 frames or if a new plate is detected
# #             if frame_count % 5 == 0 or plate_text != last_detected_text:
# #                 plate_filename = os.path.join(output_folder, f"plate_{plate_count}.jpg")
# #                 cv2.imwrite(plate_filename, plate_area)
# #                 plate_count += 1
# #                 plate_hashes.add(plate_hash)  # Store hash to avoid duplicates

# #             # Display detected text on frame
# #             cv2.putText(frame, plate_text, (x, y - 10), 
# #                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# #         # Show processed video in Streamlit
# #         video_placeholder.image(frame, channels="BGR", use_column_width=True)

# #     cap.release()

# #     # Remove duplicate detected plates
# #     detected_plates = list(set(detected_plates))

# # def stop():
# #     detected_plates=list(set(detected_plates))
# #     if detected_plates:
# #         st.success("✅ **Detected Number Plates:**")
# #         for plate in detected_plates:
# #             st.write(f"📌 {plate}")
# #     else:
# #         st.warning("❌ No number plates detected.")

# #     # Show saved plate images
# #     st.subheader("📁 Saved License Plate Images:")
# #     # if plate_count > 0:
# #     #     for i in range(plate_count):
# #     #         st.write(f"- `plate_{i}.jpg` saved in `{output_folder}/`")
# #     # else:
# #     #     st.warning("❌ No license plate images were saved.")
    

# # # Streamlit UI
# # st.title("Real-Time License Plate Detection")

# # # Start/Stop Buttons
# # if "stop" not in st.session_state:
# #     st.session_state.stop = False

# # if st.button("Start Video Processing"):
# #     start()

# # if st.button("Stop Video Processing"):
# #         st.session_state.stop = True
# #         stop()

# # # Function to compute image hash

    



# # # Display results after stopping

    



################################################ Process-3 ###############################################################
# # import streamlit as st
# # import cv2
# # import easyocr
# # import numpy as np
# # import os
# # import hashlib

# # # Initialize OCR reader
# # reader = easyocr.Reader(['en'])

# # # Load Haar cascade for number plate detection
# # vehicle_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

# # # Create folder for saving plates
# # output_folder = "plates"
# # os.makedirs(output_folder, exist_ok=True)

# # # Streamlit UI
# # st.title("Real-Time License Plate Detection")

# # # Initialize session state variables
# # if "stop" not in st.session_state:
# #     st.session_state.stop = False
# # if "detected_plates" not in st.session_state:
# #     st.session_state.detected_plates = set()  # Use set to remove duplicates
# # if "plate_hashes" not in st.session_state:
# #     st.session_state.plate_hashes = set()
# # if "plate_count" not in st.session_state:
# #     st.session_state.plate_count = 0
# # if "last_detected_text" not in st.session_state:
# #     st.session_state.last_detected_text = ""

# # # Function to compute image hash
# # def compute_image_hash(image):
# #     return hashlib.md5(image.tobytes()).hexdigest()

# # # Start button logic
# # if st.button("Start Video Processing"):
# #     st.session_state.stop = False  # Reset stop state

# #     # Start video capture
# #     cap = cv2.VideoCapture(0)
# #     video_placeholder = st.empty()

# #     frame_count = 0

# #     while cap.isOpened() and not st.session_state.stop:
# #         has_frame, frame = cap.read()
# #         if not has_frame:
# #             break

# #         # Detect number plates
# #         plates = vehicle_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=4)

# #         for (x, y, w, h) in plates:
# #             frame_count += 1
# #             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

# #             # Extract number plate area (with safe boundary checks)
# #             plate_area = frame[max(0, y+7): min(y+h-7, frame.shape[0]), 
# #                                max(0, x+7): min(x+w-7, frame.shape[1])]
# #             if plate_area.size == 0:
# #                 continue

# #             # Convert plate to grayscale
# #             plate_gray = cv2.cvtColor(plate_area, cv2.COLOR_BGR2GRAY)
# #             plate_hash = compute_image_hash(plate_gray)

# #             # Check for duplicate images
# #             if plate_hash in st.session_state.plate_hashes:
# #                 continue
            
# #             # OCR Processing
# #             results = reader.readtext(plate_area)
# #             plate_text = "Unknown"
# #             for (_, text, _) in results:
# #                 plate_text = text.strip()
            
# #             # Avoid duplicate plate detections
# #             if plate_text == st.session_state.last_detected_text:
# #                 continue

# #             # Store detected text
# #             st.session_state.detected_plates.add(plate_text)
# #             st.session_state.last_detected_text = plate_text  # Update last detected text

# #             # Save unique plate image every 5 frames or if a new plate is detected
# #             if frame_count % 5 == 0 or plate_text != st.session_state.last_detected_text:
# #                 plate_filename = os.path.join(output_folder, f"plate_{st.session_state.plate_count}.jpg")
# #                 cv2.imwrite(plate_filename, plate_area)
# #                 st.session_state.plate_count += 1
# #                 st.session_state.plate_hashes.add(plate_hash)  # Store hash to avoid duplicates

# #             # Display detected text on frame
# #             cv2.putText(frame, plate_text, (x, y - 10), 
# #                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# #         # Show processed video in Streamlit
# #         video_placeholder.image(frame, channels="BGR", use_column_width=True)

# #     cap.release()

# # # Stop button logic
# # if st.button("Stop Video Processing"):
# #     st.session_state.stop = True

# # # Display results after stopping
# # if st.session_state.stop:
# #     if st.session_state.detected_plates:
# #         st.success("✅ **Detected Number Plates:**")
# #         for plate in st.session_state.detected_plates:
# #             st.write(f"📌 {plate}")
# #     else:
# #         st.warning("❌ No number plates detected.")

# #     # Show saved plate images
# #     st.subheader("📁 Saved License Plate Images:")
# #     if st.session_state.plate_count > 0:
# #         for i in range(st.session_state.plate_count):
# #             st.write(f"- `plate_{i}.jpg` saved in `{output_folder}/`")
# #     else:
# #         st.warning("❌ No license plate images were saved.")


################################################ Process-4 ###############################################################

# # import streamlit as st
# # import cv2
# # import easyocr
# # import numpy as np
# # import os
# # import hashlib

# # # Initialize OCR reader
# # reader = easyocr.Reader(['en'])

# # # Load Haar cascade for number plate detection
# # vehicle_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

# # # Create folder for saving plates
# # output_folder = "plates"
# # os.makedirs(output_folder, exist_ok=True)

# # # Streamlit UI
# # st.title("Real-Time License Plate Detection")

# # # Initialize session state variables
# # if "stop" not in st.session_state:
# #     st.session_state.stop = False
# # if "detected_plates" not in st.session_state:
# #     st.session_state.detected_plates = set()  # Use set to avoid duplicates
# # if "plate_hashes" not in st.session_state:
# #     st.session_state.plate_hashes = set()
# # if "plate_count" not in st.session_state:
# #     st.session_state.plate_count = 0
# # if "last_detected_text" not in st.session_state:
# #     st.session_state.last_detected_text = ""

# # # Function to compute image hash
# # def compute_image_hash(image):
# #     return hashlib.md5(image.tobytes()).hexdigest()

# # # Start/Stop Buttons
# # if st.button("Start Video Processing"):
# #     st.session_state.stop = False  # Reset stop state

# #     # Start video capture
# #     cap = cv2.VideoCapture(0)
# #     video_placeholder = st.empty()

# #     frame_count = 0

# #     while cap.isOpened() and not st.session_state.stop:
# #         has_frame, frame = cap.read()
# #         if not has_frame:
# #             break

# #         # Detect number plates
# #         plates = vehicle_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=4)

# #         for (x, y, w, h) in plates:
# #             frame_count += 1
# #             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

# #             # Extract number plate area (safe cropping)
# #             plate_area = frame[max(0, y+7): min(y+h-7, frame.shape[0]), 
# #                                max(0, x+7): min(x+w-7, frame.shape[1])]
# #             if plate_area.size == 0:
# #                 continue

# #             # Convert to grayscale
# #             plate_gray = cv2.cvtColor(plate_area, cv2.COLOR_BGR2GRAY)
# #             plate_hash = compute_image_hash(plate_gray)

# #             # Skip if already detected
# #             if plate_hash in st.session_state.plate_hashes:
# #                 continue
            
# #             # OCR Processing
# #             results = reader.readtext(plate_area)
# #             plate_text = "Unknown"
# #             for (_, text, _) in results:
# #                 plate_text = text.strip()
            
# #             # Avoid duplicate detections
# #             if plate_text in st.session_state.detected_plates:
# #                 continue

# #             # Store detected text
# #             st.session_state.detected_plates.add(plate_text)
# #             st.session_state.last_detected_text = plate_text

# #             # Save plate image every 5 frames or for a new plate
# #             if frame_count % 5 == 0 or plate_text != st.session_state.last_detected_text:
# #                 plate_filename = os.path.join(output_folder, f"plate_{st.session_state.plate_count}.jpg")
# #                 cv2.imwrite(plate_filename, plate_area)
# #                 st.session_state.plate_count += 1
# #                 st.session_state.plate_hashes.add(plate_hash)

# #             # Display detected text on frame
# #             cv2.putText(frame, plate_text, (x, y - 10), 
# #                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# #         # Show processed video in Streamlit
# #         video_placeholder.image(frame, channels="BGR", use_column_width=True)

# #     cap.release()

# # # Stop button logic
# # if st.button("Stop Video Processing"):
# #     st.session_state.stop = True

# # # Display results after stopping
# # if st.session_state.stop:
# #     if st.session_state.detected_plates:
# #         st.success("✅ **Detected Number Plates:**")
# #         for plate in st.session_state.detected_plates:
# #             st.write(f"📌 {plate}")
# #     else:
# #         st.warning("❌ No number plates detected.")

# #     # Show saved plate images
# #     st.subheader("📁 Saved License Plate Images:")
# #     if st.session_state.plate_count > 0:
# #         for i in range(st.session_state.plate_count):
# #             st.write(f"- `plate_{i}.jpg` saved in `{output_folder}/`")
# #     else:
# #         st.warning("❌ No license plate images were saved.")

################################################ Process-6 ###############################################################


# # import streamlit as st
# # import cv2
# # import easyocr
# # import numpy as np
# # import os
# # import hashlib

# # # Initialize OCR reader
# # reader = easyocr.Reader(['en'])

# # # Load Haar cascade for number plate detection
# # vehicle_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

# # # Create folder for saving plates
# # output_folder = "plates"
# # os.makedirs(output_folder, exist_ok=True)
# # frame_count = -1

# # # Lists to store detected plates & hashes for unique images
# # detected_plates = []
# # plate_hashes = set()  # Stores unique hashes
# # plate_count = 0  # Counter for saved images
# # last_saved_plate = None  # Store last saved image for comparison
# # last_detected_text = ""

# # # Streamlit UI
# # st.title("Real-Time License Plate Detection")

# # # Stop button handling
# # if "stop" not in st.session_state:
# #     st.session_state.stop = False

# # if st.button("Stop Video Processing"):
# #     st.session_state.stop = True

# # # Function to compute hash of an image
# # def compute_image_hash(image):
# #     """Compute a unique hash for an image to prevent duplicates."""
# #     return hashlib.md5(image.tobytes()).hexdigest()

# # # Start video capture
# # cap = cv2.VideoCapture(0)
# # video_placeholder = st.empty()

# # while cap.isOpened() and not st.session_state.stop:
# #     has_frame, frame = cap.read()
# #     if not has_frame:
# #         break

# #     # Detect number plates
# #     plates = vehicle_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=4)

# #     for (x, y, w, h) in plates:
# #         frame_count += 1
# #         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

# #         # Extract number plate area safely
# #         y1, y2 = max(0, y+7), min(y+h-7, frame.shape[0])
# #         x1, x2 = max(0, x+7), min(x+w-7, frame.shape[1])
# #         plate_area = frame[y1:y2, x1:x2]

# #         if plate_area.size == 0:
# #             continue

# #         plate_gray = cv2.cvtColor(plate_area, cv2.COLOR_BGR2GRAY)
# #         plate_hash = compute_image_hash(plate_gray)

# #         # Check for duplicate image
# #         if plate_hash in plate_hashes:
# #             continue
            
# #         # OCR Processing
# #         results = reader.readtext(plate_area)
# #         plate_text = "Unknown"
# #         for (_, text, _) in results:
# #             plate_text = text.strip()
# #             detected_plates.append(plate_text)

# #         # Skip redundant detections
# #         if plate_text == last_detected_text:
# #             continue  

# #         # Save unique plate image every 5 frames or if a new plate is detected
# #         if frame_count % 5 == 0 or plate_text != last_detected_text:
# #             plate_filename = os.path.join(output_folder, f"plate_{plate_count}.jpg")
# #             cv2.imwrite(plate_filename, plate_area)
# #             plate_count += 1
# #             plate_hashes.add(plate_hash)  # Store hash to avoid duplicates
# #             last_detected_text = plate_text  # Update last detected text

# #         # Display detected text on frame
# #         cv2.putText(frame, plate_text, (x, y - 10), 
# #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# #     # Show processed video in Streamlit
# #     video_placeholder.image(frame, channels="BGR", use_column_width=True)

# # cap.release()

# # # Remove duplicate detected plates
# # detected_plates = list(set(detected_plates))
