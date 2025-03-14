################################################ Process-1 ###############################################################
# from PIL import Image
# import cv2
# import pytesseract
# import streamlit as st
# import numpy as np

# # pytesseract  path
# pytesseract.pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# def detect_image(uploaded_image):

#     # Detect plates
#     image=np.array(uploaded_image)
#     st.image(image, caption="Uploaded Image", use_column_width=True)
#     vehicle_cascade=cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')
#     # Detect plates
#     plates=vehicle_cascade.detectMultiScale(image,1.1,6)
#     # Draw rectangle around the detected plate
#     c=1
#     st.write(f"Number of Number Plates detected={len(plates)}")
#     for (x,y,w,h) in plates:
#         cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),3)   # B G  R  Red Is ON
#         plate_area=image[y+7:y+h-7,x+7:x+w-7]

#         # OCR on the plate region
#         plate_text = pytesseract.image_to_string(plate_area, config='--psm 7').strip()
#         st.image(plate_area, caption=f"Number_Plate Detected {c}", use_column_width=True)
#         cv2.putText(image, plate_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)
#         st.write(f"Detected Number Plate {c}: **{plate_text}**")
#         c+=1
#     return image
    
# # Streamlit App Interface
# st.title("Vehicle Number Plate Detection")

# # Upload image for plate detection
# uploaded_image = st.file_uploader("Upload an Image in(jpg, jpeg) format", type=["jpg", "jpeg", "png"])
# if uploaded_image:
#     image = Image.open(uploaded_image)

#     # Perform license plate detection on the uploaded image
#     detected_image = detect_image(image)
#     st.image(detected_image, caption=" Image with Detected number plates", use_column_width=True)



################################################ Process-2 ###############################################################

# from PIL import Image
# import cv2
# import easyocr #import pytesseract
# import streamlit as st
# import numpy as np

# # pytesseract  path
# #pytesseract.pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# reader = easyocr.Reader(['en'])
# def detect_image(uploaded_image):

#     # Detect plates
#     image=np.array(uploaded_image)
#     st.image(image, caption="Uploaded Image", use_column_width=True)
#     image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#     vehicle_cascade=cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')
#     # Detect plates
#     plates=vehicle_cascade.detectMultiScale(image_gray,1.1,6)
#     # Draw rectangle around the detected plate
#     c=1
#     st.write(f"Number of Number Plates detected={len(plates)}")
#     for (x,y,w,h) in plates:
#         cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),3)   # B G  R  Red Is ON
#         plate_area=image[y+7:y+h-7,x+7:x+w-7]
#         results=reader.readtext(plate_area)
#         # OCR on the plate region
#         # plate_text = pytesseract.image_to_string(plate_area, config='--psm 7').strip()
#         for (bbox, text, prob) in results:
#             plate_text =text
#         st.image(plate_area, caption=f"Number_Plate Detected {c}", use_column_width=True)
#         cv2.putText(image, plate_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)
#         st.write(f"Detected Number Plate {c}: **{plate_text}**")
#         c+=1
#     return image
    
# # Streamlit App Interface
# st.title("Vehicle Number Plate Detection")

# # Upload image for plate detection
# uploaded_image = st.file_uploader("Upload an Image in(jpg, jpeg) format", type=["jpg", "jpeg", "png"])
# if uploaded_image:
#     image = Image.open(uploaded_image)

#     # Perform license plate detection on the uploaded image
#     detected_image = detect_image(image)
#     st.image(detected_image, caption=" Image with Detected number plates", use_column_width=True)


################################################ Process-3 ###############################################################
# from PIL import Image
# import cv2
# import easyocr #import pytesseract
# import streamlit as st
# import numpy as np

# # pytesseract  path
# #pytesseract.pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# reader = easyocr.Reader(['en'])
# # Streamlit App Interface
# st.title("Vehicle Number Plate Detection")

# # Upload image for plate detection
# uploaded_image = st.file_uploader("Upload an Image in(jpg, jpeg) format", type=["jpg", "jpeg", "png"])
# if uploaded_image:
#     image = Image.open(uploaded_image)
#      # Detect plates
#     image=np.array(image)
#     st.image(image, caption="Uploaded Image", use_column_width=True)
#     image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#     vehicle_cascade=cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')
#     # Detect plates
#     plates=vehicle_cascade.detectMultiScale(image_gray,1.1,6)
#     # Draw rectangle around the detected plate
#     c=1
#     st.write(f"Number of Number Plates detected={len(plates)}")
#     for (x,y,w,h) in plates:
#         cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),3)   # B G  R  Red Is ON
#         plate_area=image[y+7:y+h-7,x+7:x+w-7]

#         results=reader.readtext(plate_area)
#         for (bbox, text, prob) in results:
#             plate_text =text
#         st.image(plate_area, caption=f"Number_Plate Detected {c}", use_column_width=True)
#         cv2.putText(image, plate_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)
#         st.write(f"Detected Number Plate {c}: **{plate_text}**")
#         c+=1

#     # Perform license plate detection on the uploaded image
#     st.image(image, caption=" Image with Detected number plates", use_column_width=True)

################################################ Process-4 ###############################################################

# from PIL import Image
# import cv2
# import easyocr
# import streamlit as st
# import numpy as np

# # Initialize EasyOCR
# reader = easyocr.Reader(['en'])

# # Streamlit App Interface
# st.title("Vehicle Number Plate Detection")

# # Upload image for plate detection
# uploaded_image = st.file_uploader("Upload an Image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

# if uploaded_image:
#     # Load image using PIL and convert to OpenCV format
#     image = Image.open(uploaded_image)
#     image = np.array(image)

#     # Convert to BGR (for OpenCV) if needed
#     if len(image.shape) == 3 and image.shape[2] == 4:  # If image has alpha channel (RGBA)
#         image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
#     # Convert image to grayscale
#     image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

#     # Load Haar cascade
#     vehicle_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

#     # Detect plates
#     plates = vehicle_cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=6)

#     # Display uploaded image
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     st.write(f"### Number of Plates Detected: **{len(plates)}**")

#     # If no plates detected, notify the user
#     if len(plates) == 0:
#         st.warning("No number plates were detected in the image.")

#     # Process detected plates
#     detected_plate_texts = []
#     for idx, (x, y, w, h) in enumerate(plates, start=1):
#         # Draw rectangle around the detected plate
#         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)

#         # Ensure valid cropping
#         y1, y2 = max(0, y+7), min(y+h-7, image.shape[0])
#         x1, x2 = max(0, x+7), min(x+w-7, image.shape[1])
#         plate_area = image[y1:y2, x1:x2]

#         # Perform OCR
#         results = reader.readtext(plate_area)

#         # Extract text (handling case where OCR detects nothing)
#         plate_text = "Unknown"
#         if results:
#             plate_text = results[0][1]  # Extract text from first result
        
#         detected_plate_texts.append(plate_text)

#         # Display detected plate
#         st.image(plate_area, caption=f"Detected Plate {idx}", use_column_width=True)
        
#         cv2.putText(image, plate_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)
#         st.write(f"**Detected Plate {idx}:** `{plate_text}`")

#     # Display final image with detected plates
#     st.image(image, caption="Image with Detected Number Plates", use_column_width=True)




################################################ Process-5 ###############################################################

# from PIL import Image
# import cv2
# import easyocr
# import streamlit as st
# import numpy as np

# # Initialize EasyOCR
# reader = easyocr.Reader(['en'])

# # Streamlit App Interface
# st.title("Vehicle Number Plate Detection")

# # Upload image for plate detection
# uploaded_image = st.file_uploader("Upload an Image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

# if uploaded_image:
#     # Load image using PIL and convert to OpenCV format
#     image = Image.open(uploaded_image)
#     image = np.array(image)

#     # Convert to BGR (for OpenCV) if needed
#     if len(image.shape) == 3 and image.shape[2] == 4:  # If image has alpha channel (RGBA)
#         image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
#     # Convert image to grayscale
#     image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

#     # Load Haar cascade
#     vehicle_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

#     # Detect plates
#     plates = vehicle_cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=6)

#     # Display uploaded image
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     st.write(f"### Number of Plates Detected: **{len(plates)}**")

#     # If no plates detected, notify the user
#     if len(plates) == 0:
#         st.warning("No number plates were detected in the image.")

#     # Process detected plates
#     detected_plate_texts = []
#     for idx, (x, y, w, h) in enumerate(plates, start=1):
#         # Draw rectangle around the detected plate
#         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)

#         # Ensure valid cropping
#         y1, y2 = max(0, y+7), min(y+h-7, image.shape[0])
#         x1, x2 = max(0, x+7), min(x+w-7, image.shape[1])
#         plate_area = image[y1:y2, x1:x2]

#         # Perform OCR
#         results = reader.readtext(plate_area)

#         # Extract text (handling case where OCR detects nothing)
#         plate_text = "Unknown"
#         if results:
#             plate_text = results[0][1]  # Extract text from first result
        
#         detected_plate_texts.append(plate_text)

#         # Display detected plate
#         st.image(plate_area, caption=f"Detected Plate {idx}", use_column_width=True)
        
#         cv2.putText(image, plate_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)
#         st.write(f"**Detected Plate {idx}:** `{plate_text}`")

#     # Display final image with detected plates
#     st.image(image, caption="Image with Detected Number Plates", use_column_width=True)




idx