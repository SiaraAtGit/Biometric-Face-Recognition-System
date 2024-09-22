import cv2
import dlib
import numpy as np
from imutils import face_utils
import os
import csv

# Load Dlib's pre-trained face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Function to compute facial embeddings
def get_face_embeddings(image, face_rect):
    shape = predictor(image, face_rect)
    face_descriptor = face_rec_model.compute_face_descriptor(image, shape)
    return np.array(face_descriptor)

# Load reference images and their information
def load_reference_images(reference_images_dir, csv_file):
    reference_embeddings = {}
    reference_info = {}
    with open(csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            file_name = row['file_name']
            image_path = os.path.join(reference_images_dir, file_name)
            reference_image = cv2.imread(image_path)
            if reference_image is None:
                print(f"Error: Unable to load image at path: {image_path}")
                continue
            reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
            faces = detector(reference_image, 1)
            if len(faces) != 1:
                print(f"Error: Reference image {file_name} must contain exactly one face.")
                continue
            embedding = get_face_embeddings(reference_image, faces[0])
            reference_embeddings[file_name] = embedding
            reference_info[file_name] = {
                'name': row['name'],
                'age': row['age'],
                'occupation': row['occupation']
            }
    return reference_embeddings, reference_info

# Compare two facial embeddings
def compare_faces(embedding1, embedding2, tolerance=0.6):
    distance = np.linalg.norm(embedding1 - embedding2)
    return distance < tolerance

# Draw structured and aesthetic label with background
def draw_label(img, lines, pos, bg_color=(0, 0, 0), text_color=(255, 255, 255), font_scale=0.6, thickness=1):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    margin = 5

    # Calculate the size of the label
    max_width = max(cv2.getTextSize(line, font_face, font_scale, thickness)[0][0] for line in lines)
    total_height = sum(cv2.getTextSize(line, font_face, font_scale, thickness)[0][1] for line in lines) + margin * len(lines)

    x, y = pos
    end_x = x + max_width + margin * 2
    end_y = y - total_height - margin

    # Draw background rectangle
    cv2.rectangle(img, (x, y), (end_x, end_y), bg_color, cv2.FILLED)

    # Draw text
    y_offset = y
    for line in lines:
        cv2.putText(img, line, (x + margin, y_offset), font_face, font_scale, text_color, thickness, cv2.LINE_AA)
        y_offset -= cv2.getTextSize(line, font_face, font_scale, thickness)[0][1] + margin

# Load reference face embeddings and information
reference_images_dir = "reference_images"  # Directory containing reference images
csv_file = "reference_images_info.csv"     # CSV file with reference images information
reference_embeddings, reference_info = load_reference_images(reference_images_dir, csv_file)

# Initialize webcam
cap = cv2.VideoCapture(100)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector(frame_rgb, 0)
    
    for face in faces:
        (x, y, w, h) = face_utils.rect_to_bb(face)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        face_embedding = get_face_embeddings(frame_rgb, face)
        match_found = False
        label_lines = ["No Match"]
        
        for ref_name, ref_embedding in reference_embeddings.items():
            if compare_faces(ref_embedding, face_embedding):
                info = reference_info[ref_name]
                label_lines = [
                    f"Match: {info['name']}",
                    f"Age: {info['age']}",
                    f"Occupation: {info['occupation']}"
                ]
                match_found = True
                break
        
        draw_label(frame, label_lines, (x, y - 10), bg_color=(0, 0, 0), text_color=(255, 255, 255))
    
    cv2.imshow("Biometric Sensing System", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
