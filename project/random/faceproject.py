"""
Face Recognition System Documentation
===================================

This system implements a complete face recognition pipeline with three main phases:
1. Face Capture
2. Model Training
3. Face Recognition/Testing

Requirements:
------------
- opencv-contrib-python
- numpy
- os (built-in)

Installation:
------------
pip install opencv-contrib-python numpy
"""

import os
import cv2
import numpy as np

def capture_faces(output_dir="captured_faces", person_name="person1", num_images=10):
    """
    Captures face images from webcam and saves them to specified directory.

    Parameters:
    -----------
    output_dir : str
        Directory where captured face images will be stored
    person_name : str
        Name of the person whose face is being captured
    num_images : int
        Number of images to capture

    Usage:
    ------
    capture_faces(person_name="John", num_images=100)

    Controls:
    ---------
    - Press 'q' to quit capturing
    - Press 'c' to capture an image
    - Press 's' to show all captured images
    """
    os.makedirs(output_dir, exist_ok=True)
    cam = cv2.VideoCapture(0)  # Initialize webcam
    count = 0

    while count < num_images:
        ret, frame = cam.read()
        if not ret:
            print("Failed to capture image")
            break

        # Display the frame
        cv2.imshow("Capture Faces", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('c'):  # Capture
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            file_path = os.path.join(output_dir, f"{person_name}_{count}.jpg")
            cv2.imwrite(file_path, gray)
            print(f"Saved: {file_path}")
            count += 1

    cam.release()
    cv2.destroyAllWindows()

def train_faces(data_dir="captured_faces", model_file="face_model.yml"):
    """
    Trains the LBPH face recognizer using captured images.

    Parameters:
    -----------
    data_dir : str
        Directory containing the captured face images
    model_file : str
        File path where trained model will be saved

    Returns:
    --------
    None
        Saves the trained model to disk

    Notes:
    ------
    - Uses LBPH (Local Binary Pattern Histogram) algorithm
    - Creates a label mapping for each unique person
    - Saves label mapping to 'labels.pkl'
    """
    try:
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    except AttributeError:
        print("Error: Install opencv-contrib-python")
        return

    faces = []
    labels = []
    label_dict = {}
    current_label = 0

    # Process each image
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".jpg"):
            file_path = os.path.join(data_dir, file_name)
            label = file_name.split("_")[0]  # Extract person's name

            # Assign numerical label
            if label not in label_dict:
                label_dict[label] = current_label
                current_label += 1

            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            faces.append(img)
            labels.append(label_dict[label])

    # Train and save model
    face_recognizer.train(np.array(faces), np.array(labels))
    face_recognizer.write(model_file)

    # Save label mapping
    with open("labels.pkl", "wb") as f:
        import pickle
        pickle.dump(label_dict, f)

def test_faces(model_file="face_model.yml", labels_file="labels.pkl"):
    """
    Performs real-time face recognition using trained model.

    Parameters:
    -----------
    model_file : str
        Path to trained model file
    labels_file : str
        Path to label mapping file

    Usage:
    ------
    test_faces()

    Controls:
    ---------
    Press 'q' to quit recognition

    Display:
    --------
    Shows live video feed with:
    - Green rectangles around detected faces
    - Person's name and confidence score
    """
    # Load label mapping
    with open(labels_file, "rb") as f:
        import pickle
        label_dict = pickle.load(f)

    # Initialize recognizer
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read(model_file)

    cam = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        # Process each detected face
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            label, confidence = face_recognizer.predict(face)

            # Get name from label
            name = "Unknown"
            for key, value in label_dict.items():
                if value == label:
                    name = key
                    break

            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({confidence:.2f})", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.9, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    """
    Main execution flow of the face recognition system.
    
    Flow:
    1. Capture face images
    2. Train the model
    3. Test face recognition
    """
    # Phase 1: Capture faces
    capture_faces(person_name="John", num_images=100)

    # Phase 2: Train model
    train_faces()

    # Phase 3: Test recognition
    test_faces()
