from projectme.face_recognition import FaceRecognitionSystem

def main():
    face_recognition = FaceRecognitionSystem()
    print("Starting Face Recognition System...")
    print("Say 'capture' to take a photo or 'show' to display the last captured photo")
    print("Press 'q' to quit")
    face_recognition.run()

if __name__ == "__main__":
    main()