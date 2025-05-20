import cv2
import speech_recognition as sr
import numpy as np
from datetime import datetime
import glob
import os

class FaceRecognitionSystem:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recognizer = sr.Recognizer()
        self.current_image = None
        self.running = True  # Add a flag to control the main loop

    def capture_on_command(self):
        """Listen for voice command and capture image when triggered"""
        with sr.Microphone() as source:
            print("Listening for commands ('capture', 'show', 'show all' or 'exit')...")
            try:
                audio = self.recognizer.listen(source, timeout=5)
                command = self.recognizer.recognize_google(audio).lower()
                
                if 'capture' in command:
                    self.capture_photo()
                elif 'show all' in command:
                    self.display_all_photos()
                elif 'show' in command:
                    self.display_photo()
                elif 'exit' in command:
                    print("Exiting the program...")
                    self.running = False  # Set the flag to False to exit
                
            except sr.UnknownValueError:
                print("Could not understand audio")
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
            except Exception as e:
                print(f"Error: {e}")

    def display_all_photos(self):
        """Display all captured photos"""
        # Get all jpg files in the current directory that start with 'captured_photo_'
        photo_files = glob.glob("captured_photo_*.jpg")
        
        if not photo_files:
            print("No captured photos found!")
            return
        
        print(f"Found {len(photo_files)} photos")
        
        for photo_file in photo_files:
            img = cv2.imread(photo_file)
            if img is not None:
                cv2.imshow(photo_file, img)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ... (keep other existing methods unchanged)

    def run(self):
        """Main loop to run the face recognition system"""
        try:
            while self.running:  # Use the flag in the main loop
                ret, frame = self.cap.read()
                if ret:
                    # Draw rectangles around detected faces in the live feed
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                    cv2.imshow('Camera Feed', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    self.capture_on_command()
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
