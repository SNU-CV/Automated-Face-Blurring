import face_recognition
import cv2
import numpy as np
import os
import glob

face_cascade = None
eye_cascade = None

def initialize_detector():
    global face_cascade
    global eye_cascade
    
    cv2_path = cv2.data.haarcascades
    face_cascade = cv2.CascadeClassifier(cv2_path + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2_path + 'haarcascade_eye.xml')

# https://towardsdatascience.com/building-a-face-recognizer-in-python-7fd6630c6340
def detect_face(frame):
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
    # Display the results
    for (top, right, bottom, left) in face_locations:
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    return frame, face_locations
    

# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html
def detect_eyes(img, gray, faces):
    for (top, right, bottom, left) in faces:
        roi_gray = gray[4*top:4*bottom, 4*left:4*right]
        roi_color = img[4*top:4*bottom, 4*left:4*right]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            
    return img

def recognize_video():
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        img, faces = detect_face(frame)
        img = detect_eyes(img, gray, faces)
        cv2.imshow('frame', img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    
if __name__ == "__main__":
    initialize_detector()
    recognize_video()
    
