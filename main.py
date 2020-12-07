# references
#
# https://towardsdatascience.com/building-a-face-recognizer-in-python-7fd6630c6340
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html

import face_recognition
import cv2
import numpy as np
import os
import glob

def initialize_haarcascades():
    global face_cascade
    global eye_cascade
    
    cv2_path = cv2.data.haarcascades
    face_cascade = cv2.CascadeClassifier(cv2_path + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2_path + 'haarcascade_eye.xml')
    
def initialize_detector():
    global face_exempt_encodings
    global face_exempt_names

    face_exempt_encodings = []
    face_exempt_names = []
    cur_direc = os.getcwd()
    path = os.path.join(cur_direc, 'data/faces/')
    list_of_files = [f for f in glob.glob(path+'*.jpg')]
    number_files = len(list_of_files)
    names = list_of_files.copy()
    
    for i in range(number_files):
        globals()['image_{}'.format(i)] = face_recognition.load_image_file(list_of_files[i])
        globals()['image_encoding_{}'.format(i)] = face_recognition.face_encodings(globals()['image_{}'.format(i)])[0]
        face_exempt_encodings.append(globals()['image_encoding_{}'.format(i)])
        # Create array of known names
        names[i] = names[i].replace(cur_direc, "")  
        face_exempt_names.append(names[i])

def detect_face(frame):
    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=2)

    return face_locations
    
def recognize_face(face_locations, frame):
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces (face_exempt_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(face_exempt_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = faces_names[best_match_index]
        face_names.append(name)
            
    return face_names

def detect_eyes(img, gray, faces):
    for (top, right, bottom, left) in faces:
        roi_gray = gray[4*top:4*bottom, 4*left:4*right]
        roi_color = img[4*top:4*bottom, 4*left:4*right]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            
    return img

def recognize_video(video = 0, speed = 4):
    # video : path to video file, 0 to use webcam
    # speed : resize factor to speed up recognition.
    video_capture = cv2.VideoCapture(video)
    
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    
    while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small_frame = cv2.resize(frame, (0, 0), fx=1/speed, fy=1/speed)
            
        if process_this_frame:
            face_locations = detect_face(small_frame)
            # face_names = recognize_face(face_locations, small_frame)
            
        for (top, right, bottom, left) in face_locations:
            top *= speed
            right *= speed
            bottom *= speed
            left *= speed
            
            region = frame[top:bottom, left:right]
            kernel_size = 100 if (right - left) > 100 and (bottom - top) > 100 else min((right - left), (bottom - top)) 
            frame[top:bottom, left:right] = cv2.GaussianBlur(region, (51, 51), 0)
            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        
        # img = detect_eyes(img, gray, faces)
        cv2.imshow('frame', frame)
        process_this_frame = not process_this_frame

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    
if __name__ == "__main__":
    # initialize_haarcascades()
    initialize_detector()
    
    cur_direc = os.getcwd()
    path = os.path.join(cur_direc, 'videos/')
    list_of_videos = [f for f in glob.glob(path+'*.mp4')]
    
    if len(list_of_videos) == 0:
        recognize_video()
    else:
        print("Choose a video. Write anything other than an index to use webcam.")
        for i, v in enumerate(list_of_videos):
            print(i, v)
        choice = input()
        try:
            idx = int(choice)
            if 0 <= idx < len(list_of_videos):
                recognize_video(list_of_videos[idx])
            else:
                recognize_video()
        except (ValueError):
            recognize_video()