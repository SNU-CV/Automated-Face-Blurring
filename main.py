# references
#
# https://towardsdatascience.com/building-a-face-recognizer-in-python-7fd6630c6340

import face_recognition
import cv2
import numpy as np
import os
import glob
    
def initialize_detector():
    global face_exempt_encodings
    global face_exempt_names

    face_exempt_encodings = []
    face_exempt_names = []
    cur_direc = os.getcwd()
    path = os.path.join(cur_direc, 'photos/')
    list_of_files = [f for f in glob.glob(path+'*.jpg')]
    number_files = len(list_of_files)
    names = list_of_files.copy()
    
    for i in range(number_files):
        image = face_recognition.load_image_file(list_of_files[i])
        image_encoding = face_recognition.face_encodings(image)
        names[i] = names[i].replace(cur_direc, "")

        if len(image_encoding) == 0:
            print("Face is not found in given image: " + names[i])
            continue
        elif len(image_encoding) > 1:
            print("More than one face is found in given image: " + names[i])
            continue 

        face_exempt_encodings.append(image_encoding[0]) 
        face_exempt_names.append(names[i])

def detect_face(frame):
    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=2)

    return face_locations
    
def match_face(face_encodings, frame, temporal_encodings):
    are_matched = []

    if len(face_exempt_encodings) == 0:
        return [False] * len(face_encodings)

    for face_encoding in face_encodings:
        is_matched = False
        exempt_distances = face_recognition.face_distance(face_exempt_encodings, face_encoding)
        temporal_distances = face_recognition.face_distance(temporal_encodings, face_encoding)
        
        if len(exempt_distances) > 0 and np.amin(exempt_distances) < 0.3:
            is_matched = True

        if len(temporal_distances) > 0 and np.amin(temporal_distances) < 0.2:
            is_matched = True

        are_matched.append(is_matched)
            
    return are_matched

def recognize_video(video = 0, resize_factor = 4):
    # video : path to video file, 0 to use webcam
    # resize_factor : resize factor to speed up recognition.
    video_capture = cv2.VideoCapture(video)
    
    face_locations = []
    face_encodings = []
    face_matches = []
    process_this_frame = True
    temporal_encodings = []
    
    while True:
        ret, frame = video_capture.read()
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small_frame = cv2.resize(frame, (0, 0), fx=1/resize_factor, fy=1/resize_factor)
            
        if process_this_frame:
            face_locations = detect_face(small_frame)
            if len(face_locations) == 0:
                face_encodings = []
                face_matches = []
                temporal_encodings = []
            else:
                face_encodings = face_recognition.face_encodings(small_frame, face_locations)
                face_matches = match_face(face_encodings, small_frame, temporal_encodings)
                temporal_encodings = np.array(face_encodings)[np.array(face_matches)].tolist()
            
        for i, (top, right, bottom, left) in enumerate(face_locations):
            top *= resize_factor
            right *= resize_factor
            bottom *= resize_factor
            left *= resize_factor
            
            # apply gaussian blur
            if (not face_matches[i]):
                region = frame[top:bottom, left:right]
                kernel_size = 100 if (right - left) > 100 and (bottom - top) > 100 else min((right - left), (bottom - top)) 
                frame[top:bottom, left:right] = cv2.GaussianBlur(region, (51, 51), 0)
            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        
        cv2.imshow('frame', frame)
        process_this_frame = not process_this_frame

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    
if __name__ == "__main__":
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