# AI---Face-Recognition

Face recognition is one of the challenging problem in the Computer Vision industry. Many algorithms have been developed to address the issue of facial recognition during the last thirty years. Algorithms based on LDA, PCA, ICA and Artificial Neural Networks have been used to try to address the issue of face recognition. Facial recognition algorithms are affected by illumination thus variation in lighting as well as pose variation. As a result, hybrid methods have been developed which uses a combination of two algorithms. Face Recognition has been greatly used to develop security systems as well as surveillance systems to keep track fraud and criminal activities. In this paper,the researcher used the LBPH (Local Binary Patterns Histograms) algorithm to produce a prototype of a system that will find missing people using facial recognition. The major objective of the research is to determine the accuracy of the system as well as the recognition rate.
 The software requirements for this project is opencv and Pycharm software.
 

Keywords:

Facial Recognition, Local Binary Patterns Histograms, Artificial Neural Networks, Biometrics, Face Identification ,Opencv , Pycharm



Introduction 
Introduce the background of the project in this section

As each day passes by, more people are reported missing, some maybe hiding from serious crimes, abducted or even running away from their families’ due to some social problems. Any case of a missing person is reported to the police and all the details of the case are taken. Children who are abused by their parents tend to live in the streets and they are reported missing. Media can be used to find missing people for instance the use of newspapers. Media appeals may be the quickest and most effective way of raising awareness of your missing person and helping in the continuing search for him or her. Nevertheless, not everyone feels comfortable using the media. Different newspapers and magazines have different interviewing techniques and styles. Whilst many journalists will be sympathetic, others may appear forceful, cold or aggressive or behave in other ways, which seem insensitive to what you are going through. Some people do not trust the media or want their circumstances made public; others feel overwhelmed by the thought of dealing with journalists and being asked probing and personal questions about their missing friend or relatives.Additionally, publicity may put already vulnerable people at greater risk by forcing them further away if they do not wish to be found. Kidnappers can continue to victimize their victims, as they will be aware through media.However, the use of facial recognition technique makes it easier for us to find the missing people and this will cater for all the disadvantages of using media. Searching for a missing person using media resulted in many problems for instance publicity may put already vulnerable people at greater risk by forcing them further away if they do not wish to be found.The news of missing person is advertised on television and newspapers for a certain period. After a few number of days everyone would have forgot that news since the advertisement is not continued for long.The police find all the possibilities of finding the missing people by using posters and announcing on the media but as a result, they do not have real time solution, which means they cannot track down the missing person if the person to be found is not staying at one position.To avoid this, we use facial recognition whereby surveillance cameras are installed at convenient places to track people moving via the live video feed. This will be different from searching the whole nation for the missing person; instead, we can narrow down our search to a specific area based on the results produced by the system.




 FACE RECOGNIZATION  

DIFFERENT APPROACHES OF FACE RECOGNITION: 
There are two predominant approaches to the face recognition problem: Geometric 
(feature based) and photometric (view based). As researcher interest in face recognition 
continued, many different algorithms were developed, three of which have been well studied 
in face recognition literature. 

Recognition algorithms can be divided into two main approaches: 

Geometric:
Is based on geometrical relationship between facial landmarks, or in 
other words the spatial configuration of facial features. That means that the main 
geometrical features of the face such as the eyes, nose and mouth are first located and then 
faces are classified on the basis of various geometrical distances and angles between 
features.





Photometric stereo: 
Used to recover the shape of an object from a number of 
images taken under different lighting conditions. The shape of the recovered object is 
defined by a gradient map, which is made up of an array of surface normals, 



,






CODING PART

import face_recognition
import cv2
import numpy as np

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
pranto_image = face_recognition.load_image_file("pranto.jpg")
pranto_face_encoding = face_recognition.face_encodings(pranto_image)[0]

# Load a second sample picture and learn how to recognize it.
rakibul__image = face_recognition.load_image_file("rakibul_hafij.jpg")
rakibul_face_encoding=face_recognition.face_encodings(rakibul_hafij_image)[0]

# Load a 3th sample picture and learn how to recognize it.
imtiazz_image = face_recognition.load_image_file("imtiazz.png")
imtiazz_face_encoding=face_recognition.face_encodings(imtiazz_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    pranto_face_encoding,
    rakibul_hafij_face_encoding,
    imtiazz_face_encoding
]
known_face_names = [
    "pranto",
    "rakibul_hafij" ,
    "imtiazz"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'p' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('p'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

A short description about the code :

face_recognition.api.batch_face_locations(images, number_of_times_to_upsample=1, 
batch_size=128) 

Returns an 2d array of bounding boxes of human faces in a image using the cnn face detector If you are using a 
GPU, this can give you much faster results since the GPU can process batches of images at once. If you aren’t using a GPU, you don’t need this function. 


Parameters 
• images – A list of images (each as a numpy array) 

• number_of_times_to_upsample – How many times to upsample the image looking 
for faces. Higher numbers fifind smaller faces. 

• batch_size – How many images to include in each GPU processing batch. 
Returns A list of tuples of found face locations in css (top, right, bottom, left) order 
face_recognition.api.compare_faces(known_face_encodings, face_encoding_to_check, toler
ance=0.6) Compare a list of face encodings against a candidate encoding to see if they match. 
Parameters 

• known_face_encodings – A list of known face encodings 

• face_encoding_to_check – A single face encoding to compare against the list 

• tolerance – How much distance between faces to consider it a match. Lower is more 
strict. 0.6 is typical best performance. Returns A list of True/False values indicating which known_face_encodings match the face encod
ing to check Face Recognition Documentation, Release 1.2.3 face_recognition.api.face_distance(face_encodings, face_to_compare) 
Given a list of face encodings, compare them to a known face encoding and get a euclidean distance for each 
comparison face. The distance tells you how similar the faces are. Parameters 

• faces – List of face encodings to compare 

• face_to_compare – A face encoding to compare against 

Returns A numpy ndarray with the distance for each face in the same order as the ‘faces’ array 
face_recognition.api.face_encodings(face_image, known_face_locations=None, 
num_jitters=1) Given an image, return the 128-dimension face encoding for each face in the image. 
Parameters 

• face_image – The image that contains one or more faces 

• known_face_locations – Optional - the bounding boxes of each face if you already 
know them. 

• num_jitters – How many times to re-sample the face when calculating encoding. 
Higher is more accurate, but slower (i.e. 100 is 100x slower) 
Returns A list of 128-dimensional face encodings (one for each face in the image) 
face_recognition.api.face_landmarks(face_image, face_locations=None, model=’large’) 
Given an image, returns a dict of face feature locations (eyes, nose, etc) for each face in the image 
Parameters 

• face_image – image to search 

• face_locations – Optionally provide a list of face locations to check.

• model – Optional - which model to use. “large” (default) or “small” which only returns 5 
points but is faster. 

Returns A list of dicts of face feature locations (eyes, nose, etc) 
face_recognition.api.face_locations(img, number_of_times_to_upsample=1, model=’hog’) 
Returns an array of bounding boxes of human faces in a image 
Parameters 
• img – An image (as a numpy array) 
• number_of_times_to_upsample – How many times to upsample the image looking 
for faces. Higher numbers fifind smaller faces. 
• model – Which face detection model to use. “hog” is less accurate but faster on CPUs. 
“cnn” is a more accurate deep-learning model which is GPU/CUDA accelerated (if avail
able). The default is “hog”. 
Returns A list of tuples of found face locations in css (top, right, bottom, left) order 
face_recognition.api.load_image_file(fifile, mode=’RGB’) 
Loads an image fifile (.jpg, .png, etc) into a numpy array 
Parameters 
• file – image fifile name or fifile object to load 
• mode – format to convert the image to. Only ‘RGB’ (8-bit RGB, 3 channels) and ‘L’ (black 
and white) are supported. 
Returns image contents as numpy array

