import face_recognition
import picamera
import numpy as np
import os
import time
from datetime import datetime
from datetime import date
from servo_control import Servo

"""
Created by Ethan Lyon for ELEC574. Rice University Spring 2020
This script uses the RPi's camera and the facial recognition library to
recognize the faces of people in the 'data/whitelist/' folder.
Will create two files. A data base of names with the facial embedding features 
and another file of all of the names. Will take pictures every two seconds about
and compare the faces it finds with every known face to see if there's a match.

When a match is found, the servo is actuated via the original servo_control.py class.
"""

def open_close_door(servo, setup_new_servo = True):
    if(setup_new_servo):
        ser = Servo(pin = SERVO_PIN, angle = 0)
        ser.setup()
        ser.rotate_and_lock(180, wait_time = 0.7)
        time.sleep(1.5)
        ser.rotate_and_lock(0, wait_time = 0.7)
        ser.stop()
    else:
        servo.rotate_and_lock(180, wait_time = 0.7)
        time.sleep(1.5)
        servo.rotate_and_lock(0, wait_time = 0.7)


#Test the servo's capabilities
print("Setting up Servo..")
SERVO_PIN = 17
serv = Servo(pin = SERVO_PIN, angle = 0)
serv.setup()
serv.rotate_and_lock(180, wait_time = 0.5)
serv.rotate_and_lock(90, wait_time = 0.5)
serv.rotate_and_lock(0, wait_time = 0.5)
serv.stop()

open_door = False;

test_dir = './data/whitelist/'

camera = picamera.PiCamera()
camera.resolution = (320, 240)
output = np.empty((240, 320, 3), dtype=np.uint8)

# Load a sample picture and learn how to recognize it.
print("Loading encodings...")

if(not os.path.exists('recognized_ringers')):
    os.makedirs('recognized_ringers')
if(not os.path.exists('unknown_ringers')):
    os.makedirs('unknown_ringers')
if(os.path.exists('face_db.npy')):
    face_db = np.load('face_db.npy', allow_pickle = True).item()
    print("DB Loaded")
else:
    face_db = {}

faces_list = os.listdir(test_dir)
old_faces_list = []

if(not os.path.exists('names.npy')):
    np.save('names', faces_list)
else:
    old_faces_list = np.load('names.npy')

new_faces = []
changed_data = False
for f in faces_list:
    if(not (f in old_faces_list)):
        new_faces.append(f)
        changed_data = True
        
removed_faces = []
for of in old_faces_list:
    if(not (of in faces_list)):
        removed_faces.append(of)
        changed_data = True
    
print("Found new faces: " + str(new_faces))
print("Removed faces: " + str(removed_faces))

faces_indx = list(range(len(new_faces)))
print("Faces: " + str(faces_list))

for indx in faces_indx:
    face_name = faces_list[indx]
    f_img = face_recognition.load_image_file(test_dir + face_name)
    f_enc = face_recognition.face_encodings(f_img)
    face_db[face_name] = f_enc
    print(str(100*(indx + 1)//(len(faces_list))) + "% Done", end="\r")

for rf in removed_faces:
    face_db.pop(rf)

print("Encodings done")
if(changed_data):
    np.save('face_db', face_db)
    np.save('names', faces_list)

faces_list = list(face_db.keys())
print("68:" + str(faces_list))
faces_indx = list(range(len(faces_list)))
face_db_encodings = (list(face_db.values()))


# Initialize some variables
face_locations = []
face_encodings = []


# Begin automatic capture of image from Pi's camera every 2 seconds.
# Perfoms facial detection and recognition on very photo taken. Will print out
# The name of those it recognizes and actaute the servo if your face is
# contained in the data folder

img_cnt = 0
while True:
    print("Capturing image " + str(img_cnt))
    img_cnt = img_cnt + 1
    # Grab a single frame of video from the RPi camera as a numpy array
    camera.capture(output, format="rgb")

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(output)
    print("Found {} faces in image.".format(len(face_locations)))
    if(len(face_locations) > 1):
        open_door = False
        print("Error: More than one face detected. One at a time, please.")
    
    elif(len(face_locations) == 1):
        camera_encoding = face_recognition.face_encodings(output, face_locations)[0]
        # Loop over each face found in the frame to see if it's someone we know.
        for indx in faces_indx:
            # See if the face is a match for the known face(s)
            face_encoding = face_db_encodings[indx]
            match = face_recognition.compare_faces(camera_encoding, face_encoding)
            name = "<Unknown Person>"
            
            if match[0]:
                name = (faces_list[indx]).split(".")[0]
                print("I see someone named {}".format(name))
                print("Opening door for {}".format(name))
                open_close_door(SERVO_PIN)
                open_door = True 
            # Log the person found in the image and time of button press
            log = open('log.txt','a')
            if(name == "<Unknown Person>"):
                log.write(date.today().strftime("%B %d, %Y") + " - " + datetime.now().strftime("%H:%M:%S") + ": "+ "Unknown Person\n")
            else:
                log.write(date.today().strftime("%B %d, %Y") + " - " + datetime.now().strftime("%H:%M:%S") + ": " + name + "\n")
            log.close()
            
