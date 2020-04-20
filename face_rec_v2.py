import face_recognition
import picamera
import numpy as np
import os
import time
from servo_control import Servo

# Get a reference to the Raspberry Pi camera.
# If this fails, make sure you have a camera connected to the RPi and that you
# enabled your camera in raspi-config and rebooted first.

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
                name = faces_list[indx]
                open_close_door(SERVO_PIN)
                open_door = True #Code to open door goes here
                print("I see someone named {}!".format(name))
            
