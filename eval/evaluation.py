# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 12:30:01 2020

@author: ethan
"""

import os
import cv2
import pandas as pd
import numpy as np
import face_recognition
import time
import matplotlib.pyplot as plt

def print_time(exec_time, typeof = "Execution"):
    print( typeof + " time: " + str(exec_time//60) + "m" + str((exec_time % 60)//1.0) + "s")

time0 = time.time()
test_dir = './data/lfw_funneled/'
people_list = os.listdir(test_dir)
N_req = 5
multiple_face_names = []
for p in people_list:
    img_list = os.listdir(test_dir + p)
    if(len(img_list) >= N_req):
        multiple_face_names.append(p)

print("Loading encodings...")

#Load the existing face encoding, name : encoding, dictionary. Initialize to empty if it doesn't exist
if(os.path.exists('face_db.npy')):
    face_db = np.load('face_db.npy', allow_pickle = True).item()
    print("DB Loaded")
else:
    face_db = {}

faces_list = os.listdir(test_dir)
old_faces_list = []

#Load the old names list and save the current names list if it doesn't exist.
if(not os.path.exists('names.npy')):
    np.save('names', faces_list)
else:
    old_faces_list = np.load('names.npy')

#Find and add new people added to the data and find and remove the names that were removed.
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
print("Found removed faces: " + str(removed_faces))

faces_indx = list(range(len(new_faces)))
print("Faces: " + str(faces_list))

prev_done = -1
for indx in faces_indx:
    face_name = faces_list[indx]
    img_list = os.listdir(test_dir + face_name)
    total_images = len(img_list)
    #Only get the first image in the folder. Will modify later to evaluate model using SVM, but for now this is sufficient.
    first_image_in_folder = test_dir + face_name + '/' + os.listdir(test_dir + face_name)[0]
    f_img = face_recognition.load_image_file(first_image_in_folder)
    face_locations = face_recognition.face_locations(f_img)
    N_face_loc = len(face_locations)
    if(N_face_loc == 0):
        print("No face found in image: " + first_image_in_folder)
    elif(N_face_loc == 1):
        pass
    else:
        print("More than one face found in image: " + first_image_in_folder)
    f_enc = face_recognition.face_encodings(f_img)
    face_db[face_name] = [f_enc, N_face_loc, total_images]
    cur_done = 100*(indx + 1)//(len(faces_list))
    if(not (cur_done == prev_done)):
        print(str(100*(indx + 1)//(len(faces_list))) + "% Done ", end="\r")
    prev_done = cur_done

enc1 = time.time()
enc_time = enc1 - time0
print_time(enc_time, "Encoding")

#Remove face encodings of people that were deleted
for rf in removed_faces:
    face_db.pop(rf)

#Save the encodings and names if the data was changed
print("Encodings done")
if(changed_data):
    np.save('face_db', face_db)
    np.save('names', faces_list)


faces_list = list(face_db.keys())
faces_indx = list(range(len(faces_list)))
face_db_encodings = (list(face_db.values()))

et0 = time.time()
evaluation_db = dict(zip(face_db.keys(), []))
eval_indx = 0
print("Evaluating...")
for eval_name in faces_list:
    eval_arr = []
    eval_enc = (face_db[eval_name])[0]
    eval_n_face = (face_db[eval_name])[1]
    for test_name in faces_list:
        test_enc = (face_db[test_name])[0]
        test_n_face = (face_db[test_name])[1]
        if(test_n_face == 0 or eval_n_face == 0):
            eval_arr.append(0)
        elif(test_n_face == 1):
            match = face_recognition.compare_faces(test_enc[0], eval_enc)[0]
            if(match):
                eval_arr.append(1)
            else:
                eval_arr.append(0)
        else:
            eval_arr.append(0)
            #match = face_recognition.compare_faces(test_enc, eval_enc[0])
            
            #if(True in match):
                #eval_arr.append(1)
            #else:
                #eval_arr.append(0)
    evaluation_db[eval_name] = eval_arr
    if(eval_indx % int(.01*len(people_list)) == 0):
        print(str(100*(eval_indx + 1)//(len(people_list))) + "%...", end="\r")
    eval_indx += 1
    
eval_mat = list(evaluation_db.values())
et1 = time.time()
eval_time = et1 - et0
print_time(eval_time, "Evaluation")
row_sum = np.sum(eval_mat, axis = 1)
col_sum = np.sum(eval_mat, axis = 0)
diag = np.diag(eval_mat)
acc = 100*np.sum(diag)/len(row_sum)
false_negative = 100 - acc
false_pos = 100*(np.sum(eval_mat) - np.sum(np.diag(eval_mat)))/(len(row_sum)*len(col_sum))

print("Accuracy: " + str(acc) + "%")
print("False Positive rate: " + str(false_pos) + "%")



exec_time = time.time() - time0
print_time(exec_time, "Total")

