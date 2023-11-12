# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 17:15:22 2019
@author: Gowtham S
"""
'''--------- RUN ALL CODE TO GET THE EMBEDDINGS AND THEN IT WILL GIVE YOU AN EMBEDDING AND THEN COMPARED --------- '''
''' --------------------- DATASET ENCODING ----------------------- '''
''' This code creates a 128-d vector(embeddings) for the faces in our dataset which we will use to compare and give the output '''
import pickle
import cv2
import imutils, face_recognition
import os, glob
# All names
names=[]

# Getting the file directories
file_list = glob.glob(os.path.join(os.getcwd(), "lfw\*"))
for f in file_list:
    s = f.split(os.path.sep)[-1]
    names.append(s)

imagePath = []

for n in names:
    each_imagePath_list = glob.glob(os.path.join(os.getcwd(), "lfw\\"+n+"\*"))
    for ep in each_imagePath_list:    
        imagePath.append(ep)
    each_imagePath_list = []

known_encodings = []
known_faces = []
for i,im in enumerate(imagePath):
    print("Processes Image {}/{}".format(i+1, len(imagePath)))
    name = im.split(os.path.sep)[-2]
    image = cv2.imread(im)    
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, boxes)
    for encoding in encodings:
        known_encodings.append(encoding)
        known_faces.append(name)

# Dumping all using pickle so to use later
data = {"encodings":known_encodings,"names":known_faces}

'''------------------------- RECOGNIZING PART -------------------------- '''
''' This code is used to get the test image and Perform recognition '''
import cv2
import imutils, face_recognition
import os, glob


# Loading the input image and convert it from BGR to RGB
test_image = cv2.imread("test_image.png")
test_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

test_boxes = face_recognition.face_locations(test_rgb)
print("Performed Encoding.......")
test_encodings = face_recognition.face_encodings(test_rgb, test_boxes)

# initialize the list of names for each face detected
test_names = []

for encoding in test_encodings:
    matches = face_recognition.compare_faces(data["encodings"], encoding)
    name1 = "Unknown"
    if True in matches:
        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
        counts = {}
        for i in matchedIdxs:
            name1 = data["names"][i]
            counts[name1] = counts.get(name1, 0) + 1
        
        name1 = max(counts, key=counts.get)
    test_names.append(name1)

for ((top, right, bottom, left), name2) in zip(test_boxes, test_names):
    cv2.rectangle(test_image, (left, top), (right, bottom), (0, 255, 0), 2)
    y = top - 15 if top - 15 > 15 else top + 15
    cv2.putText(test_image, name2, (left-5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0,0,255), 2)
cv2.imwrite("result.png", test_image)
cv2.imshow("Image", test_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

""" The result I obtained is saved as result.png, you can run this above code(only) and you will get the same result """
