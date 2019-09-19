
# Importing required stuff
import cv2
import matplotlib.pyplot as plt

# Getting the Image and converting them to Gray image
imagePath = 'elen_pic.jpg'
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.axis('off')
plt.imshow(gray, cmap = 'gray')
plt.show()

# Using Cascade Classifier and Detecting faces
# This contains the magic of Viola Jones
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.08, minNeighbors=8, minSize=(30,30))
n = len(faces)
if(n<=1):
    print("{0} face is found !".format(n))
else:
    print("{0} faces are found !".format(n))

# Annotating and saving the Detected faces
for (x,y,w,h) in faces:
    cv2.rectangle(image, (x, y), (x+w,y+h), (0,255,0), 2)
status = cv2.imwrite("faces_detected.jpg", image)
print("The status of pic stored: {0}".format(status))   

# Dispalying the Detected faces
imagePath2 = "faces_detected.jpg"
faces_detected = cv2.imread(imagePath2)
faces_detected = cv2.cvtColor(faces_detected, cv2.COLOR_BGR2RGB)
plt.axis("off")
plt.imshow(faces_detected)


