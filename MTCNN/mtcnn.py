# Implementation of MTCNN
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN
import time

# The image path is read and displayed from above part
def draw_BB(image, faces):
    start = time.time()
    plt.imshow(image)
    ax = plt.gca()
    for face in faces:
        x, y, width, height = face['box']
        rect = Rectangle((x, y), width, height, fill=False, color='red')
        ax.add_patch(rect)
        for key, value in face['keypoints'].items():
            dot = Circle(value, radius=8, color='red')
            ax.add_patch(dot)
    plt.show()
    return start
    
def sep_faces(image, faces):
    
    for i in range(len(faces)):
        x, y, width, height = faces[i]['box']
        x, y = x, y
        x1, y1 = x+width, y+height
        plt.subplot(1, len(faces), i+1)
        plt.axis('off')
        plt.imshow(image[y:y1, x:x1])
    plt.show()
    end=time.time()
    print("Time: ", end-start)

image = plt.imread("men.jpg")
detector = MTCNN()
faces = detector.detect_faces(image)
start = draw_BB(image,faces)
end = sep_faces(image, faces)

