import cv2
#Load our model
from keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('BrainTumor10Epochs.h5')

#Load the images
image = cv2.imread('C:\\Users\\Pelop\\Desktop\\Brain-Tumor-Detection-Deep-Learning\\pred\\pred0.jpg')

# Covert image to array format
img= Image.fromarray(image)
img = img.resize((64,64))
img=np.array(img)

#Expand dimension of images
input_img=np.expand_dims(img,axis=0)

# Predict images based on model
result_x = model.predict(input_img)
classes_x = np.argmax(result_x,axis=1)
print(result_x)
