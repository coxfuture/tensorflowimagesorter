import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import sys
import shutil

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
from glob import glob
from tqdm import tqdm

class_names = glob("./Dataset/*")
cnames = []  
for i in class_names:
    name = i.split('\\')[1]
    cnames.append(name)
name_id_map = dict(zip(cnames, range(len(class_names))))
#print("done loading classes:",name_id_map)

model = keras.models.load_model('./Model/1/')

img_height = 256
img_width = 256
test_directory = os.path.abspath('./Data')
dest_directory = os.path.abspath('./Sorted')

for root, dirs, files in os.walk(test_directory):
  for test_image in tqdm(files,desc = "Sorting:"):

    fullpath = os.path.join(root,test_image)

    img = keras.preprocessing.image.load_img(fullpath, target_size=(img_height, img_width))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    dest_folder = class_names[np.argmax(score)].split('\\')[-1]
    destpath = os.path.join(dest_directory,dest_folder,test_image)
    # print(test_image,
    #     "This image most likely belongs to {} with a {:.2f} percent confidence."
    #     .format(class_names[np.argmax(score)], 100 * np.max(score)),
    #     "So I will put it in:",destpath 
    # ) 
    count = 1
    while os.path.exists(destpath):
      destpath = destpath
      destpath = destpath.split('.')[0] + ' ('+str(count)+').' +destpath.split('.')[1]
      
      count += 1
    
    shutil.copy(fullpath,destpath)
