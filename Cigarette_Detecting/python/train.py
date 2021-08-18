#!/usr/bin/env python
# coding: utf-8

import os,sys
import numpy as np
import scipy
from scipy import ndimage
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
#from PIL import Image
from PIL import Image
import random


def DataSet():
    
    train_path_cigarette ='../fig/faces/cigarette/train/'
    train_path_normal = '../fig/faces/normal/train/'
    
    test_path_cigarette ='../fig/faces/cigarette/test/'
    test_path_normal = '../fig/faces/normal/test/'
    
    imglist_train_cigarette = os.listdir(train_path_cigarette)
    imglist_train_normal = os.listdir(train_path_normal)
    
    imglist_test_cigarette = os.listdir(test_path_cigarette)
    imglist_test_normal = os.listdir(test_path_normal)
        
    X_train = np.empty((len(imglist_train_cigarette) + len(imglist_train_normal), 224, 224, 3))
    Y_train = np.empty((len(imglist_train_cigarette) + len(imglist_train_normal), 2))
    count = 0
    for img_name in imglist_train_cigarette:
        
        img_path = train_path_cigarette + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_train[count] = img
        Y_train[count] = np.array((1,0))
        count+=1
        
    for img_name in imglist_train_normal:

        img_path = train_path_normal + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_train[count] = img
        Y_train[count] = np.array((0,1))
        count+=1
        
    X_test = np.empty((len(imglist_test_cigarette) + len(imglist_test_normal), 224, 224, 3))
    Y_test = np.empty((len(imglist_test_cigarette) + len(imglist_test_normal), 2))
    count = 0
    for img_name in imglist_test_cigarette:

        img_path = test_path_cigarette + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_test[count] = img
        Y_test[count] = np.array((1,0))
        count+=1
        
    for img_name in imglist_test_normal:
        
        img_path = test_path_normal + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_test[count] = img
        Y_test[count] = np.array((0,1))
        count+=1
        
    index = [i for i in range(len(X_train))]
    random.shuffle(index)
    X_train = X_train[index]
    Y_train = Y_train[index]
    
    index = [i for i in range(len(X_test))]
    random.shuffle(index)
    X_test = X_test[index]    
    Y_test = Y_test[index]

    return X_train,Y_train,X_test,Y_test

X_train,Y_train,X_test,Y_test = DataSet()
print('X_train shape : ',X_train.shape)
print('Y_train shape : ',Y_train.shape)
print('X_test shape : ',X_test.shape)
print('Y_test shape : ',Y_test.shape)

# model
model = ResNet50(
    weights=None,
    classes=2
)

model.compile(optimizer=tf.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# train
#for i in range(1):
#print ('Index: ', i+1)
print ('Begin training...')
model.fit(X_train, Y_train, epochs=10, batch_size=2)
print ('Train finished!')

#evaluate
print ('Begin Evaluation...')
model.evaluate(X_test, Y_test, batch_size=2)
print ('Evaluation finished!')

# save
print ('Saving model...')
model.save('my_resnet_model.h5')
print ('Model saved!')

# restore
#print ('Loading model...')
#model = tf.keras.models.load_model('my_resnet_model.h5')
#print ('Model loaded!')


# test
img_path = "../fig/final_validate/1.jpg"
img = image.load_img(img_path, target_size=(224, 224))

#plt.imshow(img)
img = image.img_to_array(img)/ 255.0
img = np.expand_dims(img, axis=0)

print ('Image predicting...')
print (model.predict(img))
np.argmax(model.predict(img))

# test
img_path = "../fig/final_validate/2.jpg"
img = image.load_img(img_path, target_size=(224, 224))

#plt.imshow(img)
img = image.img_to_array(img)/ 255.0
img = np.expand_dims(img, axis=0)

print ('Image 2 predicting...')
print (model.predict(img))
np.argmax(model.predict(img))

# test
img_path = "../fig/final_validate/3.jpg"
img = image.load_img(img_path, target_size=(224, 224))

#plt.imshow(img)
img = image.img_to_array(img)/ 255.0
img = np.expand_dims(img, axis=0)

print ('Image 3 predicting...')
print (model.predict(img))
np.argmax(model.predict(img))
