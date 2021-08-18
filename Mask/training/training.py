import os
import sys
import random
import numpy as np
import scipy
from scipy import ndimage
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from PIL import Image

def DataSet():
    train_path_masked = '../input_imgs/train/masked/'
    train_path_unmasked = '../input_imgs/train/unmasked/'
    test_path_masked = '../input_imgs/test_/masked/'
    test_path_unmasked = '../input_imgs/test_/unmasked/'

    imglist_train_masked = os.listdir(train_path_masked)
    imglist_train_unmasked = os.listdir(train_path_unmasked)
    imglist_test_masked = os.listdir(test_path_masked)
    imglist_test_unmasked = os.listdir(test_path_unmasked)

    X_train = np.empty((len(imglist_train_masked) + len(imglist_train_unmasked), 224, 224, 3))
    Y_train = np.empty((len(imglist_train_masked) + len(imglist_train_unmasked), 2))
    X_test = np.empty((len(imglist_test_masked) + len(imglist_test_unmasked), 224, 224, 3))
    Y_test = np.empty((len(imglist_test_masked) + len(imglist_test_unmasked), 2))

    count = 0

    for img_name in imglist_train_masked:
        img_path = train_path_masked + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        X_train[count] = img
        Y_train[count] = np.array((1,0))
        count+=1
    
    for img_name in imglist_train_unmasked:

        img_path = train_path_unmasked + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_train[count] = img
        Y_train[count] = np.array((0,1))
        count+=1

    count = 0
    for img_name in imglist_test_masked:
        img_path = test_path_masked + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        X_test[count] = img
        Y_test[count] = np.array((1,0))
        count+=1
    
    for img_name in imglist_test_unmasked:

        img_path = test_path_unmasked + img_name
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

def main():
    X_train,Y_train,X_test,Y_test = DataSet()
    print('X_train shape : ',X_train.shape)
    print('Y_train shape : ',Y_train.shape)
    print('X_test shape : ',X_test.shape)
    print('Y_test shape : ',Y_test.shape)

    model = ResNet50(
        weights = None,
        classes = 2
        )

    model.compile(optimizer=tf.optimizers.Adam(0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    print ("begin training")
    model.fit(X_train, Y_train, epochs=1, batch_size=1)
    print ("training finished")
    #print ("begin testing")
    #model.evaluate(X_test, Y_test, batch_size=1)
    #print ("testing finished")

    print ('begin saving model...')
    #model.save('mask_model.h5')
    print ('saving finished!')

    img_path = "../validate/mask_validate.jpg"
    img = image.load_img(img_path, target_size=(224, 224))

    #plt.imshow(img)
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0) 

    print(model.predict(img))
    np.argmax(model.predict(img))

    img_path = "../validate/unmask_validate.jpg"
    img = image.load_img(img_path, target_size=(224, 224))

    #plt.imshow(img)
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    print(model.predict(img))
    np.argmax(model.predict(img))

if __name__=="__main__":
    main()
