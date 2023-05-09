# coding: utf-8

# In[ ]:
import os

# import tensorflow as tf

# import keras
# from keras.engine.saving import load_model
# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
# from keras.layers import Dense, Activation, Dropout, Flatten
#
# from keras.preprocessing import image
# from keras.preprocessing.image import ImageDataGenerator

import numpy as np

#------------------------------
# sess = tf.Session()
# keras.backend.set_session(sess)
#------------------------------
#variables
# from src.dbconnection import iud

num_classes =13
batch_size = 40
epochs = 50


# import pywt.data

import  cv2

#------------------------------

# import os, cv2, keras
# import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten
# from keras.layers import Conv2D, MaxPooling2D
# from keras.engine.saving import load_model
# manipulate with numpy,load with panda
import numpy as np
# import pandas as pd

# data visualization
import cv2
# import matplotlib
# import matplotlib.pyplot as plt
# import seaborn as sns

# get_ipython().run_line_magic('matplotlib', 'inline')

# my_list = anolist=['Abuse', 'Arrest','Arson','Assault','Burglary','Explosion','Fightning','Normal Vedio', 'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting', 'Stealing', 'Vandalism']
# newlist=['Assault','Burglary','Fightning','Normal Vedio', 'Robbery', 'Shoplifting', 'Vandalism']
# # Data Import
# def read_dataset(path):
#     data_list = []
#     label_list = []
#     my_list = os.listdir(r'C:\Users\User\Downloads\Train')
#     i=-1
#     ii=0
#     for pa in my_list:
#
#         print(pa,"==================")
#         i=i+1
#         for root, dirs, files in os.walk(r'C:\Users\User\Downloads\Train\\' + pa):
#
#          for f in files:
#
#
#              file_path = os.path.join(r'C:\Users\User\Downloads\Train\\'+pa, f)
#
#
#              img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
#              res = cv2.resize(img, (48, 48), interpolation=cv2.INTER_CUBIC)
#              data_list.append(res)
#              # label = dirPath.split('/')[-1]
#              label = i
#              label_list.append(label)
#              # label_list.remove("./training")
#              ii+=1
#              if ii>=1000:
#                 ii=0
#                 break
#     return (np.asarray(data_list, dtype=np.float32), np.asarray(label_list))
#
# def read_dataset1(path):
#     data_list = []
#     label_list = []
#
#     file_path = os.path.join(path)
#     img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
#     res = cv2.resize(img, (48, 48), interpolation=cv2.INTER_CUBIC)
#     data_list.append(res)
#     # label = dirPath.split('/')[-1]
#
#             # label_list.remove("./training")
#     return (np.asarray(data_list, dtype=np.float32))

# from sklearn.model_selection import train_test_split
# # load dataset
# x_dataset, y_dataset = read_dataset(r"C:\Users\User\OneDrive\Desktop\Test")
# X_train, X_test, y_train, y_test = train_test_split(x_dataset, y_dataset, test_size=0.2, random_state=0)
#
# y_train1=[]
# for i in y_train:
#     emotion = keras.utils.to_categorical(i, num_classes)
#     print(i,emotion)
#     y_train1.append(emotion)
#
# y_train=y_train1
# x_train = np.array(X_train, 'float32')
# y_train = np.array(y_train, 'float32')
# x_test = np.array(X_test, 'float32')
# y_test = np.array(y_test, 'float32')
#
# x_train /= 255  # normalize inputs between [0, 1]
# x_test /= 255
# print("x_train.shape",x_train.shape)
# x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
# x_train = x_train.astype('float32')
# x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
# x_test = x_test.astype('float32')
#
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')
# # ------------------------------
# # construct CNN structure
#
# model = Sequential()
#
# # 1st convolution layer
# model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48, 48, 1)))
# model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))
#
# # 2nd convolution layer
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
#
# # 3rd convolution layer
# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
#
# model.add(Flatten())
#
# # fully connected neural networks
# model.add(Dense(1024, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(1024, activation='relu'))
# model.add(Dropout(0.2))
#
# model.add(Dense(num_classes, activation='softmax'))
# # ------------------------------
# # batch process
#
# print(x_train.shape)
#
# gen = ImageDataGenerator()
# train_generator = gen.flow(x_train, y_train, batch_size=batch_size)
#
# # ------------------------------
#
# model.compile(loss='categorical_crossentropy'
#               , optimizer=keras.optimizers.Adam()
#               , metrics=['accuracy']
#               )
#
# # ------------------------------
#
# if not os.path.exists("model1.h5"):
#
#     model.fit_generator(train_generator, steps_per_epoch=batch_size, epochs=epochs)
#     model.save("model1.h5")  # train for randomly selected one
# else:
#     model = load_model("model1.h5")  # load weights
# from sklearn.metrics import confusion_matrix
# yp=model.predict_classes(x_test,verbose=0)
# cf=confusion_matrix(y_test,yp)
# print(cf)
#
# from sklearn.metrics import classification_report
# target_names = ['COVID', 'Fibrosis','Normal',"Viral Pneumonia"]
#
#
# print(classification_report(y_test, yp, target_names=target_names))


# def predict(fn):
#     dataset=read_dataset1(fn)
#     (mnist_row, mnist_col, mnist_color) = 48, 48, 1
#
#     dataset = dataset.reshape(dataset.shape[0], mnist_row, mnist_col, mnist_color)
#     dataset = dataset / 255
#
#     mo = load_model("model1.h5")
#
#
#     # predict probabilities for test set
#
#     yhat_classes = mo.predict_classes(dataset, verbose=0)
#     return yhat_classes[0]
#
#     print(yhat_classes)

#predict(r"D:\vehicle classification with deep learning\src\semi-semitrailer-truck-tractor-highway.jpg")

# importing libraries
import cv2
import numpy as np

# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video file")

# Read until video is completed
i=0
while (cap.isOpened()):

    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        i=i+1

        # Display the resulting frame
        cv2.imwrite("sample.png",frame)
        if i==30:
            i=0
            # res=predict("sample.png")
            # # print(res)
            # if my_list[res] in newlist:
            #     iud("INSERT INTO anomaly VALUES(NULL,%s,%s,curdate(),curdate(),%s)",("1",my_list[res],'pending'))
            #     print(my_list[res])
        cv2.imshow('Frame', frame)

        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

            # Break the loop
    else:
        break

# When everything done, release
# the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
