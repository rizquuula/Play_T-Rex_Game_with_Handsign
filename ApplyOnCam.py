import tensorflow as tf 
import os 
import cv2
import numpy as np 
import time 
from PIL import Image
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import pyautogui

cam = cv2.VideoCapture(0)
cam.set(3,200)
cam.set(4,200)
#cam.set(cv2.CAP_PROP_FPS, 3)
img_size = 224
boolean = 1
KEY = 0

convnet = input_data(shape=[None, img_size, img_size, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')
model.load('/home/linkgish/Desktop/MachLearn5/SignZeroFive.model')

def preprocessing(img_source):
    #img_source = str(img_source)
    #img = cv2.imread(img_source)
    img = img_source
    old_size = img.shape[:2]       #Original size
    #print(old_size)     
    # => (288, 352)
    ratio = float(img_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])      #Changed to the new size in same ratio
    #print(ratio,' and ',new_size)      
    # => 0.6363636363636364  and  (183, 224)#
    img = cv2.resize(img, (new_size[1], new_size[0]))

    delta_h = img_size - new_size[0]
    delta_w = img_size - new_size[1]
    #print(delta_w,' and ',delta_h)
    # => 0 and 41
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    #print (top,bottom,left,right)
    # => 20 21 0 0                  // is for integer divide

    #color = [255, 255, 255]
    color = [0, 0, 0]

    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    #print(new_img.shape)
    #print(new_img)
    #new_img.reshape(-1,img_size,img_size,1)
    #new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    #print(new_img.shape)
    '''a = np.array([[1,2,3], [4,5,6], [7,8,9]])
    print(a)
    a.reshape(-1,2,2,2)
    print(a)'''
    '''
    cv2.imshow('This is image',new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''
    return new_img

while boolean:
    if KEY==1:
        print('pressed')
        #pyautogui.keyDown('up')
        pyautogui.press('up')
    else:
        pass
        
    ret, frame = cam.read()
    min_color = np.array([0, 10, 10], dtype = "uint8")
    max_color = np.array([35, 255, 255], dtype = "uint8")
    HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(HSV, min_color, max_color)
    mask224 = preprocessing(mask)
    newmask = np.reshape(mask224, (-1,224,224,1))
    #newmask = np.expand_dims(mask224, axis=-1)
    #binary = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    prediction = model.predict(newmask)
    #print(prediction)

    cv2.namedWindow("img 0")
    cv2.imshow('img 0', frame)
    cv2.namedWindow("img 1")
    cv2.imshow('img 1', HSV)
    cv2.namedWindow("img 2")
    cv2.imshow('img 2', mask)
    
    if prediction[0][0] < prediction[0][1]:
        print('Five ',(prediction[0][1]/prediction[0][0])*100, ' %')
        KEY = 0
    elif prediction[0][0] >= prediction[0][1]:
        print('Zero', (prediction[0][0]/prediction[0][1])*100, ' %')
        KEY = 1
        #pyautogui.keyDown('up')
        #pyautogui.keyUp('up')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        boolean = 0

cam.release()
cv2.destroyAllWindows()