import cv2
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Activation, Dense, Conv2D, MaxPooling2D, ZeroPadding2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import time 
import os
from tqdm import tqdm
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

GENERATE_NAME = "Mach-Learning-{}".format(int(time.time()))
callbacks = TensorBoard(log_dir='./Graph/{}'.format(GENERATE_NAME))

dirTrain = 'TrainingZF/'
dirValidation = 'ValidatingZF/'
dirTrainZ = 'TrainingZF/Close/'
dirTrainF = 'TrainingZF/Open/'
dirValidationZ = 'ValidatingZF/Close/'
dirValidationF = 'ValidatingZF/Open/'
training = 900*2
validating = 100*2
batch_size = 32
img_size = 224
epoch = 30

def preprocessing(img_source):
    #img_source = str(img_source)
    img = cv2.imread(img_source)

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
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
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


'''cv2.imshow('This is image',preprocessing('Resizeme.jpg'))
cv2.waitKey(0)
cv2.destroyAllWindows()'''

def layerprocess(size,dimension):
    convnet = input_data(shape=[None, size, size, dimension], name='input')

    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')
    return model

#Import image
load_Z_img_train = os.listdir(dirTrainZ)
load_F_img_train = os.listdir(dirTrainF)

load_Z_img_val = os.listdir(dirValidationZ)
load_F_img_val = os.listdir(dirValidationF)
'''
for img in load_Z_img_train:
        #print(str(load_Z_img_train)+str(img))
        print(str(dirTrainZ)+str(img))

for img in tqdm(load_Z_img_train):
        #print(str(load_Z_img_train+img))
        img_array = preprocessing(str(dirTrainZ)+str(load_Z_img_train)+str(img))
        cv2.imshow('This is image',img_array)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
'''

def create_train_data():
    training_data = []
    for img in tqdm(load_Z_img_train):
        #print(str(load_Z_img_train+img))
        img_array = preprocessing(str(dirTrainZ)+str(img))
        label = [1,0]
        training_data.append([np.array(img_array) , np.array(label)])

    for img in tqdm(load_F_img_train):
        img_array = preprocessing(str(dirTrainF)+str(img))
        label = [0,1]
        training_data.append([np.array(img_array) , np.array(label)])

    np.save('training_data.npy',training_data)
    return training_data

def create_val_data():
    validation_data = []
    for img in tqdm(load_Z_img_val):
        #print(str(load_Z_img_val+img))
        img_array = preprocessing(str(dirValidationZ)+str(img))
        label = [1,0]
        validation_data.append([np.array(img_array) , np.array(label)])

    for img in tqdm(load_F_img_val):
        img_array = preprocessing(str(dirValidationF)+str(img))
        label = [0,1]
        validation_data.append([np.array(img_array) , np.array(label)])
    np.save('val_data.npy', validation_data)
    return validation_data

train = create_train_data()
val = create_val_data()

#If you already have database
#train = np.load('training_data.npy')
#val = np.load('val_data.npy')

X = np.array([i[0] for i in train]).reshape(-1,img_size,img_size,1)
Y = [i[1] for i in train]

val_x = np.array([i[0] for i in val]).reshape(-1,img_size,img_size,1)
val_y = [i[1] for i in val]

model = layerprocess(img_size,1)

adam = Adam(lr=0.0001)
#model.compile(
#    optimizer=adam, 
#    loss='categorical_crossentropy', 
#    metrics=['accuracy'])

model.fit({'input': X}, {'targets': Y}, n_epoch=epoch, 
    validation_set=({'input': val_x}, {'targets': val_y}), 
    snapshot_step=500, show_metric=True, run_id=GENERATE_NAME)
#model.fit({'input': X}, {'targets': Y}, 
#    n_epoch=3, 
#    validation_set=({'input': test_x}, {'targets': test_y}), 
#    snapshot_step=500, 
#    show_metric=True, 
#    callbacks=[callbacks])

#model.save_weights('SignZeroFive.h5')
model.save('SignZeroFive.model')
#model.make_model("SignZeroFive.json")
