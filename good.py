import os
from PIL import Image
dir='./MiniResized'
folders=os.listdir(dir)
for folder in folders:
    folderdir=dir+'/'+folder
    filelist=(os.listdir(folderdir))
    for file in filelist:
        filedir=folderdir+'/'+file
        im = Image.open(filedir)
        im=im.resize((600,600))
        im.save(filedir)
    print('-> '+folder+" : All Images In Folder Resize Success")
    

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import  img_to_array
from keras import backend as K
from sklearn.model_selection import train_test_split
#5926
import numpy as np
import os
from PIL import Image
print("OK")

file='MiniResized'
m=n=96

classes=os.listdir("./"+file)
x=[]
y=[]
for classname in classes:
    print(classname)
    imagefiles=os.listdir("./"+file+"/"+classname+'/')
    for img in imagefiles:
        im=Image.open("./"+file+"/"+classname+'/'+img);
        im=im.convert(mode='RGB')
        imrs=im.resize((m,n))
        imrs=img_to_array(imrs)/255;
        imrs=imrs.transpose(2,0,1);
        imrs=imrs.reshape(3,m,n);
        x.append(imrs)
        y.append(classname)
        

nb_classes=len(classes)
nb_filters=32

nb_pool=2
nb_conv=3

import tensorflow
x=np.array(x)
y=np.array(y)
x=np.array(tensorflow.transpose(x,[0,2,3,1]))
print(np.shape(x))
x_train, x_test,y_train, y_test= train_test_split(x,y,test_size=0.2,random_state=42)

print(np.shape(np.array(x_train)))

uniques, id_train=np.unique(y_train,return_inverse=True)
Y_train=np_utils.to_categorical(id_train,nb_classes)
uniques, id_test=np.unique(y_test,return_inverse=True)
Y_test=np_utils.to_categorical(id_test,nb_classes)


model=Sequential()

model.add(Convolution2D(nb_filters,(nb_conv,nb_conv),padding='same',activation="relu",data_format='channels_last',input_shape=(m,n,3)));

model.add(Convolution2D(nb_filters,(nb_conv,nb_conv), activation="relu"));
model.add(MaxPooling2D(pool_size=(nb_pool,nb_pool)));


model.add(Dense(256,activation="relu"))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation="softmax"))

model.summary()

model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

model.fit(x_train, Y_train,
          batch_size=32,
          epochs=8,
          verbose=1,
          validation_data=(x_test, Y_test))
#model.fit(x_train,Y_train,epochs=nb_epoch,validation_data=(x_test, Y_test))
model.evaluate(x_test, Y_test);
im = Image.open("./"+file+"/"+"normal-pylorus"+'/'+'0a0ce428-e7ee-4a42-9e60-b58b7cb6a06f.jpg');
imrs = im.resize((m,n))
imrs=img_to_array(imrs)/255;
imrs=imrs.transpose(2,0,1);
imrs=imrs.reshape(3,m,n);

x=[]
x.append(imrs)
x=np.array(x);
x=np.array(tensorflow.transpose(x,[0,2,3,1]))

predictions = model.predict(x)
out = np.concatenate(predictions).ravel().tolist()
print(classes)
list=predictions[0]
print(list)
maxval=max(list)
print(maxval)
n=0
for val in list:
    if val==maxval:
        print(classes[n])
    n=n+1

