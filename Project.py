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

import numpy as np
import os
from PIL import Image
print("OK")

m,n=600,600

classes=os.listdir("./MiniResized")
x=[]
y=[]
for classname in classes:
    print(classname)
    imagefiles=os.listdir("./MiniResized/"+classname+'/')
    for img in imagefiles:
        im=Image.open("./MiniResized/"+classname+'/'+img);
        im=im.convert(mode='RGB')
        imrs=im.resize((m,n))
        imrs=img_to_array(imrs)/255;
        imrs=imrs.transpose(2,0,1);
        imrs=imrs.reshape(3,m,n);
        x.append(imrs)
        y.append(classname)
        

batch_size=32
nb_classes=len(classes)
nb_epoch=20
nb_filters=32
nb_pool=2
nb_conv=3
import tensorflow
x=np.array(x)
y=np.array(y)
x=np.array(tensorflow.transpose(x,[0,2,3,1]))
print(np.shape(x))
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2,random_state=4)

print(np.shape(np.array(x_train)))

uniques, id_train=np.unique(y_train,return_inverse=True)
Y_train=np_utils.to_categorical(id_train,nb_classes)
uniques, id_test=np.unique(y_test,return_inverse=True)
Y_test=np_utils.to_categorical(id_test,nb_classes)


model=Sequential()

model.add(Convolution2D(nb_filters,(nb_conv,nb_conv),padding='same',data_format='channels_last',input_shape=(m,n,3)));
model.add(Activation('relu'));

model.add(Convolution2D(nb_filters,(nb_conv,nb_conv)));
model.add(Activation('relu'));

model.add(MaxPooling2D(pool_size=(nb_pool,nb_pool)));

model.add(Dropout(0.5));
model.add(Flatten());
model.add(Dense(128));
model.add(Dropout(0.5));
model.add(Dense(nb_classes));
model.add(Activation('softmax'));
model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])

nb_epoch=3;
batch_size=20;
model.fit(x_train,Y_train,batch_size=batch_size,epochs=nb_epoch,verbose=1)

im = Image.open("./MiniResized/"+"normal-pylorus"+'/'+'0a0ce428-e7ee-4a42-9e60-b58b7cb6a06f.jpg');
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

