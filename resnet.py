import sys
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from keras.layers import Input, Add, Dense, Activation, BatchNormalization, Flatten, Conv2D, MaxPooling2D, Dropout, AveragePooling2D
from keras.models import Model
from keras.utils.np_utils import to_categorical
from keras.initializers import glorot_uniform

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import keras.backend as K
K.set_image_data_format('channels_last')

class resnet:
    def __init__(self):
        self.size = 0
        self.classes = 0
        self.x = None
        self.y = None
        self.v = None
        self.epx = None
        self.eid = None
        self.model = None
    
    def preprocess(self, path, file, resize, crop):
        if resize == [0,0] and crop == [0,0]:
            print('Image Should Be Preprocessed.')
            sys.exit()
        if resize != [0,0]:
            image= Image.open(path+file).convert('RGB').resize(resize)
            self.size = resize[0]
        if crop != [0,0]:
            w = image.size[0]
            h = image.size[1]
            image= Image.open(path+file).convert('RGB').crop((int(w-crop[0]/2),int(h-crop[1]/2),w-int(w-crop[0]/2),h-int(h-crop[1]/2)))
            self.size = crop[0]
            pixels=[]
            for w in range(image.size[0]):
                for h in range(image.size[1]):
                    pixels.append(image.getpixel((w,h)))
            pixels = np.array(pixels, dtype = int).reshape(-1).T
            return pixels.reshape((self.size,self.size,3))

    def input(self, ipath, lpath, resize, crop):
        if os.path.exists(ipath):
            if lpath != '' and os.path.exists(lpath):
                with open(lpath, 'r') as f:
                    lines = f.readlines() [1:]
                    items = [l.rstrip().split(',') for l in lines]
                    self.classes = len(set([label for _, label in items]))-1
                    idx_labels = dict((idx, int(label)) for idx, label in items)
                    train_c = int(len(idx_labels))
                split_r = 0.1
                train_s = int(train_c * (1 - split_r))
                count_t, count_v = 0,0
                t_labels, v_labels = [],[]
                t_pixels, v_pixels = [],[] 
                for train_file in os.listdir(ipath):
                    ext = os.path.splitext(train_file)[-1]
                    if ext in ('.jpg','.png','.tif'):
                        idx = train_file.replace(ext,'')
                        if idx in idx_labels.keys():    
                            if os.path.getsize(ipath+train_file) != 0 and count_t <= train_s:
                                t_pixels.append(self.preprocess(ipath,train_file,resize,crop))
                                t_labels.append(idx_labels[idx])
                                count_t+=1
                            elif os.path.getsize(ipath+train_file) != 0 and count_v <= (train_c-train_s):
                                v_pixels.append(self.preprocess(ipath,train_file,resize,crop))
                                v_labels.append(idx_labels[idx])
                                count_v+=1
                self.x = np.array(t_pixels)
                self.y = to_categorical(np.array(t_labels))
                self.v = (np.array(v_pixels),to_categorical(np.array(v_labels)))
                print('Trainset Preprocess Finished.')
            elif lpath == '':
                e_pixels, idys = [],[]
                for test_file in os.listdir(ipath):
                    ext = os.path.splitext(train_file)[-1]
                    if ext in ('.jpg','.png','.tif'):
                        idy = test_file.replace(ext,'')
                        if os.path.getsize(ipath+test_file) != 0:    
                            e_pixels.append(self.preprocess(ipath,test_file,resize,crop))
                            idys.append(idy)
                    self.epx = np.array(e_pixels)
                    self.eid = np.append(idys)
                    print('Testset Preprocess Finished.')
            else:
                print('Labelpath Not Found.')
                sys.exit()
        else:
            print('Imagepath Not Found.')
            sys.exit()

    def identity_block(self, X, f, filters, stage, block):
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
    
        F1, F2, F3 = filters
    
        X_shortcut = X
    
        X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
        X = Activation('relu')(X)
    
        X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)
        
        return X

    def convolutional_block(self, X, f, filters, stage, block, s = 2):
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
    
        F1, F2, F3 = filters

        X_shortcut = X

        X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        X = Conv2D(F2, (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        X = Conv2D(F3, (1, 1), strides = (1,1),name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

        X_shortcut = Conv2D(F3, (1, 1), strides = (s,s), name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
        X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)

        return X

    def create_model(self, input_shape, classes):
        X_input = Input(input_shape)
        nsize = self.size
        while (nsize-4)%4 != 0:
            nsize-=1
        i_size = nsize-4
    
        X = Conv2D(i_size, (3, 3), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X_input)
        X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
        X = Activation('tanh')(X)
        X = MaxPooling2D((3, 3), strides=(2, 2))(X)
        
        conv_size = i_size
        for stage in range(2,2+int(i_size/16)):
            X = self.convolutional_block(X, f = 3, filters = [conv_size*2, conv_size*2, conv_size], stage = stage, block='a', s = 2)
            X = self.identity_block(X, 3, [conv_size*2, conv_size*2, conv_size], stage=stage, block='b')
            X = self.identity_block(X, 3, [conv_size*2, conv_size*2, conv_size], stage=stage, block='c')
            conv_size*=2

        X =  AveragePooling2D((2,2),name="avg_pool")(X)
        X = Flatten()(X)
        cnt=1
        for fc_size in np.logspace(i_size, classes, base=0.25):
            X = Dense(fc_size, activation='softmax', name='fc'+str(cnt), kernel_initializer = glorot_uniform(seed=0))(X)
            X = Dropout(0.1)(X)
            cnt+=1

        model = Model(inputs = X_input, outputs = X, name='ResNet')
        
        return model

    def compile_model(self):
        X_shape = (self.size,self.size,3)
        self.model = self.create_model(X_shape, self.classes)
        self.model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
        print('Model Compiled.')
        self.model.summary()

    def train(self):
        self.model.fit(x=self.x, y=self.y, epochs=100, batch_size=32, validation_data= self.v)

    def test(self, opath):
        e_labels = self.model.predict(self.epx)
        e_len = len(e_labels)
        e_labels = np.array(e_labels).reshape((e_len,1))
        new_e_labels = []
        for i in range(e_len):
            new_e_labels.append(float(e_labels[i,0]))
        new_e_labels = np.array(new_e_labels)
        results = pd.DataFrame({'id':self.eid, 'label':list(new_e_labels)})
        try:
            results.to_csv(opath, header=True, index=False)
            print('Results Stored In %s' % opath)
        except:
            print('Outputpath Not Found.')
            sys.exit()
        
