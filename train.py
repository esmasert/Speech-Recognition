from __future__ import print_function
import tensorflow as tf
import tensorflow.python.keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten, add, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, Callback
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras import models
from keras import optimizers
from  keras.utils import multi_gpu_model
import math, json, os, sys
from resnet import residual_network
import json
from collections import Counter
from sklearn.utils import class_weight
import numpy as np
#from tensorflow.python.keras.datasets import cifar10

import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.python.keras import losses
from tensorflow.python.keras.applications import nasnet
from tensorflow.python.client import device_lib
import pickle

from keras_efficientnets import EfficientNetB0

os.environ["CUDA_VISIBLE_DEVICES"]="0"
with tf.device('GPU:1'):
    def check_available_gpus():
        local_devices = device_lib.list_local_devices()
        gpu_names = [x.name for x in local_devices if x.device_type == 'GPU']
        gpu_num = len(gpu_names)

        print('{0} GPUs are detected : {1}'.format(gpu_num, gpu_names))

        return gpu_num



    DATA_DIR = '/Users/esmasert/Desktop/Diarization/spk_dataset_spectrogram'
    #TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    #VALID_DIR = os.path.join(DATA_DIR, 'val')
    SIZE = (256, 256,1)
    NUM_GPU = check_available_gpus()
    BATCH_SIZE = 128
    modelname = 'efficient_net'



    if __name__ == "__main__":

    ########################### GET DATA FROM FOLDER #################
    ########################### GET DATA FROM FOLDER #################
        num_train_samples = sum([len(files) for r, d, files in os.walk(DATA_DIR)])
        #num_valid_samples = sum([len(files) for r, d, files in os.walk(VALID_DIR)])

        #num_train_steps = math.floor(num_train_samples/BATCH_SIZE)
       # num_valid_steps = math.floor(num_valid_samples/BATCH_SIZE)

        datagen = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=0.2, rescale=1./255)

        #val_gen = tf.keras.preprocessing.image.ImageDataGenerator()

        train_generator = datagen.flow_from_directory(
                                                DATA_DIR,
                                                subset='training',
                                                target_size=(256, 256),
                                                color_mode="grayscale",
                                                batch_size=BATCH_SIZE,
                                                class_mode='categorical',
                                                shuffle=False,
                                                seed=42
                                                )

        val_generator = datagen.flow_from_directory(
                                        DATA_DIR,
                                        subset='validation',
                                        target_size=(256, 256),
                                        color_mode="grayscale",
                                        batch_size=BATCH_SIZE,
                                        class_mode="categorical",
                                        shuffle=True,
                                        seed=42
                                        )


        print(val_generator.class_indices)
        exDict=train_generator.class_indices
        with open('class_lbl.txt', 'w') as file:
            file.write(json.dumps(exDict))

        counter = Counter(train_generator.classes)
        max_val = float(max(counter.values()))
        #class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}

        class_weights = class_weight.compute_class_weight(
                                   'balanced',
                                    np.unique(train_generator.classes),
                                    train_generator.classes)
        print(class_weights)

        num_train_steps = train_generator.n//train_generator.batch_size
        num_valid_steps = val_generator.n//val_generator.batch_size
        print(num_train_steps, num_valid_steps)

    ########################### BUILD MODEL ############################
        #model = tf.keras.applications.resnet50.ResNet50()
        #model = nasnet.NASNetLarge(input_shape = (256,256,1), weights= None, include_top= False)
        #model = build_resnet_50((256,256,1), 61)
        model = EfficientNetB0(input_shape=(256, 256, 1), classes=3, include_top=True, weights=None)
        #input = Input(shape=(256,256,3))
        #out = residual_network(input)
        #out = base_model.outputs
        #print(out)
        #out = Flatten()(out)
        #out = Dense(512, activation='relu', kernel_initializer='he_normal')(out)
        #out = Dense(3, activation='softmax', kernel_initializer='he_normal')(out)
        #out = highway(input, 61)
        #out = testmodel(input, 61)
        #model = RDNet(input_shape = input)
        #model = Model(inputs=base_model.inputs, outputs=out, name='eff_net')##################################################################

        #model = multi_gpu_model(model,gpus = check_available_gpus() )
        model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=True),
                                loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
    ########################### MAKE DIRECTORY ############################
        # Prepare model model saving directory.
        save_dir = os.path.join(os.getcwd(), 'checkpoints/'+modelname)
        #model_name = "resnet{epoch:03d}.pkl"
        model_name = "train_2.h5"
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, model_name)

    ########################### CALLBACKS #################################
        checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True)
        tensorboard = TensorBoard(log_dir='./checkpointsBURAYABAK/'+modelname+'/logs', histogram_freq=0,
                                     write_graph=True,write_images=True)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                      patience=1, min_lr=0.00001)

        early_stopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=0,
                                    mode='auto')

        class save_weight(Callback):
            def __init__(self, model):
                self.model = model
            def on_epoch_end(self, epoch, logs={}):
                weigh = self.model.get_weights()
                print('\n')
                print('----------------saving weights------------------')
                print('\n')
                print(epoch)
                print('\n')
                try:
                    fpkl= open(filepath+"_"+str(epoch)+".pkl", 'wb')	#Python 3
                    pickle.dump(weigh, fpkl, protocol= pickle.HIGHEST_PROTOCOL)
                    fpkl.close()
                except:
                    fpkl= open(filepath+"_"+str(epoch)+".pkl", 'w')	#Python 2
                    pickle.dump(weigh, fpkl, protocol= pickle.HIGHEST_PROTOCOL)
                    fpkl.close()


    ##########################################################################
        class_weight = {0: 50,
                        1:20,
                        2:0.5 }


        model.fit_generator(generator=train_generator,
                            steps_per_epoch=num_train_steps,
                            epochs=30,
                            verbose=1,
                            callbacks=[early_stopping,reduce_lr,checkpoint,tensorboard],
                            validation_data=val_generator,
                            validation_steps= num_valid_steps,
                            class_weight=class_weights)
    #model.save('resnet50_final_new.h5')
