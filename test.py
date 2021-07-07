from keras.models import load_model
from keras.preprocessing import image
from keras_efficientnets import custom_objects
import numpy as np
import os


os.environ["CUDA_VISIBLE_DEVICES"]="1"

# dimensions of our images
img_width, img_height = 320, 240

# load the model we saved
model = load_model('/Users/esmasert/Desktop/Diarization/train_spk_dir/checkpoints/efficient_net/train_2.h5')
# model.compile(loss='binary_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])
#
# # predicting images
# img = image.load_img('test1.jpg', target_size=(img_width, img_height))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
#
# images = np.vstack([x])
# classes = model.predict_classes(images, batch_size=10)
# print classes

path = '/Users/esmasert/Desktop/Diarization/speech/81fbc2be-5ec8-45f9-95f4-b51fc1b35962-0-recording_292.24.png'
def pred(d):
    img = image.load_img(d, target_size=(256, 256),grayscale=True)
    print('girdi')
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    y = model.predict(x)
    print(y)
    return y

pred(path)