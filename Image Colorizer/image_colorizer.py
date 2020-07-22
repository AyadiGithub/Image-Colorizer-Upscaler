"""
Author: Mohamed Ayadi
"""

# In[1]: Imports

import tensorflow as tf
print("Tensorflow version " + tf.__version__)
device_name = tf.test.gpu_device_name()
import os
import re
from skimage.transform import resize, rescale
from skimage.color import rgb2lab, lab2rgb, rgb2gray, grey2rgb
from skimage.io import imsave
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras as keras
from livelossplot import PlotLossesKeras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.python.client import device_lib
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D
from tensorflow.keras.layers import Conv2DTranspose, UpSampling2D, add
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras import backend
device_lib.list_local_devices()
np.random.seed(0)

# In[2]: Colab relevant imports

# Import google drive to colab
# from google.colab import drive
# drive.mount('/content/drive')

# Download and Unzip File
# !wget dataset_url

# !unzip '/content/file'

# In[3]: Encoder for the Autoencoder

# Input layer will be 256x256x1 (Grayscale LAB Image)
input_img = Input(shape=(256, 256, 1))  # Using Input class from Keras

layer1 = Conv2D(128, (3, 3),
                padding='same',
                activation='relu',
                kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(10e-7))(input_img)  # Applied to the output of previous layer


layerm1 = MaxPooling2D(padding='same')(layer1)  # MaxPooling over previous layer output


layer2 = Conv2D(128, (3, 3),
                padding='same',
                activation='relu',
                kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(10e-7))(layerm1)   # Applied to the output of previous layer

layer3 = Conv2D(128, (3, 3),
                padding='same',
                activation='relu',
                kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(10e-7))(layer2)   # Applied to the output of previous layer

layerm2 = MaxPooling2D(padding='same')(layer3)  # MaxPooling over previous layer output

layer4 = Conv2D(128, (3, 3),
                padding='same',
                activation='relu',
                kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(10e-7))(layerm2)  # Applied to the output of previous layer


layer5 = Conv2D(256, (3, 3),
                padding='same',
                activation='relu',
                kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(10e-7))(layer4)  # Applied to the output of previous layer

layer6 = Conv2D(256, (3, 3),
                padding='same',
                activation='relu',
                kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(10e-7))(layer5)  # Applied to the output of previous layer

layerm3 = MaxPooling2D(padding='same')(layer6)  # MaxPooling over previous layer output

layer7 = Conv2D(256, (3, 3),
                padding='same',
                activation='relu',
                kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(10e-7))(layerm3)  # Applied to the output of previous layer

layer8 = Conv2D(512, (3, 3),
                padding='same',
                activation='relu',
                kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(10e-7))(layer7)  # Applied to the output of previous layer

layer9 = Conv2D(512, (3, 3),
                padding='same',
                activation='relu',
                kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(10e-7))(layer8)  # Applied to the output of previous layer

# Model class from keras to put the model together
encoder = Model(input_img, layer9)

# Encoder model summary
encoder.summary()

# In[4]: Decoder

# Reverse the Encoder for the Decoder

layer10 = Conv2DTranspose(256, (3, 3),
                         strides=(2, 2),
                         padding='same',
                         activation='relu',
                         kernel_initializer='he_normal',
                         kernel_regularizer=regularizers.l2(10e-7))(layer9)  # Applied to the output of previous layer

layeradd = add([layer10, layer6])  # Add features from encoder layer to decoder layer

layer11 = Conv2D(256, (3, 3),
                padding='same',
                activation='relu',
                kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(10e-7))(layeradd)

layer12 = Conv2DTranspose(128, (3, 3),
                         strides=(2, 2),
                         padding='same',
                         activation='relu',
                         kernel_initializer='he_normal',
                         kernel_regularizer=regularizers.l2(10e-7))(layer11)  # Applied to the output of previous layer

layeradd1 = add([layer12, layer3])  # Add features from encoder layer to decoder layer

layer13 = Conv2D(128, (3, 3),
                padding='same',
                activation='relu',
                kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(10e-7))(layeradd1)

layer14 = Conv2DTranspose(128, (3, 3),
                          strides=(2, 2),
                          padding='same',
                          activation='relu',
                          kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(10e-7))(layer13)  # Applied to the output of previous layer

layeradd2 = add([layer14, layer1])  # Add features from encoder layer to decoder layer

layer15 = Conv2D(64, (3, 3),
                 padding='same',
                 activation='relu',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(10e-7))(layeradd2)  # Applied to the output of previous layer

# Decoder output, 2 channels for colored LAB Image
decoder = Conv2D(2, (3, 3),
                 padding='same',
                 activation='tanh',
                 kernel_initializer='glorot_normal')(layer15)  # Applied to the output of previous layer

# In[5]: Encoder + Decoder

# combine the encoder and decoder to make an Autoencoder using Keras Model class
autoencoder = Model(input_img, decoder)

# Autoencoder summary
autoencoder.summary()

# In[6]: if Model already trained

# To load trained model or Network weights
# autoencoder = tf.keras.models.load_model(r'C:\Colorizer.h5')
# autoencoder.load_weights(r'C:\autoencoder_weightsv.h5')
# encoder.load_weights(r'C:\encoder_weightsv.h5')

# In[7]: Loss function

# Combined Loss function Average SSIM and MSE
# Alpha is the weight attributed to the loss function
alpha = 0.10


def CustomLoss(y_true, y_pred):
    ssim = tf.image.ssim(y_true, y_pred, 1.0)
    dssim = backend.mean(1 - ssim)
    mse = backend.mean(backend.square(y_true - y_pred), axis=-1)
    loss = (alpha*dssim)+((1-alpha)*mse)

    return loss

  
# In[8]

# Plotting during training using livelossplot Library
# plotlosses = PlotLossesKeras()

# In[9]: Model Initialization

# Define optimizer and compile Model Autoencoder
# Reducing learning rate if no decrease in Val_loss - Non adaptive optimizer
# reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, min_lr=1e-6, mode='min')


autoencoder.compile(optimizer='adam', loss=dssimloss)

# Checkpoint to save best weights

# checkpoint_filepath = r'C:\Colorizer\checkpoints'
# model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_filepath,
#     save_weights_only=True,
#     monitor='val_loss',
#     mode='min',
#     save_best_only=True)


# In[10]: Training

# Train function


def train_batches(just_load_dataset=False):

    # Training on Tesla P-100 (16GB VRAM)
    # Batch size
    batches = 64
    # point in current batch
    batch = 0

    # Current batch number
    batch_nb = 0

    max_batches = -1  # No limit to number of batches

    # number of epochs
    epochs = 50

    x_train_orig = []
    y_train_orig = []

    x_train = []
    y_train = []

    # Dataset Path
    data_set_path = (path)

    for dirpath, dirnames, filenames in os.walk(data_set_path):

        for filename in filenames:
            if re.search("\.(jpg|jpeg|JPEG|png)$", filename):
                if batch_nb == max_batches:
                    return x_train, y_train

                filepath = os.path.join(dirpath, filename)  # set path for each image

                image = plt.imread(filepath)  # read image from path

                if len(image.shape) > 2:

                  lab = rgb2lab(image)  # Convert image from RGB to LAB
                  image_resized = resize(lab, (256, 256))  # Resizing images
                  x_train_orig.append(image_resized[:,:,0])
                  y_train_orig.append(image_resized[:,:,1:] / 128)
                  batch += 1

                  if batch == batches:

                      batch_nb += 1

                      y_train = np.array(y_train_orig)  # Convert to numpy array
                      x_train = np.array(x_train_orig)  # Convert to numpy array
                      x_train = x_train.reshape(x_train.shape+(1,))  # Reshape

                      if just_load_dataset:  # If just_load_dataset is set to True
                          return x_train , y_train  # return arrays

                      print('Training batch', batch_nb, '(', batches, ')')

                      autoencoder.fit(x_train, y_train, epochs=epochs, batch_size=32, validation_split=0.10, shuffle=True)
                      # callbacks=[plotlosses, model_checkpoint_callback, reduce_lr]

                      x_train_orig = []
                      y_train_orig = []

                      batch = 0

    return x_train, y_train

x_train_orig, y_train_orig = train_batches()

# In[11] Save the Model

autoencoder.save(path)

# Saving model to json
autoencoder_json = autoencoder.to_json()
with open("model.json", "w") as json_file:  # Store model.json
    json_file.write(autoencoder_json)

# Save model weights
autoencoder.save_weights(path)
encoder.save_weights(path)

# In[12]: Test Model

# Test images to evaluate the model
image = img_to_array(load_img(path)
image = resize(image, (256, 256))
colored = []
colored.append(image)

colored = np.array(colored, dtype=float)
colored = rgb2lab(1.0/255*colored)[:, :, :, 0]
colored = colored.reshape(colored.shape+(1,))

# Autoencoder prediction
output = autoencoder.predict(colored)
output = output*128

prediction = np.zeros((256, 256, 3))
prediction[:, :, 0] = colored[0][:, :, 0]
prediction[:, :, 1:] = output[0]

# Convert Image from LAB to RGB
result = lab2rgb(prediction)

# Rescaling Image
result = rescale(result, 2.0, multichannel=True)
plt.imshow(result)
# imsave("result.jpg", lab2rgb(result))
