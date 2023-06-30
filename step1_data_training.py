# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install and import TensorFlow
!pip install tensorflow==2.9.1
import tensorflow as tf
print(tf.__version__)

# Import necessary libraries
import numpy as np
import random as python_random
import keras
import matplotlib.pyplot as plt
import os
from tensorflow.keras.layers.experimental import preprocessing
from keras.models import Sequential, Model
from keras_preprocessing.image import load_img
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.applications import EfficientNetB3, EfficientNetB5, EfficientNetB0
from keras.applications.mobilenet import MobileNet
from tensorflow.keras import layers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.utils.vis_utils import plot_model
import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from keras import models
import h5py
from keras import metrics
import cv2 as cv

# Check GPU availability
import tensorflow as tf
print(tf.__version__)
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

# Define a function to set up data for training, validation and testing
def set_data(train,test, batchSize, image_size):
 np.random.seed(1234)
 python_random.seed(1234)
 tf.random.set_seed(1234)

 Image_size = [image_size,image_size]
 # Define a function for contrast limited adaptive histogram equalization (CLAHE)
 def clhe(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    gray = gray.astype(np.uint8)
    eq = cv.equalizeHist(gray)
    eq = cv.cvtColor(eq, cv.COLOR_GRAY2RGB)
    eq = eq.astype(np.float32)
    return eq

 # Create ImageDataGenerators for train and test datasets
 train_datagen= ImageDataGenerator(validation_split=0.3,rotation_range=10,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=0.5,
                                   preprocessing_function=clhe
                                   )

 test_datagen = ImageDataGenerator(preprocessing_function=clhe)

 # Load images and labels from directories
 train_set = train_datagen.flow_from_directory(
                train,
                target_size=Image_size,
                batch_size=batchSize,
                color_mode="rgb",
                interpolation='bicubic',
                class_mode='categorical'
                )

 test_set= test_datagen.flow_from_directory(
              test,
              target_size=Image_size,
              color_mode = "rgb", interpolation='bicubic',
              class_mode='categorical'
             )
 validation_set = train_datagen.flow_from_directory(
    train, # same directory as training data
    target_size=Image_size,color_mode = "rgb",interpolation='bicubic',
    batch_size=batchSize)
 return train_set, test_set, validation_set;

# Function to plot training history
def plot_hist(hist):
    plt.figure(3)
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()

    plt.figure(4)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

# Function to create and compile the model
def create_model():
  inputs = layers.Input(shape=(188, 188, 3))
  x = inputs
  basemodel = EfficientNetB3(include_top=False, input_tensor=x, weights="imagenet")
  basemodel.trainable = False
  basemodel = unfreeze_model(basemodel, -200)
  x = basemodel.output
  x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
  x = layers.Dense(512,activation='relu')(x)
  x = layers.Dropout(0.2)(x)
  x = layers.Dense(256,activation='relu')(x)
  x = layers.Dropout(0.2)(x)
  outputs = layers.Dense(5, activation="softmax", name="pred")(x)
  model = tf.keras.Model(basemodel.input, outputs)
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
  model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
  return model

# Function to unfreeze some layers of the model
def unfreeze_model(model, num_of_layers):
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[num_of_layers:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
    return model

# Set up parameters and paths for training
batchSize = 8
epoches = 50;
image_size = 188;
train_path = '/content/drive/MyDrive/skin_ntd_research/skin_ntd_dataset/train'
test_path = '/content/drive/MyDrive/skin_ntd_research/skin_ntd_dataset/test'
train_set, test_set, validation_set = set_data(train_path,test_path,batchSize, image_size)
model = create_model()
checkpoint_path = "/content/drive/MyDrive/skin_ntd_research/weights_best.hdf5"
checkpoint_dir = os.path.dirname(checkpoint_path)
metric='val_accuracy'
# Set up callback for model checkpointing
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  monitor=metric,mode='max',
                                                save_best_only=True,
                                                verbose=1)
# Train the model
history=model.fit(train_set, epochs = epoches, validation_data= validation_set, callbacks=[cp_callback], shuffle=True)
# Save the model
model.save('/content/drive/MyDrive/skin_ntd_research/model/mymodel')
# Evaluate the model
results = model.evaluate(test_set,batch_size=8)
accuracy = results[1]
predict_labels=model.predict(test_set,batch_size=batchSize)
test_labels=test_set.classes
# Plot training history
#plot_hist(history)
print(accuracy)

# Print out labels and predicted labels
print(test_labels)
print(predict_labels.argmax(axis=1))
# Print classification report and confusion matrix
from sklearn.metrics import classification_report
print('\nClassification Report\n')
print(classification_report(test_labels, predict_labels.argmax(axis=1), target_names=['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5']))
confusion = confusion_matrix(test_labels, predict_labels.argmax(axis=1))
print('Confusion Matrix\n')
print(confusion)

# Load the model
mode_loaded = keras.models.load_model('/content/drive/MyDrive/skin_ntd_research/model/mymodel')

# Set up parameters and paths for evaluation
batchSize = 8
epoches = 50;
image_size = 188;
train_path = '/content/drive/MyDrive/skin_ntd_research/skin_ntd_dataset/train'
test_path = '/content/drive/MyDrive/skin_ntd_research/skin_ntd_dataset/test'
train_set, test_set, validation_set = set_data(train_path,test_path, batchSize, image_size)
# Evaluate the loaded model
results = mode_loaded.evaluate(test_set,batch_size=8)
accuracy = results[1]
predict_labels=mode_loaded.predict(test_set,batch_size=batchSize)

# Function to set up data and load the model for evaluation
def set_data(test):
 Image_size = [188,188]
 batchSize = 8
 numClasses = 5

  # Function to preprocess image using CLAHE
  def clhe(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    gray = gray.astype(np.uint16)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    eq = clahe.apply(gray)
    eq = cv.cvtColor(eq, cv.COLOR_GRAY2RGB)
    eq = eq.astype(np.float32)
    return eq

 # Set up test data generator
 test_datagen = ImageDataGenerator( preprocessing_function=clhe)
 test_set= test_datagen.flow_from_directory(
              test,
              target_size=Image_size,
              batch_size=batchSize,
              interpolation='bicubic',
              class_mode='categorical',
              shuffle=False
             )
 # Load the model
 m = tf.keras.models.load_model("/content/drive/MyDrive/skin_ntd_research/model/mymodel")
 accuracy = m.evaluate(test_set)
 r=m.predict(test_set)
 k=test_set.classes
 return accuracy,r,k

# Set up path for test data and evaluate the model
test_path = "/content/drive/MyDrive/skin_ntd_research/skin_ntd_dataset/test"
start=time.time();
accuracy,r,k = set_data(test_path)
end=time.time();

# Get predicted labels
predIdxs = np.argmax(r,axis=-1)
# Compute confusion matrix and plot it
cnf_matrix = confusion_matrix(k, predIdxs)
ax= plt.subplot()
sns.heatmap(cnf_matrix, annot=True,cmap='Blues');

# Print classification report
from sklearn.metrics import classification_report
print('\nClassification Report\n')
print(classification_report(k, predIdxs))
