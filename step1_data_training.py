# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install specific tensorflow version and print its version to confirm
!pip install tensorflow==2.9.1
import tensorflow as tf
print(tf.__version__)

# Check if a GPU is available for training the model
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

# Import required modules
import numpy as np
import tensorflow as tf
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
from tensorflow.keras.applications import EfficientNetB3
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

# Set data processing parameters and define the data generator functions
def set_data(train,test, batchSize, image_size):
 # Set seeds for reproducibility
 np.random.seed(1234)
 python_random.seed(1234)
 tf.random.set_seed(1234)

 # Set image size and contrast equalization function
 Image_size = [image_size,image_size]

 def clhe(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    gray = gray.astype(np.uint8)
    eq = cv.equalizeHist(gray)
    eq = cv.cvtColor(eq, cv.COLOR_GRAY2RGB)
    eq = eq.astype(np.float32)
    return eq

 # Data augmentation and preparation for training dataset
 train_datagen= ImageDataGenerator(validation_split=0.3,rotation_range=10,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=0.5,
                                   preprocessing_function=clhe
                                   )

 # Preprocessing for test dataset
 test_datagen = ImageDataGenerator(preprocessing_function=clhe)

 # Create data generators
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
    train, 
    target_size=Image_size,color_mode = "rgb",interpolation='bicubic',
    batch_size=batchSize)
 return train_set, test_set, validation_set;

# Plotting function for accuracy and loss over training epochs
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

# Function to create the model
def create_model():
  inputs = layers.Input(shape=(188, 188, 3))
  x = inputs
  # Load pre-trained EfficientNetB3 as base model
  basemodel = EfficientNetB3(include_top=False, input_tensor=x, weights="imagenet")
  basemodel.trainable = False
  # Unfreeze the last 200 layers of the base model
  basemodel = unfreeze_model(basemodel, -200)
  x = basemodel.output
  x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
  x = layers.Dense(512,activation='relu')(x)
  x = layers.Dropout(0.2)(x)
  x = layers.Dense(256,activation='relu')(x)
  x = layers.Dropout(0.2)(x)
  # Output layer with softmax activation for multi-class classification
  outputs = layers.Dense(5, activation="softmax", name="pred")(x)
  model = tf.keras.Model(basemodel.input, outputs)
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
  model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
  return model

# Function to unfreeze the last 'num_of_layers' of the base model
def unfreeze_model(model, num_of_layers):
    for layer in model.layers[num_of_layers:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
    return model

# Define parameters for the training
batchSize = 8
epoches = 50;
image_size = 188;
train_path = '/content/drive/MyDrive/skin_ntd_research/skin_ntd_dataset/train'
test_path = '/content/drive/MyDrive/skin_ntd_research/skin_ntd_dataset/test'
train_set, test_set, validation_set = set_data(train_path,test_path,batchSize, image_size)
# Create the model
model = create_model()
# Define the checkpoint path and create the checkpoint directory
checkpoint_path = "/content/drive/MyDrive/skin_ntd_research/eff_versions/models/b3/weights_best.hdf5"
checkpoint_dir = os.path.dirname(checkpoint_path)
# Define the metric to monitor and create a ModelCheckpoint callback
metric='val_accuracy'
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  monitor=metric,mode='max',
                                                save_best_only=True,
                                                verbose=1)
# Train the model
history=model.fit(train_set, epochs = epoches, validation_data= validation_set, callbacks=[cp_callback], shuffle=True)
# Save the trained model
model.save('/content/drive/MyDrive/skin_ntd_research/eff_versions/models/b3/mymodel')
# Evaluate the trained model on the test set
results = model.evaluate(test_set,batch_size=8)
accuracy = results[1]
# Predict the labels of the test set
predict_labels=model.predict(test_set,batch_size=batchSize)
# Get the true labels of the test set
test_labels=test_set.classes
print(accuracy)

print(test_labels)
print(predict_labels.argmax(axis=1))
# Print classification report
from sklearn.metrics import classification_report
print('\nClassification Report\n')
print(classification_report(test_labels, predict_labels.argmax(axis=1), target_names=['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5']))
# Print confusion matrix
confusion = confusion_matrix(test_labels, predict_labels.argmax(axis=1))
print('Confusion Matrix\n')
print(confusion)

# Load the saved model
mode_loaded = keras.models.load_model('/content/drive/MyDrive/skin_ntd_research/eff_versions/models/b3/mymodel')

# Redefine parameters for the training and get the data generators
batchSize = 8
epoches = 50;
image_size = 188;
train_path = '/content/drive/MyDrive/skin_ntd_research/skin_ntd_dataset/train'
test_path = '/content/drive/MyDrive/skin_ntd_research/skin_ntd_dataset/test'
train_set, test_set, validation_set = set_data(train_path,test_path, batchSize, image_size)
# Evaluate the loaded model on the test set
results = mode_loaded.evaluate(test_set,batch_size=8)
accuracy = results[1]
# Predict the labels of the test set with the loaded model
predict_labels=mode_loaded.predict(test_set,batch_size=batchSize)

# Function to set up the data for testing
def set_data(test):
  # Setting image size and batch size parameters
  Image_size = [188,188]
  batchSize = 8
  numClasses = 5

  # Defining a function to apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to the images
  def clhe(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    gray = gray.astype(np.uint16)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    eq = clahe.apply(gray)
    eq = cv.cvtColor(eq, cv.COLOR_GRAY2RGB)
    eq = eq.astype(np.float32)
    return eq

  # Create image data generator for test data
  test_datagen = ImageDataGenerator()

  # Create iterator for the test data
  test_set= test_datagen.flow_from_directory(
              test,
              target_size=Image_size,
              batch_size=batchSize,
              interpolation='bicubic',
              class_mode='categorical',
              shuffle=False
             )
  # Load the best model 
  m = tf.keras.models.load_model("/content/drive/MyDrive/skin_ntd_research/eff_versions/models/b3/weights_best.hdf5")
  
  # Evaluate model accuracy on test data
  accuracy = m.evaluate(test_set)
  r = m.predict(test_set)
  
  # Getting the ground truth labels from the test data
  k = test_set.classes

  return accuracy, r, k

# Set path for test data
test_path = "/content/drive/MyDrive/skin_ntd_research/skin_ntd_dataset/test"
start = time.time()

# Call the function to setup data and perform evaluation and prediction
accuracy, r, k = set_data(test_path)

end = time.time()

# Finding the class with maximum probability
predIdxs = np.argmax(r,axis=-1)

# Creating a confusion matrix to visualize the performance of the model
cnf_matrix = confusion_matrix(k, predIdxs)
ax= plt.subplot()
sns.heatmap(cnf_matrix, annot=True,cmap='Blues');

# Printing a classification report to understand performance for each class
from sklearn.metrics import classification_report
print('\nClassification Report\n')
print(classification_report(k, predIdxs))
