# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Check if a GPU is available for training the model
import tensorflow as tf
print(tf.__version__)
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

# Import required modules
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from skimage.transform import resize

# Load the model
model_path = '/content/drive/MyDrive/skin_ntd_research/stable/stable_model/model/mymodel'
model = load_model(model_path)

# Define the last convolutional layer name
last_conv_layer_name = 'top_conv'

# Get the class names
class_names = ['buruli_ulcer', 'leishmaniasis', 'leprosy', 'mycetoma', 'scabies']

def generate_gradcam(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(188, 188))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Predict the class probabilities
    preds = model.predict(x)
    predicted_class = np.argmax(preds[0])

    # Print the predicted class and the prediction accuracy
    print(f"Predicted disease: {class_names[predicted_class]}")
    print(f"Probability: {round(preds[0][predicted_class]*100, 2)}%")

    # Get the last convolutional layer output and the predicted class output
    last_conv_layer = model.get_layer(last_conv_layer_name)
    classifier_output = model.output[:, predicted_class]

    # Calculate the gradients of the predicted class output with respect to the last convolutional layer output
    grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(x)
        loss = predictions[:, predicted_class]

    output = conv_output[0]
    grads = tape.gradient(loss, conv_output)[0]

    # Perform Global Average Pooling (GAP) on the gradients
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, output), axis=-1)

    # Postprocessing to generate heatmap
    cam = np.maximum(cam, 0)  # apply ReLU to the heatmap
    heatmap = (cam - cam.min()) / (cam.max() - cam.min())  # normalize heatmap

    # Resize the heatmap to match the size of the original image
    heatmap = resize(heatmap, (img.size[1], img.size[0]))

    # Superimpose the heatmap on the original image
    heatmap = np.uint8(255 * heatmap)
    superimposed_img = np.uint8(heatmap[..., np.newaxis] * 0.4 + img_to_array(img))

    # Display the original image, heatmap, and superimposed image
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap='jet')
    plt.title('Heatmap')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(superimposed_img)
    plt.title('Superimposed Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Test the function
generate_gradcam('/content/drive/MyDrive/skin_ntd_research/images_for_grad_cam/mycetoma/2.jpg')
generate_gradcam('/content/drive/MyDrive/skin_ntd_research/images_for_grad_cam/buruli_ulcer/3.jpg')
generate_gradcam('/content/drive/MyDrive/skin_ntd_research/images_for_grad_cam/leishmaniasis/2.jpg')
generate_gradcam('/content/drive/MyDrive/skin_ntd_research/images_for_grad_cam/leprosy/3.jpg')
generate_gradcam('/content/drive/MyDrive/skin_ntd_research/images_for_grad_cam/scabies/3.jpg')
