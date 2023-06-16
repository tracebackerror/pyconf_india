import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions

# Load the InceptionV3 model
model = inception_v3.InceptionV3(weights='imagenet')

# Load and preprocess the input image
img_path = 'path_to_input_image.jpg'
img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = inception_v3.preprocess_input(x)

# Predict the class probabilities for the input image
preds = model.predict(x)
decoded_preds = decode_predictions(preds, top=3)[0]
print("Original Predictions:")
for _, label, prob in decoded_preds:
    print(f"{label}: {prob:.4f}")

# Define the adversarial attack function using FGSM
def fgsm_attack(image, epsilon):
    # Gradient tape records the gradients to calculate the adversarial perturbation
    with tf.GradientTape() as tape:
        tape.watch(image)
        preds = model(image)
        loss = tf.keras.losses.CategoricalCrossentropy()(tf.one_hot(np.argmax(preds[0]), 1000), preds)
    # Calculate gradients
    gradient = tape.gradient(loss, image)
    # Generate adversarial perturbation
    perturbation = epsilon * tf.sign(gradient)
    # Create adversarial image by adding perturbation to the original image
    adversarial_image = tf.clip_by_value(image + perturbation, -1, 1)
    return adversarial_image

# Generate the adversarial image
epsilon = 0.01  # Perturbation magnitude
adversarial_x = fgsm_attack(x, epsilon)

# Predict the class probabilities for the adversarial image
adversarial_preds = model.predict(adversarial_x)
decoded_adversarial_preds = decode_predictions(adversarial_preds, top=3)[0]
print("\nAdversarial Predictions:")
for _, label, prob in decoded_adversarial_preds:
    print(f"{label}: {prob:.4f}")

# Display the original image and the adversarial image
image.array_to_img(x[0]).show()
image.array_to_img(adversarial_x[0]).show()
