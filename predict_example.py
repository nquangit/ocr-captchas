import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from keras.saving import register_keras_serializable

"""
## Define constants
"""
img_width = 80
img_height = 30
max_length = 4  # Adjust this based on your dataset


@register_keras_serializable()
class CTCLayer(layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred

    def get_config(self):
        config = super().get_config()
        return config


# Characters present in the dataset must be as same as the one used in training
characters = sorted(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])

# Mapping characters to integers
char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)

# Mapping integers back to original characters
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)

"""
## Load the model
"""
model = load_model("model/captcha_ocr_model.keras", compile=False)

# for layer in model.layers:
#     print(layer.name)

# Extract the prediction model
prediction_model = tf.keras.models.Model(
    model.input[0], model.get_layer(name="dense2").output
)
prediction_model.summary()

"""
## Preprocessing functions
"""


# Define a function to preprocess the image
def preprocess_image(img_path):
    # Read image
    img = tf.io.read_file(img_path)
    # Decode and convert to grayscale
    img = tf.io.decode_png(img, channels=1)
    # Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # Resize to the desired size
    img = tf.image.resize(img, [img_height, img_width])
    # Transpose the image because we want the time dimension to correspond to the width of the image.
    img = tf.transpose(img, perm=[1, 0, 2])
    # Expand dims to add batch size
    img = tf.expand_dims(img, axis=0)
    return img


# Define a function to decode the prediction
def decode_prediction(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][
        0
    ][:, :max_length]
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


"""
## Inference
"""

import requests


def download_file(url, filename="Show.png"):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check if the request was successful

        with open(filename, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print(f"File downloaded successfully as {filename}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")


# Example usage
download_file("change_url_here")


# Load and preprocess the image
img_path = "Show.png"
img = preprocess_image(img_path)

# Make the prediction
pred = prediction_model.predict(img)
pred_text = decode_prediction(pred)

# Print the prediction
print("Predicted text:", pred_text[0])

# Visualize the image and prediction
plt.imshow(img[0, :, :, 0].numpy().T, cmap="gray")
plt.title(f"Prediction: {pred_text[0]}")
plt.axis("off")
plt.show()
