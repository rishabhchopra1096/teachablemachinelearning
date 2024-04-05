from fastapi import FastAPI, HTTPException, Query
from keras.models import load_model
import numpy as np
import tensorflow as tf
import requests

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model_v2.h5", compile=False)

# Load the labels
class_names = open("labels_v2.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

def read_tensor_from_image_url(url,
                               input_height=224,
                               input_width=224,
                               input_mean=0,
                               input_std=255):
    image_reader = tf.image.decode_jpeg(
        requests.get(url).content, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize(dims_expander,[input_height,input_width], method='bilinear',antialias=True, name = None)
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])

    return normalized


app = FastAPI()

@app.get("/classify/")  # Use GET method
async def classify_image(image_url: str = Query(..., description="The URL of the image to classify")):
    try:
        image = read_tensor_from_image_url(image_url)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array

        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()  # Ensure class name doesn't include newline characters
        confidence_score = prediction[0][index]
        print("Class:", class_name, end="\n")
        return {"class": class_name[2:], "confidence": float(confidence_score)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# For locally running main.py
# # Decode and resize the image from the URL
# image = read_tensor_from_image_url(image_url)

# # Turn the image into a numpy array
# image_array = np.asarray(image)

# # Normalize the image
# normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# # Load the image into the array
# data[0] = normalized_image_array

# # Predicts the model
# prediction = model.predict(data)
# index = np.argmax(prediction)
# class_name = class_names[index]
# confidence_score = prediction[0][index]

# return {"class": class_name[2:], "confidence": float(confidence_score)}


# #decode and resize the image from the URL
# image = read_tensor_from_image_url(r'https://i.pinimg.com/736x/a8/32/60/a83260ededd887b78794f1569e2ba8da.jpg')

# # # Replace this with the path to your image
# # image_path = "avenger_images/test/robert_downey_jr/robert_downey_jr13.png"
# # image = Image.open(image_path).convert("RGB")
# # # resizing the image to be at least 224x224 and then cropping from the center
# # size = (224, 224)
# # image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# # turn the image into a numpy array
# image_array = np.asarray(image)

# # Normalize the image
# normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# # Load the image into the array
# data[0] = normalized_image_array

# # Predicts the model
# prediction = model.predict(data)
# print(prediction)
# index = np.argmax(prediction)
# class_name = class_names[index]
# confidence_score = prediction[0][index]

# # Print prediction and confidence score
# print("Class:", class_name[2:], end="")
# print("Confidence Score:", confidence_score)