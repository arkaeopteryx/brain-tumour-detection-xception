import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load model
model = tf.keras.models.load_model("brain_tumor_model.keras")

# Labels
class_names = ['no', 'yes']

# Prediction function
def predict(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return class_names[int(prediction[0][0] > 0.5)]

# Launch app
gr.Interface(fn=predict, inputs="image", outputs="label", title="Brain Tumor Detection").launch()
