
import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model("brain_tumor_model.keras")

# Define class names
class_names = ["No Tumor", "Tumor"]

# Define prediction function
def predict(img):
    img = img.resize((224, 224))  # Resize for the model
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)
    return f"{predicted_class} ({confidence*100:.2f}%)"

# Define the Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Brain Tumor Detection",
    description="Upload an MRI image to detect presence of a brain tumor."
)

# Launch the app (for local testing)
if __name__ == "__main__":
    interface.launch()
