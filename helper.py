import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model

# Function to preprocess the image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Function to make predictions
def predict(model, image_path):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    return predictions

# Load models from the static directory
models = {
    "VGG16": load_model('static/vgg16_model.h5'),
    "VGG19": load_model('static/vgg19_model.h5')
}

print("Models loaded")
