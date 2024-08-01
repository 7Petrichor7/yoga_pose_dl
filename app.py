import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helper import preprocess_image, predict, models

st.title('Yoga Pose Classification')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
model_name = st.selectbox('Select Model', ['VGG16', 'VGG19'])

# Assuming we have a list of class names corresponding to the model's output
# Replace this list with your actual class names
class_names = [f'class_{i}' for i in range(1, 108)]

if uploaded_file is not None:
    # Save the uploaded file to a temporary directory
    temp_file_path = os.path.join("static", uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Predict the class of the image
    model = models.get(model_name)
    if model:
        predictions = predict(model, temp_file_path)
        predicted_class = np.argmax(predictions, axis=1)[0]
        st.write(f"Model: {model_name}")
        st.write(f"Predicted class index: {predicted_class}")  # Debugging line
        if predicted_class < len(class_names):
            st.write(f"Prediction: {class_names[predicted_class]}")
        else:
            st.write("Error: Predicted class index is out of range.")
        st.image(temp_file_path)

        # Plot the predictions
        pred_df = pd.DataFrame(predictions, columns=class_names).transpose()
        pred_df.columns = ['Probability']
        pred_df = pred_df.sort_values(by='Probability', ascending=False)
        
        st.bar_chart(pred_df)

    else:
        st.write("Model not found")

# Make sure to create the "static" directory to store the uploaded files
if not os.path.exists("static"):
    os.makedirs("static")
