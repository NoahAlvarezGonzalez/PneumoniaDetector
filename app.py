import cv2 as cv
import os
import streamlit as st
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
import PIL
import base64
import io


st.title("Pneumonia Detector")
model = load_model("pneumonia_A88_R94_AUC95_128x128.h5")


uploaded_file = st.file_uploader("Choose an image...", type="jpeg")
if uploaded_file is not None:
    opencv_image = cv.imdecode(np.frombuffer(uploaded_file.read(), dtype=np.uint8), 1)
    resized = cv.resize(opencv_image, (128, 128))
    normalized = resized / 255
    reshaped = np.expand_dims(normalized, axis=0)
    result = model.predict(reshaped)
    if result < 0.5:
        res = 100 - (result[0][0]*100)
        cv.putText(opencv_image, "Normal", (100, 100), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)
        cv.putText(opencv_image, str(res), (100, 200), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)
    else:
        res = (result[0][0] * 100)
        cv.putText(opencv_image, "Pneumonia", (100, 100), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)
        cv.putText(opencv_image, str(res), (100, 200), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)
    st.image(opencv_image, width=800)
    download = st.button('Download JPEG File')
    if download:
        dl = PIL.Image.fromarray(opencv_image)
        buffered = io.BytesIO()
        dl.save(buffered, format="JPEG")
        dl_str = base64.b64encode(buffered.getvalue()).decode()
        href = f'<a href="data:file/jpg;base64,{dl_str}" download="result.jpg">Click here to download your file</a>'
        st.markdown(href, unsafe_allow_html=True)