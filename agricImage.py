# Import modules
import streamlit as st
import google.generativeai as google_genai
import numpy as np
import tracemalloc
import keras
import os
from PIL import Image

tracemalloc.start()

width = 256
height = 256

# Gemini API setup
GOOGLE_API_KEY=os.getenv('API_KEY')

google_genai.configure(api_key=GOOGLE_API_KEY)

gen_model = google_genai.GenerativeModel('gemini-pro-vision')

# Streamlit app/Layout

# Overview
st.set_page_config(
    page_title="Crop image classification",
)

st.title('Image classification (Crops)')

st.subheader('Overview')

st.markdown('''
         This **machine learning** project is for detecting crop image of the face in the given image .
         It is an application of **computer vision** using Tensorflow. 
    ''')

img_path = st.file_uploader('Please upload an image(Face)', type=['png','jpg', 'jpeg'])

# load trained model
pretrained_model_path = 'Inception-V1.h5'

@st.cache_resource
def loadmodel():
    image_classifier = keras.models.load_model(pretrained_model_path)
    return image_classifier 


# Image processing for classification
def process_image(imgpath):
    fileImage = Image.open(imgpath).convert("RGB").resize([width, height],Image.LANCZOS) # type: ignore
    image = np.array(fileImage)
    img_array = image.reshape(1, width,height,3)
    img_array = img_array.astype('float32')
    img_array = img_array/255.
    
    return img_array
     
    
# Classifier function
def predict(imgpath):
    processed_image = process_image(imgpath)
    prediction = image_classifier.predict(processed_image) # type: ignore
    
    return prediction

def gemini_output():
    img = Image.open(img_path)
    gemini_response = gen_model.generate_content(["Give agricultural and nutritional info based on this picture. It should include a description of the crop, how to cultivate it and its optimal growth conditions. If its not a crop image, respond saying that its not a crop image", img], stream=True)
    gemini_response.resolve()
    
    return gemini_response.text

    

classes = {0: 'Cherry', 1: 'Coffee-plant', 2: 'Cucumber', 3: 'Fox_nut(Makhana)', 4: 'Lemon', 5: 'Olive-tree',
               6: 'Pearl_millet(bajra)', 7: 'Tobacco-plant', 8: 'almond', 9: 'banana', 10: 'cardamom', 11: 'chilli',
               12: 'clove', 13: 'coconut', 14: 'cotton', 15: 'gram', 16: 'jowar', 17: 'jute', 18: 'maize',
               19: 'mustard-oil', 20: 'papaya', 21: 'pineapple', 22: 'rice', 23: 'soyabean', 24: 'sugarcane',
               25: 'sunflower', 26: 'tea', 27: 'tomato', 28: 'vigna-radiati(Mung)', 29: 'wheat'}
if img_path:
    image_classifier = loadmodel()

prediction = predict(img_path)

predicted_class = prediction.argmax()
certainty = 100 * prediction.max()


if st.button('Classify image'):
    with st.spinner("Classifying..."):
        st.markdown(f'''
            <div style="background-color: black; color: white; font-weight: bold; padding: 1rem; border-radius: 10px;">
            <h4>Results</h4>
            <img href={img_path}/>
                  Predicted crop => <span style="font-weight: bold;">{classes[predicted_class]} </span> with <span style="font-weight: bold;">{certainty:.2f}% </span>certainty
            </div>
                </p>
                ''', unsafe_allow_html=True)
        st.success('Successful')
        
gen_button = st.button('Generate Gemini output')

if gen_button:
    output = gemini_output()

    st.markdown(f'''
            <div style="background-color: black; color: white; font-weight: bold; padding: 1rem; border-radius: 10px;">
            <h4>Output</h4>
            {output}
            </div>
                </p>
                ''', unsafe_allow_html=True)
    st.success('Successful')



# Footer
st.write('')
st.write('')
st.write('')
st.write('')


st.markdown("<hr style='border: 1px dashed #ddd; margin: 2rem;'>", unsafe_allow_html=True) #Horizontal line

st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        Project by <a href="https://github.com/ChibuzoKelechi" target="_blank" style="color: white; font-weight: bold; text-decoration: none;">
         kelechi_tensor</a>
    </div>
    
    <div style="text-align: center; padding: 1rem;">
        Data from <a href="https://kaggle.com" target="_blank" style="color: lightblue; font-weight: bold; text-decoration: none;">
         Kaggle</a>
    </div>
""",
unsafe_allow_html=True)

# Peace Out :)