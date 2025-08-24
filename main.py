import cv2
import numpy as np 
import streamlit as st 
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from PIL import Image 

def load_model():       #load model
    model = MobileNetV2(weights = "imagenet")
    return model

def preprocess_image(image):        #takes image and transforms it for model to understand
    
    img = np.array(image)       #converts image into array of arrays with numbers (pixels)
    img = cv2.resize(img,(224,224)) #resizes image
    
    img = preprocess_input(img.astype(np.float32))
    img = np.expand_dims(img, axis = 0) #converts image into format with multiple images
    
    return img

def classify_image(model,image):
    try:
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)        #make predictions
        
        #converts numeric predictions into strings; takes top 5 predictions; [0] for 1 response
        decoded_predictions = decode_predictions(predictions, top=5)[0]             
        return decoded_predictions
    
    except Exception as e:      #error case
        st.error(f"Error classifying image: {str(e)}")
        return None

def main():     #function for streamlit UI
    st.set_page_config(page_title = "AI Image Classifier", page_icon = "🏞️", layout = "centered")
    
    st.title("AI Image Classifier")
    st.write("Upload an image and let's see if AI can inform you what it is. Please be aware that this classifier leverages MobileNetV2 which is a previously trained light weight machine learning model. It responds quickly but it has limited accuracy due to limited training. For a list of things it can identify, check out the following link: https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json")
    
    @st.cache_resource
    def load_cached_model():        #uses cache; fastens loading previously loaded models
        return load_model()
    
    model = load_cached_model()
    
    uploaded_file = st.file_uploader("Choose an image from your device (jpg, png, jpeg):", type = ["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        btn = st.button("Classify Image")   #button requesting model to analyze
        
        if btn:
            with st.spinner("Analyzing Image..."):
                image = Image.open(uploaded_file)
                predictions = classify_image(model, image)     #calls on predictions
                
                if predictions:
                    st.subheader("Predictions")
                    for _, label, score in predictions:     #format of predictions (1st value doesn't matter)
                        st.write(f"**{label}**: {score:.2%}")       #shows confidence in prediction
                        
            
if __name__ == "__main__":
    main()
    
    
