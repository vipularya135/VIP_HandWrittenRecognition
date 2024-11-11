import streamlit as st
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from googletrans import Translator

# Load the processor and model
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# Initialize the translator
translator = Translator()

# Streamlit app
st.title("Handwritten Text Recognition")
st.write("Upload an image containing handwritten text:")

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    try:
        # Preprocess the image
        pixel_values = processor(image, return_tensors="pt").pixel_values
        
        # Generate text from the image
        generated_ids = model.generate(pixel_values)
        
        # Decode the generated text
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Display recognized text
        st.subheader("Recognized Text:")
        st.write(generated_text)
        
        # Translate to Kannada
        translated_text = translator.translate(generated_text, dest='kn').text
        
        # Display translated text
        st.subheader("Translated Text (Kannada):")
        st.write(translated_text)
        
    except Exception as e:
        st.error("An error occurred during text recognition or translation.")
        st.write(str(e))
