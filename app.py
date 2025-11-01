# app.py

import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import io

# Set page config
st.set_page_config(page_title="Text-to-Image Generator", layout="centered")

st.title("Text-to-Image Generator")
st.write("Enter a prompt and generate an image!")

# Input: text prompt
prompt = st.text_input("Enter your image description:", "")

# Input: image size
width = st.slider("Width", min_value=256, max_value=1024, value=512, step=64)
height = st.slider("Height", min_value=256, max_value=1024, value=512, step=64)

# Optional: select style
style = st.selectbox(
    "Choose style",
    ("Realistic", "Cartoon", "Anime", "Digital Art")
)

# Button to generate image
if st.button("Generate Image"):
    if not prompt:
        st.error("Please enter a prompt to generate an image.")
    else:
        with st.spinner("Generating image..."):
            try:
                # Load the model (this may take some time on first run)
                pipe = StableDiffusionPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-2-base",
                    torch_dtype=torch.float16
                )
                pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

                # Modify prompt based on selected style
                full_prompt = f"{prompt}, {style} style"

                # Generate image
                image = pipe(full_prompt, width=width, height=height).images[0]

                # Display image
                st.image(image, caption="Generated Image", use_column_width=True)

                # Download button
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="Download Image",
                    data=byte_im,
                    file_name="generated_image.png",
                    mime="image/png"
                )

            except Exception as e:
                st.error(f"An error occurred: {e}")
