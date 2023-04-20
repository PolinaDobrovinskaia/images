import streamlit as st
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
from torchvision.utils import save_image

@st.cache
def main():
    # Set up the model
    model_id = "stabilityai/stable-diffusion-2-1"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    st.title("Image Generation App")

    # Get user input
    prompt = st.text_input("Enter a prompt:")

    # Generate and display the image
    if st.button("Generate"):
        image = pipe(prompt).images[0].cpu()
        save_image(image, "output.png")
        st.image("output.png", use_column_width=True)
        

if __name__ == "main":
    main()