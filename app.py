import pandas as pd
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import timm
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)
import streamlit as st
# ------------------------------------------------------------------------------
# 4) Streamlit app
# ------------------------------------------------------------------------------
def run_app():
    # load artifacts
    tokenizer = DistilBertTokenizerFast.from_pretrained('text_model')
    text_model = DistilBertForSequenceClassification.from_pretrained('text_model')
    image_model = timm.create_model('mobilenetv3_small_100', pretrained=False, num_classes=4)
    image_model.load_state_dict(torch.load('image_model.pt', map_location='cpu'))
    image_model.eval()

    st.title("MediScout Pakistan")
    st.sidebar.header("Patient Input")
    symptoms = st.sidebar.text_area("Symptoms")
    img_file = st.sidebar.file_uploader("Image", type=['jpg','png'])
    # predict text
    if symptoms:
        enc = tokenizer(symptoms, return_tensors='pt', truncation=True, padding=True)
        out = text_model(**enc).logits.softmax(-1).detach().numpy()[0]
        cond_idx = np.argmax(out)
        cond = list(pd.read_csv('synthetic_data.csv')['condition'].unique())[cond_idx]
        st.write("**Condition:**", cond, f"({out[cond_idx]*100:.1f}%)")
    # predict image
    if img_file:
        img = Image.open(img_file).convert('RGB')
        tf = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        x = tf(img).unsqueeze(0)
        probs = image_model(x).softmax(-1).detach().numpy()[0]
        img_classes = ['dengue_rash','measles_rash','fungal_rash','no_rash']
        idx = np.argmax(probs)
        st.write("**Image class:**", img_classes[idx], f"({probs[idx]*100:.1f}%)")

# ------------------------------------------------------------------------------
# Main entry
# ------------------------------------------------------------------------------
if __name__=='__main__':
        run_app()