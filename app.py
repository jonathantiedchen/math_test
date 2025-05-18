import os

import streamlit as st
from huggingface_hub import hf_hub_download
from unsloth import FastLanguageModel,is_bfloat16_supported
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import importlib
import random
from datasets import load_dataset

path = 'jonathantiedchen/MistralMath-CPT-IFT'

#Sidebar Text
st.sidebar.write("📥 Downloading models from Hugging Face...")
try:
    model, tokenizer = FastLanguageModel.from_pretrained(
                        model_name=path,
                        max_seq_length=2048,
                        dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
                        load_in_4bit=True
                    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    FastLanguageModel.for_inference(model)
    st.sidebar.write("Model Downloaded Successfully")
except Exception as e:
    st.sidebar.error(f"⚠️ Failed to load Mistral model with Unsloth: {e}")

# Streamlit UI
st.title("🧠 Math LLM Demo")
st.write("💬 Ask me anything!")