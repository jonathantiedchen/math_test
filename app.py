import os

import streamlit as st
from huggingface_hub import hf_hub_download
from unsloth import FastLanguageModel,is_bfloat16_supported
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import importlib
import random
from datasets import load_dataset


#Sidebar Text
st.sidebar.write("📥 Downloading models from Hugging Face...WIP")

# Streamlit UI
st.title("🧠 Math LLM Demo")
st.write("💬 Ask me anything!")