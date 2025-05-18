import os
import streamlit as st
from huggingface_hub import hf_hub_download
from unsloth import FastLanguageModel,is_bfloat16_supported
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteriaList
import importlib
import random
from datasets import load_dataset

# Streamlit UI
st.title("🧠 Math LLM Demo")
st.write("💬 Please prompt me something!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Some Specifications
generation_util = [
        "Q:",
        "</s>",
        "<|im_end|>"
    ]
mistral_path = 'jonathantiedchen/MistralMath-CPT-IFT'

#LOAD MISTRAL
@st.cache_resource
def load_mistral():
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
                            model_name=mistral_path,
                            max_seq_length=2048,
                            dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
                            load_in_4bit=True
                        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        FastLanguageModel.for_inference(model)
    except Exception as e:
        st.sidebar.error(f"⚠️ Failed to load Mistral model with Unsloth: {e}")
        
    return model, tokenizer

st.sidebar.write("📥 Load all Models.")
with st.sidebar:
    mistral, mistral_tokenizer = load_mistral()
st.sidebar.write(f"✅ Successfully loaded Mistral.")



prompt = st.text_area("Enter your math prompt:", "Jasper has 5 apples and eats 2 of them. How many apples does he have left?")

if st.button("Generate Response", key="manual"):
    with st.sidebar:
        with st.spinner("🔄 Generating..."):
            
            #MISTRAL PROMPTING
            inputs = mistral_tokenizer(prompt, return_tensors="pt").to(mistral.device)
            with torch.no_grad():
                outputs = mistral.generate(
                    **inputs, 
                    max_new_tokens=512, 
                    pad_token_id=mistral_tokenizer.eos_token_id, 
                    eos_token_id=mistral_tokenizer.eos_token_id
                )
            generated_text = mistral_tokenizer.decode(outputs[0], skip_special_tokens=True)
            if generated_text.startswith(prompt):
                response_only = generated_text[len(prompt):].strip()
            else:
                response_only = generated_text.strip()

    st.subheader("🔎 Prompt")
    st.code(prompt)
    st.subheader("🧠 Model Output")
    st.code(generated_text)
    st.subheader("✂️ Response Only")
    st.success(response_only)