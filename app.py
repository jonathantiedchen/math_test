import os
import streamlit as st
from huggingface_hub import hf_hub_download
from unsloth import FastLanguageModel,is_bfloat16_supported
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteriaList
import importlib
import random
from datasets import load_dataset
from utils import SpecificStringStoppingCriteria

# Streamlit UI
st.title("🧠 Math LLM Demo")
st.write("💬 Please prompt something!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Some Specifications
generation_util = [
        "Q:",
        "</s>",
        "<|im_end|>"
    ]
mistral_path = 'jonathantiedchen/MistralMath-CPT-IFT'

### LOAD MODELS FUNCTIONS
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

#LOAD ALL MODELS EXECUTION
with st.sidebar:
    with st.spinner:("📥 Load all Models. That might take a while.")
        mistral, mistral_tokenizer = load_mistral()
    st.sidebar.write(f"✅ Successfully loaded Mistral.")



# BEGIN THE PROMPTING
# TO ADD: 
# - add a tab where gsm8k questions are randomly prompted. 
# - in the gsm8k tab the user has a drop down or other method to choose a gsm8k question
# - add a history of the prompts similar to chat format 
prompt = st.text_area("Enter your math prompt:", "Jasper has 5 apples and eats 2 of them. How many apples does he have left?")

if st.button("Generate Response", key="manual"):
    with st.sidebar:
        with st.spinner("🔄 Generating..."):
            
            #MISTRAL PROMPTING
            inputs = mistral_tokenizer(prompt, return_tensors="pt").to(mistral.device)
            stop_criteria = SpecificStringStoppingCriteria(mistral_tokenizer, generation_util, len(prompt))
            stopping_criteria_list = StoppingCriteriaList([stop_criteria])
            with torch.no_grad():
                outputs = mistral.generate(
                    **inputs, 
                    max_new_tokens=512, 
                    pad_token_id=mistral_tokenizer.eos_token_id, 
                    stopping_criteria=stopping_criteria_list
                )
            generated_text = mistral_tokenizer.decode(outputs[0], skip_special_tokens=True)
            if generated_text.startswith(prompt):
                response_only = generated_text[len(prompt):].strip()
            else:
                response_only = generated_text.strip()

    st.subheader("🔎 Prompt")
    st.write(prompt)
    st.subheader("🧠 Model Output")
    st.write(generated_text)
    st.subheader("✂️ Response Only")
    st.success(response_only)