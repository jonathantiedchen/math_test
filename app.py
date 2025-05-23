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

# MODELS
# Need a method which loads all those models and their tokenizer
# GPT2 Small, GPT2 Large and Mistral need own functions as Large uses LoRA, and Mistral uses unsloth
gpt_models = {
    "Vanilla GPT-2": "openai-community/gpt2",
    "GPT2-Small-CPT-CL-IFT": "jonathantiedchen/GPT2-Small-CPT-CL-IFT"
}

mistral_models = {
    "Mistral 7B+CPT+CL+IFT": "jonathantiedchen/MistralMath-CPT-IFT"
}

all_models = gpt_models | mistral_models

gpt_path = 'jonathantiedchen/GPT2-Small-CPT-CL-IFT'
mistral_path = 'jonathantiedchen/MistralMath-CPT-IFT'


### LOAD MODELS FUNCTIONS
#LOAD MISTRAL
@st.cache_resource
def load_mistral(mistral_path, models):
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
        models[name] = {"tokenizer": tokenizer, "model": model}
    except Exception as e:
        st.sidebar.error(f"⚠️ Failed to load Mistral model with Unsloth: {e}")
    
    return models

@st.cache_resource
def load_gpts(path):
    try: 
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path).to(device)
        model.eval()
        models[name] = {"tokenizer": tokenizer, "model": model}
    except Exception as e:
        st.sidebar.error(f"⚠️ Failed to load GPT model: {e}")
        
    return models


###########################
#LOAD ALL MODELS EXECUTION
models={}
with st.sidebar:
    with st.spinner:("📥 Load all Models. That might take a while.")
        models = load_mistral(mistral_path, models)
        models = load_mistral(gpt_path, models)
    st.sidebar.write(f"✅ Successfully loaded Mistral.")

##########################
#Select a model
model_choice = st.selectbox("Choose a model:", list(all_models.keys()))
tokenizer = models[model_choice]["tokenizer"]
model = models[model_choice]["model"]

# BEGIN THE PROMPTING
# TO ADD: 
# - add GPT model
# - prompting needs to be done different to MISTRAL
# - add a tab where gsm8k questions are randomly prompted. 
# - in the gsm8k tab the user has a drop down or other method to choose a gsm8k question
# - add a history of the prompts similar to chat format 
prompt = st.text_area("Enter your math prompt:", "Jasper has 5 apples and eats 2 of them. How many apples does he have left?")

if st.button("Generate Response", key="manual"):
    with st.sidebar:
        with st.spinner("🔄 Generating..."):

            # Configuration needed for all models
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            stop_criteria = SpecificStringStoppingCriteria(tokenizer, generation_util, len(prompt))
            stopping_criteria_list = StoppingCriteriaList([stop_criteria])
            
            # Statement to check model version, different model need different prompting strategy
            if 'mistral' in model_choice.lower():
                #MISTRAL PROMPTING
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                stop_criteria = SpecificStringStoppingCriteria(tokenizer, generation_util, len(prompt))
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
           
            #gpt2 small prompting        
            elif 'small' in model_choice.lower(): 
                output = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=1,
                    pad_token_id=tokenizer.eos_token_id,
                    stopping_criteria=stopping_criteria_list
                )
                generated_text = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                response_only = generated_text[len(prompt):].strip()

            else: 
                "⚠️ Problems in identifying the model."

    st.subheader("🔎 Prompt")
    st.write(prompt)
    st.subheader("🧠 Model Output")
    st.write(generated_text)
    st.subheader("✂️ Response Only")
    st.success(response_only)