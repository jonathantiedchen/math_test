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
from cot import EIGHT_SHOT_PROMPT, FOUR_SHOT_PROMPT

# Streamlit UI
st.title("🧠 Math LLM Demo")
st.write("💬 Please prompt something!")
use_cot = st.toggle("Use Chain-of-Thought Prompt")

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
    "GPT-2 Small BL": "openai-community/gpt2",
    "GPT-2 Small CPT+CL+IFT": "jonathantiedchen/GPT2-Small-CPT-CL-IFT"
}

mistral_models = {
    "Mistral 7B BL": "unsloth/mistral-7b-bnb-4bit",
    "Mistral 7B CPT+CL": "jonathantiedchen/Mistral-7B-CPT-CL",
    "Mistral 7B CPT+IFT": "jonathantiedchen/MistralMath-CPT-IFT"
}


all_models = gpt_models | mistral_models



### LOAD MODELS FUNCTIONS
#LOAD MISTRAL
@st.cache_resource
def load_mistral(mistral_path, _models):
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
        _models[mistral_path] = {"tokenizer": tokenizer, "model": model}
    except Exception as e:
        st.sidebar.error(f"⚠️ Failed to load Mistral model with Unsloth: {e}")
    
    return _models

@st.cache_resource
def load_gpts(path, _models):
    try:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path).to(device)
        model.eval()
        _models[path] = {"tokenizer": tokenizer, "model": model}
    except Exception as e:
        st.sidebar.error(f"⚠️ Failed to load GPT model: {e}")
        
    return _models


###########################
#LOAD ALL MODELS EXECUTION
models = {}
with st.sidebar:
    with st.spinner("📥 Load all Models. That might take a while."):
        # Load each model using the paths from the dictionaries
        for model_path in mistral_models.values():
            models = load_mistral(model_path, models)
        for model_path in gpt_models.values():
            models = load_gpts(model_path, models)
    st.write("✅ Successfully loaded all models.")

##########################
#Select a model
model_choice = st.selectbox("Choose a model:", list(all_models.keys()))

# Get the actual model path from the dictionary
model_path = all_models[model_choice]

tokenizer = models[model_path]["tokenizer"]
model = models[model_path]["model"]

# BEGIN THE PROMPTING
# TO ADD: 
# - add drop down where user can choose gsm8k question and a button which selects a random gsm8k question
# - the selected or random gsm8k question gets put into the prompt text area
# - (if possible) add a history of the prompts similar to chat format 
prompt = st.text_area("Enter your math prompt:", "Jasper has 5 apples and eats 2 of them. How many apples does he have left?")

if use_cot: 
    if 'mistral' in model_choice.lower():
        #use 8 shot prompt
        prompt_template = EIGHT_SHOT_PROMPT
        input_text = prompt_template.format(question=prompt)

    elif 'small' in model_choice.lower() or 'gpt' in model_choice.lower(): 
        #use 4s shot prompt
        prompt_template = FOUR_SHOT_PROMPT
        input_text = prompt_template.format(question=prompt)
else: 
    input_text = prompt

if st.button("Generate Response", key="manual"):
    with st.sidebar:
        with st.spinner("🔄 Generating..."):

            # Configuration needed for all models
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            stop_criteria = SpecificStringStoppingCriteria(tokenizer, generation_util, len(input_text))
            stopping_criteria_list = StoppingCriteriaList([stop_criteria])
            
            # Statement to check model version, different model need different prompting strategy
            if 'mistral' in model_choice.lower():
                #MISTRAL PROMPTING
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, 
                        max_new_tokens=512, 
                        pad_token_id=tokenizer.eos_token_id, 
                        stopping_criteria=stopping_criteria_list
                    )
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                if generated_text.startswith(input_text):
                    response_only = generated_text[len(input_text):].strip()
                else:
                    response_only = generated_text.strip()
           
            #gpt2 small prompting        
            elif 'small' in model_choice.lower() or 'gpt' in model_choice.lower(): 
                output = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=1,
                    pad_token_id=tokenizer.eos_token_id,
                    stopping_criteria=stopping_criteria_list
                )
                generated_text = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                response_only = generated_text[len(input_text):].strip()

            else: 
                st.error("⚠️ Problems in identifying the model.")
                generated_text = "Error: Model not recognized"
                response_only = "Error: Model not recognized"

    st.subheader("🔎 Prompt")
    st.write(input_text)
    #st.subheader("🧠 Model Output")
    #st.write(generated_text)
    st.subheader("🧠 Model Output")
    st.success(response_only)