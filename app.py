import os
import random
import streamlit as st
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteriaList
from unsloth import FastLanguageModel, is_bfloat16_supported
from utils import SpecificStringStoppingCriteria
from cot import EIGHT_SHOT_PROMPT, FOUR_SHOT_PROMPT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generation_util = [
    "Q:",
    "</s>",
    "<|im_end|>"
]

# GPT-2 and Mistral model registry
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


### Load GSM8K once
@st.cache_resource
def load_gsm8k_dataset():
    return load_dataset("openai/gsm8k", "main")["test"]


### Load Mistral
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


### Load GPT-2
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


# Load models
st.title("🧠 Math LLM Demo")
models = {}
with st.sidebar:
    with st.spinner("📥 Load all Models. That might take a while."):
        for model_path in mistral_models.values():
            models = load_mistral(model_path, models)
        for model_path in gpt_models.values():
            models = load_gpts(model_path, models)
    st.write("✅ Successfully loaded all models.")


# Load GSM8K dataset and allow selection
st.sidebar.write("📥 Load GSM8K")
gsm8k_data = load_gsm8k_dataset()
st.sidebar.write("📊 GSM8K loaded:", len(gsm8k_data), "samples")

question_index = st.selectbox("🔢 Select GSM8K question index", range(len(gsm8k_data)))

if st.button("🎲 Pick Random Question"):
    question_index = random.randint(0, len(gsm8k_data) - 1)
    # Update the query parameters instead of calling it like a function
    st.query_params.update(question_index=question_index)

default_prompt = "Jasper has 5 apples and eats 2 of them. How many apples does he have left?"
selected_question = gsm8k_data[question_index]["question"] if question_index is not None else default_prompt


# Prompt options
st.write('##')
use_cot = st.toggle("Use Chain-of-Thought Prompt")
model_choice = st.selectbox("Choose a model:", list(all_models.keys()))
model_path = all_models[model_choice]
tokenizer = models[model_path]["tokenizer"]
model = models[model_path]["model"]

# Prompt input
prompt = st.text_area("Enter your math prompt:", selected_question)

# Generation
if st.button("Generate Response", key="manual"):
    with st.sidebar:
        with st.spinner("🔄 Generating..."):

            if use_cot:
                if 'mistral' in model_choice.lower():
                    prompt_template = EIGHT_SHOT_PROMPT
                else:
                    prompt_template = FOUR_SHOT_PROMPT
                input_text = prompt_template.format(question=prompt)
            else:
                input_text = prompt

            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            stop_criteria = SpecificStringStoppingCriteria(tokenizer, generation_util, len(input_text))
            stopping_criteria_list = StoppingCriteriaList([stop_criteria])

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=1,
                    pad_token_id=tokenizer.eos_token_id,
                    stopping_criteria=stopping_criteria_list
                )
                generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
                response_only = generated_text[len(input_text):].strip() if generated_text.startswith(input_text) else generated_text.strip()

    st.subheader("🔎 Prompt")
    st.write(input_text)
    st.subheader("🧠 Model Output")
    st.success(response_only)
