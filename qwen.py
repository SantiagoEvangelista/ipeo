# chat_qwen3_32b.py
import os
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def load_model():
    model_name = "Qwen/Qwen3-32B"
    os.makedirs("offload", exist_ok=True)  # Needed if you spill to CPU

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_threshold=6.0,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",         # Automatically split layers across GPUs + RAM
        offload_folder="offload",   # Use RAM and disk as needed
        trust_remote_code=True
    )

    return tokenizer, model

tokenizer, model = load_model()

def chat(user_message, history):
    # Build the input prompt with history
    prompt = ""
    for user, assistant in history:
        prompt += f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant}<|im_end|>\n"
    prompt += f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
    history.append((user_message, response.strip()))
    return history, history

with gr.Blocks() as demo:
    gr.Markdown("# Chat with Qwen3-32B\nRunning locally (4-bit quantized)")
    chatbot = gr.Chatbot()
    user_input = gr.Textbox(placeholder="Type your message...", show_label=False)
    user_input.submit(chat, [user_input, chatbot], [chatbot, chatbot])
    demo.launch(server_name="0.0.0.0", server_port=7860)
