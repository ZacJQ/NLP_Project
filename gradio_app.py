
import os
os.environ['HF_HOME'] = r"D:\NLP_Project\HuggingFace"
os.environ['HOMEDRIVE'] = "D:"

import gradio as gr
import cv2
import requests
import pandas as pd
import os
import torch
import time
from PIL import Image
from io import BytesIO
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
from transformers import BitsAndBytesConfig
from transformers import GenerationConfig
from peft import LoraConfig
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics2ForConditionalGeneration
from transformers import AutoModelForSeq2SeqLM,Idefics2ForConditionalGeneration, Idefics2Processor,BitsAndBytesConfig
import peft
from peft import PeftModel, PeftConfig
from peft import LoftQConfig, LoraConfig, get_peft_model, AutoPeftModel
import transformers



model_trans_sr_en = "facebook/nllb-200-distilled-1.3B"
model_vqa_name = "HuggingFaceM4/idefics2-8b"
cache_dir = r"D:\NLP_Project\HuggingFace"
model_name = "ZacJQ/idefics2-8b-docvqa-finetuned-museum-v2"
peft_model_id = "ZacJQ/idefics2-8b-docvqa-finetuned-museum-v2"
model_trans_en_sr = ""
image_split = False
max_len_token_trans = 800
context_window_turns = 5
no_turns = 1
global chat_history
chat_history = []
source_lang = "mar_Deva"
language = ['Bhojpuri', 'Gujarati', 'Hindi', 'Marathi', 'Urdu', 'English']


"""
Initializes the models
"""
global device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    quantization_config = BitsAndBytesConfig(
                                            load_in_4bit=True,
                                            bnb_4bit_quant_type="nf4",
                                            bnb_4bit_use_double_quant=True,
                                            bnb_4bit_compute_dtype=torch.float16
                                            )
else:
    quantization_config = BitsAndBytesConfig(
                                            load_in_4bit=True,
                                            bnb_4bit_quant_type="nf4",
                                            bnb_4bit_use_double_quant=True,
                                            bnb_4bit_compute_dtype=torch.float32
                                            )
# model_base = AutoModelForVision2Seq.from_pretrained(model_vqa_name,
#                                                         torch_dtype=torch.float16,
#                                                         quantization_config=quantization_config,
#                                                         cache_dir=cache_dir,
#                                                         )


config = PeftConfig.from_pretrained(peft_model_id)
model_base = Idefics2ForConditionalGeneration.from_pretrained(config.base_model_name_or_path, 
                                                    cache_dir=cache_dir,
                                                    torch_dtype=torch.float16,
                                                    quantization_config=quantization_config,
                                                    )
# model_vqa = AutoPeftModel.from_pretrained(model_base, 
#                                       peft_model_id)

processor_vqa = Idefics2Processor.from_pretrained(config.base_model_name_or_path,
                                                # model_name,
                                                do_image_splitting=False,
                                                cache_dir=cache_dir 
                                                )


model_vqa = model_base
model_vqa.add_adapter(adapter_config=config, adapter_name="Finetuned")
model_vqa.enable_adapters()
print(model_vqa.active_adapter())
tokenizer_sr_en = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B", cache_dir=cache_dir)
model_trans_sr_en = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B", quantization_config=quantization_config ,cache_dir=cache_dir)
model_language = "eng_Latn"


def get_last_n_conversation_turns(message: dict, no_turns: int):
    """
    Retrieve the last n conversation turns from the given messages.

    Args:
    messages (list): List of conversation messages.
    n (int): Number of conversation turns to retrieve.

    Returns:
    list: List of the last n conversation turns.
    """
    no_turns = min(no_turns, len(message))
    conversation_turns = message[-no_turns:]
    return conversation_turns

def get_text(grad_message: dict)-> str:
    """
    Returns the current message from the user User
    """
    user_input = grad_message['text']
    return user_input

def get_image(grad_message: dict)-> tuple[list,int]:
    """
    Returns the list of images and no. of images from the User
    """
    user_image = grad_message['files']
    print(user_image)   # added for debugging
    no_image = len(user_image)
    return (user_image, no_image)

def get_translation(text: str, source_lang: str) -> str:
    """
    Converts input query into the target LLM language (English)
    """
    source_lang = "mar_Deva"  # Hard coded for now
    task = "translation"  # Hard coded for now
    translator = pipeline(task,
                          model= model_trans_sr_en,
                          tokenizer= tokenizer_sr_en,
                          src_lang=source_lang,
                          tgt_lang= model_language,
                          max_length = max_len_token_trans
                          )
    output = translator(text)
    trans_text_sr_en = output[0]["translation_text"]
    trans_text_sr_en = trans_text_sr_en
    print("Marathi to english - ",trans_text_sr_en)
    return trans_text_sr_en

def return_translation(text:str ,source_lang: str)-> str:
    """
    Converts LLM output (English) to the original language
    """
    source_lang = "mar_Deva"  # Hard coded for now
    task = "translation"  # Hard coded for now
    translator = pipeline(task,
                          model= model_trans_sr_en,
                          tokenizer= tokenizer_sr_en,
                          src_lang=model_language,
                          tgt_lang=source_lang ,
                          max_length = max_len_token_trans
                          )
    output = translator(text)
    trans_text_en_sr = output[0]["translation_text"]
    print("Englsih to Marathi-",trans_text_en_sr)
    return trans_text_en_sr

def get_template_user(grad_message: dict , chat_history: list)-> list:
    """
    Converts the input message from user into template and appends to chat history
    """
    text_sr = get_text(grad_message)
    text = get_translation(text_sr, "mar_Deva")

    image_list, no_image = get_image(grad_message)
    if no_image == 0:
        chat_history.append({"role": "user", "content": [{"type": "text", "text": text},]})
    else:
        chat_history.append({"role": "user","content": [{"type": "image"},{"type": "text", "text": text},]})
    return chat_history

def get_template_assistant(output_llm: str, chat_history: list)-> list:
    """
    Converts the LLM output into the the given template and appends to chat history
    """
    chat_history.append({"role": "assistant", "content": [{"type": "text", "text": output_llm},]})
    return chat_history

def give_output(output: list)-> str:
    """

    """
    temp = output[0]
    extra = "Assistant:"
    assistant_reply = temp.split(extra)
    return assistant_reply[-1]

def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)

def get_chat_history(history: list)-> list:
    
    return chat_history

def chat_engine(chat_history: list,image: Image, max_new_token: int) -> str:
    """
    Generates assistant replies to given input
    """

    gen_config = GenerationConfig(do_sample= True, temperature= 1.15, num_beams=3, repetition_penalty= 1.4, top_p=0.97)
    prompt = processor_vqa.apply_chat_template(chat_history,
                                                    add_generation_prompt=True)
    if image!= None:
        inputs = processor_vqa(text=prompt,
                                    images=[image],
                                    return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        start = time.time()
        generated_ids = model_vqa.generate(**inputs, max_new_tokens=max_new_token, generation_config=gen_config)
        print("time for generations:", (time.time() - start))
        print("max memory allocated:", (torch.cuda.max_memory_allocated())/1024*1024)
        print("number of tokens generated:", len(generated_ids[:,
                                                            inputs["input_ids"].size(1):][0]
                                                                ))
        output = processor_vqa.batch_decode(generated_ids, skip_special_tokens=True)
        # print(processor_vqa.batch_decode(generated_ids, skip_special_tokens=True))
        output_reply = give_output(output)
        print(output_reply)
        return output_reply
    else:
        inputs = processor_vqa(text=prompt,
                                    return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        start = time.time()
        generated_ids = model_vqa.generate(**inputs, max_new_tokens=max_new_token, generation_config=gen_config)
        print("time for generations:", (time.time() - start))
        print("max memory allocated:", (torch.cuda.max_memory_allocated())/1024*1024)
        print("number of tokens generated:", len(generated_ids[:,
                                                            inputs["input_ids"].size(1):][0]
                                                                ))
        output = processor_vqa.batch_decode(generated_ids, skip_special_tokens=True)
        # print(processor_vqa.batch_decode(generated_ids, skip_special_tokens=True))
        output_reply = give_output(output)
        print(output_reply)
        return output_reply   # changed this for testing
    

def gradio_interface_llm(current_message_gradio: dict, history: any, language: str):
    """
    Uses all functions defined above and 
    """
    global chat_history
    try:
        if len(chat_history) == 0:
            input = get_template_user(current_message_gradio, chat_history)
            input = get_last_n_conversation_turns(input,7)
            image, no_image = get_image(current_message_gradio)
            if no_image == 0:
                output_en = chat_engine(input, None ,max_new_token=512)
                output = return_translation(output_en, source_lang)
            else:
                output_en = chat_engine(input, image, max_new_token=512)
                output = return_translation(output_en, source_lang)
            history = get_template_assistant(output_en,chat_history)    # changed this for testing get_template_assistant(output,chat_history) 
            chat_history = history
            print("1")
        else:
            input = get_template_user(current_message_gradio, chat_history)
            input = get_last_n_conversation_turns(input,7)
            image, no_image = get_image(current_message_gradio)
            if no_image == 0:
                output_en = chat_engine(input, None ,max_new_token=512)
                output = return_translation(output_en, source_lang)
            else:
                output_en = chat_engine(input, image, max_new_token=512)
                output = return_translation(output_en, source_lang)
            history = get_template_assistant(output_en,chat_history)   # changed this for testing get_template_assistant(output,chat_history) 
            chat_history = history
            print("2")
        for i in range(len(output)):   # changed this for testing
            time.sleep(0.05)
            yield output[:i+1]
    except Exception as e:
        chat_history = []
        if len(chat_history) == 0:
            input = get_template_user(current_message_gradio, chat_history)
            input = get_last_n_conversation_turns(input,7)
            image, no_image = get_image(current_message_gradio)
            if no_image == 0:
                output_en = chat_engine(input, None ,max_new_token=512)
                output = return_translation(output_en, source_lang)
            else:
                output_en = chat_engine(input, image, max_new_token=512)
                output = return_translation(output_en, source_lang)
            history = get_template_assistant(output_en,chat_history)    # changed this for testing get_template_assistant(output,chat_history) 
            chat_history = history
            print("1")
        else:
            input = get_template_user(current_message_gradio, chat_history)
            input = get_last_n_conversation_turns(input,7)
            image, no_image = get_image(current_message_gradio)
            if no_image == 0:
                output_en = chat_engine(input, None ,max_new_token=512)
                output = return_translation(output_en, source_lang)
            else:
                output_en = chat_engine(input, image, max_new_token=512)
                output = return_translation(output_en, source_lang)
            history = get_template_assistant(output_en,chat_history)   # changed this for testing get_template_assistant(output,chat_history) 
            chat_history = history
            print("2")
        for i in range(len(output)):   # changed this for testing
            time.sleep(0.05)
            yield output[:i+1]


demo = gr.ChatInterface(fn=gradio_interface_llm, 
                        # chatbot = gr.Chatbot([],
                        #             elem_id="chatbot",
                        #             bubble_full_width=False), 
                        # examples=[{"text": "Hello, tell me about Chatrapti Shivaji Maharaj",
                        #            "file":[]}, 
                        #            {"text": "Could you give me a brief description about this article",
                        #            "file":[]}, 
                        #            {"text": "Tell me more about the atifact",
                        #            "file":[]}, 
                        #           ],
                        additional_inputs= gr.Dropdown(choices=language, 
                                                       multiselect=False,
                                                       max_choices=1
                                                       ),
                        chatbot = gr.Chatbot([],
                                    show_copy_button=True,
                                    show_share_button=True,
                                    elem_id="chatbot",
                                    likeable=True,
                                    bubble_full_width=True),
                        textbox= gr.MultimodalTextbox(file_types=['image'], 
                                                      info="Type your query here"),
                        title="Chatbot - VQA for CSMVS", 
                        multimodal=True)

demo.launch(share=True)