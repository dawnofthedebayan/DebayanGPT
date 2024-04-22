import gradio as gr
import random
import time

import os
import torch

import transformers 
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


from transformers import BitsAndBytesConfig

from langchain.text_splitter import CharacterTextSplitter
from langchain.document_transformers import Html2TextTransformer
from langchain.document_loaders import AsyncChromiumLoader

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain

import nest_asyncio
#################################################################
# Tokenizer
#################################################################

custom_cache_dir = "/data/Bhattacharya/.cache"
# Create the cache directory if it doesn't exist
os.makedirs(custom_cache_dir, exist_ok=True)

#model_name='mistralai/Mistral-7B-Instruct-v0.1'
model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
model_config = transformers.AutoConfig.from_pretrained(
    model_name,
    cache_dir=custom_cache_dir
)

from transformers import LlamaTokenizerFast

#tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,cache_dir=custom_cache_dir)
# check the tokenizer
#print(tokenizer("Hello, this is a test sentence."), "AutoTokenizer")

#tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")

# check the tokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,cache_dir=custom_cache_dir)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print(tokenizer("Hello, this is a test sentence."), "LlamaTokenizerFast") 

save_directory = "/data/Bhattacharya/"
model_config.save_pretrained(save_directory)


#################################################################
# bitsandbytes parameters
#################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

#################################################################
# Set up quantization config
#################################################################
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

#################################################################
# Load pre-trained config
#################################################################
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    cache_dir=custom_cache_dir
)


def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

print(print_number_of_trainable_model_parameters(model))

text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    #task="text-generation",
    task="text-generation",
    temperature=0.1,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=3000,
)

mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

#!playwright install 
#!playwright install-deps 

#import nest_asyncio
#nest_asyncio.apply()

# Load from local vector store FAISS


db = FAISS.load_local("/home/Bhattacharya/RAG_with_Debayan/documents/vectorstore/faiss_index",HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'),allow_dangerous_deserialization=True)

retriever = db.as_retriever(search_kwargs={"k": 5})

"""
standalone_query_generation_pipeline = pipeline(
 model=model,
 tokenizer=tokenizer,
 task="text-generation",
 temperature=0.0,
 repetition_penalty=1.1,
 return_full_text=True,
 max_new_tokens=10000,
)
standalone_query_generation_llm = HuggingFacePipeline(pipeline=standalone_query_generation_pipeline)

from langchain.retrievers.multi_query import MultiQueryRetriever
retriever = MultiQueryRetriever.from_llm(retriever=db.as_retriever(), llm=standalone_query_generation_llm)
"""

# Create prompt template
prompt_template = """
### [INST] Instruction: Answer the question based on the context provided and chat history. If chat history is not provided, answer strictly based on context. If chat history and context is provided, answer based on context and chat history. Here is chat history to help:

{chat_history}

Here is the context to help:

{context} 

### QUESTION:
{question} [/INST]
 """

# Create prompt from prompt template 
prompt = PromptTemplate(
    input_variables=[ "chat_history","context", "question"],
    template=prompt_template,
)
from operator import itemgetter

# Create llm chain 
llm_chain = LLMChain(llm=mistral_llm, prompt=prompt)

rag_chain = ( 
 {"chat_history": itemgetter("chat_history") , "context": retriever, "question": itemgetter("question")}
    | llm_chain
)


#response = rag_chain.invoke("Explain the main idea of the paper \"Learning Robust Representation for Laryngeal Cancer Classification in Vocal Folds from Narrow Band Images\"")



with gr.Blocks() as demo:

    textbox = gr.Textbox(lines=5, label="Welcome to DebayanGPT", value="I am an AI chatbot created by my master Debayan Bhattacharya. I am here to help you with your queries related to my master's research papers. \n You can ask me questions like:\n \n \"What is the main idea of the paper \"Learning Robust Representation for Laryngeal Cancer Classification in Vocal Folds from Narrow Band Images\"?\", \n \n \"What is the main novelty of the paper \"Squeeze and multi-context attention for polyp segmentation?\" \n \n \"What were the main findings of the paper  Computer-Aided Diagnosis of Maxillary Sinus Anomalies: Validation and Clinical Correlation\"?\" \n \n Please note that I am a work in progress and I may not be able to answer all your questions. \n Please be patient with me. \n Thank you for your understanding. \n How can I help you today?")


    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Enter your question here and press Shift+ENTER", lines=2, placeholder="What is the main idea of the paper \"Learning Robust Representation for Laryngeal Cancer Classification in Vocal Folds from Narrow Band Images\"?")
    clear = gr.ClearButton([msg, chatbot])

    def respond(message, chat_history):
        
        chat_history_text = ["" for _ in range(len(chat_history))] 
        for i in range(len(chat_history)):
            chat_history_text[i] = "User: " + chat_history[i][0] + "\n" + "Chatbot: " + chat_history[i][1] + "\n"

        chat_history_text = "".join(chat_history_text)

        data = {"chat_history": chat_history_text, "question": message}

        bot_message = rag_chain.invoke(data)

       


        bot_message = bot_message["text"].split("[/INST]")[-1]

        print("Chat history: ", chat_history_text)
        print("Bot Message", bot_message)


        chat_history.append((message, bot_message))
        time.sleep(2)
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

    
demo.queue()
demo.launch(share=True)

