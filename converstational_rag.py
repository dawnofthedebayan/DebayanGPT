import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
from datasets import load_dataset
from peft import LoraConfig, PeftModel

from langchain.text_splitter import CharacterTextSplitter
from langchain.document_transformers import Html2TextTransformer
from langchain.document_loaders import AsyncChromiumLoader

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.llms import HuggingFacePipeline

from langchain.retrievers.multi_query import MultiQueryRetriever

from huggingface_hub import login
login(token="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")

from utils import return_tokenizer_and_model, return_template


#################################################################
# Initialise LLM model and tokenizer 

tokenizer, model = return_tokenizer_and_model(name='mistralai/Mistral-7B-Instruct-v0.2') 


#################################################################
# Build text generation pipelines


standalone_query_generation_pipeline = pipeline(
 model=model,
 tokenizer=tokenizer,
 task="text-generation",
 temperature=0.0,
 repetition_penalty=1.1,
 return_full_text=True,
 max_new_tokens=5000,
)
standalone_query_generation_llm = HuggingFacePipeline(pipeline=standalone_query_generation_pipeline)

response_generation_pipeline = pipeline(
 model=model,
 tokenizer=tokenizer,
 task="text-generation",
 temperature=0.2,
 repetition_penalty=1.1,
 return_full_text=True,
 max_new_tokens=5000,
)
response_generation_llm = HuggingFacePipeline(pipeline=response_generation_pipeline)



#################################################################
# Load from local vector store FAISS
db = FAISS.load_local("/home/Bhattacharya/RAG_with_Debayan/documents/vectorstore/faiss_index",HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'),allow_dangerous_deserialization=True)

retriever = MultiQueryRetriever.from_llm(retriever=db.as_retriever(), llm=standalone_query_generation_llm)

#print(retriever.invoke( "How was supervised contrastive learning used?"))


#assert False 

#################################################################
# Create prompt template and LLM 

from langchain.schema import format_document
from langchain_core.messages import get_buffer_string
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.memory import ConversationBufferMemory


from operator import itemgetter


CONDENSE_QUESTION_PROMPT,ANSWER_PROMPT = return_template()



DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)




# Instantiate ConversationBufferMemory
memory = ConversationBufferMemory(
 return_messages=True, output_key="answer", input_key="question"
)

# First we add a step to load memory
# This adds a "memory" key to the input object
loaded_memory = RunnablePassthrough.assign(
    chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
)
# Now we calculate the standalone question
standalone_question = {
    "standalone_question": {
        "question": lambda x: x["question"],
        "chat_history": lambda x: get_buffer_string(x["chat_history"]),
    }
    | CONDENSE_QUESTION_PROMPT
    | standalone_query_generation_llm,
}
# Now we retrieve the documents
retrieved_documents = {
    "docs": itemgetter("standalone_question") | retriever,
    "question": lambda x: x["standalone_question"],
}
# Now we construct the inputs for the final prompt
final_inputs = {
    "context": lambda x: _combine_documents(x["docs"]),
    "question": itemgetter("question"),
}
# And finally, we do the part that returns the answers
answer = {
    "answer": final_inputs | ANSWER_PROMPT | response_generation_llm,
    "question": itemgetter("question"),
    "context": final_inputs["context"]
}
# And now we put it all together!
final_chain = loaded_memory | standalone_question | retrieved_documents | answer



def call_conversational_rag(question, chain, memory):
    """
    Calls a conversational RAG (Retrieval-Augmented Generation) model to generate an answer to a given question.

    This function sends a question to the RAG model, retrieves the answer, and stores the question-answer pair in memory 
    for context in future interactions.

    Parameters:
    question (str): The question to be answered by the RAG model.
    chain (LangChain object): An instance of LangChain which encapsulates the RAG model and its functionality.
    memory (Memory object): An object used for storing the context of the conversation.

    Returns:
    dict: A dictionary containing the generated answer from the RAG model.
    """
    
    # Prepare the input for the RAG model
    inputs = {"question": question}

    # Invoke the RAG model to get an answer
    result = chain.invoke(inputs)



    # Save the current question and its answer to memory for future context
    memory.save_context(inputs, {"answer": result["answer"]})
    
    # Return the result
    return result


question = "What are the main contributions in Supervised Contrastive Learning by Debayan Bhattacharya?"

result = call_conversational_rag(question, final_chain, memory)
print("##############################################")

print(result["answer"])






