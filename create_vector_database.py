import os
import torch


from langchain.document_loaders import PyPDFLoader

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings



# read all pdfs in a directory
directory = "/home/Bhattacharya/RAG_with_Debayan/documents/"
pdfs = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pdf')]

loaders = [PyPDFLoader(pdf) for pdf in pdfs]
docs = []
for loader in loaders:
    docs.extend(loader.load())

# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)

splits = text_splitter.split_documents(docs)

"""
print(splits)

#assert False
print("Length of splits: ", len(splits))
splits = []
for doc in docs:
    # remove every split after encountering "References" or "Bibliography" or "Reference" or "Bibliographies" 

    split = text_splitter.split_documents([doc])

    print(split)

    #print("Length of split: ", len(split.page_content)) 
     
    for i, s in enumerate(split):
        print(s)
        if "References" in s or "Bibliography" in s or "Reference" in s or "Bibliographies" in s or "BIBLIOGRAPHY" in s or "REFERENCES" in s or "REFERENCE" in s or "BIBLIOGRAPHIES" in s: 
            split = split[:i]
            break
        
    break 
    splits.extend(split)

    #print(splits)


"""


#splits = text_splitter.split_documents(docs)

print(f"Number of splits after removing references: {len(splits)}")

# print split text
for i, split in enumerate(splits[-1:]):
    print("Length of split: ", len(split.page_content))
    print(f"Split {i}: {split.page_content[:]}", end="\n\n")
    

# Load chunked documents into the FAISS index
db = FAISS.from_documents(splits, 
                          HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))

db.save_local("/home/Bhattacharya/RAG_with_Debayan/documents/vectorstore/faiss_index") 
print("Saved the vectorstore to /home/Bhattacharya/RAG_with_Debayan/documents/vectorstore/faiss_index")


retriever = db.as_retriever()

