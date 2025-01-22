
!pip install huggingface_hub
!pip install chromadb

!huggingface-cli login

# !pip install langchain_community
!pip install langchain

!pip install PyPDF2
from PyPDF2 import PdfReader
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
# import falcon
import chromadb

# get a token: https://huggingface.co/docs/api-inference/quicktour#get-your-api-token

from getpass import getpass

HUGGINGFACEHUB_API_TOKEN = getpass()

import os
os.environ["HUGGINGFACEHUB_API_TOKEN"]=HUGGINGFACEHUB_API_TOKEN

# provide the path of  pdf file/files.
pdfreader = PdfReader('/content/Generative AI with Large Language Models..pdf')

from typing_extensions import Concatenate
# read text from pdf
raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

raw_text

# We need to split the text using Character Text Split such that it sshould not increse token size
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 800,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

len(texts)

!pip install sentence-transformers

# Download embeddings from OpenAI
embeddings = HuggingFaceEmbeddings()

document_search = Chroma.from_texts(texts, embeddings)

document_search

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceHub

# repo_id = "google/flan-t5-xxl"
repo_id="tiiuae/falcon-7b"

task = 'text2text-generation'

# # Load the question-answering chain
chain = load_qa_chain(HuggingFaceHub(repo_id=repo_id, task=task), chain_type="stuff")

query = "What is LLM"
docs = document_search.similarity_search(query)
chain.run(input_documents=docs, question=query)

query = "What are Transformers"
docs = document_search.similarity_search(query)
chain.run(input_documents=docs, question=query)

