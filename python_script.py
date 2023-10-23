# RUN 
# pip install pinecone-client langchain sentence-transformers openai pypdf IProgress -q
# 01: CONFIGURE
http_proxy='http://lgn304-v304:53128'
PINECONE_API_KEY='20163887-a4fa-44e7-98d2-ab1eb38937f6'
PINECONE_API_ENV='gcp-starter'
PINECONE_index_name="cjz-medical"
MODEL_ID="/work/u00cjz00/slurm_jobs/github/models/Llama-2-7b-chat-hf"
PDF_DIR="data"

# 02: Load LIBRARY
import os, timeit, sys
import pinecone
import transformers
import torch
import warnings
from pinecone.core.client.configuration import Configuration as OpenApiConfiguration
from langchain import PromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer

warnings.filterwarnings('ignore')


# 03: Extract Data From the PDF File
def load_pdf_file(data):
    loader= DirectoryLoader(data, glob="*.pdf",loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents

extracted_data=load_pdf_file(data='data/')
#print(extracted_data)

# 04: Split the Data into Text Chunks
def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks=text_split(extracted_data)
#print("Length of Text Chunks", len(text_chunks))

# 05. Download the Embeddings from Hugging Face
def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings

start = timeit.default_timer()
embeddings = download_hugging_face_embeddings()

query_result = embeddings.embed_query("Hello world")
#print("Length", len(query_result))

# 06. pinecone
openapi_config = OpenApiConfiguration.get_default_copy()
openapi_config.proxy = http_proxy
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV,openapi_config=openapi_config)
pinecone.list_indexes()

# 07. If we already have an index we can load it like this
docsearch=Pinecone.from_existing_index(PINECONE_index_name, embeddings)
query = "What are Allergies"
docs=docsearch.similarity_search(query, k=3)
#print("Result", docs)

# 08. Load model
tokenizer=AutoTokenizer.from_pretrained(MODEL_ID)
pipeline=transformers.pipeline(
    "text-generation",
    model=MODEL_ID,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_length=1000,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
    )
llm=HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature':0})

# 9. prompt_template
prompt_template="""
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}

# 10. question
#qa=RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={'k': 2}),return_source_documents=True, chain_type_kwargs=chain_type_kwargs)
#query="What are Allergies"
#print("Response",qa.run(query))

# 11. RUN
qa=RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={'k': 2}),return_source_documents=True, chain_type_kwargs=chain_type_kwargs)
while True:
    user_input=input(f"Input Prompt:")
    if user_input=='exit':
        print('Exiting')
        sys.exit()
    if user_input=='':
        continue
    result=qa({"query": user_input})
    print("Response : ", result["result"])
    print("Source Documents : ", result["source_documents"])

end=timeit.default_timer()
print(f"Time to retrieve response: {end-start}")
