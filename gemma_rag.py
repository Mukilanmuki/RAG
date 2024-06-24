pip install langchain-community==0.2.4 langchain==0.2.3 faiss-cpu==1.8.0 unstructured==0.14.5 unstructured[pdf]==0.14.5 transformers==4.41.2 sentence-transformers==3.0.1

pip install accelerate bitsandbytes

import os

from langchain_community.llms import Ollama
from langchain.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
import torch

os.environ["HF_TOKEN"] = #paste your hugging face token

#loading the LLM
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b",quantization_config=quantization_config)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
pipeline = pipeline(
    "text-generation",
    model=model,
    max_new_tokens=100,
    tokenizer= tokenizer
)

from langchain import HuggingFacePipeline
gemma_llm = HuggingFacePipeline(
    pipeline=pipeline,
    model_kwargs={"temperature": 0.7},
)
#locate you file
loader = UnstructuredFileLoader("# add the location of the file you want to pdf you want to do Q and A")
documents = loader.load()

text_splitter = CharacterTextSplitter(separator="/n",
                                      chunk_size=1000,
                                      chunk_overlap=200)

text_chunks = text_splitter.split_documents(documents)

# loading the vector embedding model
embeddings = HuggingFaceEmbeddings()

knowledge_base = FAISS.from_documents(text_chunks, embeddings)

qa_chain = RetrievalQA.from_chain_type(
    gemma_llm,
    retriever=knowledge_base.as_retriever(),
    chain_type="stuff" )

#change the query
question = "your query"
response = qa_chain.invoke({"query": question})

output = response["result"]
start_index = output.find("Helpful Answer")
if start_index != -1:
    # Extract the text after the answer
    relevant_text = output[start_index + len("Helpful Answer"):].strip()

    # Print or use the relevant text
    print(relevant_text)
else:
    print("Answer not found in the text.")