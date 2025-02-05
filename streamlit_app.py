from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import streamlit as st

llm = Ollama(model="llama2-uncensored")

@st.cache_resource
class PdfGpt():
    def __init__(self, file_path):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        chunks = text_splitter.split_documents(documents=PyMuPDFLoader(file_path=file_path).load())
        
        embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device':'cpu'},
            encode_kwargs = { 'normalize_embeddings': True }
        )
        vectorstore = FAISS.from_documents(chunks, embedding_model)
        vectorstore.save_local("vectorstore")
        
        template = """
        ### System:
        You are an respectful and honest assistant. You have to answer the user's questions using only the context \
        provided to you. If you don't know the answer, just say you don't know. Don't try to make up an answer.

        ### Context:
        {context}

        ### User:
        {question}

        ### Response:
        """

        self.hey = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            chain_type="stuff",
            return_source_documents=True, 
            chain_type_kwargs={'prompt': PromptTemplate.from_template(template) } 
        )

oracle = PdfGpt("reinventing_your_life.pdf") # PDF file name
ask = st.text_input("What's up?", key="ask", label_visibility='hidden')

A,B = st.columns([.05, .95])
C,D = st.columns([.05, .95])
with A:
    st.caption("🦙")
with C:
    st.caption("📓")

if ask not in [None, "", []]:  
    with B:
        st.markdown( llm.predict(ask) )
    with D:
        response = oracle.hey({'query': ask})
        st.markdown( response['result'] )
