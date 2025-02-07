# importing required modules
import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from transformers import AutoTokenizer
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores.faiss import FAISS
from dotenv import load_dotenv
import tempfile

# loading the .env file to get the api key
load_dotenv()

# getting the openai key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Building RAG pipeline

# Since the pdf files have been extracted already - using md files 
def build_rag_pipeline(uploaded_file):
    # saving the uploaded file to a temporary location as docling's document converter required a path and not file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.md') as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

    # processing the saved file
    st.info("Processing the uploaded file.....")
    converter = DocumentConverter()
    docling_doc = converter.convert(source=temp_file_path).document

    # chunking using HybridChunker
    embed_model_id = 'sentence-transformers/all-MiniLM-L6-V2' # this model is used for tokenization during chunking not for gnerating embeddings
    tokenizer = AutoTokenizer.from_pretrained(embed_model_id)
    chunker = HybridChunker(tokenizer=tokenizer, max_tokens=1000)
    chunked_docs = []
    chunks = list(chunker.chunk(dl_doc=docling_doc))
    for i, chunk in enumerate(chunks):
        chunked_docs.append(Document(
            page_content=chunk.text,
            metadata = {
                "chunk_id":i,
                "source_file": uploaded_file.name,
                "token_count": len(tokenizer(chunk.text)['input_ids'])            }
        ))
    
    # generate embeddings for chunks
    st.info('Processing... ')
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    # vector store - faiss
    vc_db = FAISS.from_documents(chunked_docs,embeddings)

    # create retriever
    retriever = vc_db.as_retriever(search_kwargs={"k":3})

    # defining llm
    llm = ChatOpenAI(model_name='gpt-3.5-turbo',temperature=0)

    # define prompt
    template = """
Utilize the retrieved context below to answer each question.  If the user tries to start a conversation with greetings respond politely.
    Question: {question}
    Context: {context}
    """

    prompt = ChatPromptTemplate.from_template(template)

    # build rag pipeline
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# streamlit app
st.title("DocBot: RAG QA System for your documents")
st.info("Upload a file to build pipeline for answering questions based on the file.")

# file uploader
uploaded_file = st.file_uploader("Upload the file", type=['md'])

if uploaded_file:
    try:
        rag_chain = build_rag_pipeline(uploaded_file)
        st.success("Enter your query below")
        query = st.text_input("Enter your question:")
        if query:
            with st.spinner("Generating response...."):
                response = rag_chain.invoke(query)
            st.subheader("Answer:")
            st.write(response)
    except Exception as e:
        st.error(f"Error building RAG pipeline: {e}")
        