from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.schema import Document  # Import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils import load_env
from pdf_loader import load_pdf
import os
import sys

# Ensure the script can find local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load API key
api_key = load_env()

def main():
    # Load PDF text
    pdf_text = load_pdf("D:/rag/insurance.pdf")
    
    # Split text into chunks and create Document objects
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = [Document(page_content=chunk) for chunk in splitter.split_text(pdf_text)]
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vector_store = Chroma.from_documents(documents, embeddings)
    
    # Initialize RetrievalQA
    retriever = vector_store.as_retriever()
    llm = OpenAI(api_key=api_key)
    qa_chain = RetrievalQA(llm=llm, retriever=retriever)
    
    # Example Query
    query = "What is the name of the insurance company?"
    result = qa_chain.run(query)
    print("Answer:", result)

if __name__ == "__main__":
    main()
