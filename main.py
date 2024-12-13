from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from pdf_loader import load_pdf
from utils import load_env

def main():
    # Load environment variables and API key
    api_key = load_env()
    
    # Load PDF text
    pdf_text = load_pdf("data/insurance.pdf")
    
    # Split text into chunks and create Document objects
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = [Document(page_content=chunk) for chunk in splitter.split_text(pdf_text)]
    
    # Create embeddings
    embeddings = OpenAIEmbeddings(api_key=api_key)
    
    # Create FAISS vector store
    faiss_index = FAISS.from_documents(documents, embeddings)
    
    # Create retriever
    retriever = faiss_index.as_retriever()
    
    # Initialize ChatOpenAI LLM
    llm = ChatOpenAI(api_key=api_key, temperature=0)

    # Create retrieval-based QA chain
    retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True  # Include source documents in the response
    )
    
    # Run example query
    query = "What is the name of the insurance company?"
    inputs = {"query": query}  # Use a dictionary for inputs
    result = retrieval_chain.invoke(inputs)  # Use `invoke` instead of `run`
    
    # Extract the result and source documents
    answer = result["result"]
    source_documents = result["source_documents"]
    
    print("Answer:", answer)
    print("\nSource Documents:")
    for doc in source_documents:
        print(doc.page_content)

if __name__ == "__main__":
    main()
