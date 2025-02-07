# importing required modules
import os
import streamlit as st
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

load_dotenv()

# getting the openai key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Building RAG pipeline for multiple files
def build_rag_pipeline(uploaded_files):
    all_chunked_docs = []

    # iterate over all uploaded files
    for uploaded_file in uploaded_files:
        # save each uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.md') as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

        converter = DocumentConverter()
        docling_doc = converter.convert(source=temp_file_path).document

        # chunking using HybridChunker
        embed_model_id = 'sentence-transformers/all-MiniLM-L6-V2'  # tokenization model for chunking
        tokenizer = AutoTokenizer.from_pretrained(embed_model_id)
        chunker = HybridChunker(tokenizer=tokenizer, max_tokens=1000)
        chunks = list(chunker.chunk(dl_doc=docling_doc))

        for i, chunk in enumerate(chunks):
            all_chunked_docs.append(Document(
                page_content=chunk.text,
                metadata={
                    "chunk_id": i,
                    "source_file": uploaded_file.name,
                    "token_count": len(tokenizer(chunk.text)['input_ids'])
                }
            ))

    # generate embeddings for all chunks
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    # vector store - faiss
    vc_db = FAISS.from_documents(all_chunked_docs, embeddings)

    # create retriever
    retriever = vc_db.as_retriever(search_kwargs={"k": 3})

    # defining llm
    llm = ChatOpenAI(model_name='gpt-4o', temperature=0)

    # define prompt
    template = """
Utilize the retrieved context below to answer each question. If the user tries to start a conversation with greetings respond politely. Keep the responses concise and avoid repeating information.
    Question: {question}
    Context: {context}

    Answer:
    """

    prompt = ChatPromptTemplate.from_template(template)

    # build RAG pipeline
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

with st.sidebar:
    st.title("DocBot Settings")
    st.info("Upload your files and start interacting!")
    st.markdown("🔧 **Built with AI-powered solutions**")

# main app
st.title("DocBot: RAG QA System for Your Documents")
st.info("Upload your files to create a personalized document QA system.")

# uploader
uploaded_files = st.file_uploader("Upload files", type=['md'], accept_multiple_files=True)

if uploaded_files:
    st.success(f"Uploaded {len(uploaded_files)} files:")
    for file in uploaded_files:
        st.write(f"📄 {file.name} - {len(file.getvalue()) / 1024:.2f} KB")

    # pipeline for rrag
    try:
        with st.spinner("Building RAG pipeline..."):
            rag_chain = build_rag_pipeline(uploaded_files)
        st.success("RAG pipeline built successfully! Start chatting with your documents.")
        # interface for chat
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []

        user_input = st.text_input("Your question:", key="user_input")

        if user_input:
            response = rag_chain.invoke(user_input)
            cleaned_response = response.replace(f"Question: {user_input}", "").strip()

            st.session_state['chat_history'].append((user_input, cleaned_response))

        if st.session_state['chat_history']:
            st.subheader("Chat History")
            for question, answer in st.session_state['chat_history']:
                with st.container():
                    st.markdown(f"**You:** {question}")
                    st.markdown(f"💬 **DocBot:** {answer}")

        # reset button
        if st.button("Reset Chat History"):
            st.session_state['chat_history'] = []
            st.success("Chat history reset!")

    except Exception as e:
        st.error(f"Error building RAG pipeline: {e}")
else:
    st.warning("Please upload files to start.")
