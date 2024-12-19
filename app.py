import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Title and Description
st.title("Chat with Your Document")
st.write("Upload a PDF file to start interacting with its content. Ask questions and get detailed answers.")

# File Upload Widget
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    try:
        # Save the uploaded file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        # Load the PDF file
        st.write("Processing your document...")
        loader = PyPDFLoader("temp.pdf")
        data = loader.load()

        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        text_chunks = text_splitter.split_documents(data)

        # Generate embeddings
        st.write("Generating embeddings...")
        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

        # Create a FAISS vector store
        db = FAISS.from_documents(text_chunks, embeddings)

        # Set up the retriever
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        # Set up the LLM (Groq in this case)
        llm = ChatGroq(model_name='llama3-70b-8192')

        # Set up memory for conversation history
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=False)

        # Define the prompt template
        prompt_template = PromptTemplate(
            template='''You are a helpful assistant. Greet the user before answering.\n\n{context}\n{question}''',
            input_variables=['context', 'question']
        )

        # Create the Conversational Retrieval Chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            memory=memory,
            retriever=retriever,
            return_source_documents=False,
            output_key='answer'
        )

        # Query Input
        query = st.text_input("Ask a question about the document:")

        if query:
            # Get the response
            result = qa_chain(query)
            st.write("### Answer:")
            st.write(result['answer'])

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
else:
    st.info("Please upload a PDF file to begin.")

# Clear button to reset the interface
if st.button("Clear Chat"):
    st.experimental_rerun()
