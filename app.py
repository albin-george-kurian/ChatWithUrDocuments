import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from io import BytesIO


# Define headers for any API that requires authorization
headers = {
    "authorization": st.secrets["API_KEY"],
    "content-type": "application/json"
}

# Title and Description
st.title("Chat with Your Document")
st.write("Upload a PDF file to start interacting with its content. Ask questions and get detailed answers.")

# File Upload Widget
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    try:
        # Read the uploaded file into memory
        pdf_bytes = uploaded_file.read()
        pdf_file = BytesIO(pdf_bytes)

        # Extract text from PDF using PyPDF2
        st.write("Processing your document...")
        reader = PdfReader(pdf_file)
        raw_text = ""
        for page in reader.pages:
            raw_text += page.extract_text()

        if not raw_text.strip():
            raise ValueError("The PDF does not contain extractable text.")

        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        text_chunks = text_splitter.create_documents([raw_text])

        # Generate embeddings
        st.write("Generating embeddings...")
        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5", headers=headers)

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
    st.rerun()
