import json
import os
import boto3
import requests
import streamlit as st
from langchain_aws import ChatBedrock
import json


from langchain_aws import BedrockEmbeddings
from langchain_community.llms.bedrock import Bedrock


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

## Bedrock Clients
bedrock=boto3.client(service_name="bedrock-runtime",region_name="us-east-1")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0",client=bedrock, region_name="us-east-1")

prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use atleast summarize with 
250 words with detailed explantions. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


def generate_blog(url, blog_topic):
    payload = {"blog_topic": blog_topic}
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        
        if response.status_code == 200:
            st.success("Blog generated successfully!")
            st.json(response.json())
        else:
            st.error(f"Error: {response.status_code}")
            st.write(response.text)
    except Exception as e:
        st.error(f"Request failed: {str(e)}")

def upload_pdfs_to_s3(directory, bucket_name):
    s3 = boto3.client('s3')
    
    try:
        for filename in os.listdir(directory):
            if filename.endswith(".pdf"):
                file_path = os.path.join(directory, filename)
                s3.upload_file(file_path, bucket_name, filename)
                st.success(f"Uploaded {filename} to {bucket_name}")
    except Exception as e:
        st.error(f"Upload failed: {str(e)}")

# def list_pdf_files():
#     with st.sidebar:
#         st.title("List PDF Files from Directory")
        
#         uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True, type="pdf")
        
#         if uploaded_files:
#             st.write("Uploaded PDF files:")
#             for uploaded_file in uploaded_files:
#                 st.write(uploaded_file.name)
def data_ingestion(directory):
    st.write(f"Loading PDFs from directory: {directory}")
    loader = PyPDFDirectoryLoader(directory)
    documents = loader.load()
    st.write(f"Loaded {len(documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    st.write(f"Split documents into {len(docs)} chunks")
    return docs

def load_and_split_pdfs(directory):
    if directory:
        st.write(f"Directory provided: {directory}")
        docs = data_ingestion(directory)
        st.write(f"Number of documents after splitting: {len(docs)}")
        get_vector_store(docs)
    else:
        st.error("Please enter a directory path.")
    st.success("Done")

## Vector Embedding and vector store
def get_vector_store(docs):
    if not docs:
        st.error("No documents to process.")
        return

    st.write("Generating embeddings for documents...")
    try:
        vectorstore_faiss = FAISS.from_documents(
            docs,
            bedrock_embeddings
        )
        vectorstore_faiss.save_local("faiss_index")
        st.success("Vector store created and saved locally.")
    except Exception as e:
        st.error(f"Failed to create vector store: {str(e)}")

def get_llama3():
    llm = ChatBedrock(
        model_id="us.meta.llama3-2-90b-instruct-v1:0",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region="us-east-1"
    )
    return llm

def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 100}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer=qa.invoke({"query":query})
    return answer['result']

def main():
    # st.title("Blog Generation Request - Feb 21")
    st.set_page_config("Chat PDF")
    
    st.header("Literature Retrieval Evidence Summarization")

    user_question = st.text_input("Provide the parameters to generate evidence")
    
    # # Define the API endpoint
    # url = "https://iccp1m5kmk.execute-api.us-east-1.amazonaws.com/dev/blog-generation"
    
    # # Input for blog topic (Optional)
    # blog_topic = st.text_input("Enter Blog Topic", "Machine Learning and Generative AI")
    
    # if st.button("Generate Blog"):
    #     generate_blog(url, blog_topic)
    
    # # Upload PDF files to S3
    # st.title("Upload PDF Files to S3")
    
    # directory = st.text_input("Enter Directory Path", "/path/to/pdf/files")
    # bucket_name = st.text_input("Enter S3 Bucket Name", "your-s3-bucket-name")
    
    # if st.button("Upload PDFs"):
    #     upload_pdfs_to_s3(directory, bucket_name)
    # List PDF files from directory
    with st.sidebar:
        st.title("Vector Update")
        
        directory = st.text_input("Enter Folder name with Publications ", "", key="load_pdfs_directory")
        if st.button("Load PDFs", key="load_pdfs_button"):    
            if directory:
                docs = data_ingestion(directory)
                # st.write(f"Number of documents: {len(docs)}")
                get_vector_store(docs)
                # for i, doc in enumerate(docs):
                #     st.write(f"Document {i+1}: {doc.to_json}...")  # Display first 200 characters of each document
            else:
                st.error("Please enter a directory path.")
            st.success("Done")

    if st.button("Llama3 Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings,allow_dangerous_deserialization=True)
            llm=get_llama3()
            
            #faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")
    


if __name__ == "__main__":
    main()