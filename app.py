
import streamlit as st
import requests
import json

def main():
    st.title("Blog Generation Request - Feb 21")
    
    # Define the API endpoint
    url = "https://iccp1m5kmk.execute-api.us-east-1.amazonaws.com/dev/blog-generation"
    
    # Input for blog topic (Optional)
    blog_topic = st.text_input("Enter Blog Topic", "Machine Learning and Generative AI")
    
    if st.button("Generate Blog"):
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
    
    # Upload PDF files to S3
    st.title("Upload PDF Files to S3")
    
    directory = st.text_input("Enter Directory Path", "/path/to/pdf/files")
    bucket_name = st.text_input("Enter S3 Bucket Name", "your-s3-bucket-name")
    
    if st.button("Upload PDFs"):
        s3 = boto3.client('s3')
        
        try:
            for filename in os.listdir(directory):
                if filename.endswith(".pdf"):
                    file_path = os.path.join(directory, filename)
                    s3.upload_file(file_path, bucket_name, filename)
                    st.success(f"Uploaded {filename} to {bucket_name}")
        except Exception as e:
            st.error(f"Upload failed: {str(e)}")

if __name__ == "__main__":
    main()
