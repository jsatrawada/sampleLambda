
import streamlit as st
import requests
import json

def main():
    st.title("Blog Generation Request - Feb 20")
    
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

if __name__ == "__main__":
    main()
