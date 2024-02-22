from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks import get_openai_callback

import streamlit as st
import os as os
import pandas as pd

__directory = r"./Data/"
__vector_directory = r"./VectorStore/"

if 'question_answer' not in st.session_state:
    st.session_state['question_answer'] = []

if 'store_name' not in st.session_state:
    st.session_state['store_name'] = ''
    
load_dotenv() # read local .env file

# function to upload file
def upload_file(file):
    # file path
    file_path = os.path.join(__directory, file.name)

    # write file on disk
    with open(file_path,"wb") as f:
      f.write(file.getbuffer())

    st.success("Saved File")
    return True

def read_and_textify_pdf(pdf):
    # concate pdf path
    pdf_path = os.path.join(__directory, pdf.name)
    # read file file and get file text
    file_reader = PDFMinerLoader(pdf_path)
    file_text = file_reader.load()

    text_splitter = RecursiveCharacterTextSplitter(
      chunk_size = int(os.environ['chunk_size']),
      chunk_overlap = int(os.environ['chunk_overlap']),
      length_function = len
    )

    chunks = text_splitter.split_text(text = file_text[0].page_content)
    return chunks

def save_vector_db(store_name, chunks):
    if (chunks):
        # file __name__
        store_path = os.path.join(__vector_directory, store_name)
        # embeddings
        if not os.path.exists(store_path):
            embedding = OpenAIEmbeddings(model = "text-embedding-3-small")
            # create vector db
            vector_db = FAISS.from_texts(chunks, embedding = embedding)
            # save vector db
            vector_db.save_local(store_path)

def get_vector_db(store_name):
    vector_db = ''
    # file __name__
    store_path = os.path.join(__vector_directory, store_name)
    # embeddings
    if os.path.exists(store_path):
        # load vector db
        vector_db = FAISS.load_local(store_path, OpenAIEmbeddings(model = "text-embedding-3-small"))

    return vector_db
        
def search_similarities():
    # input field value
    query = st.session_state.widget_txt
    
    # vector_db from args
    store_name = st.session_state.store_name
    vector_db = get_vector_db(store_name)
    
    # if query has value and 
    if query and vector_db:
        # run similarity search
        docs = vector_db.similarity_search(query = query, k = 3)
        # create llm object
        llm = OpenAI()
        # Q&A model
        chain = load_qa_chain(llm = llm, chain_type = "stuff")
        # cost of the requests
        with get_openai_callback() as cb:
            response = chain.run(input_documents = docs, question = query)
            print(cb)

        # show response
        clear_text(response)

def clear_text(response):
    st.session_state.search_text = st.session_state.widget_txt     
    st.session_state.question_answer +=  [[st.session_state.search_text, response]]    
    st.session_state.widget_txt = ""        
    
# function to display input field for ask question
def ask_question():
    # Accept user questions/query
    st.write('Enter text here:')

    col1, col2 = st.columns([2,1])
    with col1:
        st.text_input(label = 'search', key = 'widget_txt', label_visibility = 'collapsed')   
    with col2:
        st.button(label = 'search', key = 'search_btn', on_click = search_similarities)
      
# Sidebar contents
with st.sidebar:
    st.title('LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
    ''')

def main():
    st.header("Chat with PDF 💬")

    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type = 'pdf')

    # if file is selected
    if pdf is not None:
      # uploaded files
      if upload_file(pdf):
        # get chunks from uploaded file
        chunks = read_and_textify_pdf(pdf)

        # vector store name
        store_name = f"{pdf.name[:-4]}.faiss"
        # save vectore store name in session state 
        st.session_state.store_name = store_name
        # generate embeddings
        save_vector_db(store_name, chunks)
        
        # show result in data frame
        df = pd.DataFrame(st.session_state.question_answer)
        if not df.empty:
            st.write(df)
        
        # show UI for ask questions
        ask_question()        

if __name__ == '__main__':
    main()