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
import json 

__directory=r"./data/"
__vector_directory=r"./vector_store/"

load_dotenv() # read local .env file

if 'question_answer' not in st.session_state:
    st.session_state['question_answer']=[ {"role": "system", "content": "You are a helpful assistant, please get me related information to the query I have posted."}]

if 'store_name' not in st.session_state:
    st.session_state['store_name']=''
    
# function to upload file
def upload_file(file):
    # file path
    file_path=os.path.join(__directory, file.name)

    # write file on disk
    with open(file_path,"wb") as f:
      f.write(file.getbuffer())

    st.success("Saved File")
    return True

def read_and_textify_pdf(pdf):
    # concate pdf path
    pdf_path=os.path.join(__directory, pdf.name)
    # read file file and get file text
    file_reader=PDFMinerLoader(pdf_path)
    file_text=file_reader.load()

    text_splitter=RecursiveCharacterTextSplitter(
      chunk_size=int(os.environ['chunk_size']),
      chunk_overlap=int(os.environ['chunk_overlap']),
      length_function=len
    )

    chunks=text_splitter.split_text(text=file_text[0].page_content)
    return chunks

def save_vector_db(store_name, chunks):
    if (chunks):
        # file __name__
        store_path=os.path.join(__vector_directory, store_name)
        # embeddings
        if not os.path.exists(store_path):
            embedding=OpenAIEmbeddings(model=os.environ['OPENAI_API_MODEL'])
            # create vector db
            vector_db=FAISS.from_texts(chunks, embedding=embedding)
            # save vector db
            vector_db.save_local(store_path)

def get_vector_db(store_name):
    vector_db=''
    # file __name__
    store_path=os.path.join(__vector_directory, store_name)
    # embeddings
    if os.path.exists(store_path):
        # load vector db
        vector_db=FAISS.load_local(store_path, OpenAIEmbeddings(model=os.environ['OPENAI_API_MODEL']))

    return vector_db
        
def search_similarities(search):
    # store name from session state
    store_name=st.session_state.store_name
    # get vector_db
    vector_db=get_vector_db(store_name)
    
    # if query has value and 
    if search and vector_db:
        query = json.dumps(search, separators=(',', ':'))        
        st.write(query)        
        # run similarity search
        docs=vector_db.similarity_search(query=query, k=3)
        # create llm object
        llm=OpenAI(temperature=0.2)
        # Q&A model
        chain=load_qa_chain(llm=llm, chain_type="stuff")
        # cost of the requests
        with get_openai_callback() as cb:
            response=chain.run(input_documents=docs, question=search)
            print(cb)
            
        # Display user message in chat message container
        st.chat_message("assistant").markdown(response)
        # Add assistant response to chat history
        st.session_state.question_answer.append({"role": "assistant", "content": response})
      
# function to display input field for ask question
def question_answer():    
    # Display chat messages from history on app rerun
    for qna in st.session_state.question_answer:
        if (qna["role"]=="user" or qna["role"]=="assistant"):
            with st.chat_message(qna["role"]):
                st.markdown(qna["content"])

    # React to user input
    if prompt := st.chat_input("Enter text here"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.question_answer.append({"role": "user", "content": prompt})
        # search for similarities
        search_similarities(st.session_state.question_answer)
      
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
    pdf=st.file_uploader("Upload your PDF", type='pdf')

    # if file is selected
    if pdf is not None:
      # uploaded files
      if upload_file(pdf):
        # get chunks from uploaded file
        chunks=read_and_textify_pdf(pdf)
        # vector store name
        store_name=f"{pdf.name[:-4]}.faiss"
        # save vectore store name in session state 
        st.session_state.store_name=store_name
        # generate embeddings
        save_vector_db(store_name, chunks)        
        # show UI for ask questions
        question_answer()        

if __name__ == '__main__':
    main()