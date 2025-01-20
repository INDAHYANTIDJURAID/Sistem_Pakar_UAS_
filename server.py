from flask import Flask, request, jsonify
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

app = Flask(__name__)

# Inisialisasi komponen RAG
load_dotenv()

def initialize_rag():
    loader = PyPDFLoader("data.pdf")
    data = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = text_splitter.split_documents(data)
    
    vectorstore = Chroma.from_documents(
        documents=docs, 
        embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    )
    
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Inisialisasi model
retriever = initialize_rag()
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None
)

# Template prompt dengan memory
system_message = """
Anda adalah asisten untuk SISTEM PAKAR DIAGNOSIS GANGGUAN MENSTRUASI 
MENGGUNAKAN METODE NATURAL LANGUAGE PROCESSING (NLP).
Gunakan potongan konteks yang diberikan berikut untuk menjawab
pertanyaan terkait. Jika Anda tidak tahu jawabannya, katakan bahwa Anda
tidak tahu. Gunakan maksimal tiga kalimat dan berikan jawaban secara ringkas.

Konteks: {context}
"""

prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(system_message),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)

# Inisialisasi memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Buat chain dengan memory
rag_chain = create_retrieval_chain(retriever, create_stuff_documents_chain(
    llm=llm,
    prompt=ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{input}")
    ])
))

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        
        # Dapatkan konteks dari RAG
        rag_response = rag_chain.invoke({"input": user_message})
        
        # Gunakan memory untuk mendapatkan riwayat chat
        memory_variables = memory.load_memory_variables({})
        chat_history = memory_variables.get("chat_history", [])
        
        # Buat response menggunakan LLM dengan konteks dan history
        response = llm.invoke(
            prompt.format(
                context=rag_response["answer"],
                chat_history=chat_history,
                question=user_message
            )
        )
        
        # Simpan percakapan ke memory
        memory.save_context(
            {"question": user_message},
            {"answer": response.content}
        )
        
        return jsonify({
            "status": "success",
            "response": response.content
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)