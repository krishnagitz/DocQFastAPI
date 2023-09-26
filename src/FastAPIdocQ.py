import os
import sys
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from fastapi import FastAPI, HTTPException

load_dotenv('.env')
print(os.getenv('OPENAI_API_KEY'))
app = FastAPI()


documents = []
# Create a List of Documents from all of our files in the ./docs folder
for file in os.listdir("docs"):
    if file.endswith(".pdf"):
        pdf_path = "./docs/" + file
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
    elif file.endswith('.docx') or file.endswith('.doc'):
        doc_path = "./docs/" + file
        loader = Docx2txtLoader(doc_path)
        documents.extend(loader.load())
    elif file.endswith('.txt'):
        text_path = "./docs/" + file
        print(text_path)
        loader = TextLoader(text_path)
        documents.extend(loader.load())

# Split the documents into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
documents = text_splitter.split_documents(documents)

# Convert the document chunks to embedding and save them to the vector store
vectordb = Chroma.from_documents(documents, embedding=OpenAIEmbeddings(), persist_directory="./data")
vectordb.persist()

# create our Q&A chain
pdf_qa = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(temperature=0.7, model_name='gpt-3.5-turbo'),
    retriever=vectordb.as_retriever(search_kwargs={'k': 6}),
    return_source_documents=True,
    verbose=False
)

@app.get("/")
async def root():
    return {"message": "Hello, hackathon team!"}
chat_history = []

# Define an endpoint to receive questions and return answers
MAX_TOKENS = 4097
@app.post("/ask")
async def ask_question(question: str):
    while True:
        query = question
        if query == "exit" or query == "quit" or query == "q" or query == "f":
            print('Exiting')
            sys.exit()
            # Calculate the total tokens in the conversation so far
        total_tokens = sum(len(query) + len(answer) for query, answer in chat_history)

        # Check if adding the new question exceeds the token limit
        if total_tokens + len(query) > MAX_TOKENS:
            # If it does, remove old messages until it fits
            while total_tokens + len(query) > MAX_TOKENS:
                removed_query, removed_answer = chat_history.pop(0)
                total_tokens -= len(removed_query) + len(removed_answer)

        result = pdf_qa(
            {"question": query, "chat_history": chat_history})
        print(f"{white}Answer: " + result["answer"])
        chat_history.append((query, result["answer"]))
        answer = result["answer"]
        return {"question": query, "answer": answer}
