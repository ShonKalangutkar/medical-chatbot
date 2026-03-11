from flask import Flask, render_template, request
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import system_prompt
import os


app = Flask(__name__)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


embeddings = download_embeddings()


index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)


retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


chatModel = ChatOllama(
    model="llama3.1:8b",
    temperature=0
)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)


question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():

    msg = request.form["msg"]
    print("User:", msg)

    response = rag_chain.invoke({"input": msg})

    print("Bot:", response["answer"])

    return response["answer"]


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)