import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

import gradio as gr

# Load environment variables from the .env file
load_dotenv()

print("Loading PDF document...")
loader = PyPDFLoader("octopuses.pdf")
document = loader.load()

print(document[0].page_content)

print("Chunking text...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(document)

print(chunks[0].page_content)

print("Creating vector database...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_db = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings, 
    persist_directory="./chroma_db"
)

# Configure the database to act as a document retriever
retriever = vector_db.as_retriever(search_kwargs={"k": 2})

# Define the hidden prompt structure for the LLM
template = """
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Context: {context}

Question: {question}

Answer:
"""
prompt = PromptTemplate.from_template(template)

# Initialize the free Gemini model tier
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

# Helper function to stitch retrieved chunks into a single text block
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Connect everything together using LangChain Expression Language (LCEL)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)


"""
user_question = "What days can I work from home?"
print(f"\nQuestion: {user_question}")

response = rag_chain.invoke(user_question)
# print(f"Answer: {response.content}")

# Clean up the output if Gemini returns a list of content blocks
if isinstance(response.content, list):
    clean_answer = response.content[0]['text']
else:
    clean_answer = response.content

print(f"Answer: {clean_answer}")
"""


def answer_question(question):
    if question.lower() in ['exit', 'quit']:
        return "Goodbye!"
    response = rag_chain.invoke(question)
    if isinstance(response.content, list):
        clean_answer = response.content[0]['text']
    else:
        clean_answer = response.content
    return f"Answer: {clean_answer}"

# Gradio UI
demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(placeholder="Ask a question about the PDF..."),
    outputs="text",
    title="PDF RAG Chatbot",
    description="Chat with the octopuses.pdf document",
    examples=["What makes octopuses intelligent?", "How many hearts do octopuses have?", "Can octopuses use tools?"],
)

if __name__ == "__main__":
    demo.launch()
