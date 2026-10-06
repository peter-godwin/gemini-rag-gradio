# PDF RAG Chatbot

A Retrieval-Augmented Generation application that lets you chat with PDF documents.

## Features

- Loads and chunks PDF documents
- Creates vector embeddings using Google Gemini
- Stores vectors in Chroma DB
- Interactive Q&A about document contents
- Three-sentence concise answers

## Current PDF

The project now includes `octopuses.pdf` - a document about the fascinating world of octopuses, covering their intelligence, three hearts, blue blood, and escape artist abilities.

## Setup

1. Install dependencies:
   ```bash
   pip install langchain langchain-community langchain-google-genai python-dotenv
   ```

2. Set up your Google API key in `.env`:
   ```
   GOOGLE_API_KEY=your_key_here
   ```

3. Run the application:
   ```bash
   python rag_app.py
   ```

## Usage

Type questions about the document contents. Type `exit` or `quit` to stop.

Example: "What makes octopuses intelligent?"