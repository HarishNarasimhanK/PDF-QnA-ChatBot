# Conversational RAG with PDF Uploads

This Streamlit app enables a user to upload a PDF file, and ask questions about its content in a conversational format. The app leverages Groq's API, LangChain's retrieval-augmented generation (RAG), Hugging Face embeddings, and FAISS for vector database management.

## Features

1. **PDF Upload and Parsing:**
   - Users can upload a PDF, and the app processes it to extract content.

2. **Conversational RAG Interface:**
   - Stateful question answering using LangChain RAG.
   - Tracks session-specific question-answer history.

3. **Audio Input:**
   - Supports voice-based questions using Groq's transcription service.

4. **Key Validation:**
   - Validates Groq API key before usage.

## Prerequisites

### API Keys
- Obtain a Groq API key from the [Groq Developer Portal](https://groq.com/).
- Obtain a Hugging Face token for embedding models.

### Installation

Ensure the following Python packages are installed:

```bash
pip install streamlit langchain langchain-huggingface langchain-groq langchain-community langchain-core faiss-cpu groq
```

## How to Run the App

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/conversational-rag
   cd conversational-rag
   ```

2. Create a `.env` file with the Hugging Face token:
   ```bash
   echo "HF_TOKEN=your_huggingface_token" > .env
   ```

3. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. Open the app in your browser at [http://localhost:8501](http://localhost:8501).

## Usage Instructions

1. **Set the Groq API Key:**
   - Enter your Groq API key in the sidebar.

2. **Upload a PDF File:**
   - Use the file uploader to upload a PDF file.

3. **Ask Questions:**
   - Enter questions in the text input field.
   - Alternatively, use the audio input to ask questions.

4. **View Results:**
   - The app displays concise answers and maintains a history of your conversation.

## Error Handling

1. **Invalid API Key:**
   - The app verifies the Groq API key before proceeding. Invalid keys will prompt an error.

2. **PDF Processing Issues:**
   - If there is an error while processing the uploaded PDF, a descriptive error message will be shown.

3. **Empty Input:**
   - If no input is provided, the app displays a reminder to enter a question.

## Code Overview

1. **Embedding Models:**
   - Uses `all-MiniLM-L6-v2` from Hugging Face with FAISS vector database for efficient retrieval.

2. **LangChain RAG Components:**
   - Utilizes recursive character splitting for document chunking.
   - Implements history-aware retrievers and QA chains.

3. **Groq API:**
   - Integrates with Groq for audio transcription and conversational language models.

4. **State Management:**
   - Tracks session history using Streamlit's `session_state`.

## Troubleshooting

1. **Groq API Key Validation Failure:**
   - Ensure the key is valid and entered correctly.
   - Check your network connection and Groq API availability.

2. **Hugging Face Embeddings Error:**
   - Verify that the Hugging Face token is correct.
   - Check for proper installation of required libraries.

3. **PDF Upload Issues:**
   - Confirm the uploaded file is a valid PDF.

4. **Session History Not Saving:**
   - Ensure you have a unique session ID.

## References

- [Groq Documentation](https://groq.com/docs/)
- [LangChain Documentation](https://python.langchain.com/)
- [Hugging Face Models](https://huggingface.co/models)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
