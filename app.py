import tempfile
from groq import Groq
from langchain.schema import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import streamlit as st

## HuggingFaceEmbeddings
hf_token = "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"
embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2",encode_kwargs={'token': hf_token} )

## setting up streamlit
st.title("Conversational RAG with PDF uploads")
st.write("Upload a PDF and Ask the ChatBot about the context..")

## Input the GROQ API key
groq_api_key = st.sidebar.text_input("Enter the Groq API key ", type = "password")


def validate_groq_api_key(api_key: str) -> bool:
    """
    Validate Groq API key by attempting to make a simple API call
    
    Parameters:
    api_key (str): The Groq API key to validate
    
    Returns:
    bool: True if valid, False otherwise
    """
    try:
        # Initialize the ChatGroq client
        chat = ChatGroq(
            groq_api_key=api_key,
            model_name="gemma2-9b-it"  # Using a known model
        )
        
        # Make a simple test request
        messages = [
            HumanMessage(content="Test connection")
        ]
        
        # Attempt to get a response
        response = chat.invoke(messages)
        
        # If we get here, the API key is valid
        return True
        
    except Exception as e:
        return False

## Checking if GROQ API key is provided
if groq_api_key:
    if validate_groq_api_key(groq_api_key):
        ## LLM model definition
        model = ChatGroq(model_name = 'gemma2-9b-it',groq_api_key = groq_api_key)
        ## Chat Interface
        ## for stateful managing of chat history
        session_id = st.text_input("Session ID",value = "default_session")

        if 'store' not in st.session_state:
            st.session_state.store = {}

        ## File Uploader
        uploaded_file = st.file_uploader("Choose a PDF file", type = "pdf", accept_multiple_files = False)
        if uploaded_file:
            docs = []
            try:
                # Create a temporary file with explicit permissions
                with tempfile.NamedTemporaryFile(mode='wb+', delete=False, suffix='.pdf') as temp_pdf:
                    # Write the uploaded file content to the temporary file
                    temp_pdf.write(uploaded_file.getvalue())
                    temp_pdf.flush()  # Ensure all data is written
                    temp_file_path = temp_pdf.name  # Store the path
                    
                # Load PDF using the temporary file path
                loader = PyPDFLoader(temp_file_path)
                docs = loader.load()
                
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
                
            finally:
                # Clean up: Remove the temporary file
                try:
                    import os
                    if 'temp_file_path' in locals():
                        os.unlink(temp_file_path)
                except Exception as e:
                    print(f"Error cleaning up temporary file: {str(e)}")

            ## split the documents into chunk and storing a vector database
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 1500,
                chunk_overlap = 500
            )
            msg = st.empty()
            msg.info("PDF is getting uploaded...Please wait")
            docs = text_splitter.split_documents(docs)
            vectorstore = FAISS.from_documents(
                docs,
                embedding = embeddings
                )
            retriever = vectorstore.as_retriever()
            msg = st.empty()

            ## Creating the prompt template
            contextualize_q_system_prompt = """
                Given a chat history and the latest user 
                question which might reference a context 
                which is present in the chat history, 
                formulate a standalone question which can be 
                understood without the chat history. Do NOT 
                answer the question, just reformulate it 
                if needed, otherwise return it as it is
                """
            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system",contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ])
            history_aware_retriever = create_history_aware_retriever(
                model,
                retriever,
                contextualize_q_prompt
            )

            ## Answer Question Prompt (based on provided context)
            system_prompt = """
                you are a helpful chat assistant for question
                answering tasks. use the following pieces of 
                retrieved context to answer the following question in 
                a very concise way by using atmost 4 sentences. 
                If you dont know the answer, say that you dont know.
                Use ONLY atmost four sentences and keep the answer very 
                concise.\n\n{context}
            """
            qa_prompt = ChatPromptTemplate.from_messages([
                ("user",system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ])

            ## creating the RAG chain
            qa_chain = create_stuff_documents_chain(model, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever,qa_chain)

            def get_session_history(session_id : str) -> BaseChatMessageHistory:
                if session_id not in st.session_state.store:
                    st.session_state.store[session_id] = ChatMessageHistory()
                return st.session_state.store[session_id]

            ## creating the Conversational RAG chain
            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key = "input",
                output_messages_key = "answer",
                history_messages_key = "chat_history"
            )

            user_input = st.text_input("Ask the question through text \'or\' voice")
            # Initialize Groq client
            client = Groq(api_key=groq_api_key)
            audio_input = st.audio_input("ASK", label_visibility="collapsed")
            if user_input or audio_input:
                session_history = get_session_history(session_id)
            if user_input:
                response = conversational_rag_chain.invoke(
                    {
                        "input":user_input,
                    },
                    config = {
                        "configurable" : {"session_id" : session_id}
                    }
                )
                st.write("QUESTION")
                st.write(user_input)
                st.write(f"ANSWER")
                st.write(response["answer"])
                st.markdown("""---""")
                st.write("MESSAGE HISTORY")
                for i,msg in enumerate(session_history.messages):
                    if i%2 == 0:
                        st.write(f"Quesion {(i + 2) //2}:")
                        st.write(msg.content)
                    else:
                        st.write(f"Answer {(i + 2) //2}:")
                        st.write(msg.content)
            elif audio_input:
                # Save the audio input to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as temp_file:
                    temp_file.write(audio_input.getbuffer())
                    temp_file_path = temp_file.name
                with open(temp_file_path, "rb") as file:
                    st.text("Processing audio...")
                    transcription = client.audio.transcriptions.create(
                        file=(temp_file_path, file.read()),
                        model="whisper-large-v3-turbo",
                        prompt="Just transcribe",
                        temperature=0.4,
                        language="en",
                        response_format="verbose_json",
                    )
                os.remove(temp_file_path)
                user_input = transcription.text
                if user_input:
                    response = conversational_rag_chain.invoke(
                        {
                            "input":user_input,
                        },
                        config = {
                            "configurable" : {"session_id" : session_id}
                        }
                    )
                    st.write("QUESTION")
                    st.write(user_input)
                    st.write(f"ANSWER")
                    st.write(response["answer"])
                    st.markdown("""---""")
                    st.write("MESSAGE HISTORY")
                    for i,msg in enumerate(session_history.messages):
                        if i%2 == 0:
                            st.write(f"Quesion {(i + 2) //2}:")
                            st.write(msg.content)
                        else:
                            st.write(f"Answer {(i + 2) //2}:")
                            st.write(msg.content)
            else:
                st.error("Note: Please enter some questions before asking..")
    else:
        st.error("INVALID API KEY")
else:
    st.error("GROQ API KEY IS NOT PROVIDED")
    st.error("To provide GROQ API key, Click\t>")
