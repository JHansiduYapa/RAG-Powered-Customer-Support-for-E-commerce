import streamlit as st
import os
import re

# llama index for RAG system
from llama_index.core import VectorStoreIndex, load_index_from_storage, StorageContext
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# Initialize LLM
llm = Ollama(model="deepseek-r1:1.5b", temperature=0.1, request_timeout=120)

# Initialize embedding model
ollama_embedding = OllamaEmbedding(
    model_name="nomic-embed-text:latest",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)

# Define a custom sentence splitter
def custom_sentence_splitter(text):
    return re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s+", text)

# Initialize the node parser
node_parser = SentenceWindowNodeParser.from_defaults(
    sentence_splitter=custom_sentence_splitter,
    window_size=2,
    window_metadata_key="window",
    original_text_metadata_key="original_sentence",
)

# Index directory
index_dir = "D:/RAG-Powered-Customer-Support-for-E-commerce/app/sentence_index"

# Load the index
sentence_index = load_index_from_storage(
    StorageContext.from_defaults(persist_dir=index_dir),
    embed_model=ollama_embedding
)

# Create a query engine with streaming enabled
query_engine = sentence_index.as_query_engine(
    llm=llm,
    similarity_top_k=5,
    node_postprocessors=[MetadataReplacementPostProcessor(target_metadata_key="window")],
    streaming=True  # Enable streaming responses
)
#----------------------------------------------------------------------app-----------------------------------------------------------------------
with st.sidebar:
    "🌟 Features:"
    "✅ Instant responses to customer queries"
    "✅ 24/7 availability for support"
    "✅ User-friendly and interactive experience"

    "Need help? Chat with Tonic.lk Support Bot now and get instant assistance! 💙"

    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Streamlit UI
st.title("🤖💬Tonic.lk Customer Support(RAG)")

# Display chat messages in a conversational format
for chat in st.session_state.chat_history:
    with st.chat_message("user" if chat["role"] == "user" else "assistant"):
        st.markdown(chat["message"])

# User input field
if user_input := st.chat_input("Type your message..."):
    # Add user's message to history
    st.session_state.chat_history.append({"role": "user", "message": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get response from the model in streaming mode
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_container = st.empty()  # Placeholder for streaming response
            response_text = ""
            response_obj = query_engine.query(user_input)
            
            # Check if the streaming response supports a 'stream' method.
            if hasattr(response_obj, "stream"):
                for chunk in response_obj.stream():
                    response_text += chunk
                    response_container.markdown(response_text)
            else:
                # Fallback if no streaming method exists:
                response_text = response_obj.response if hasattr(response_obj, "response") else str(response_obj)
                response_container.markdown(response_text)

    # Append assistant's response to chat history
    st.session_state.chat_history.append({"role": "assistant", "message": response_text})
