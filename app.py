import streamlit as st
import os
import re
import time

# llama index for RAG system
from llama_index.core import VectorStoreIndex, load_index_from_storage, StorageContext
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import PromptTemplate
from llama_index.core.indices.postprocessor import SentenceTransformerRerank


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
# initialize re-ranker
# BAAI/bge-reranker-base
# link: https://huggingface.co/BAAI/bge-reranker-base
rerank = SentenceTransformerRerank(
    top_n=2, model="BAAI/bge-reranker-base"
)

# Create a query engine with streaming enabled
query_engine = sentence_index.as_query_engine(
    llm=llm,
    similarity_top_k=3,
    node_postprocessors=[
        rerank,
        MetadataReplacementPostProcessor(target_metadata_key="window")]
)

# customer service chatbot
qa_prompt_tmpl_str = (
    "You are a helpful and professional customer service chatbot. Your job is to provide accurate, "
    "concise, and friendly responses to customer inquiries based on the provided context."
    "find the most relevant parts on the Context Information and answer as professional customer service.\n"
    "---------------------\n"
    "Context Information:\n"
    "{context_str}\n"
    "---------------------\n"
    "Please ensure your response is relevant, clear, and aligns with company policies.\n"
    "If the context does not provide sufficient information, politely ask for clarification.\n"
    "only give the answer for the question do not use words like based on the provided context.\n"
    "Use a professional yet friendly tone.\n\n"
    "Customer Query: {query_str}\n"
    "Response: "
)

qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

query_engine.update_prompts(
    {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
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
# simulate streaming function 
def stream_response(response):
    for word in response.response.split(" "):
        yield word + " "
        time.sleep(0.02)

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
            response_container.write_stream(stream_response(response_obj))
            
    # Append assistant's response to chat history
    st.session_state.chat_history.append({"role": "assistant", "message": response_obj.response})
