---

# Tonic.lk Customer Support (RAG) Chatbot

This repository contains the source code for a customer support chatbot built using a Retrieval-Augmented Generation (RAG) system. The chatbot leverages a custom sentence window retriever combined with a large language model (LLM) to generate accurate, contextually relevant responses for customer inquiries.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture & Components](#architecture--components)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

The project integrates several components to build an effective customer support chatbot:
- **LLM (Ollama)**: Generates conversational responses using a dedicated model.
- **Ollama Embedding**: Converts text into vector representations for effective retrieval.
- **Custom Sentence Splitting**: Uses a tailored sentence splitter to ensure accurate segmentation of text.
- **Sentence Window Node Parser**: Processes text in overlapping windows to maintain context during retrieval.
- **Re-Ranking**: Applies a SentenceTransformer-based re-ranker to enhance answer relevancy.
- **Prompt Templating**: Guides the LLM to produce professional and clear customer support responses.

## Features

- **Accurate and Relevant Responses:** Retrieves contextually relevant information from a pre-built sentence index to ensure helpful customer support.
- **Customizable Query Engine:** Easily update prompts and postprocessors to fine-tune the chatbot’s responses.
- **Evaluation and Testing:** Includes evaluation tests based on deep evaluation techniques (context relevance, goodness, and faithfulness).

## Architecture & Components

### LLM & Embeddings
- **Ollama LLM:** Configured with a dedicated model (e.g., `deepseek-r1:1.5b`) to generate responses.
- **Ollama Embedding:** Utilizes an embedding model (e.g., `nomic-embed-text:latest`) to process text inputs.

### Data Retrieval and Processing
- **Custom Sentence Splitter:** Uses regular expressions to split text into sentences for better contextual grouping.
- **Sentence Window Node Parser:** Constructs overlapping windows of sentences to maintain context.
- **Index Storage & Loading:** Loads a pre-built sentence index from a specified directory to be queried in real time.

### Query Engine
- **Similarity Search:** Retrieves the top-k similar nodes from the index.
- **Re-Ranking:** Uses a SentenceTransformer-based re-ranker (e.g., `BAAI/bge-reranker-base`) to optimize the relevance of retrieved content.
- **Prompt Template:** Defines a custom prompt to ensure that the generated answers maintain a professional and concise tone.

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

After starting, a web interface will open where you can interact with the chatbot. Simply type your query, and the chatbot will process the input through the sentence index and generate a contextually relevant response.

### Code Overview

Below is a simplified snippet of the key components:

```python
import streamlit as st
import re
import time

from llama_index.core import load_index_from_storage, StorageContext
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import PromptTemplate
from llama_index.core.indices.postprocessor import SentenceTransformerRerank

# Initialize LLM and embedding model
llm = Ollama(model="deepseek-r1:1.5b", temperature=0.1, request_timeout=120)
ollama_embedding = OllamaEmbedding(
    model_name="nomic-embed-text:latest",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)

# Custom sentence splitter function
def custom_sentence_splitter(text):
    return re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s+", text)

node_parser = SentenceWindowNodeParser.from_defaults(
    sentence_splitter=custom_sentence_splitter,
    window_size=2,
    window_metadata_key="window",
    original_text_metadata_key="original_sentence",
)

# Load the sentence index
index_dir = "D:/RAG-Powered-Customer-Support-for-E-commerce/app/sentence_index"
sentence_index = load_index_from_storage(
    StorageContext.from_defaults(persist_dir=index_dir),
    embed_model=ollama_embedding
)

# Initialize re-ranker
rerank = SentenceTransformerRerank(
    top_n=2, model="BAAI/bge-reranker-base"
)

# Create the query engine with postprocessors
query_engine = sentence_index.as_query_engine(
    llm=llm,
    similarity_top_k=3,
    node_postprocessors=[
        rerank,
        MetadataReplacementPostProcessor(target_metadata_key="window")
    ]
)

# Define a custom prompt template for the chatbot
qa_prompt_tmpl_str = (
    "You are a helpful and professional customer service chatbot. Your job is to provide accurate, "
    "concise, and friendly responses to customer inquiries. "
    "Find the most relevant parts of the context and answer as a professional customer service representative.\n"
    "---------------------\n"
    "Context Information:\n"
    "{context_str}\n"
    "---------------------\n"
    "Please ensure your response is relevant, clear, and aligns with company policies.\n"
    "Only provide the answer for the question; avoid redundant phrases.\n"
    "Customer Query: {query_str}\n"
    "Response: "
)
qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
query_engine.update_prompts(
    {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
)
```

## Screenshots

### Evaluation Tests

Below are screenshots showcasing the evaluation tests conducted to verify the system's performance using evaluation techniques (context relevance, goodness, and faithfulness):

![Evaluation Test Screenshot](https://github.com/janith99hansidu/RAG-Powered-Customer-Support-for-E-commerce/blob/main/src/eval.png)  
*Figure 1: Evaluation test results on metrics Contextual Relevance/Faithfulness/Groundedness*

![Evaluation Test Screenshot](https://github.com/janith99hansidu/RAG-Powered-Customer-Support-for-E-commerce/blob/main/src/eval_results.png)  
*Figure 2: Evaluation test results based on LLM evaluation*

### Sample Input and Output

These screenshots illustrate sample inputs and outputs:

Below is an example of a markdown table that displays the sample input and sample output screenshots side-by-side:

| **Sample Input** | **Sample Output** |
|------------------|-------------------|
| ![Sample Input Screenshot](https://github.com/janith99hansidu/RAG-Powered-Customer-Support-for-E-commerce/blob/main/src/q1.png) <br> | ![Sample Output Screenshot](https://github.com/janith99hansidu/RAG-Powered-Customer-Support-for-E-commerce/blob/main/src/ans1.png) <br>|
| ![Sample Input Screenshot](https://github.com/janith99hansidu/RAG-Powered-Customer-Support-for-E-commerce/blob/main/src/q2.png) <br> | ![Sample Output Screenshot](https://github.com/janith99hansidu/RAG-Powered-Customer-Support-for-E-commerce/blob/main/src/ans2.png) <br>|
| ![Sample Input Screenshot](https://github.com/janith99hansidu/RAG-Powered-Customer-Support-for-E-commerce/blob/main/src/q3.png) <br> | ![Sample Output Screenshot](https://github.com/janith99hansidu/RAG-Powered-Customer-Support-for-E-commerce/blob/main/src/ans3.png) <br>|
| ![Sample Input Screenshot](https://github.com/janith99hansidu/RAG-Powered-Customer-Support-for-E-commerce/blob/main/src/q4.png) <br> | ![Sample Output Screenshot](https://github.com/janith99hansidu/RAG-Powered-Customer-Support-for-E-commerce/blob/main/src/ans4.png) <br>|
| ![Sample Input Screenshot](https://github.com/janith99hansidu/RAG-Powered-Customer-Support-for-E-commerce/blob/main/src/q5.png) <br> | ![Sample Output Screenshot](https://github.com/janith99hansidu/RAG-Powered-Customer-Support-for-E-commerce/blob/main/src/ans5.png) <br>|

## Evaluation

The chatbot has been rigorously evaluated using techniques that assess:
- **Contextual Relevance:** Ensuring the system retrieves the most pertinent sentences.
- **Faithfulness:** Verifying that the answers are both helpful and adhere closely to the provided context.
- **Groundedness:** LLMs assesses how well a model's generated responses align with information from the input source.

These evaluation measures ensure that the chatbot reliably meets customer support standards.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- **Streamlit:** For providing the framework to build an interactive UI.
- **Llama Index and Ollama:** For their advanced retrieval and language modeling capabilities.
- **BAAI/bge-reranker-base:** For enhancing the relevancy of the system’s responses.
- Special thanks to the open-source community for their continued support and contributions.

---