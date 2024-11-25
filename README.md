# GitChat: A GitHub Repository Q&A Chatbot

GitChat is a chatbot-based application that answers questions based on the GitHub repository provided by users. It leverages advanced embeddings and a powerful vector database for quick retrieval and LLM capabilities to generate accurate responses. Whether you're exploring documentation, looking for answers on code issues, or diving into project insights, GitChat makes it easy to interact with a GitHub repository.

## Features
- **GitHub Repository Integration**: Allows users to provide a GitHub repository URL, which is then processed to enable Q&A.
- **Embeddings with Gemini**: Uses **Gemini-Embeddings** to create rich, high-quality embeddings from the repository content.
- **Efficient Search with FAISS**: Utilizes **FAISS** as a vector database to quickly search through embeddings for the most relevant answers.
- **LLM with Groq API**: Powered by **Llama** and **groq_api** for answering user questions in a conversational manner.
- **Interactive Interface**: Provides a user-friendly interface using **Streamlit**, enabling users to easily interact with the chatbot.

## Live Demo

You can try out the app directly on the following link:

[GitChat Demo](https://gitchats.streamlit.app)

![image](https://github.com/user-attachments/assets/fbba0ab5-69cf-4933-9ac4-447e669343e2)

## Installation

To run this application locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/rugvedp/gitchat.git
    cd gitchat
    ```

2. Create a virtual environment (optional but recommended):
    ```bash
    python -m venv gitchat-env
    source gitchat-env/bin/activate  # For Linux/Mac
    gitchat-env\Scripts\activate     # For Windows
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Requirements

The following Python libraries are required for this application:

- **langchain-google-genai**: To use Google's Generative AI capabilities.
- **google.generativeai**: Google Generative AI SDK.
- **faiss-cpu**: A library for efficient similarity search and clustering of dense vectors.
- **langchain**: To handle chainable models and components.
- **tiktoken**: Tokenizer for efficiently handling large inputs for language models.
- **beautifulsoup4**: For web scraping and parsing HTML content.
- **streamlit**: Framework to create the web interface for interaction.
- **groq**: Used for connecting to and interacting with the Groq API.
- **langchain-community**: Extended community features of LangChain.
- **langchain-groq**: LangChain components specifically designed for Groq API.

## Usage

1. Run the application:
    ```bash
    streamlit run app.py
    ```

2. Open your browser and go to `http://localhost:8501` to interact with GitChat.

3. Provide a GitHub repository URL in the input field.

4. Ask questions related to the repository, and GitChat will process and provide relevant answers based on the repository’s contents.

## How It Works

1. **Embedding**: Gemini-Embeddings processes the content from the provided GitHub repository.
2. **Vector Database**: FAISS stores and searches these embeddings to efficiently retrieve related content.
3. **Groq and LLM**: The chatbot, powered by the Llama model and Groq API, generates human-like answers based on the content retrieved from the repository.

## Contributing

If you'd like to contribute to the project, feel free to fork the repository, make changes, and create a pull request. Please ensure that your code follows the existing code style and passes any existing tests.

### Bug Reports & Feature Requests

If you encounter any issues or have suggestions for new features, please open an issue on the GitHub repository.

## License

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)

This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.
