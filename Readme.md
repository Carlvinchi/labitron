# Proof of Concept Chatbot

## Description

In this project I tried to build a chatbot that can analyze test results, and assist lab scientist to generate lab reports.

## Implementation

I used open sourced AI models via ollama, so if you want to run this project you must have ollama running locally or via docker.

Langchain was used to handle conversations and memory

Streamlit was used to develop chatGPT like UI

A simple JSON storage was used to persist conversations to disk

You can find out more about ollama here, --> [https://ollama.com/](https://ollama.com/)

This is optional, create a Python virtual environment as well, use the commmands below

python3 -m venv name_of_virtual_environment

source name_of_env/bin/activate on Linux

After you setup ollama, you can use the command below to install the packages required to run the project.

pip install -r requirements.txt, make sure you are in the project directory.

Run the chatbot using the command below

streamlit run app.py

Follow the link to view in your browser

Enjoy!!

#### ~ By Carlvinchi
