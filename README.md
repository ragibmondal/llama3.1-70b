README
======

Llama3 Chat App
===============

**What is it?**
----------------

A Streamlit app that uses the Groq API to generate chat responses based on user input.

**Requirements**
---------------

* `streamlit`
* `groq`
* `dotenv`
* Python 3.x
* `GROQ_API_KEY` environment variable

**Config**
---------

The app uses a `config.yaml` file to store model information and prompt templates.

**Usage**
------

### Install

`pip install streamlit groq dotenv`

### Configure

Create a `.env` file with your Groq API key: `GROQ_API_KEY=your_api_key_here`

### Run

`streamlit run app.py`

### Features
---------

* Chat interface with user input and AI responses
* Adjustable max tokens slider to control response length
* Display of model information and prompt templates
* Ability to clear chat history and download chat history as a text file
* Prompt templates can be loaded from the `config.yaml` file

**Notes**
------

* The app uses a generator function to yield chat response content from the Groq API response.
* `st.write_stream` is used to display the chat responses in real-time.
* The app stores the chat history in the `st.session_state` object.
* Error handling is implemented to catch any exceptions raised by the Groq API.
