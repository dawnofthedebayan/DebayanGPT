Sure, here's a sample README file for your GitHub repository:

---

# DebayanGPT Research Papers Chatbot

## Overview

DebayanGPT Research Papers Chatbot is an AI-powered chatbot created by Debayan Bhattacharya to assist with queries related to his research papers. The chatbot utilizes Llama models to understand and respond to questions about specific research papers authored by Debayan Bhattacharya.

## Features

- **Question Answering:** Ask questions about specific research papers authored by Debayan Bhattacharya, such as the main idea, main findings, or novelty.
- **Context Understanding:** The chatbot can incorporate context provided by the user to generate more accurate responses. 
- **Intelligent Prompting:** Utilizes a prompt template to guide users on how to structure their questions, ensuring the chatbot understands the user's intent.

## Usage

1. **Input Your Question:** Enter your question in the provided textbox.
2. **Submit:** Press Shift+ENTER to submit your question.
3. **Receive Response:** The chatbot will provide a response based on the question and any provided context.

## How to Run

To run the chatbot locally, follow these steps:

1. Clone the repository:

```
git clone https://github.com/dawnofthedebayan/DebayanGPT.git
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Create the vector store. I use FAISS for this:

```
python create_vector_database.py 
```


4. Run the chatbot:

```
python chatbot.py
```

4. Access the chatbot interface via your web browser at `127.0.0.1:7860`.

## Disclaimer

Please note that the DebayanGPT Research Papers Chatbot is a work in progress and may not be able to answer all questions accurately. Your patience and understanding are appreciated.

## Contributing

Contributions to the chatbot's development are welcome! If you'd like to contribute, please fork the repository and submit a pull request with your changes.

