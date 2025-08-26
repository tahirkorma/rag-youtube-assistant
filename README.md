# YouTube RAG Q&A with Google Gemini

🎬 **YouTube Video Q&A powered by Retrieval-Augmented Generation (RAG) and Google Gemini LLM**.  
Ask questions about YouTube videos and get answers directly from the transcript.

---

## Introduction

The **YouTube RAG Q&A with Google Gemini** project is an interactive web application that allows users to ask questions about YouTube videos and get precise answers based solely on the video transcript.  

Using a **Retrieval-Augmented Generation (RAG)** approach, the app splits video transcripts into smaller chunks, converts them into embeddings, and stores them in a **vector database (FAISS)**. When a user asks a question, the system retrieves the most relevant chunks and uses **Google Gemini LLM** to generate context-aware, accurate responses.  

This project demonstrates the combination of **video processing, natural language understanding, and large language models** to create an intelligent and interactive video assistant.  

---

## Live Demo

🌐 [Try the app](https://ragbotbytahirkorma.streamlit.app/)

---
##  📸 Screenshot
   ![screenshot](https://github.com/user-attachments/assets/f5cc0cfc-3a7d-49f4-be2b-3555c7cd94b4)
   
   
--- 

## Features

- Automatic **video validation** and transcript fetching from YouTube.  
- **RAG-based retrieval**: answers are generated using only the video transcript.  
- Powered by **Google Gemini LLM** for high-quality answers.  
- Supports multiple languages in transcripts (e.g., English & Hindi).  
- Built with **Streamlit** for an interactive web interface.  

---

## How It Works

1. **Video Validation:** The app checks if the YouTube URL is valid and accessible.  
2. **Transcript Fetching:** Captions are fetched using `YouTubeTranscriptApi`.  
3. **Text Chunking:** The transcript is split into smaller chunks for better retrieval.  
4. **Embeddings & Vector Store:** Each chunk is converted into embeddings using Google Gemini, and stored in FAISS.  
5. **Retrieval-Augmented Generation (RAG):** When you ask a question, relevant chunks are retrieved from FAISS.  
6. **LLM Answering:** Google Gemini LLM generates answers based only on the retrieved transcript chunks.

---

## How to Use

1. Enter your **Google AI Studio API key** in the input field.  
   - To get a free API key: [Google AI Studio](https://aistudio.google.com/).  
2. Paste the full **YouTube video URL** you want to analyze.  
3. Type your **question** about the video.  
4. Click **Submit Question** to get the AI-generated answer.  

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/tahirkorma/rag-youtube-assistant.git
cd <rag-youtube-assistant>
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the App
```bash
streamlit run app.py
```

## Requirements
<pre>
Python 3.8+
Streamlit
Pytube
youtube-transcript-api
LangChain
FAISS
langchain-google-genai
(All dependencies are listed in requirements.txt)
</pre>
