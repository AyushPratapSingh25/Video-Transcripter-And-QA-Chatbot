# 🎥 YouTube Video Transcription Agent (RAG Chatbot)

An AI-powered chatbot that lets you **ask questions about a YouTube video**.
The system automatically extracts the **video transcript**, stores it in a **vector database**, and uses a **Retrieval-Augmented Generation (RAG)** pipeline to generate accurate answers.

This project demonstrates how to build an end-to-end AI application using modern LLM tooling.

---

## 🚀 Features

* 📺 Extracts transcripts from YouTube videos
* 🧠 Converts transcripts into embeddings
* 🗂 Stores embeddings in a vector database
* 🔎 Retrieves relevant transcript chunks for queries
* 🤖 Generates answers using an LLM
* 💬 Interactive chat interface built with Streamlit
* ⚡ Fast semantic search using vector similarity

---

## 🧠 Architecture

```
User Question
      ↓
Retriever (Vector Search)
      ↓
Relevant Transcript Chunks
      ↓
LLM (Answer Generation)
      ↓
Final Response + Source
```

This system uses a **Retrieval-Augmented Generation (RAG)** pipeline.

---

## 🛠 Tech Stack

* Python
* LangChain
* Ollama (LLM + embeddings)
* Pinecone (Vector Database)
* Streamlit (UI)
* YouTube Transcript API

---

## 📂 Project Structure

```
youtube-video-chatbot/
│
├── ingestion.py        # Extracts transcript & stores embeddings
├── core.py             # RAG pipeline (retrieval + generation)
├── app.py              # Streamlit chatbot interface
├── logger.py           # Logging utilities
├── .env                # Environment variables
├── .gitignore
└── README.md
```

---

## 💡 Example Questions

* What is the video about?
* What are the main points discussed?
* What does the speaker explain about AI?
* Summarize the video.

---

## 🔍 How It Works

1. The YouTube transcript is extracted using the YouTube Transcript API.
2. The transcript is split into smaller chunks.
3. Each chunk is converted into embeddings.
4. Embeddings are stored in a vector database.
5. When a user asks a question, relevant chunks are retrieved.
6. The LLM generates an answer using the retrieved context.

---

## 📌 Future Improvements

* Support multiple videos in one database
* Add timestamp references for answers
* Allow users to paste any YouTube link
* Improve retrieval accuracy
* Deploy the chatbot online

---

## 🤝 Contributing

Contributions are welcome!

If you'd like to improve the project:

1. Fork the repository
2. Create a new branch
3. Submit a pull request

---

## ⭐ Support

If you found this project helpful, consider giving it a **star ⭐ on GitHub**.

---