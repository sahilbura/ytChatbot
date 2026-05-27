# ytChatbot

A lightweight chatbot that lets you interact with the content of any YouTube video using Retrieval-Augmented Generation (RAG) through a browser frontend.

Paste a video URL, and the app:

* Fetches the transcript (even non-English ones!)
* Translates to English using an LLM (if needed)
* Indexes the transcript into a vector store
* Lets you ask questions about the video via a chatbot UI

Built using **LangChain**, **OpenAI**, **FastAPI**, and **Vue**.

---

## ✨ Features

* ✅ Auto-fetches YouTube transcripts
* 🌐 Auto-translates non-English transcripts to English
* 🔍 RAG pipeline using LangChain
* 📚 FAISS-powered vector search
* 💬 Browser-based chat UI for interactive chatting

---

## 🧰 Tech Stack

| Tool                     | Purpose                                                |
| ------------------------ | ------------------------------------------------------ |
| `LangChain`              | Chaining LLM calls, building RAG pipeline              |
| `OpenAI`                 | GPT for translation + answering, embeddings generation |
| `FAISS`                  | Vector store for semantic search                       |
| `FastAPI`                | API backend for transcript and chat sessions           |
| `Vue`                    | Frontend user interface                                |
| `Vite`                   | Frontend build and dev tooling                         |
| `youtube-transcript-api` | Fetches YouTube video transcripts                      |
| `langdetect`             | Detects language of transcripts                        |
| `python-dotenv`          | Loads environment variables                            |

---

## 📦 Installation

1. **Clone the repository**

```bash
git clone https://github.com/sahilbura/ytChatBot.git
cd ytChatBot
```

2. **Install Python dependencies**

```bash
pip install -r requirements.txt
```

3. **Install frontend dependencies**

```bash
cd frontend
npm install
```

4. **Add your OpenAI API key** to a `.env` file:

```
OPENAI_API_KEY=your_openai_api_key_here
```

5. **Run the backend**

```bash
uvicorn server:app --reload
```

6. **Run the frontend** in a second terminal

```bash
cd frontend
npm run dev
```

---

## 🧠 How It Works

1. **Transcript Extraction**

   * Uses `youtube-transcript-api` to fetch captions
   * Detects language via `langdetect`
   * If not English, translates via OpenAI GPT

2. **Embedding + Indexing**

   * Transcripts are chunked with `RecursiveCharacterTextSplitter`
   * Embeddings are generated with `text-embedding-3-small`
   * Indexed into FAISS for semantic retrieval

3. **Chat Chain**

   * Retrieves relevant chunks from FAISS
   * Injects them into a LangChain prompt template
   * GPT model returns an answer grounded in video context

---

## 🧪 Example Questions

Try asking:

* "Summarize the video."
* "What is the main argument of the video?"
* "Does the video explain how to solve this issue?"

---

## 🙋‍♀️ Author

Made with ❤️ by sahil
Let me know what you think or suggest ideas to build next!
