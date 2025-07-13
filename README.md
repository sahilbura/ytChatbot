# YouTube RAG Chatbot

A lightweight chatbot that lets you interact with the content of any YouTube video using Retrieval-Augmented Generation (RAG).

Paste a video URL, and the app:

* Fetches the transcript (even non-English ones!)
* Translates to English using an LLM (if needed)
* Indexes the transcript into a vector store
* Lets you ask questions about the video via a chatbot UI

Built using **LangChain**, **OpenAI**, and **Streamlit**.

---

## ✨ Features

* ✅ Auto-fetches YouTube transcripts
* 🌐 Auto-translates non-English transcripts to English
* 🔍 RAG pipeline using LangChain
* 📚 FAISS-powered vector search
* 💬 Streamlit UI for interactive chatting

---

## 🧰 Tech Stack

| Tool                     | Purpose                                                |
| ------------------------ | ------------------------------------------------------ |
| `LangChain`              | Chaining LLM calls, building RAG pipeline              |
| `OpenAI`                 | GPT for translation + answering, embeddings generation |
| `FAISS`                  | Vector store for semantic search                       |
| `Streamlit`              | UI for user interaction                                |
| `youtube-transcript-api` | Fetches YouTube video transcripts                      |
| `langdetect`             | Detects language of transcripts                        |
| `python-dotenv`          | Loads environment variables                            |

---

## 📦 Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/YoutubeChatbot.git
cd YoutubeChatbot
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Add your OpenAI API key** to a `.env` file:

```
OPENAI_API_KEY=your_openai_api_key_here
```

4. **Run the app**

```bash
streamlit run app.py
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

Made with ❤️ by [Adiba Anjum](https://www.linkedin.com/in/adiba-anjum-/)
Let me know what you think or suggest ideas to build next!
