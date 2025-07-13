from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
import streamlit as st
from youtube_transcript_api._api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from langdetect import detect
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI(model='gpt-4o-mini')

# 1. indexing 
# Document Ingestion

def get_translated_transcript(video_id: str, model) -> str:
  """
    Fetches a YouTube video's transcript, detects language, and translates it to English if needed.
    
    Args:
        video_id (str): The YouTube video ID.
        model: A LangChain-compatible LLM like ChatOpenAI.
    
    Returns:
        str: Final English transcript (original or translated).
  """
  try:
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

    try:
      transcript = transcript_list.find_manually_created_transcript(['en'])
    except:
      transcript = transcript_list.find_transcript([t.language_code for t in transcript_list])
    
    transcript_chunks = transcript.fetch()

    transcript_text = " ".join(chunk.text for chunk in transcript_chunks)

    detected_lang = detect(transcript_text)

    if detected_lang != 'en':
      if len(transcript_text) > 100:
        transcript_text = transcript_text[:100]
      response = model.invoke(f"You are a professional translator. Translate the following video transcript into English. Keep the meaning accurate but don't add commentary. {transcript_text}")
      transcript_text = response.content

    return transcript_text

  except TranscriptsDisabled:
    print("Transcripts disabled")
  except NoTranscriptFound:
    print("no transcripts found")
  except Exception as e:
    print("error: ",e)

  return "error"

english_transcript = get_translated_transcript("6ZpNISm6umA" , model)

# text splitting

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([english_transcript])

# generate embedding and store in vectore store

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_documents(chunks, embeddings)

# 2. Retrieval

retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={"k": 4})

# 3. Augmentation
prompt = PromptTemplate(
  template="""
  You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      Context: {context}
      Question: {question}
""",
input_variables=['context', 'question']
)



# 4. Generation

# 5. Building chain

