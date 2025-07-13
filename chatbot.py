from youtube_transcript_api._api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from langdetect import detect
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

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
      if len(transcript_text) > 10000:
        transcript_text = transcript_text[:10000]
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

def build_chat_chain(transcript_text, model, embeddings):
  """
  Builds a LangChain pipeline (Retriever + Prompt + LLM) based on transcript.
  """
  splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
  chunks = splitter.create_documents([transcript_text])

  vector_store = FAISS.from_documents(chunks, embeddings)

  retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={"k": 4})

  prompt = PromptTemplate(
    template="""
    You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, Say that you don't have sufficient context most probably because the video doesn't talk about this topic but try to answer the question in a general way

        Context: {context}
        Question: {question}
  """,
  input_variables=['context', 'question']
  )

  def format_docs(retrived_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrived_docs)
    return context_text

  # Building the chain 

  parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
  })

  parser = StrOutputParser()

  main_chain = parallel_chain | prompt | model | parser

  return main_chain

