from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langdetect import detect
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate


class ChatChain:
  """Simple chat chain that uses a retriever and an LLM with `invoke()`."""

  def __init__(self, retriever, prompt_template: PromptTemplate, model, default_context: str = ""):
    self.retriever = retriever
    self.prompt_template = prompt_template
    self.model = model
    self.default_context = default_context

  def invoke(self, question: str) -> str:
    # Use retriever context when available; otherwise use transcript context directly.
    if self.retriever is not None:
      try:
        docs = self.retriever.get_relevant_documents(question)
        context_text = "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)
      except Exception:
        context_text = self.default_context
    else:
      context_text = self.default_context

    prompt_text = self.prompt_template.format(context=context_text, question=question)

    # Some models (e.g., ChatOpenAI) expose `invoke()` returning an object with `.content`
    resp = self.model.invoke(prompt_text)
    try:
      return resp.content
    except Exception:
      return str(resp)

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
    english_codes = ['en', 'en-US', 'en-GB']
    ytt_api = YouTubeTranscriptApi()

    # 1) Try direct fetch prioritizing English
    try:
      fetched = ytt_api.fetch(video_id, languages=english_codes)
    except Exception:
      fetched = None

    # 2) If not found, list and pick best available (manual EN -> generated EN -> any)
    if fetched is None:
      try:
        transcript_list = ytt_api.list(video_id)
        transcript = None
        try:
          transcript = transcript_list.find_manually_created_transcript(english_codes)
        except Exception:
          transcript = None
        if transcript is None:
          try:
            transcript = transcript_list.find_generated_transcript(english_codes)
          except Exception:
            transcript = None
        if transcript is None:
          try:
            transcript = next(iter(transcript_list))
          except Exception:
            transcript = None
        if transcript is not None:
          # Translate to English if needed
          try:
            if getattr(transcript, "language_code", "") not in english_codes and getattr(transcript, "is_translatable", False):
              fetched = transcript.translate('en').fetch()
            else:
              fetched = transcript.fetch()
          except Exception:
            fetched = transcript.fetch()
      except Exception:
        fetched = None

    if fetched is None:
      print("no transcripts found")
      return "error"

    # Build text robustly from FetchedTranscript
    try:
      raw = fetched.to_raw_data()
      transcript_text = " ".join(snippet.get("text", "") for snippet in raw if snippet.get("text"))
    except Exception:
      # Fallback iteration (snippet objects)
      transcript_text = " ".join(getattr(snippet, "text", "") for snippet in fetched)

    detected_lang = detect(transcript_text)

    if detected_lang != 'en':
      try:
        translation_input = transcript_text[:10000] if len(transcript_text) > 10000 else transcript_text
        response = model.invoke(
          f"You are a professional translator. Translate the following video transcript into English. Keep the meaning accurate but don't add commentary. {translation_input}"
        )
        translated_text = getattr(response, "content", "").strip()
        if translated_text:
          transcript_text = translated_text
      except Exception as translation_error:
        print("translation failed, using original transcript:", translation_error)

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

  retriever = None
  if embeddings is not None:
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={"k": 4})

  prompt = PromptTemplate(
    template="""
    You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        if insufficient context, respond: 'Insufficient context to answer' and provide related general info labeled as such.

        Context: {context}
        Question: {question}
  """,
  input_variables=['context', 'question']
  )

  # If no retriever is available, provide compact transcript context directly.
  fallback_context = transcript_text[:12000]

  # Return a simple ChatChain object compatible with existing usage (`chain.invoke(question)`).
  return ChatChain(retriever, prompt, model, default_context=fallback_context)

