import streamlit as st
from urllib.parse import urlparse, parse_qs
from chatbot import get_translated_transcript, build_chat_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

st.title("Youtube Chatbot")

video_url = st.text_input("Paste the URL of the Youtube Video")

if st.button("Let's Go!"):
  if not video_url:
    st.error("Please enter a URL")
  else:
    with st.spinner("Fetching Transcript..."):
      try:
        parsed_url = urlparse(video_url)
        video_id = parse_qs(parsed_url.query).get("v")

        if not video_id:
          st.error("Invalid Youtube URL format")
        else:
          video_id = video_id[0]
          transcript = get_translated_transcript(video_id, model)

          if transcript and transcript != "error":
            st.success("Transcript fetched successfully!")

            # Build and store the LangChain chatbot in session state
            st.session_state.chain = build_chat_chain(transcript, model, embeddings)
            st.session_state.ready = True
            
          else:
            st.error("Could not fetch transcript")

      except Exception as e:
        st.error(f"Something went wrong: {e}")

if st.session_state.get("ready"):
  question = st.text_input("Ask your question about the video")

  if question:
    with st.spinner("Thinking..."):
      answer = st.session_state.chain.invoke(question)
      st.markdown(f"**Answer:** {answer}")