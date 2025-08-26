# Import libraries
import os
import streamlit as st
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from pytube import YouTube
from pytube.exceptions import VideoUnavailable, RegexMatchError, PytubeError
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda

# App Title 
st.markdown(
    """
    <h1 style='text-align: center; font-size: 40px; font-weight: bold;'> RAG-based
        <img src="https://upload.wikimedia.org/wikipedia/commons/0/09/YouTube_full-color_icon_%282017%29.svg" 
             width="60" height="60" style="vertical-align: middle; margin-left:5px; margin-right:5px;">
        YouTube Video Assistant with Gemini
    </h1>
    """,
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)

# Instructions block
st.markdown(
    """
    ### How to use this tool:
    1. Enter your **Google AI Studio API key** in the field below.  
       - To get one **for free**, go to [Google AI Studio](https://aistudio.google.com/),  
         sign in with your Google account, and generate an API key under **Get API key**.  
       - Copy the key and paste it into this app.  
    2. Paste the full **YouTube video URL** you want to analyze.  
    3. Once the video is validated ✅, type your **question** about the video in the text box.  
    4. Click **Submit Question** and the AI will answer based only on the video transcript.  
    5. If the transcript is unavailable, you’ll see an error message.  
    """,
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)


# Streamlit app
#Step 1: User enters API key
api_key = st.text_input("Enter your Google AI Studio API key here")
if api_key:
  os.environ['GOOGLE_API_KEY'] = api_key

# Step 2: User enters full YouTube URL
video_url = st.text_input("Paste full YouTube URL")


# Step 3: Helper function to extract video ID
def extract_video_id(url: str):
    try:
        parsed_url = urlparse(url)
        if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
            vid = parse_qs(parsed_url.query).get("v", [None])[0]
        elif parsed_url.hostname == "youtu.be":
            vid = parsed_url.path[1:]
        else:
            return None
        if vid and len(vid) == 11:
            return vid
        return None
    except:
        return None

# Step 4: extract video id

if video_url:
    video_id = extract_video_id(video_url)
       
    if not video_id:
         st.error("Invalid url or video not available") 
         st.stop()
    else:
        try:
            yt = YouTube(video_url)  # This will raise an exception if the video is unavailable
            st.success("✅ Video found!")
            # st.write(f"Video ID: {yt.video_id}")
        except VideoUnavailable:
            st.error("❌ Video unavailable")
        except RegexMatchError:
            st.error("❌ Invalid YouTube URL format")
        except PytubeError as e:
            st.error(f"⚠️ Pytube error: {e}")
        except Exception as e:
            st.error(f"⚠️ An unexpected error occurred: {e}")

question = st.text_area ("Write your question here.")
button = st.button("Submit Question")  
if button:
    if not video_id:
        st.error("Cannot process question because video is invalid or unavailable.")
    elif not question:
        st.error("Please enter a question.")
    else:
        st.info("Processing your question... This may take a few seconds.") 
        # Step 5: LLM Process
        # Step 5.1a: Indexing (Document Ingestion)            
        try:
            transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=['en','hi']) # If you don’t care which language, this returns the “best” one
            transcript = " ".join(chunk.text for chunk in transcript_list) # Flatten it to plain text
            print(transcript)
        except TranscriptsDisabled:
            print("No captions available for this video.")


        #  Step 5.1b: Indexing (Text Splitting) 

        #create splitter object for chunking and initializing
        splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
        chunks = splitter.create_documents([transcript])


        # Step 5.1c & 5.1d: Indexing (Embedding Generation and Storing in Vector Store)

        #create embedding and vector objects and initializing
        embeddings = GoogleGenerativeAIEmbeddings(model = "models/gemini-embedding-001")
        vector_store = FAISS.from_documents(chunks, embeddings)

        #map vectors to documents
        # vector_store.index_to_docstore_id


        # Step 5.2: Retrieval
        #create a retriever object
        retriever =  vector_store.as_retriever(search_type = "similarity", search_kwargs = {"k": 4})

        # Step 5.3: Augmentation
        llm = ChatGoogleGenerativeAI(model = "gemini-2.0-flash", temperature = 0.2) #model define
        prompt = PromptTemplate(  #create prompt template
            template = """You are a helpful assistant.
            Answer ONLY from the provided transcript context.
            If the context is insufficient, just say you don't know.
            Context: {context}
            Question: {question}
            """,
            input_variables = ["context", "question"]
        )


        retrieved_docs = retriever.invoke(question)

        #create function to fetch docs for context
        def format_docs(retrieved_docs):
            context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
            return context_text

        #create parallel chain
        parallel_chain = RunnableParallel({
            'context': retriever | RunnableLambda(format_docs),
            'question': RunnablePassthrough()
        })

        parser = StrOutputParser()

        main_chain = parallel_chain | prompt | llm | parser
        response = main_chain.invoke(question)
        st.write(response)