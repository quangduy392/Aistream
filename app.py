import streamlit as st
import requests
import os
import time
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from docx import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import json
# from heygen_app import fetch_presenters1 as heygen_fetch_presenters
# from heygen_app import voice_presenters as heygen_voice_presenters
# from heygen_app import generate_video as heygen_generate_video

# ---- Functionality from code1.py ----

# def fetch_presenters(idi_key):
#     url = "https://api.d-id.com/clips/presenters?limit=100"
#     headers = {
#         "Authorization": f"Basic {idi_key}"
#     }
#     response = requests.get(url, headers=headers)
    
#     if response.status_code == 200:
#         data = response.json()
#         return data.get("presenters", [])
#     else:
#         st.error(f"Error fetching presenters: {response.status_code}")
#         return []

def heygen_fetch_presenters(file_path):
    try:
        # Open and load the JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        # Check if there is any error in the JSON data
        if data.get("error") is None:
            # Extract and return the list of avatars
            avatars = data.get("data", {}).get("avatars", [])
            return avatars
        else:
            print(f"Error in JSON data: {data['error']}")
            return []
    
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {file_path}")
        return []

def heygen_voice_presenters(gender=None):
    try:
        with open('heygenvoice_response.json', 'r') as f:
            data = json.load(f)
            
            # Check for errors in the JSON response
            if data.get("error") is None:
                # Get the list of voices
                voices = data.get("data", {}).get("voices", [])
                
                # Filter voices based on language (English) and provider (Microsoft is no longer specified in the new structure)
                voices = [voice for voice in voices if voice['language'] == 'English']
                
                # If gender is specified, filter by gender
                if gender:
                    voices = [voice for voice in voices if voice.get('gender').lower() == gender.lower()]
                    print("Filtering by gender:", gender)
                
                return voices
            else:
                print(f"Error in JSON data: {data['error']}")
                return []
    
    except FileNotFoundError:
        print("Voice JSON file not found.")
        return []
    except json.JSONDecodeError:
        print("Error decoding JSON from file.")
        return []

def heygen_generate_video(prompt, avatar_id, voice, idi_key):
    url = "https://api.heygen.com/v2/video/generate"
    print(avatar_id)

    payload = {
        "test": True,
        "caption": False,
        # "title": "",
        # "callback_id": "",
        "dimension": {
            "width": 480,
            "height": 720
        },
        "video_inputs": [
            {
                "character": {
                "type": "avatar",
                "avatar_id": avatar_id,
                    "scale": 1,
                    "avatar_style": "normal",
                    "offset": {
                        "x": 0,
                        "y": 0
                    }
                },
                "voice": {
                    "type": "text",
                    "input_text": "Enabling FULL-HD photorealistic avatars, medium-shot, with body and hands movements using just text or audio as input. Premium Presenters (Clips) is an easy-to-use endpoint that lets users and developers supercharge training presentations, corporate communications, sales, marketing content and more. You can also create a custom HQ Presenter in Full-HD resolution based on your own video footage.",
                    "voice_id": voice["voice_id"]
                },
                "background": {
                    "type": "image",
                    "url":"https://gotrangtri.vn/wp-content/uploads/2020/03/bia-7.jpg"
                }
            }
        ]
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "x-api-key": f"{idi_key}"
    }

    # response = requests.post(url, json=payload, headers=headers)


    video_url = "error"
    try:
        response = requests.post(url, json=payload, headers=headers)
        print(response.text)
        if response.status_code == 200:
            res = response.json()
            id = res["data"]["video_id"]
            getresponse = requests.get(f"https://api.heygen.com/v1/video_status.get?video_id={id}", headers=headers)
            if getresponse.status_code == 200:
                res = getresponse.json()
                while res["data"]["status"] == "processing" or res["data"]["status"] == "pending" or res["data"]["status"] == "waiting":
                    time.sleep(10)
                    getresponse = requests.get(f"https://api.heygen.com/v1/video_status.get?video_id={id}", headers=headers)
                    res = getresponse.json()
                if res["data"]["status"] == "failed":
                    status = res["status"]
                    video_url = "error"
                elif res["data"]["status"] == "completed":
                    video_url = res["data"]["video_url"]
                # else:
                    
            else:
                status = "error"
                video_url = "error"
        # else:
        #     video_url = "error"
    except Exception as e:
        video_url = "error"
        
    return video_url

def fetch_presenters():
    try:
        with open('presenter_list.json', 'r') as f:
            data = json.load(f)
            presenters = data.get("presenters", [])
        return presenters
    except FileNotFoundError:
        st.error("Presenter JSON file not found.")
        return []

def voice_presenters(gender=None):
    try:
        with open('voice_response.json', 'r') as f:
            data = json.load(f)
            # Filter only English voices from Microsoft provider
            voices = [voice for voice in data if 
                      voice['provider'] == 'microsoft' and 
                      any(lang['locale'] == 'en-US' for lang in voice['languages'])]
            # If gender is specified, use it to filter the voices
            if gender:
                voices = [voice for voice in voices if voice.get('gender') == gender]
                print("GENDERRRRRRRR",gender)
        return voices
    except FileNotFoundError:
        st.error("Voice JSON file not found.")
        return []

def generate_video(prompt, presenter_id, voice, idi_key):
    url = "https://api.d-id.com/clips"
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Basic {idi_key}"
    }

    payload = {
        "presenter_id": presenter_id,
        "script": {
            "type": "text",
            "subtitles": "false",
            "provider": {
                "type": "microsoft",
                "voice_id": voice['id']
            },
            "input": prompt,
            "ssml": "false"
        },
        "config": { "result_format": "mp4" },
        # "presenter_config": {
        #     "crop": {
        #     "type": "rectangle",
        #     "rectangle": {
        #         "bottom": 1,
        #         "right": 0.8,
        #         "left": 0.2,
        #         "top": 0
        #     }
        #     }
        # },
        "presenter_config": {
        "crop": {
        "type": "wide"
                }
        },
        "background": {
            "source_url": "https://gotrangtri.vn/wp-content/uploads/2020/03/bia-7.jpg"
        }
    }

    video_url = "error"
    try:
        response = requests.post(url, json=payload, headers=headers)
        print("response",response.text)
        if response.status_code == 201:
            res = response.json()
            id = res["id"]
            status = "created"
            while status == "created" or status == "started" :
                getresponse = requests.get(f"{url}/{id}", headers=headers)
                if getresponse.status_code == 200:
                    status = res["status"]
                    res = getresponse.json()
                    if res["status"] == "done":
                        video_url = res["result_url"]
                        print("video_url done",video_url)
                    elif status == "created" or status == "started":
                        time.sleep(10)
                    elif status == "error":
                        print("Error fetching video status")
                        status = "error"
                        video_url = "error"
                    elif status == "rejected":
                        print("rejected")
                        status = "rejected"
                        video_url = "error"  
                else:
                    print("Error fetching video status")
                    status = "error"
                    video_url = "error"
        else:
            print("Error creating video")
            video_url = "error"
    except Exception as e:
        video_url = "error"
        
    return video_url

# ---- Functionality from code2.py ----

def get_docx_text(docx_docs):
    text = ''
    for docx in docx_docs:
        doc = Document(docx)
        for para in doc.paragraphs:
            text += para.text
    return text

def get_pdf_text(pdf_docs):
    text = ''
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store = FAISS.from_texts(
        text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, 
    if the answer is not in provided context just say, "answer is not available in the context", don't provide any wrong answers\n\n
    
    Context :\n {context}?\n
    Question : \n{question}\n
    
    Answer :
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    new_db = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.session_state.chat_output = response["output_text"]  # Store the output in session state
    
    return st.session_state.chat_output

def display_tiktok_video(video_url):
    tiktok_style = f"""
    <style>
    .tiktok-video-container {{
        position: relative;
        width: 280px;
        height: 500px;  
        overflow: hidden;
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        margin: 20px auto;
        background-color: black; /* Optional: background behind the video */
    }}

    .tiktok-video-container video {{
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 100%;
        height: 100%;
        object-fit: cover; /* Ensures video fills the container */
    }}
    </style>

    <div class="tiktok-video-container">
        <video controls autoplay>
            <source src="{video_url}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
    </div>
    """
    
    # Render the HTML inside the Streamlit app
    st.markdown(tiktok_style, unsafe_allow_html=True)

def display_presenter_selection(presenters):
    presenter_style = """
    <style>
    .presenter-container {
        padding: 20px;
        border: 2px solid #ddd;
        border-radius: 10px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        background-color: #f9f9f9;
        margin-top: 20px;
    }
    .presenter-header {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 20px;
        text-align: center;
    }
    </style>
    """
    
    st.markdown(presenter_style, unsafe_allow_html=True)

    # Create a container for the presenter selection
    with st.container():
        st.markdown('<div class="presenter-container">', unsafe_allow_html=True)
        st.markdown('<div class="presenter-header">Choose Presenter</div>', unsafe_allow_html=True)

        # Display the presenters in the container
        num_per_row = 1  # Number of presenters per row
        num_rows = (len(presenters) // num_per_row) + 1
        for i in range(num_rows):
            cols = st.columns([1] * num_per_row)
            for j in range(num_per_row):
                index = i * num_per_row + j
                if index < len(presenters):
                    presenter = presenters[index]
                    with cols[j]:
                        st.image(presenter['image_url'], width=70)  # Adjust the image size
                    with cols[j]:
                        if st.button(presenter['name'], key=f"button_{presenter['presenter_id']}"):
                            st.session_state.selected_presenter_id = presenter['presenter_id']
                            st.session_state.selected_presenter_name = presenter['name']

        st.markdown('</div>', unsafe_allow_html=True)

# ---- Main App ----
def main():
    st.set_page_config(page_title="Multi-Functional App", page_icon=":robot:")
    
    if 'chat_output' not in st.session_state:
        st.session_state.chat_output = ""  # Initialize with an empty string
    
    if 'idi_key' not in st.session_state:
        st.session_state.idi_key = ""  # Initialize with an empty string

    # Input box cho API key
    st.sidebar.header("Enter  Gen API Key")
    idi_key_input = st.sidebar.text_input("Gen API Key", type="password", value=st.session_state.idi_key)

    # Submit button to update API key
    if st.sidebar.button("Submit", key="submit_gen_key"):
        st.session_state.idi_key = idi_key_input  # Lưu key vào session state
        st.sidebar.success("API Key updated")
    else:
        st.session_state.idi_key = st.session_state.idi_key

    idi_key = st.session_state.idi_key  # Sử dụng API key từ session state



    if 'chat_key' not in st.session_state:
        st.session_state.chat_key = ""  # Initialize with an empty string

    # Input box cho API key
    st.sidebar.header("Enter Chat API Key")
    chat_key_input = st.sidebar.text_input("Chat API Key", type="password", value=st.session_state.idi_key)

    # Submit button to update API key
    if st.sidebar.button("Submit", key="submit_chat_key"):
        st.session_state.chat_key = chat_key_input  # Lưu key vào session state
        st.sidebar.success("API Key updated")
    else:
        st.session_state.chat_key = st.session_state.chat_key

    # chat_key = st.session_state.chat_key  # Sử dụng API key từ session state
    os.environ["GOOGLE_API_KEY"] = st.session_state.chat_key

    if not idi_key:
        st.sidebar.error("Please enter an API key to proceed.")
    else:
        # Configure Google API Key (you can replace this part with your actual logic for Google API key if needed)
        
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        # print(chat_key)

        # Tabs for different functionality
        page = st.sidebar.radio("Choose a page", ["Generate Avatar Video", "HeyGen Generate Avatar Video", "Chat with Documents"])
        
        # Avatar Video Generator Tab
        if page == "Generate Avatar Video":
            st.title("Generate Avatar Video")

            # If there is output from Chat with Documents, use it as the prompt
            if 'chat_output' in st.session_state:
                prompt = st.text_area("Enter Text Prompt", st.session_state.chat_output)
            else:
                prompt = st.text_area("Enter Text Prompt", "Once upon a time...")

            # presenters = fetch_presenters(idi_key)
            presenters = fetch_presenters()

            if presenters:
                # Create 2-column layout: 1/3 for presenters, 2/3 for results
                left_col, right_col = st.columns([2, 3])

                with left_col:
                    st.header("Choose Presenter")
                    num_per_row = 2
                    for i in range(0, len(presenters), num_per_row):
                        cols = st.columns(num_per_row, vertical_alignment="bottom")
                        for j in range(num_per_row):
                            index = i + j
                            if index < len(presenters):
                                presenter = presenters[index]
                                with cols[j]:
                                    st.image(presenter['image_url'], width=60)
                                    # Add key to button to prevent multiple selections
                                    if st.button(presenter['name'], key=f"button_{presenter['presenter_id']}"):
                                        st.session_state.selected_presenter_id = presenter['presenter_id']
                                        st.session_state.selected_presenter_gender = presenter.get('gender')  # Lưu giới tính

                with right_col:
                    if 'selected_presenter_id' in st.session_state:
                        selected_presenter = next(p for p in presenters if p['presenter_id'] == st.session_state.selected_presenter_id)
                        st.image(selected_presenter['image_url'], caption=selected_presenter['name'], width=250)
                        gender = st.session_state.selected_presenter_gender
                        voices = voice_presenters(gender) 

                        if st.button("Generate Video", key="generate_video"):
                            st.text("Generating video...")

                            if voices:
                                try:   
                                    video_url = generate_video(prompt, st.session_state.selected_presenter_id, voices[0], idi_key)
                                    if video_url != "error":
                                        display_tiktok_video(video_url)
                                    else:
                                        st.text("Sorry... Try again")
                                except Exception as e:
                                    st.text("Sorry... Try again")
                            else:
                                st.text("No voices found for the selected gender.")
        # Avatar Video Generator Tab
        elif page == "HeyGen Generate Avatar Video":
            st.title("HeyGen Generate Avatar Video")

            # Nếu có output từ Chat with Documents, sử dụng nó làm prompt
            if 'chat_output' in st.session_state:
                prompt = st.text_area("Enter Text Prompt", st.session_state.chat_output)
            else:
                prompt = st.text_area("Enter Text Prompt", "Once upon a time...")

            presenters = heygen_fetch_presenters('listavt.json')

            if presenters:
                # Tạo layout 2 cột: 1/3 cho presenters, 2/3 cho kết quả
                left_col, right_col = st.columns([2, 3])

                with left_col:
                    st.header("Choose Presenter")
                    num_per_row = 2
                    for i in range(0, len(presenters), num_per_row):
                        cols = st.columns(num_per_row, vertical_alignment="bottom")
                        for j in range(num_per_row):
                            index = i + j
                            if index < len(presenters):
                                presenter = presenters[index]
                                with cols[j]:
                                    st.image(presenter['preview_image_url'], width=60)
                                    # Thêm key để tránh xung đột ID
                                    if st.button(presenter['avatar_name'], key=f"button_{presenter['avatar_id']}"):
                                        st.session_state.selected_presenter_id = presenter['avatar_id']
                                        st.session_state.selected_presenter_gender = presenter.get('gender')  # Lưu giới tính

                with right_col:
                    if 'selected_presenter_id' in st.session_state:
                        selected_presenter = next(p for p in presenters if p['avatar_id'] == st.session_state.selected_presenter_id)
                        st.image(selected_presenter['preview_image_url'], caption=selected_presenter['avatar_name'], width=250)
                        gender = st.session_state.selected_presenter_gender
                        voices = heygen_voice_presenters(gender) 
                        print(voices[0])

                        if st.button("Generate Video", key="generate_video"):
                            st.text("Generating video...")

                            if voices:
                                try:   
                                    video_url = heygen_generate_video(prompt, st.session_state.selected_presenter_id, voices[0], idi_key)
                                    if video_url != "error":
                                        display_tiktok_video(video_url)
                                    else:
                                        st.text("Sorry... Try again")
                                except Exception as e:
                                    st.text("Sorry... Try again")
                            else:
                                st.text("No voices found for the selected gender.")

        # Chat with Documents Tab
        elif page == "Chat with Documents":
            st.header("Chat with PDF or DOCX files")

            docs = st.file_uploader("Upload your PDF(s) or DOCX file(s)", type=["pdf", "docx"], accept_multiple_files=True)

            if st.button("Submit", key="submit_docs"):
                raw_text = ''
                pdf_docs = [doc for doc in docs if doc.name.endswith(".pdf")]
                docx_docs = [doc for doc in docs if doc.name.endswith(".docx")]

                if pdf_docs:
                    raw_text += get_pdf_text(pdf_docs)
                if docx_docs:
                    raw_text += get_docx_text(docx_docs)

                if raw_text:
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Completed")
            user_question = st.text_input("Ask a Question from the uploaded files")
     
            if st.button("Submit Question", key="submit_question"):
                ans = user_input(user_question)
            st.write("Reply: ", st.session_state.chat_output)
            # Generate Video in Chat with Documents Tab
            if 'chat_output' in st.session_state and st.button("Generate Video from Chat", key="generate_video_chat"):
                st.text("Generating video using default presenter...")
                # presenters = fetch_presenters(idi_key)
                presenters = fetch_presenters()
                
                if presenters:
                    default_presenter_id = presenters[0]['presenter_id']
                    gender = presenters[0].get('gender')
                    voices = voice_presenters(gender) 
                    video_url = generate_video(st.session_state.chat_output, default_presenter_id, voices[0], idi_key)
                    if video_url != "error":
                        display_tiktok_video(video_url)
                    else:
                        st.text("Sorry... Try again")

if __name__ == "__main__":
    main()

