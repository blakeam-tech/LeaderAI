"""
LeaderAI: Explore World Leaders' Strategies on Global Issues
"""
import datetime
import time
import streamlit as st
from utils.output_prediction_class import Output
from database.database import FirestoreService


# Streamlit page configuration
st.set_page_config(page_title="LeaderAI", page_icon=":robot:")


def add_custom_css():
    """Add custom CSS for chat interface."""
    st.markdown(
        """
    <style>
        body {
            overflow-x: hidden; /* Ensure no horizontal overflow */
        }
        .chatbox {
            background-color: #fafafa;
            border-radius: 10px;
            padding: 10px;
            word-wrap: break-word;
            overflow-wrap: break-word;
            width: 100%;
            max-width: 90vw; /* Maximum width */
            margin: auto; /* Centering */
            opacity: 0; /* For fade-in effect */
            transition: opacity 0.5s ease-in; /* Smooth transition */
        }
        .chatbox.show {
            opacity: 1; /* Make visible */
        }
        .message {
            color: #333;
            font-size: 16px;
            padding: 8px;
            border-radius: 8px;
            border: 1px solid #ddd;
            display: block;
            white-space: pre-wrap; /* Wrap text */
            overflow-x: hidden; /* Prevent horizontal overflow */
        }
        .bot-message {
            background-color: #e1f5fe;
        }
        .user-message {
            background-color: #c8e6c9;
        }
        .stButton > button {
            width: 100%;
            border-radius: 20px;
            background-color: #4CAF50;
            color: white;
        }
    </style>
    <script>
        function fadeInElements() {
            setTimeout(() => {
                const elements = document.querySelectorAll('.chatbox');
                elements.forEach(element => {
                    element.classList.add('show');
                });
            }, 500); // Delay for elements to be ready
        }
        window.addEventListener('load', fadeInElements);
    </script>
    """,
        unsafe_allow_html=True,
    )


add_custom_css()

st.title("LeaderAI")
st.subheader("Explore World Leaders' Strategies on Global Issues")

leader = st.selectbox(
    "Select A World Leader",
    options=["Xi Jinping"],  # Currently only Xi Jinping is available
    index=0,  # Default selection
)


@st.cache_resource
def initialize_firestore():
    """Initialize Firestore service."""
    return FirestoreService("firebase_credentials.json")


firebase_client = initialize_firestore()
leader_ai = Output()


def get_chatbot_response(user_input):
    """Get chatbot response based on user input."""
    json_response = leader_ai.generate_text(user_input)
    return json_response["text"]


def typing_animation(message):
    """Display typing animation for the chatbot response."""
    placeholder = st.empty()
    for i in range(1, len(message) + 1):
        placeholder.markdown(
            f"<span style='word-wrap: break-word; display: block; white-space: pre-wrap;'>"
            f"{message[:i]}</span>",
            unsafe_allow_html=True,
        )
        time.sleep(0.005)
    placeholder.markdown(
        f"<span style='word-wrap: break-word; display: block; white-space: pre-wrap;'>{message}</span>",
        unsafe_allow_html=True,
    )


if "messages" not in st.session_state:
    st.session_state["messages"] = []

user_input = st.text_area(
    "Enter your message:",
    key="user_input",
    height=100,
    max_chars=None,
    help="Type your message here.",
)
if st.button("Send"):
    if user_input:
        response = get_chatbot_response(user_input)
        st.session_state["messages"].append(
            {"role": "user", "content": user_input, "response": response}
        )
        typing_animation(response)

for msg in st.session_state["messages"]:
    with st.container():
        st.markdown(
            f'<div class="chatbox"><div class="message user-message">{msg["content"]}</div>'
            f'<div class="message bot-message">{msg["response"]}</div></div>',
            unsafe_allow_html=True,
        )

with st.sidebar:
    with st.form("feedback_form", clear_on_submit=False):
        qualitative_feedback = st.text_area(
            "Enter your qualitative feedback:", height=100
        )
        quantitative_feedback = st.slider("Rate the response:", 0, 10, 1)
        submitted = st.form_submit_button("Submit Feedback")
        if submitted:
            feedback_data = {
                "model_output": st.session_state["messages"][-1]["response"],
                "prompt": st.session_state["messages"][-1]["content"],
                "qualitative_feedback": qualitative_feedback,
                "quantitative_feedback": quantitative_feedback,
                "time": datetime.datetime.now().isoformat(),
            }
            firebase_client.add_document("leader-ai", feedback_data)
            st.success("Feedback submitted successfully!")