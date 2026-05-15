import streamlit as st
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt")

st.set_page_config(
    page_title="FAQ Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 FAQ Chatbot")
st.write("Ask questions related to CodeAlpha AI Internship and get instant answers.")

faqs = {
    "What is CodeAlpha?":
    "CodeAlpha is a software development company that provides internships and project-based learning opportunities.",

    "What is the Artificial Intelligence internship about?":
    "The Artificial Intelligence internship helps students gain practical experience in AI, machine learning, NLP, and real-world projects.",

    "How many tasks do I need to complete?":
    "You need to complete at least two or three tasks to be eligible for the internship certificate.",

    "Will I get a certificate?":
    "Yes, you will receive a completion certificate after successfully completing the required tasks.",

    "What are the perks of the internship?":
    "The perks include an internship offer letter, completion certificate, unique ID certificate, letter of recommendation based on performance, placement support, and resume building support.",

    "Where should I upload my project?":
    "You should upload your complete source code to GitHub in a repository named CodeAlpha_ProjectName.",

    "Do I need to post my project on LinkedIn?":
    "Yes, you need to post a video explanation of your project on LinkedIn with the GitHub repository link.",

    "How do I submit my completed task?":
    "You need to submit your completed task using the submission form shared in your WhatsApp group.",

    "What is Task 1?":
    "Task 1 is a Language Translation Tool where users can enter text, select source and target languages, and get translated output using a translation API.",

    "What is Task 2?":
    "Task 2 is a Chatbot for FAQs. It answers user questions by matching them with the most similar FAQ.",

    "What is Task 3?":
    "Task 3 is Music Generation with AI. It uses MIDI music data and deep learning models like LSTM to generate new music sequences.",

    "What is Task 4?":
    "Task 4 is Object Detection and Tracking. It uses OpenCV and models like YOLO or Faster R-CNN to detect and track objects in video.",

    "What is NLP?":
    "NLP stands for Natural Language Processing. It helps computers understand and process human language.",

    "What is tokenization?":
    "Tokenization is the process of breaking text into smaller parts such as words or sentences.",

    "What is text preprocessing?":
    "Text preprocessing means cleaning text before using it in a machine learning model. It can include lowercasing, removing punctuation, and tokenization.",

    "What is TF-IDF?":
    "TF-IDF stands for Term Frequency-Inverse Document Frequency. It converts text into numerical vectors based on word importance.",

    "What is cosine similarity?":
    "Cosine similarity is a technique used to measure how similar two text documents are.",

    "Which technologies are used in this chatbot?":
    "This chatbot is built using Python, Streamlit, NLTK, Scikit-learn, TF-IDF, and cosine similarity."
}

questions = list(faqs.keys())
answers = list(faqs.values())

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

processed_questions = [preprocess_text(question) for question in questions]

def get_response(user_question):
    user_question = preprocess_text(user_question)

    all_questions = processed_questions + [user_question]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_questions)

    similarity = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    best_match_index = similarity.argmax()
    best_score = similarity[0][best_match_index]

    if best_score < 0.2:
        return "Sorry, I could not find a relevant answer. Please ask a question related to CodeAlpha or AI internship tasks."

    return answers[best_match_index]

user_input = st.text_input("Enter your question:")

if st.button("Ask"):
    if user_input.strip() == "":
        st.warning("Please enter a question first.")
    else:
        response = get_response(user_input)

        st.subheader("Chatbot Response:")
        st.success(response)

st.subheader("Sample Questions")
st.write("You can ask questions like:")

st.write("- What is CodeAlpha?")
st.write("- What is Task 2?")
st.write("- How many tasks do I need to complete?")
st.write("- Will I get a certificate?")
st.write("- What is NLP?")
st.write("- What is cosine similarity?")

with st.expander("View FAQ Knowledge Base"):
    for question, answer in faqs.items():
        st.write("**Question:**", question)
        st.write("**Answer:**", answer)
        st.write("---")