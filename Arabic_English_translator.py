import streamlit as st
from loadmodel import model, tokenizer  

# Set page config for a better look
st.set_page_config(page_title="Tarjuman - Arabic to English Translator", layout="centered")

# Custom CSS for dark mode
st.markdown(
    """
    <style>
        /* Dark background */
        body, .stApp {
            background-color: #121212;
            color: white;
        }
        
        /* Title and subtitles */
        h1, h2, h3 {
            color: #f8f9fa;
            text-align: center;
        }

        /* Text input area */
        textarea {
            background-color: #222;
            color: white;
            border-radius: 10px;
        }

        /* Button styling */
        .stButton>button {
            background-color: #ff6b6b;
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
            font-size: 16px;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #ff4757;
            transform: scale(1.05);
        }

        /* Success box */
        .stSuccess {
            background-color: #1e7e34;
            color: white;
            font-size: 18px;
            padding: 15px;
            border-radius: 10px;
        }
        
        /* Warning box */
        .stWarning {
            background-color: #f39c12;
            color: white;
            font-size: 16px;
            padding: 10px;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Translation function
def translate(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model.generate(**inputs, max_length=128, num_beams=5, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Title
st.title("🌙 Tarjuman - Arabic to English Translator")
st.write("Enter Arabic text below and get an English translation.")

# Text input
text = st.text_area("✍️ أدخل النص العربي هنا:", "")

# Translate button
if st.button("Translate 🔄"):
    if text.strip():
        english_translation = translate(text)
        st.markdown("### 🌐 **English Translation:**")
        st.success(english_translation)
    else:
        st.warning("⚠️ يرجى إدخال نص للترجمة.")  # Please enter some text.
