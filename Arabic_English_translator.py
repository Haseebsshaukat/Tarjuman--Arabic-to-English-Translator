import streamlit as st
from loadmodel import model, tokenizer  # Now it will work correctly

# Translation function
def translate(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model.generate(**inputs, max_length=128, num_beams=5, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

st.title("Tarjuman - Arabic to English Translator")
st.write("Enter Arabic text below and get an English translation.")

text = st.text_area("✍️ أدخل النص العربي هنا:", "")

if st.button("Translate"):
    if text.strip():
        english_translation = translate(text)
        st.write("### 🌐 English Translation:")
        st.success(english_translation)
    else:
        st.warning("⚠️ يرجى إدخال نص للترجمة.")  # Please enter some text.
