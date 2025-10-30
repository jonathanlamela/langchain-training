import streamlit as st
from streamlit_prepare import qa

st.title("Bible QA Bot")
question = st.text_input("Domanda")
if st.button("Invia"):
    answer = qa.invoke(question)
    st.write(answer)
