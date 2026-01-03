import streamlit as st
import requests

st.title("Memo-Agent")

def create_note(title, content):
    response = requests.post("http://localhost:8000/notes/", json={"title": title, "content": content})
    return response.json()

def get_notes():
    response = requests.get("http://localhost:8000/notes/")
    return response.json()

def chat(query):
    response = requests.post("http://localhost:8000/chat/", json={"query": query})
    return response.json()

st.header("新增筆記")
title = st.text_input("標題")
content = st.text_area("內容")
if st.button("新增"):
    create_note(title, content)

st.header("所有筆記")
notes = get_notes()
for note in notes:
    st.write(note)

st.header("聊天")
query = st.text_input("問題")
if st.button("送出"):
    response = chat(query)
    st.write(response)
