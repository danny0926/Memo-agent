import streamlit as st
import requests
import os

# API URL 從環境變數取得，預設為 localhost
API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Memo-Agent",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Memo-Agent")
st.markdown("*你的個人知識庫助手*")


def create_note(title: str, content: str):
    """建立新筆記"""
    try:
        response = requests.post(
            f"{API_URL}/notes/",
            json={"title": title, "content": content},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def get_notes():
    """取得所有筆記"""
    try:
        response = requests.get(f"{API_URL}/notes/", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"無法取得筆記: {e}")
        return []


def chat(query: str):
    """與 AI 對話"""
    try:
        response = requests.post(
            f"{API_URL}/chat/",
            json={"query": query},
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"answer": f"錯誤: {e}", "sources": []}


# 側邊欄 - 新增筆記
with st.sidebar:
    st.header("📝 新增筆記")
    title = st.text_input("標題", placeholder="輸入筆記標題...")
    content = st.text_area("內容 (支援 Markdown)", placeholder="輸入筆記內容...", height=200)
    
    if st.button("新增筆記", type="primary", use_container_width=True):
        if title and content:
            with st.spinner("正在處理..."):
                result = create_note(title, content)
                if "error" in result:
                    st.error(f"建立失敗: {result['error']}")
                else:
                    st.success("✅ 筆記建立成功！")
                    st.info(f"📋 摘要: {result.get('summary', 'N/A')}")
                    st.info(f"🏷️ 標籤: {result.get('tags', 'N/A')}")
        else:
            st.warning("請填寫標題和內容")

# 主頁面 - 分成兩個 Tab
tab1, tab2 = st.tabs(["💬 AI 對話", "📚 所有筆記"])

# Tab 1: AI 對話
with tab1:
    st.header("與你的知識庫對話")
    
    # 初始化對話歷史
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 顯示對話歷史
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("📖 參考來源"):
                    for source in message["sources"]:
                        st.markdown(f"- **{source['title']}** (相似度: {source['score']:.2%})")
    
    # 對話輸入
    if prompt := st.chat_input("輸入你的問題..."):
        # 顯示使用者訊息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 取得 AI 回答
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                result = chat(prompt)
                answer = result.get("answer", "抱歉，無法取得回答")
                sources = result.get("sources", [])
                
                st.markdown(answer)
                if sources:
                    with st.expander("📖 參考來源"):
                        for source in sources:
                            st.markdown(f"- **{source['title']}** (相似度: {source['score']:.2%})")
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })

# Tab 2: 所有筆記
with tab2:
    st.header("筆記清單")
    
    if st.button("🔄 重新整理"):
        st.rerun()
    
    notes = get_notes()
    
    if not notes:
        st.info("目前沒有任何筆記，請先新增筆記！")
    else:
        for note in notes:
            with st.expander(f"📄 {note['title']}", expanded=False):
                st.markdown(f"**摘要:** {note['summary']}")
                st.markdown(f"**標籤:** {note['tags']}")
                st.markdown(f"**建立時間:** {note['created_at']}")
                st.divider()
                st.markdown("**內容:**")
                st.markdown(note['content'])

