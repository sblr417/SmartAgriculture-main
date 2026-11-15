import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
import streamlit as st
from dotenv import load_dotenv
import requests

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# -------------------------
# 🔐 Load API Key
# -------------------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Check Streamlit Cloud secrets if not set via environment variable
if not OPENROUTER_API_KEY:
    try:
        OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY")
    except Exception:
        pass

if not OPENROUTER_API_KEY:
    st.error("❌ OPENROUTER_API_KEY not found. Please provide it in your .env file or Streamlit Cloud Secrets.")
    st.stop()

# -------------------------
# 🎨 Streamlit UI
# -------------------------
st.set_page_config(page_title="🌾 Smart Agriculture RAG Bot", layout="wide")

st.title("🌾 Smart Agriculture Crop Process Explainer Bot")
st.markdown("AI-powered educational assistant for crop lifecycle and farming processes.")

# -------------------------
# 📦 Load Vector Store
# -------------------------
@st.cache_resource
def load_vector_store():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    current_dir = os.path.dirname(os.path.abspath(__file__))
    vector_store_path = os.path.join(current_dir, "vector_store")
    
    # Fallback to relative "vector_store" if not at script path
    if not os.path.exists(vector_store_path) and os.path.exists("vector_store"):
        vector_store_path = "vector_store"

    index_file = os.path.join(vector_store_path, "index.faiss")
    if os.path.exists(index_file):
        return FAISS.load_local(vector_store_path, embedding_model, allow_dangerous_deserialization=True)

    # Automatic fallback: if index file was not committed to git, build it on the fly from knowledge_base!
    try:
        from knowledge_base import KNOWLEDGE_BASE
        from langchain_core.documents import Document

        documents = []
        for item in KNOWLEDGE_BASE:
            content = f"{item['text']}\n\nSource: {item['source']}\nReference: {item['url']}\nTopic: {item['topic']}"
            documents.append(
                Document(
                    page_content=content,
                    metadata={"topic": item["topic"], "source": item["source"]}
                )
            )
        vs = FAISS.from_documents(documents, embedding_model)
        try:
            vs.save_local(vector_store_path)
        except Exception:
            pass
        return vs
    except Exception as e:
        raise RuntimeError(f"Failed to load or initialize vector store: {e}")

vectorstore = load_vector_store()

# -------------------------
# 🚨 Safety Filter
# -------------------------
def is_restricted(query):
    restricted_terms = [
        "fertilizer",
        "chemical dosage",
        "pesticide amount",
        "yield prediction",
        "how much urea",
        "profit estimate"
    ]
    return any(term in query.lower() for term in restricted_terms)

# -------------------------
# 🤖 LLM Function (Stable + Safe)
# -------------------------
def generate_answer(query, context):

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://smartagriculture.streamlit.app",
        "X-Title": "Smart Agriculture RAG Bot"
    }

    prompt = f"""
    You are an agriculture education assistant.

    Only provide informational explanations.
    Do NOT provide fertilizer advice, chemical dosage, or yield prediction.

    Context:
    {context}

    Question:
    {query}

    Provide a simple explanation suitable for farmers.
    """

    model_name = os.getenv("OPENROUTER_MODEL", "openrouter/free")

    data = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )

        if response.status_code != 200:
            return f"⚠️ API Error ({response.status_code}): {response.text}"

        result = response.json()

        if "choices" not in result:
            return f"⚠️ Unexpected API response: {result}"

        return result["choices"][0]["message"]["content"]

    except Exception as e:
        return f"⚠️ System Error: {str(e)}"

# -------------------------
# 🔍 Query Interface
# -------------------------
user_query = st.text_input("Ask your agriculture-related question:")

if user_query:

    if is_restricted(user_query):
        st.warning("⚠️ This system provides educational explanations only.")
    else:
        with st.spinner("🔎 Searching knowledge base..."):
            docs = vectorstore.similarity_search(user_query, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])

        with st.spinner("🤖 Generating response..."):
            answer = generate_answer(user_query, context)

        st.success("✅ Answer")
        st.write(answer)

        st.markdown("### 📖 Sources")
        for doc in docs:
            st.write(f"- {doc.metadata.get('source', 'Knowledge Base')}")
