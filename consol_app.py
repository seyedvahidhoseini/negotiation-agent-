import os
import re
from typing import List, Optional, Generator

from flask import Flask, request, jsonify
from rank_bm25 import BM25Okapi

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

# -----------------------------
# کلاس BM25 Reranker
# -----------------------------
class BM25Reranker:
    base_retriever = None
    top_k = 3

    def __init__(self, base_retriever):
        self.base_retriever = base_retriever

    def _tokenize(self, text: str) -> List[str]:
        persian = "۰۱۲۳۴۵۶۷۸۹"
        arabic = "٠١٢٣٤٥٦٧٨٩"
        trans = {ord(p): ord('0')+i for i,p in enumerate(persian)}
        trans.update({ord(a): ord('0')+i for i,a in enumerate(arabic)})
        text = text.translate(trans).lower()
        tokens = re.findall(r"[a-zA-Z\u0600-\u06FF]+|\d+|[%٪]", text)
        return tokens

    def invoke(self, query):
        docs = self.base_retriever.invoke(query)
        if not docs:
            return []
        if len(docs) <= self.top_k:
            return docs

        tokenized_docs = [self._tokenize(d.page_content) for d in docs]
        bm25 = BM25Okapi(tokenized_docs)
        scores = bm25.get_scores(self._tokenize(query))
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:self.top_k]
        return [doc for doc, _ in ranked]

# -----------------------------
# پرامپت سیستم
# -----------------------------
system_template = """
تو یک دستیار مذاکره انسانی هستی.
اگر از تو اطلاعات عمومی مانند نام و ایمیل خواستن بر اساس متن های مرجع پاسخ بده
وظیفه تو:
- کارفرما با تو صحبت می‌کند و می‌خواهد بداند فرد چه مهارت‌هایی دارد و چگونه می‌تواند ارزش ایجاد کند.
- تو بر اساس متن‌های مرجع (مهارت‌های فرد) پاسخ می‌دهی.
- اگر موضوعی خارج از مهارت‌های فرد باشد، صادقانه بگو که نمی‌توانی انجام دهی و پیشنهادی منطقی ارائه بده.
- لحن پاسخ‌ها باید طبیعی، دوستانه و انسانی باشد، شبیه یک شخص واقعی.

متن‌های مرجع (مهارت‌های فرد):
{rag_context}

تاریخچه گفتگو:
{history}

پرسش فعلی کارفرما:
{input}

راهنما برای پاسخ:
1. فقط بر اساس مهارت‌ها و اطلاعات متن‌های مرجع صحبت کن.
2. اگر نمی‌توانی کار خاصی انجام دهی، با لحنی انسانی و محترمانه بیان کن.
3. از جملات کلیشه‌ای یا پاسخ‌های غیرواقعی خودداری کن.
4. هدف این است که کارفرما راضی شود، اما با صداقت و رعایت محدودیت‌ها.
5سعی کن کوتاه و خلاصه جواب بدهی
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# -----------------------------
# متغیرهای سراسری
# -----------------------------
_embedding_model = None
_llm = None
_vectorstore = None
_retriever = None

# ذخیره تاریخچه چت
chat_store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in chat_store:
        chat_store[session_id] = InMemoryChatMessageHistory()
    return chat_store[session_id]

# -----------------------------
# مقداردهی اولیه مدل‌ها
# -----------------------------
def initialize_models(
    embeddings_name="qwen3-embedding:8b",
    llm_name="gpt-oss:20b",
):
    global _embedding_model, _llm, _vectorstore, _retriever

    _embedding_model = OllamaEmbeddings(model=embeddings_name)
    _llm = ChatOllama(model=llm_name, temperature=0)

    index_dir = "my_faiss_index"
    index_path = os.path.join(index_dir, "index.faiss")

    if os.path.exists(index_path):
        _vectorstore = FAISS.load_local(
            index_dir,
            embeddings=_embedding_model,
            allow_dangerous_deserialization=True  # ⚠️ فقط اگر ایندکس امن است
        )
        base_retr = _vectorstore.as_retriever(search_type="mmr", search_kwargs={"k":6, "fetch_k":15})
        _retriever = BM25Reranker(base_retr)
        print("✅ RAG فعال است")
    else:
        print("⚠️ RAG غیرفعال است (فایل index پیدا نشد)")

# -----------------------------
# تابع استریم پاسخ با تاریخچه
# -----------------------------
def ask_with_history(session_id: str, question: str, use_rag=True):
    context = ""
    if use_rag and _retriever:
        docs = _retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)

    chain = prompt | _llm

    with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )

    output = with_history.invoke(
        {"input": question, "rag_context": context},
        config={"configurable": {"session_id": session_id}}
    )

    return output.content if hasattr(output, "content") else str(output)

# -----------------------------
# API با Flask
# -----------------------------
app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    q = data.get("question", "").strip()
    sess = data.get("session_id", "default_session")
    use_rag = data.get("use_rag", True)

    if not q:
        return jsonify({"error": "سوال نمی‌تواند خالی باشد."}), 400

    answer = ask_with_history(sess, q, use_rag)
    return jsonify({"session_id": sess, "answer": answer})

# -----------------------------
# اجرای اصلی
# -----------------------------
if __name__ == "__main__":
    initialize_models()
    app.run(host="0.0.0.0", port=8000)