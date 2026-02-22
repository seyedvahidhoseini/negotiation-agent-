# ============================================
# Persian RAG - FAISS Index Builder (JSON Version)
# Optimized for rag_ready.json
# ============================================

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from hazm import Normalizer, word_tokenize

import faiss
import numpy as np
import json
import re
import time


# =====================================
# 1️⃣ Load RAG JSON (Already Chunked)
# =====================================

json_path = "output/rag_ready.json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"[INFO] Loaded {len(data)} chunks from JSON")

docs = [
    Document(
        page_content=item["content"],
        metadata=item["metadata"]
    )
    for item in data
]


# =====================================
# 2️⃣ Shared Persian preprocessing
# =====================================

normalizer = Normalizer()

def preprocess(text: str) -> str:
    try:
        text = text.lower()
        text = normalizer.normalize(text)

        # حفظ علائم مهم جمله
        text = re.sub(r"[^\w\s\u0600-\u06FF?.!]", " ", text)

        tokens = word_tokenize(text)
        return " ".join(tokens)

    except Exception as e:
        print(f"[WARNING] preprocessing error: {e}")
        return ""


print("[INFO] Preprocessing chunks...")
texts = [preprocess(doc.page_content) for doc in docs]


# =====================================
# 3️⃣ Embedding (Ollama)
# =====================================

embedding_model = OllamaEmbeddings(
    model="qwen3-embedding:8b"
)

print("[INFO] Computing embeddings...")
start_time = time.perf_counter()

embeddings = embedding_model.embed_documents(texts)

elapsed = time.perf_counter() - start_time
print(f"[INFO] Embeddings created in {elapsed:.2f} seconds")


# =====================================
# 4️⃣ FAISS (Cosine Similarity)
# =====================================

FAISS_SETTINGS = {
    "metric": "Cosine",
    "index_type": "Flat",
    "k": 3
}

dimension = len(embeddings[0])

# تبدیل به numpy
embeddings = np.array(embeddings).astype("float32")

# Cosine normalization (صحیح و بهینه)
faiss.normalize_L2(embeddings)

index = faiss.IndexFlat(dimension, faiss.METRIC_INNER_PRODUCT)


# =====================================
# 5️⃣ Docstore mapping
# =====================================

docstore = InMemoryDocstore({
    str(i): doc for i, doc in enumerate(docs)
})

index_to_docstore_id = {
    i: str(i) for i in range(len(docs))
}


# =====================================
# 6️⃣ Build VectorStore
# =====================================

vectorstore = FAISS(
    embedding_model,
    index,
    docstore,
    index_to_docstore_id,
)

print("[INFO] Adding vectors to FAISS...")
vectorstore.index.add(embeddings)


# =====================================
# 7️⃣ Retriever
# =====================================

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": FAISS_SETTINGS["k"]}
)


# =====================================
# 8️⃣ Save FAISS index
# =====================================

print("[INFO] Saving FAISS index...")
vectorstore.save_local("my_faiss_index")

print("[SUCCESS] Index saved successfully.")


# =====================================
# 9️⃣ Example Query (Aligned)
# =====================================

def search(query: str):
    query = preprocess(query)
    results = retriever.get_relevant_documents(query)

    print("\n🔎 Results:\n")
    for i, doc in enumerate(results, 1):
        print(f"Result {i}")
        print("Page:", doc.metadata.get("page"))
        print("Section:", doc.metadata.get("section"))
        print("Type:", doc.metadata.get("type"))
        print("Content:", doc.page_content[:300])
        print("-" * 50)


# تست نمونه
# search("موضوع اصلی این سند چیست؟")
