# %%
import os
import sys
import subprocess
import shutil
import io


if os.path.isdir("chromadb"):
    print("Đang xóa thư mục chromadb cũ...")
    shutil.rmtree("chromadb")


def ensure_package(pkg_name, import_name=None):
    try:
        __import__(import_name or pkg_name)
    except ImportError:
        print(f"Thiếu thư viện '{pkg_name}', đang cài đặt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])
        __import__(import_name or pkg_name)

pkgs = [
    ("chromadb", None),
    ("langchain", None),
    ("ollama", None),
    ("tiktoken", None),
    ("PyPDF2", None)
]

for pkg, imp in pkgs:
    ensure_package(pkg, imp)

# %%

import os
from PyPDF2 import PdfReader
from langchain.schema import Document

pdf_path = r"D:\Project_self\pdf_place\CleanCode.pdf"  
reader = PdfReader(pdf_path)
docs = []

for i, page in enumerate(reader.pages):
    raw_text = page.extract_text() or ""
    if isinstance(raw_text, (bytes, bytearray)):
        text = raw_text.decode("utf-8", errors="ignore")
    else:
        text = raw_text
    docs.append(Document(
        page_content=text,
        metadata={"source": f"{os.path.basename(pdf_path)}_page_{i+1}"}
    ))

# %%
from PyPDF2 import PdfReader
from langchain.schema import Document
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# %%
embed_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# %%
# --- 7. Tạo hoặc tải lại vector database local với fallback khi embed thất bại ---
try:
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embed_model,
        persist_directory="chromadb"
    )
    vectordb.persist()
    print("🎉 VectorDB khởi tạo thành công với model embedding hiện tại.")
    
except ValueError as e:
    print(f"[Warning] Embed thất bại: {e}")
    print("Chuyển sang model embedding thay thế: sentence-transformers/all-MiniLM-L6-v2")
    from langchain.embeddings import SentenceTransformerEmbeddings
    fallback_embed = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=fallback_embed,
        persist_directory="chromadb"
    )
    vectordb.persist()
    embed_model = fallback_embed  # cập nhật cho phần truy vấn sau
    print("🎉 VectorDB đã khởi tạo lại với MiniLM embedding.")

# %%
from dotenv import load_dotenv
import os, requests

load_dotenv()  

API_KEY    = os.getenv("GEMINI_API_KEY")
MODEL_ID   = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
PROMPT     = os.getenv(
    "GEMINI_PROMPT",
    "Bạn là một trợ lý hữu ích. Hãy trả lời đầy đủ và chi tiết. Đừng lặp lại câu trả lời nếu không cần thiết."
)
MAX_TOKENS = int(os.getenv("GEMINI_MAX_TOKENS", "256"))
TEMP       = float(os.getenv("GEMINI_TEMPERATURE", "0.7"))

# Kiểm tra
print("API Key   :", "✓" if API_KEY else "✗")
print("Model     :", MODEL_ID)
print("MaxTokens :", MAX_TOKENS)
print("Temperature:", TEMP)

# %%

import time
import requests

last_call_time = 0

# Danh sách model ưu tiên
MODEL_FALLBACKS = [
    "gemini-1.5-pro",
    "gemini-2.0",
    "gemini-2.0-flash",
    "gemini-1.5-flash"
]

def call_gemini(prompt: str) -> str:
    global last_call_time

    if not API_KEY:
        raise ValueError("Please set GEMINI_API_KEY in .env")

    for model in MODEL_FALLBACKS:
        now = time.time()
        if now - last_call_time < 0.5:
            time.sleep(0.5 - (now - last_call_time))
        last_call_time = time.time()

        print(f"\n;-; Đang thử model: {model}")

        # Xác định version
        if model.startswith("gemini-2.0") or model.startswith("gemini-1.5"):
            version = "v1"
        else:
            version = "v1beta"  # cho exp cũ (nếu thêm sau)
        endpoint = ":generateContent"

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": MAX_TOKENS,
                "temperature": TEMP
            }
        }

        url = f"https://generativelanguage.googleapis.com/{version}/models/{model}{endpoint}?key={API_KEY}"
        print(f"[DEBUG] Gọi URL: {url}")

     
        for attempt in range(3):
            resp = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )

            if resp.status_code == 429:
                print(f"[429] Hết quota hoặc quá tải ở {model}. Thử lại sau...")
                break  

            if resp.status_code == 404:
                print(f"[404] Model {model} không tồn tại. Bỏ qua.")
                break

            try:
                resp.raise_for_status()
                data = resp.json()
                cand = data.get("candidates", [{}])[0]
                out = cand.get("output", "") or cand.get("content", "")
                if isinstance(out, str):
                    return out
                parts = out.get("parts", [])
                return "".join([p.get("text", "") for p in parts])
            except Exception as e:
                print(f"[ERROR] Model {model} gặp lỗi: {e}")
                break

    raise RuntimeError("❌ Tất cả model đều hết quota hoặc lỗi.")


# %%
# Cell 4: Test lại call_gemini
# test_prompt = "Tóm tắt chương 1 của tài liệu PDF cho tôi."
# print("👉 Prompt:", test_prompt)
# print("\n🎉 Gemini trả lời:\n", call_gemini(test_prompt))

# %%
# Cell 5: Định nghĩa wrapper GeminiLLM dùng call_gemini
from langchain.llms.base import LLM
from pydantic import BaseModel
from typing import Optional, List

class GeminiLLM(LLM, BaseModel):
    model_name: str
    api_key:    str
    max_output_tokens: int = 5000
    temperature:       float = 0.6

    @property
    def _llm_type(self) -> str:
        return "gemini"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # Gọi hàm call_gemini đã định nghĩa ở Cell 3
        return call_gemini(prompt)

    async def _acall(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return self._call(prompt, stop)

# %%
# Cell 6: Khởi tạo pipeline RAG với GeminiLLM và vectordb
from langchain.chains import RetrievalQA

# 1) Khởi tạo GeminiLLM
gemini_llm = GeminiLLM(
    model_name=MODEL_ID,            # từ Cell 2
    api_key=API_KEY,                # từ Cell 2
    max_output_tokens=MAX_TOKENS,   # từ Cell 2
    temperature=TEMP                # từ Cell 2
)

# 2) Tạo RetrievalQA chain
retriever = vectordb.as_retriever(search_kwargs={"k": 10})
qa_chain = RetrievalQA.from_chain_type(
    llm=gemini_llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 3) Định nghĩa hàm ask để hỏi và in kết quả + nguồn
def ask(question: str):
    result = qa_chain.invoke({"query": question})
    print("=== ANSWER ===")
    print(result["result"])
    print("\n=== SOURCES ===")
    for doc in result["source_documents"]:
        print(f"- {doc.metadata.get('source', '')}")


# %% test run
ask("describe details as much as you can about Smell and Heuristic General.")


