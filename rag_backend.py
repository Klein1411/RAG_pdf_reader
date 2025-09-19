# %%
import os
import sys
import subprocess
import shutil
import io

# --- 1. Xoá thư mục chromadb cũ nếu tồn tại ---
if os.path.isdir("chromadb"):
    print("Đang xóa thư mục chromadb cũ...")
    shutil.rmtree("chromadb")

# --- 3. Kiểm tra và cài đặt các thư viện cần thiết ---
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
# --- Đọc PDF với decoder UTF-8 để tránh lỗi tuple ---
import os
from PyPDF2 import PdfReader
from langchain.schema import Document

pdf_path = r"D:\Project_self\pdf_place\CleanCode.pdf"  # Thay đường dẫn tới file PDF của bạn
reader = PdfReader(pdf_path)
docs = []

for i, page in enumerate(reader.pages):
    raw_text = page.extract_text() or ""
    # Nếu raw_text là bytes, decode bằng utf-8 và bỏ ký tự không hợp lệ
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
# Sử dụng embedding model local
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
    # Cài và dùng MiniLM embedding local thay thế
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

# ✅ đọc .env ngay trong thư mục project (mặc định)
load_dotenv()

API_KEY    = os.getenv("OPENROUTER_API_KEY")
MODEL_ID   = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")
PROMPT     = os.getenv(
    "OPENROUTER_PROMPT",
    "Bạn là một trợ lý hữu ích. Hãy trả lời đầy đủ và chi tiết. Đừng lặp lại câu trả lời nếu không cần thiết."
)
MAX_TOKENS = int(os.getenv("OPENROUTER_MAX_TOKENS", "512"))
TEMP       = float(os.getenv("OPENROUTER_TEMPERATURE", "0.7"))

# Kiểm tra
print("API Key   :", "✓" if API_KEY else "✗")
print("Model     :", MODEL_ID)
print("MaxTokens :", MAX_TOKENS)
print("Temperature:", TEMP)



# %%
# Cell 3: Định nghĩa hàm call_gemini(prompt) – đảm bảo luôn return str
def call_openrouter(prompt: str) -> str:
    global last_call_time

    if not API_KEY:
        raise ValueError("❌ Thiếu OPENROUTER_API_KEY trong .env")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY.strip()}",
        "Content-Type": "application/json",
        "X-Title": "MyRAGApp"
    }
    payload = {
        "model": "meta-llama/llama-3.1-405b-instruct",   # 👈 thay bằng model mạnh
        "messages": [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMP,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        print("❌ HTTP", resp.status_code)
        print("Resp error:", resp.text)
        resp.raise_for_status()

    data = resp.json()
    return data["choices"][0]["message"]["content"]



# %%
# Cell 4: Test lại call_gemini
# test_prompt = "Tóm tắt chương 1 của tài liệu PDF cho tôi."
# print("👉 Prompt:", test_prompt)
# print("\n🎉 Gemini trả lời:\n", call_gemini(test_prompt))

# %%
# Cell 5: Định nghĩa wrapper GeminiLLM dùng call_gemini
from typing import List, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.outputs import ChatResult, ChatGeneration
import requests

class OpenRouterLLM(BaseChatModel):
    model: str
    api_key: str
    max_tokens: int = 512
    temperature: float = 0.7
    url: str = "https://openrouter.ai/api/v1/chat/completions"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs
    ) -> ChatResult:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": m.content}
                for m in messages
                if isinstance(m, HumanMessage)
            ],
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }

        resp = requests.post(self.url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))]
        )

    @property
    def _llm_type(self) -> str:
        return "openrouter_custom"


# %%
# Cell 6: Khởi tạo pipeline RAG với GeminiLLM và vectordb
from langchain.chains import RetrievalQA

# 1) Khởi tạo GeminiLLM
llm = OpenRouterLLM(
    model="meta-llama/llama-3.1-405b-instruct",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    max_tokens=512,
    temperature=0.7,
)
retriever = vectordb.as_retriever(search_kwargs={"k": 4})
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
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
# ask("describe details as much as you can about Smell and Heuristic General.")


