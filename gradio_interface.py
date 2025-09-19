import gradio as gr
from rag_backend import qa_chain

def respond(history, question):
    # Gọi RAG
    result = qa_chain.invoke({"query": question})
    answer = result["result"]
    sources = [doc.metadata.get("source","") for doc in result["source_documents"]]
    # Thêm cặp mới vào history
    history = history + [(question, answer)]
    # Trả về (cập nhật) lịch sử chat và chuỗi nguồn
    return history, "\n".join(sources)

with gr.Blocks(theme=gr.themes.Default(), css="""
#header {text-align: center; margin-bottom: 1em;}
#query-box {width: 100%;}
#answer-box {height: 300px; overflow-y: auto;}
#source-box {height: 150px; overflow-y: auto; white-space: pre-wrap;}
""") as demo:

    # --- Header ---
    gr.Markdown("## 📄 RAG PDF Reader", elem_id="header")
    gr.Markdown("> Nhập câu hỏi, hệ thống sẽ tìm trong PDF và trả lời kèm trang nguồn.")

    # --- Main content: two columns ---
    with gr.Row():
        with gr.Column(scale=2):
            question = gr.Textbox(
                label="❓ Câu hỏi",
                placeholder="Ví dụ: Hãy tóm tắt Clean Code...",
                elem_id="query-box",
                lines=3
            )
            submit_btn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="💬 Trả lời", elem_id="answer-box")
            source_box = gr.Textbox(label="📑 Các trang nguồn", elem_id="source-box")

    # --- Footer: Clear & Flag ---
    with gr.Row():
        clear_btn = gr.Button("Clear", variant="secondary")
        flag_btn  = gr.Button("Flag", variant="danger")

    # --- Events ---
    submit_btn.click(
        fn=respond,
        inputs=[chatbot, question],          # truyền vào history + câu hỏi
        outputs=[chatbot, source_box]        # cập nhật lại history và sources
    )
    # Clear: reset chat và source, đồng thời clear question
    clear_btn.click(lambda: ([], ""), None, [chatbot, source_box])
    clear_btn.click(lambda: "", None, question)

demo.launch(share=True)
