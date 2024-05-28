import os
os.chdir("src")
import numpy as np
import gradio as gr
from for_demo.example import get_example
from src.evaluate import model_predict

def dialogue_classification(dialogue_) :
    pred = model_predict(dialogue_)
    return pred

markdown_content = """
다산콜센터 : 대중교통 안내, 생활하수도 관련 문의, 일반행정 문의, 코로나19 관련 상담\n
금융/보험 : 사고 및 보상 문의, 상품 가입 및 해지, 이체출〮금대〮출서비스, 잔고 및 거래내역
"""
examples = get_example()

with gr.Blocks() as demo:
    gr.Markdown("# 카테고리 분류")
    gr.Markdown(markdown_content)
    
    input_text = gr.Textbox(label="입력")
    output_text = gr.Textbox(label="예측 결과")
    
    btn = gr.Button("분류")
    btn.click(fn=dialogue_classification, inputs=input_text, outputs=output_text)

    gr.Examples(examples=examples, inputs=input_text)

demo.launch(server_name="0.0.0.0", server_port=11031)
