import torch
import os
import argparse
import gradio as gr
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from tqdm import tqdm
import tempfile
import shutil
import zipfile
import time
import gc
import sys
from transformers.utils import logging
import glob
from pathlib import Path

DEFAULT_PROMPT = "Write a long descriptive caption for this image in a formal tone."
MODELS_DIR = "./models"

# Hugging Face Hub上的模型ID
HF_MODELS = {
    "Beta One": "fancyfeast/llama-joycaption-beta-one-hf-llava",
    "Alpha Two": "fancyfeast/llama-joycaption-alpha-two-hf-llava"
}

# 本地模型路径
LOCAL_MODELS = {
    "Beta One": os.path.join(MODELS_DIR, "llama-joycaption-beta-one-hf-llava"),
    "Alpha Two": os.path.join(MODELS_DIR, "llama-joycaption-alpha-two-hf-llava")
}

# 优先使用本地模型，如果不存在则使用Hugging Face Hub模型
AVAILABLE_MODELS = {}
for model_name, local_path in LOCAL_MODELS.items():
    if os.path.exists(local_path):
        AVAILABLE_MODELS[model_name] = local_path
    else:
        AVAILABLE_MODELS[model_name] = HF_MODELS[model_name]

DEFAULT_MODEL_NAME = AVAILABLE_MODELS["Beta One"]

# 自定义下载进度条
class DownloadProgressBar:
    def __init__(self):
        self.current_file = None
        self.total_size = 0
        self.downloaded = 0
        self.pbar = None

    def __call__(self, info):
        if info["file_name"] != self.current_file:
            if self.pbar is not None:
                self.pbar.close()
            self.current_file = info["file_name"]
            self.total_size = info.get("total_size", 0)
            self.downloaded = 0
            if self.total_size > 0:
                self.pbar = tqdm(total=self.total_size, unit='B', unit_scale=True, 
                                desc=f"下载 {os.path.basename(self.current_file)}")
        
        chunk_size = info.get("chunk_size", 0)
        self.downloaded += chunk_size
        if self.pbar is not None:
            self.pbar.update(chunk_size)

def load_model(model_name):
    print(f"正在加载模型: {model_name}")
    
    # 检查是否是本地路径
    is_local = os.path.exists(model_name)
    
    if is_local:
        print(f"从本地路径加载模型: {model_name}")
        processor = AutoProcessor.from_pretrained(model_name)
        llava_model = LlavaForConditionalGeneration.from_pretrained(model_name, torch_dtype="bfloat16", device_map=0)
    else:
        print(f"从Hugging Face Hub下载模型: {model_name}")
        progress_callback = DownloadProgressBar()
        
        # 设置日志级别为ERROR，减少不必要的输出
        logging.set_verbosity_error()
        
        print("下载模型处理器...")
        processor = AutoProcessor.from_pretrained(model_name, use_auth_token=False, 
                                                force_download=False, resume_download=True,
                                                proxies=None, local_files_only=False,
                                                token=None, revision=None,
                                                user_agent=None, _progress_callback=progress_callback)
        
        print("\n下载模型权重...")
        llava_model = LlavaForConditionalGeneration.from_pretrained(model_name, torch_dtype="bfloat16", 
                                                                  device_map=0, use_auth_token=False,
                                                                  force_download=False, resume_download=True,
                                                                  proxies=None, local_files_only=False,
                                                                  token=None, revision=None,
                                                                  user_agent=None, _progress_callback=progress_callback)
        
        # 恢复日志级别
        logging.set_verbosity_warning()
    
    llava_model.eval()
    return processor, llava_model

def generate_caption(image, processor, llava_model, prompt, max_tokens, temperature, top_p):
    try:
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        else:
            image = Image.fromarray(image).convert('RGB')
    except Exception as e:
        print(f"打开图片失败: {e}")
        return f"处理失败: {e}"

    convo = [
        {
            "role": "system",
            "content": "You are a helpful image captioner.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    convo_string = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
    assert isinstance(convo_string, str)
    inputs = processor(text=[convo_string], images=[image], return_tensors="pt").to('cuda')
    inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

    generate_ids = llava_model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        suppress_tokens=None,
        use_cache=True,
        temperature=temperature,
        top_k=None,
        top_p=top_p,
    )[0]

    generate_ids = generate_ids[inputs['input_ids'].shape[1]:]

    caption = processor.tokenizer.decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return caption.strip()

def process_directory(input_dir, processor, llava_model, prompt, max_tokens, temperature, top_p, progress=gr.Progress()):
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_files = [f for f in os.listdir(input_dir) if os.path.splitext(f.lower())[1] in supported_formats]
    
    if not image_files:
        return "在指定目录中没有找到支持的图片文件"
    
    results = []
    saved_files = []
    
    for i, image_file in enumerate(progress.tqdm(image_files)):
        image_path = os.path.join(input_dir, image_file)
        
        base_name = os.path.splitext(image_file)[0]
        output_file = os.path.join(input_dir, f"{base_name}.txt")
        
        caption = generate_caption(image_path, processor, llava_model, prompt, max_tokens, temperature, top_p)
        results.append(f"图片: {image_file}\n标注: {caption}\n")
        
        # 保存标注到与图片同名的txt文件
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(caption)
            saved_files.append(output_file)
        except Exception as e:
            results.append(f"保存 {output_file} 失败: {e}\n")
    
    summary = f"已处理 {len(image_files)} 张图片，保存了 {len(saved_files)} 个标注文件到源图片目录。"
    return "\n".join(results) + "\n\n" + summary

# 全局变量存储加载的模型
processor = None
llava_model = None

def load_model_interface(model_path):
    global processor, llava_model
    try:
        processor, llava_model = load_model(model_path)
        return f"模型加载成功: {model_path}"
    except Exception as e:
        return f"模型加载失败: {e}"

def release_model_resources():
    global processor, llava_model
    if processor is not None or llava_model is not None:
        processor = None
        if llava_model is not None:
            del llava_model
        llava_model = None
        torch.cuda.empty_cache()
        gc.collect()
        return "模型资源已释放，显存已清理"
    return "没有加载的模型资源需要释放"

def process_single_image(image, prompt, max_tokens, temperature, top_p, save_path=None):
    global processor, llava_model
    if processor is None or llava_model is None:
        return "请先加载模型"
    
    with torch.no_grad():
        caption = generate_caption(image, processor, llava_model, prompt, max_tokens, temperature, top_p)
    
    # 如果提供了保存路径，则保存标注到文本文件
    if save_path and isinstance(save_path, str) and os.path.isfile(save_path):
        # 获取图片文件的目录和文件名（不含扩展名）
        img_dir = os.path.dirname(save_path)
        img_name = os.path.splitext(os.path.basename(save_path))[0]
        # 创建与图片同名的txt文件路径
        txt_path = os.path.join(img_dir, f"{img_name}.txt")
        # 保存标注到txt文件
        try:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(caption)
            return f"{caption}\n\n标注已保存到: {txt_path}"
        except Exception as e:
            return f"{caption}\n\n保存标注失败: {e}"
    
    return caption

def process_directory_interface(directory, prompt, max_tokens, temperature, top_p, progress=gr.Progress()):
    global processor, llava_model
    if processor is None or llava_model is None:
        return "请先加载模型"
    
    if not directory or not os.path.isdir(directory):
        return "请提供有效的输入目录"
    
    try:
        with torch.no_grad():
            results = process_directory(directory, processor, llava_model, prompt, max_tokens, temperature, top_p, progress)
        
        return results
    except Exception as e:
        return f"批处理时出错: {str(e)}"

# 缓存管理功能
def get_cache_dir():
    """获取Hugging Face缓存目录"""
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    if not os.path.exists(cache_dir):
        # 尝试其他可能的缓存位置
        alt_cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")
        if os.path.exists(alt_cache_dir):
            cache_dir = alt_cache_dir
    return cache_dir

def get_cache_size():
    """计算缓存大小"""
    cache_dir = get_cache_dir()
    if not os.path.exists(cache_dir):
        return "缓存目录不存在", "0 MB"
    
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(cache_dir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    
    # 转换为MB或GB
    if total_size > 1024 * 1024 * 1024:
        size_str = f"{total_size / (1024 * 1024 * 1024):.2f} GB"
    else:
        size_str = f"{total_size / (1024 * 1024):.2f} MB"
    
    # 获取模型文件列表
    model_files = []
    for model_id in HF_MODELS.values():
        model_name = model_id.split("/")[-1]
        model_files.extend(glob.glob(f"{cache_dir}/**/*{model_name}*", recursive=True))
    
    model_info = f"缓存目录: {cache_dir}\n"
    model_info += f"已缓存的模型文件: {len(model_files)}\n"
    
    return model_info, size_str

def clear_cache():
    """清除缓存"""
    cache_dir = get_cache_dir()
    if not os.path.exists(cache_dir):
        return "缓存目录不存在"
    
    try:
        # 只删除与我们的模型相关的缓存文件
        deleted_files = 0
        for model_id in HF_MODELS.values():
            model_name = model_id.split("/")[-1]
            model_files = glob.glob(f"{cache_dir}/**/*{model_name}*", recursive=True)
            for file_path in model_files:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    deleted_files += 1
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    deleted_files += 1
        
        return f"成功清除缓存，删除了 {deleted_files} 个文件/目录"
    except Exception as e:
        return f"清除缓存时出错: {str(e)}"

def create_ui():
    with gr.Blocks(title="Joy Caption 图像标注工具") as demo:
        gr.Markdown("# Joy Caption 图像标注工具")
        gr.Markdown("使用 Joy Caption 模型为图像生成描述性标注")
        
        with gr.Tab("模型加载"):
            model_dropdown = gr.Dropdown(
                label="选择模型", 
                choices=list(AVAILABLE_MODELS.keys()),
                value="Beta One"
            )
            with gr.Row():
                load_btn = gr.Button("加载模型")
                release_btn = gr.Button("释放模型资源", variant="secondary")
            model_status = gr.Textbox(label="模型状态", interactive=False)
            
            def load_selected_model(model_name):
                model_path = AVAILABLE_MODELS[model_name]
                return load_model_interface(model_path)
            
            load_btn.click(
                fn=load_selected_model,
                inputs=[model_dropdown],
                outputs=[model_status]
            )
            
            release_btn.click(
                fn=release_model_resources,
                inputs=[],
                outputs=[model_status]
            )
        
        with gr.Tab("单张图片处理"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(label="上传图片")
                    prompt_input = gr.Textbox(label="提示词", value=DEFAULT_PROMPT)
                    
                    with gr.Row():
                        max_tokens_input = gr.Slider(minimum=50, maximum=1000, value=300, step=10, label="最大Token数")
                        temperature_input = gr.Slider(minimum=0.1, maximum=2.0, value=0.6, step=0.1, label="温度")
                        top_p_input = gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.05, label="Top-P")
                    
                    process_btn = gr.Button("生成标注")
                
                with gr.Column():
                    caption_output = gr.Textbox(label="生成的标注", lines=10)
            
            process_btn.click(
                fn=process_single_image,
                inputs=[image_input, prompt_input, max_tokens_input, temperature_input, top_p_input],
                outputs=[caption_output]
            )
        
        with gr.Tab("批量处理"):
            with gr.Row():
                with gr.Column():
                    dir_input = gr.Textbox(label="图片目录路径")
                    batch_prompt_input = gr.Textbox(label="提示词", value=DEFAULT_PROMPT)
                    
                    with gr.Row():
                        batch_max_tokens_input = gr.Slider(minimum=50, maximum=1000, value=300, step=10, label="最大Token数")
                        batch_temperature_input = gr.Slider(minimum=0.1, maximum=2.0, value=0.6, step=0.1, label="温度")
                        batch_top_p_input = gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.05, label="Top-P")
                    
                    batch_process_btn = gr.Button("批量处理")
                
                with gr.Column():
                    batch_results = gr.Textbox(label="处理结果", lines=15)
            
            batch_process_btn.click(
                fn=process_directory_interface,
                inputs=[dir_input, batch_prompt_input, batch_max_tokens_input, batch_temperature_input, batch_top_p_input],
                outputs=[batch_results]
            )
            
        with gr.Tab("缓存管理"):
            with gr.Row():
                with gr.Column():
                    cache_info = gr.Textbox(label="缓存信息", lines=5, interactive=False)
                    cache_size = gr.Textbox(label="缓存大小", interactive=False)
                    
                    with gr.Row():
                        refresh_cache_btn = gr.Button("刷新缓存信息")
                        clear_cache_btn = gr.Button("清除模型缓存", variant="secondary")
            
            # 初始化缓存信息
            def init_cache_info():
                info, size = get_cache_size()
                return info, size
            
            # 刷新缓存信息
            refresh_cache_btn.click(
                fn=get_cache_size,
                inputs=[],
                outputs=[cache_info, cache_size]
            )
            
            # 清除缓存
            clear_cache_btn.click(
                fn=clear_cache,
                inputs=[],
                outputs=[cache_info]
            )
            
            # 页面加载时初始化缓存信息
            demo.load(
                fn=init_cache_info,
                inputs=[],
                outputs=[cache_info, cache_size]
            )
        
        gr.Markdown("### 使用说明")
        gr.Markdown("""
        1. 首先在"模型加载"标签页加载模型
        2. 在"单张图片处理"标签页上传单张图片并生成标注
        3. 在"批量处理"标签页输入包含多张图片的目录路径进行批量处理
        4. 标注文件将直接保存在源图片目录中
        5. 使用完毕后，点击"释放模型资源"按钮释放显存
        """)
    
    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Joy Caption 图像标注WebUI')
    parser.add_argument('--share', action='store_true', help='创建公共链接以便远程访问')
    parser.add_argument('--server_name', type=str, default='0.0.0.0', help='服务器名称')
    parser.add_argument('--server_port', type=int, default=7860, help='服务器端口')
    args = parser.parse_args()
    
    demo = create_ui()
    demo.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)