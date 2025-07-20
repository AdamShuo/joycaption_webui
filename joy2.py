import torch
import os
import argparse
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from tqdm import tqdm
import gc
import sys
from transformers.utils import logging

DEFAULT_PROMPT = "Write a long descriptive caption for this image in a formal tone."
MODELS_DIR = "./models"

# 添加Hugging Face Hub模型ID
HF_MODELS = {
    "Beta One": "fancyfeast/llama-joycaption-beta-one-hf-llava",
    "Alpha Two": "fancyfeast/llama-joycaption-alpha-two-hf-llava"
}

# 本地模型路径
LOCAL_MODELS = {
    "Beta One": os.path.join(MODELS_DIR, "llama-joycaption-beta-one-hf-llava"),
    "Alpha Two": os.path.join(MODELS_DIR, "llama-joycaption-alpha-two-hf-llava")
}

# 可用模型（优先使用本地模型，如果不存在则使用Hugging Face Hub模型）
AVAILABLE_MODELS = {
    model_name: local_path if os.path.exists(local_path) else HF_MODELS[model_name]
    for model_name, local_path in LOCAL_MODELS.items()
}

DEFAULT_MODEL_NAME = AVAILABLE_MODELS["Beta One"]


def parse_args():
    parser = argparse.ArgumentParser(description='Joy Caption 批量图片标注工具')
    parser.add_argument('--input_dir', type=str, help='输入图片目录路径')
    parser.add_argument('--output_dir', type=str, help='输出标注文本目录路径(默认为输入目录)')
    parser.add_argument('--prompt', type=str, default=DEFAULT_PROMPT, help='标注提示词')
    parser.add_argument('--model', type=str, choices=list(AVAILABLE_MODELS.keys()), default="Beta One", help='选择模型')
    parser.add_argument('--max_tokens', type=int, default=300, help='生成的最大token数')
    parser.add_argument('--temperature', type=float, default=0.6, help='生成温度')
    parser.add_argument('--top_p', type=float, default=0.9, help='top-p采样参数')
         
    args = parser.parse_args()
    # 将模型名称转换为模型路径
    args.model_name = AVAILABLE_MODELS[args.model]
    return args


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


def release_model_resources(processor, llava_model):
    """释放模型资源和清理显存"""
    if processor is not None or llava_model is not None:
        processor = None
        if llava_model is not None:
            del llava_model
        torch.cuda.empty_cache()
        gc.collect()
        print("模型资源已释放，显存已清理")


def generate_caption(image_path, prompt, llava_model, processor, max_tokens=300, temperature=0.6, top_p=0.9):
    try:
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            # 如果已经是PIL.Image对象
            image = image_path.convert('RGB')
    except Exception as e:
        print(f"处理图片失败: {e}")
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


def process_directory(input_dir, output_dir, processor, llava_model, prompt, max_tokens, temperature, top_p):
    os.makedirs(output_dir, exist_ok=True)
    
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_files = [f for f in os.listdir(input_dir) if os.path.splitext(f.lower())[1] in supported_formats]
    
    if not image_files:
        print(f"在 {input_dir} 中没有找到支持的图片文件")
        return
    
    print(f"找到 {len(image_files)} 个图片文件，开始处理...")
    
    for image_file in tqdm(image_files, desc="处理图片"):
        image_path = os.path.join(input_dir, image_file)
        
        base_name = os.path.splitext(image_file)[0]
        output_file = os.path.join(output_dir, f"{base_name}.txt")
        
        caption = generate_caption(image_path, prompt, llava_model, processor, max_tokens, temperature, top_p)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(caption)


def main():
    args = parse_args()
    
    if args.input_dir is None:
        while True:
            args.input_dir = input("请输入图片目录路径(按Enter回车键开始打标):").strip()
            if os.path.exists(args.input_dir) and os.path.isdir(args.input_dir):
                break
            print(f"错误：路径 '{args.input_dir}' 不存在或不是目录，请重新输入！")

    processor, llava_model = load_model(args.model_name)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.input_dir
        
    with torch.no_grad():
        process_directory(
            args.input_dir, 
            output_dir, 
            processor, 
            llava_model, 
            args.prompt, 
            args.max_tokens, 
            args.temperature, 
            args.top_p
        )
    
    print(f"处理完成！标注结果已保存到 {output_dir}")
    
    # 释放模型资源
    release_model_resources(processor, llava_model)


if __name__ == "__main__":
     args = parse_args()
         
     main()