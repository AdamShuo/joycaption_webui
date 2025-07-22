# Joy Caption WebUI

Joy Caption WebUI 是一个基于 LLaVA 模型的图像标注工具，可以为图像生成详细的描述性标注。

## 关于原项目

本项目是基于 [fpgaminer/joycaption](https://github.com/fpgaminer/joycaption) 项目进行的二次开发和界面优化。在此特别感谢原项目作者的开源贡献，为图像标注领域提供了优秀的基础工具。原项目实现了核心的图像标注功能，本项目在此基础上增强了用户界面和使用体验。

## 功能特点

- 支持单张图片处理和批量处理
- 提供友好的 Web 界面
- 支持多种模型选择
- 可自定义提示词和生成参数
- 显示模型下载进度
- 缓存管理功能

## 没有算力？没关系，来云平台试用本项目！

来 优云智算⚡️，一键部署:秋叶丹炉（Lora-Scripts）1.12+Joy Cation 3批量自动打标工具
----------
🔗[![ucloud template](https://www-s.ucloud.cn/2025/07/bc6c365e4924e5788150c051b7929098_1753151104323.png)](https://www.compshare.cn/images/175oU3fDxmR2?referral_code=LjvXLkWsflPBezgjC8H2xJ&sharetype=Markdown)

在 腾讯Cloud Studio⚡️，复刻使用:秋叶丹炉（Lora-Scripts）1.12+Joy Cation 3批量自动打标工具
----------
[![Cloud Studio Template](https://cs-res.codehub.cn/common/assets/icon-badge.svg)](https://cloudstudio.net/a/28501271478620160?channel=share&sharetype=Markdown)

在 仙宫云⚡️，部署:秋叶丹炉（Lora-Scripts）1.12+Joy Cation 3批量自动打标工具
----------
https://www.xiangongyun.com/image/detail/e1776c2d-354e-4a3e-9148-da4669f03297?r=32YFQ1

## 安装说明

### 环境要求

- Python 3.10+
- PyTorch 2.4+
- CUDA 11.8+（用于 GPU 加速）
- NVIDIA GPU（仅支持NVIDIA系列显卡）
  - 最低需要16GB显存
  - 推荐24GB或以上显存以获得最佳性能

### 详细安装步骤

#### Linux/macOS 系统

1. 克隆仓库并进入项目目录：
   ```bash
   git clone https://github.com/AdamShuo/joycaption_webui.git
   cd joycaption_webui
   ```

2. 创建虚拟环境并激活：
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   ```

3. 安装依赖：
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. 运行程序（见下方使用方法）

#### Windows 系统

1. 克隆仓库并进入项目目录：
   ```cmd
   git clone https://github.com/AdamShuo/joycaption_webui.git
   cd joycaption_webui
   ```

2. 创建虚拟环境并激活：
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```

3. 安装依赖：
   ```cmd
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. 运行程序（见下方使用方法）

#### 使用 Conda 环境（可选）

如果您使用 Conda 管理环境，可以按照以下步骤操作：

1. 克隆仓库并进入项目目录：
   ```bash
   git clone https://github.com/AdamShuo/joycaption_webui.git
   cd joycaption_webui
   ```

2. 创建 Conda 环境并激活：
   ```bash
   conda create -n joycaption python=3.10
   conda activate joycaption
   ```

3. 安装依赖：
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### 模型文件

- 程序会自动从 Hugging Face Hub 下载模型（如果本地不存在）
- 如果您想预先下载模型，可以创建 `models` 目录（如果不存在），并将模型文件放置在以下目录：
  - Beta One 模型：`./models/llama-joycaption-beta-one-hf-llava/`
  - Alpha Two 模型：`./models/llama-joycaption-alpha-two-hf-llava/`

### 故障排除

- **依赖安装问题**：如果安装依赖时遇到问题，可以尝试逐个安装关键依赖：
  ```bash
  pip install torch torchvision
  pip install transformers
  pip install gradio
  pip install -r requirements.txt
  ```

- **CUDA 相关错误**：确保您的 PyTorch 版本与您的 CUDA 版本兼容。可以使用以下命令检查 PyTorch 是否能检测到 CUDA：
  ```python
  python -c "import torch; print(torch.cuda.is_available())"
  ```

- **内存不足**：如果运行时出现内存不足错误，可以尝试减小批处理大小或使用较小的模型。

## 使用方法

### 使用Shell脚本（推荐）

我们提供了两个Shell脚本，方便您快速启动JoyCaption2：

#### Web 界面

```bash
./run_joy2_gradio.sh
```

可选参数：
- `--share`：创建公共链接以便远程访问
- `--server_name`：服务器名称（默认：0.0.0.0）
- `--server_port`：服务器端口（默认：7860）

例如：
```bash
./run_joy2_gradio.sh --share --server_port 8080
```

#### 命令行工具

```bash
./run_joy2.sh --input_dir /path/to/images --output_dir /path/to/output --model "Beta One"
```

主要参数：
- `--input_dir`：输入图片目录路径
- `--output_dir`：输出标注文本目录路径（默认为输入目录）
- `--prompt`：标注提示词
- `--model`：选择模型（"Beta One" 或 "Alpha Two"）
- `--max_tokens`：生成的最大 token 数（默认：300）
- `--temperature`：生成温度（默认：0.6）
- `--top_p`：top-p 采样参数（默认：0.9）

这两个脚本会自动激活虚拟环境，运行相应的Python文件，然后退出虚拟环境。

### 直接使用Python

如果您已经激活了虚拟环境，也可以直接运行Python文件：

#### Web 界面

```bash
python joy2_gradio.py
```

#### 命令行工具

```bash
python joy2.py --input_dir /path/to/images --output_dir /path/to/output --model "Beta One"
```

## 模型说明

Joy Caption 支持以下模型：

1. **Beta One**：最新版本的模型，提供更准确的图像描述
   - Hugging Face地址：[fancyfeast/llama-joycaption-beta-one-hf-llava](https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava)

2. **Alpha Two**：早期版本的模型，适用于特定场景
   - Hugging Face地址：[fancyfeast/llama-joycaption-alpha-two-hf-llava](https://huggingface.co/fancyfeast/llama-joycaption-alpha-two-hf-llava)

程序会自动从Hugging Face Hub下载模型（如果本地不存在），无需手动下载。下载过程中会显示进度条，包括文件名、大小和下载速度。

## 缓存管理

Joy Caption 2 提供了缓存管理功能，您可以：

1. 在Web界面的"缓存管理"标签页查看缓存信息
2. 查看缓存目录和已缓存的模型文件数量
3. 查看缓存总大小
4. 刷新缓存信息
5. 清除模型缓存

## 示例

使用 Web 界面：
1. 打开 Web 界面
2. 在"模型加载"标签页加载模型
3. 在"单张图片处理"标签页上传图片并生成标注
4. 在"批量处理"标签页输入目录路径进行批量处理
5. 在"缓存管理"标签页管理模型缓存

## 许可证

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

本项目采用 **Apache License 2.0** 开源许可证。

完整的许可证文本请查看：
[JoyCaption许可证](https://github.com/AdamShuo/joycaption_webui/blob/main/LICENSE)

模型使用请遵循原始许可证条款。

## 联系方式

如有问题或建议，请通过以下方式联系：

- GitHub Issues: [https://github.com/AdamShuo/joycaption_webui/issues](https://github.com/AdamShuo/joycaption_webui/issues)
- 项目主页: [https://github.com/AdamShuo/joycaption_webui](https://github.com/AdamShuo/joycaption_webui)
