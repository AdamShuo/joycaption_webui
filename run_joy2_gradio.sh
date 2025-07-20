#!/bin/bash

# 激活虚拟环境
source venv/bin/activate

# 运行joy2_gradio.py
python joy2_gradio.py "$@"

# 退出虚拟环境
deactivate