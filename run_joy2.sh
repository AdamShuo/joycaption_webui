#!/bin/bash

# 激活虚拟环境
source venv/bin/activate

# 运行joy2.py
python joy2.py "$@"

# 退出虚拟环境
deactivate