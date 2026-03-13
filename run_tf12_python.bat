@echo off
:: 1. 激活你的 Anaconda 虚拟环境
call E:\Anaconda3\Scripts\activate.bat E:\Anaconda3\envs\tf_12

:: 2. 运行 Python，并将 Qt 传过来的所有参数（%*）原封不动传给 Python
python %*