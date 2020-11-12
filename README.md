零、环境要求
```bash
python 3.6及以上
GPU + CUDA 9
```

一、部署
```bash
pip install -r requirements
```

二、运行
```bash
1、移动 要处理的excel文件名.xlsx 到此目录下
    注意：文件必须有一列名为'文本详情'， 作为程序要处理的列
2、python MaskExcelProcessor.py -f 要处理的excel文件名.xlsx
```

三、结果
当前目录下会产生一个 要处理的excel文件名-MASK.xlsx 的新文件，增加了：
    1、一列"mask文本"：识别出来，决定mask掉的内容。
    2、一列"处理后文本"： 将"mask文本"的内容在原文中替换为[XX]后的文本，即为每句话经过mask后的最终内容
识别内容包括：
    电话号码（固话、手机）
    身份证号
    地址（地名）
    人名
识别内容可能不完整