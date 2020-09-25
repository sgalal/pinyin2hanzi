# pinyin2hanzi

基于深度学习的拼音转汉字。原项目地址 [ranchlai/pinyin2hanzi](https://github.com/ranchlai/pinyin2hanzi)。

示例中使用的是粤语拼音，对普通话拼音也同样适用。

## 用法

Install Python 3.8 (older versions may also work).

Copy `config.template.py` to`config.py`. Edit the configurations. Then run:

```sh
python A_preprocess.py
python B_tokenize.py
python C_train.py
python D_predict.py
```

## 原理

### 数据预处理

数据预处理步骤文件为 `A_preprocess.py`。

输入为 `corpus.txt`，输出为 `train_x.txt`, `train_y.txt`, `test_x.txt`, `test_y.txt`（注：所有输入输出均在 `data` 目录下，下同）。

输入 `corpus.txt` 的格式为：每行是一个句子，句子只可能出现汉字、字母或数字（尚未实现数字注音功能，所以数字会在后续步骤中去除）。其中，汉字既可以是繁体字，也可以是简体字，但不要混合。

将输入的句子按 8:2 分为训练集和测试集，然后分别标注拼音。如果该句含有数字，则忽略该句，继续处理。然后将汉字与拼音对齐，拼音长度大于 1 时，汉字左侧填充 `-` 符号。

将训练集的拼音存储为 `train_x.txt`，训练集的汉字（已与拼音对齐，下同）存储为 `train_y.txt`，测试集的拼音存储为 `test_x.txt`，测试集的汉字存储为 `test_y.txt`。

标注拼音时，如果全拼音长度大于 52（`PAD_TO` 为 54，减去 `<sos>` 与 `<eos>` 两个 token 得 52），则舍弃该句子。

例如，输入句子为：

```
對方又驚你騎牛搵馬
```

输出为（实际上拼音和汉字不在同一个文件）：

```
deoifongjaugingneikengauwanmaa
---對---方--又---驚--你-騎---牛--搵--馬
```

### tokenize

数据预处理步骤文件为 `B_tokenize.py`。

输入为 `train_x.txt`, `test_x.txt`，输出为 `vocab_x.txt`, `tokens_train_x.pth`, `tokens_test_x.pth`。

输入为 `train_y.txt`, `test_y.txt`，输出为 `vocab_y.txt`, `tokens_train_y.pth`, `tokens_test_y.pth`。

tokenize 时使用 char-level tokenization，即根据训练集的单字建立词表，将训练集和测试集的单字映射为正整数。

词表中 0 表示 `<unk>`，1 表示 `<sos>`，2 表示 `<eos>`，3 表示 `<pad>`，数据集的单字从 4 开始。

`vocab_x.txt` 及 `vocab_y.txt` 每行是一个字符，第一行代表第四个 token，第二行代表第五个 token，依此类推。

`tokens_train_x.pth`, `tokens_test_x.pth`, `tokens_train_y.pth`, `tokens_test_y.pth` 都是二维的 tensor，其中低维的长度为 54（`PAD_TO` 的值）。

### 训练

数据预处理步骤文件为 `C_train.py`。

### 预测

数据预处理步骤文件为 `D_predict.py`。

## TODO

文件：

- [ ] `A_preprocess.py`
- [ ] `B_tokenize.py`
- [ ] `C_train.py`
- [x] `D_predict.py`
- [x] `dataset.py`
- [x] `itos.py`
- [ ] `model.py`
- [x] `stoi.py`
- [x] `uniqueid.py`
- [x] `utils.py`

功能：

- [ ] 声母简拼
- [ ] 模型训练
   - [x] basic training process
   - [ ] scheduler
   - [ ] initialize weights
- [x] 预测
- [ ] 将预测功能封装为 Web API

## Q&A

### 如何适配普通话拼音？

拼音是程序自动标注的。要适配普通话拼音，修改 `A_preprocess.py`，将其中标注粤语拼音的代码改为普通话即可。

## 保留了一部分原项目的说明

Some of the code borrowed from https://github.com/bentrevett/pytorch-seq2seq

Model architecture was designed by myself. It is very likely that the model looks partly the same as some existing works, I apologized for not citing them. Please let me know if you think I should cite some papers.
