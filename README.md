# pinyin2hanzi

基于深度学习的拼音转汉字。原项目地址 [ranchlai/pinyin2hanzi](https://github.com/ranchlai/pinyin2hanzi)。

示例中使用的是粤语拼音，对普通话拼音也同样适用。

## 用法

### 数据预处理

数据预处理步骤文件为 `A_preprocess.py`。

输入为 `data/corpus.txt`，输出为 `train_x.txt`, `train_y.txt`, `test_x.txt`, `test_y.txt`。

输入 `data/corpus.txt` 的格式为：每行是一个句子，句子只可能出现汉字、字母或数字（尚未实现数字注音功能，故数字会在后续步骤中去除）。

将输入的句子按 8:2 分为训练集和测试集，然后分别标注拼音。如果该句含有数字，则忽略该句，继续处理。然后将汉字与拼音对齐，拼音长度大于 1 时，汉字左侧填充 `-` 符号。

将训练集的拼音存储为 `train_x.txt`，训练集的汉字（已与拼音对齐，下同）存储为 `train_y.txt`，测试集的拼音存储为 `test_x.txt`，测试集的汉字存储为 `test_y.txt`。

标注拼音时，每个句子重复三次。其中一次为全拼，两次为简拼。简拼时，随机取句子中 20%（向下取整，在代码中以 `simplify_rate` 表示）的字简拼。两次所取的字不一定相同。如果全拼长度大于 52（`PAD_TO` 为 54，减去 `<sos>` 与 `<eos>` 两个 token 得 52），则舍弃该句子的全拼和简拼。

例如，输入句子为（注：此处有错字，「番工」应为「返工/翻工」，但本项目不做错别字纠正）：

```
但係因為競爭大同番工壓力大想轉行
```

输出为（实际位于不同文件）：

```
---但--係--因--為---競---爭---大---同---番---工--壓--力---大----想---轉---行
daanhaijanwaigingzangdaaitungfaangungaatlikdaaisoengzyunhong
---但--係--因為---競---爭---大---同番工--壓--力---大----想---轉---行
daanhaijanwgingzangdaaitungfgaatlikdaaisoengzyunhong
---但--係--因--為---競---爭----大---同番---工--壓力大----想---轉---行
daanhaijanwaigingzangdaaitungfgungaatldsoengzyunhong
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

- [x] 声母简拼
- [x] 模型训练
- [x] 预测
- [ ] 将预测功能封装为 Web API
- [ ] 拆分 `data` 目录
- [ ] 将语料存入 git lfs

## 旧语料（不再使用）

- `data_aishell_transcript.txt`: <https://drive.google.com/file/d/1-9xQVprG1eg4Pfru_YRfB-dP1qpdQ_YG/view?usp=sharing>
- `data_rthk_1.txt`: <https://drive.google.com/file/d/1-6gfSDd7VjBKQJ4oS9D2P5FHeSCif7Dj/view?usp=sharing>
- `data_yuewiki.txt`: <https://drive.google.com/file/d/1--h_JAYl5_caAjUjapQxdfHs-WhjXFy9/view?usp=sharing>

## 保留了一部分原项目的说明

Some of the code borrowed from https://github.com/bentrevett/pytorch-seq2seq

Model architecture was designed by myself. It is very likely that the model looks partly the same as some existing works, I apologized for not citing them. Please let me know if you think I should cite some papers.
