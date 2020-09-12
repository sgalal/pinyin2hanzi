# pinyin2hanzi

基于深度学习的拼音转汉字。原项目地址 [ranchlai/pinyin2hanzi](https://github.com/ranchlai/pinyin2hanzi)。

示例中使用的是粤语拼音，对普通话拼音也同样适用。

## 用法

### 数据预处理

数据预处理步骤文件见 `A_preprocess.py`。

输入为 `data/corpus.txt`，输出为 `train_x.txt`, `train_y.txt`, `test_x.txt`, `test_y.txt`。

输入 `data/corpus.txt` 的格式为：每行是一个句子，句子只可能出现汉字、字母或数字（数字会在后续步骤中去除）。

首先将输入的句子按 8:2 分为训练集和测试集，然后分别标注拼音。如果该句子含有数字，则忽略该句子，继续处理。然后将汉字与拼音对齐。将训练集的拼音存储为 `train_x.txt`，训练集的汉字（已与拼音对齐，下同）存储为 `train_y.txt`，测试集的拼音存储为 `test_x.txt`，测试集的汉字存储为 `test_y.txt`。

标注拼音时，每个句子重复四次。其中一次为全拼，三次为简拼。简拼时，随机取句子中 40% 的字（向下截断）简拼。三次所取的字不一定相同。

例如，输入句子为（注：此处有错字，「番工」应为「返工/翻工」，但本项目不做错别字纠正)：

如果标注拼音的串长度大于 52（`PAD_TO` 为 54，减去 `<sos>` 与 `<eos>` 两个 token 得 52），则舍弃该句子，不对该句子进行全拼或简拼。

```
但係因為競爭大同番工壓力大想轉行
```

输出为（实际位于不同文件）：

```
---但--係--因--為---競---爭---大---同---番---工--壓--力---大----想---轉---行
daanhaijanwaigingzangdaaitungfaangungaatlikdaaisoengzyunhong
---但--係--因為---競---爭---大---同番工壓--力---大想---轉行
daanhaijanwgingzangdaaitungfgalikdaaiszyunh
---但--係因--為競爭---大---同番---工--壓力大----想---轉---行
daanhaijwaigzdaaitungfgungaatldsoengzyunhong
但係--因--為---競爭---大同---番工--壓--力---大----想---轉行
dhjanwaigingzdaaitfaangaatlikdaaisoengzyunh
```

### tokenize

数据预处理步骤文件见 `B_tokenize.py`。

输入为 `train_x.txt`, `test_x.txt`，输出为 `vocab_x.txt`, `tokens_train_x.pth`, `tokens_test_x.pth`。

输入为 `train_y.txt`, `test_y.txt`，输出为 `vocab_y.txt`, `tokens_train_y.pth`, `tokens_test_y.pth`。

tokenize 时使用 char-level tokenization，即根据训练集的单字建立词表，将训练集和测试集的单字映射为正整数。

词表中 0 表示 `<unk>`，1 表示 `<sos>`，2 表示 `<eos>`，3 表示 `<pad>`，数据集的单字从 4 开始。

## TODO

- [x] 声母简拼
- [ ] 模型训练
- [ ] 预测
- [ ] 将预测功能封装为 Web API

## 旧语料（不再使用）

- `data_aishell_transcript.txt`: <https://drive.google.com/file/d/1-9xQVprG1eg4Pfru_YRfB-dP1qpdQ_YG/view?usp=sharing>
- `data_rthk_1.txt`: <https://drive.google.com/file/d/1-6gfSDd7VjBKQJ4oS9D2P5FHeSCif7Dj/view?usp=sharing>
- `data_yuewiki.txt`: <https://drive.google.com/file/d/1--h_JAYl5_caAjUjapQxdfHs-WhjXFy9/view?usp=sharing>

## 保留了一部分原项目的说明

Some of the code borrowed from https://github.com/bentrevett/pytorch-seq2seq

Model architecture was designed by myself. It is very likely that the model looks partly the same as some existing works, I apologized for not citing them. Please let me know if you think I should cite some papers.
