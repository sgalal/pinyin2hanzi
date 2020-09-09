# pinyin2hanzi

基于深度学习的拼音转汉字。原项目地址 [ranchlai/pinyin2hanzi](https://github.com/ranchlai/pinyin2hanzi)。

示例中使用的是粤语拼音，对普通话拼音也同样适用。

## 用法

1. Download `corpus.txt` (one sentence per line)
2. Run `python preprocess.py`
3. Run `python train.py`

## 旧语料（不再使用）

- `data_aishell_transcript.txt`: <https://drive.google.com/file/d/1-9xQVprG1eg4Pfru_YRfB-dP1qpdQ_YG/view?usp=sharing>
- `data_rthk_1.txt`: <https://drive.google.com/file/d/1-6gfSDd7VjBKQJ4oS9D2P5FHeSCif7Dj/view?usp=sharing>
- `data_yuewiki.txt`: <https://drive.google.com/file/d/1--h_JAYl5_caAjUjapQxdfHs-WhjXFy9/view?usp=sharing>

## 保留了一部分原项目的说明

Some of the code borrowed from https://github.com/bentrevett/pytorch-seq2seq

Model architecture was designed by myself. It is very likely that the model looks partly the same as some existing works, I apologized for not citing them. Please let me know if you think I should cite some papers.
