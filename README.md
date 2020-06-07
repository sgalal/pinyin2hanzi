# pinyin2hanzi

Convert pinyin to hanzi using deep networks.

## TRAINING DATA

As a light-weight example, training data are downloaded from the AI shell speech recognition corpus, 
found in http://openslr.org/33/. The transcripts rather than the audio data are used. A copy of the transcript file is found in the ./data folder

## MODEL ARCHITECTURE

![](./doc/model.png)

## Run

Install Python 3.7.

Clone repository:

```cmd
git clone https://github.com/ranchlai/pinyin2hanzi.git
cd pinyin2hanzi
```

Install requirements:

```cmd
pip install -r requirements.txt
```

Convert ai-shell transcripts from 汉字 to 带声调的拼音 (pinyin with tones)

```cmd
python process_ai_shell_transcript_sd.py
```

Train the model

```cmd
python train.py
```

Do inference

```cmd
python inference_sd.py
```

## Examples

Results obtained by running inference_sd.py:

néng gòu yíng de bǐ sài zhēn de hěn kāi xīn

能够赢得比赛真的很开心

yě qǔ de le sān xiàn piāo hóng de chéng jī

也取得了三线飘红的成绩

guó yǒu qǐ yè bù bì yě wú xū jiè rù yíng lì xìng qiáng de shāng pǐn fáng kāi fā

国有企业不必也无需介入盈利性强的商品房开发

sī fǎ jiàn dìng jī gòu shì dú lì fǎ rén

司法鉴定机构是独立法人

bǎo bǎo zhòng wǔ diǎn èr jīn

宝宝重五点二斤

## Pretrained model

Pretrained model using AI-shell transcript file can be downloaded from 
[gooole drive](https://drive.google.com/open?id=186jnywHwnxqXDBxrbFRpIF7dFAWcwEx_)

## Reference

Some of the code borrowed from https://github.com/bentrevett/pytorch-seq2seq

Model architecture was designed by myself. It is very likely that the model looks partly the same as some existing works, I apologized for not citing them. Please let me know if you think I should cite some papers.
