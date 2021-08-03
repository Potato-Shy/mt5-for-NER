# Finetune HuggingFace's mT5用于NER任务 
mt5的下载地址 https://huggingface.co/google/mt5-small/tree/main

下载的文件是个.zip，直接修改文件名即可使用，**不要解压**。

我的环境是torch1.9 + transformers

## 1.embedding层的更新
使用苏神开源的中文预训练的sentencepiece model、使用该中文token id与mt5的token id 的比较结果进行修剪embedding层。
苏神项目地址
```shell
python3 slim_mt5.py \
--keep_tokens ./mt5-small/sentencepiece_cn_keep_tokens.json \
--model_config ./mt5-small/config.json \
--model ./mt5-small/mt5-small.bin \
--output_model ./mt5-small/mt5-small-cn.bin
```

## 2.生成输入输出的数据
数据采用clue的开源ner数据，数据格式如下：
```json
{"text": "浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，", "label": {"name": {"叶老桂": [[9, 11]]}, "company": {"浙商银行": [[0, 3]]}}}
{"text": "生生不息CSOL生化狂潮让你填弹狂扫", "label": {"game": {"CSOL": [[4, 7]]}}}
```
使用端到端的生成模型做ner任务，需要将输出调整为需要的文本，本次实验设计的生成文本如下：
```text
布鲁京斯研究所桑顿中国中心研究部主任李成说，东亚的和平与安全，是美国的“核心利益”之一。	address:美国 | organization:布鲁京斯研究所桑顿中国中心 | name:李成 | position:研究部主任
目前主赞助商暂时空缺，他们的球衣上印的是“unicef”（联合国儿童基金会），是公益性质的广告；	organization:unicef,联合国儿童基金会
```

运行代码,得到训练数据：
```shell
python3 prepare_data.py 
```
## 3. finetune
相关参数可在ner_config.yml中设置
运行代码：
```shell
python3 finetune_ner.py \
--project_config_path ./ner_config.yml
```
## 4.实验结果
结果不太理想，f1值大约0.66
```text
'precision': 0.6792253521126761, 'recall': 0.6473154362416107, 'f1': 0.6628865979381443
```
