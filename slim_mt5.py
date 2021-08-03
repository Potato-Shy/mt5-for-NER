# -*- coding：utf-8 -*-
# created by shy
# 用于修剪模型的embedding层，mt5的包含的tokens又25w个，其中我们需要的中文token和部分英文token大约未3w个
# 导致在进行softmax计算时存在大量不必要的计算，所以需要将其中会用到的token的embedding提取出来，组成新的embedding

import os
import json

import torch
from transformers import MT5ForConditionalGeneration
from transformers import MT5Config
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--keep_tokens", default="../generation/t5_model/mt5/sentencepiece_cn_keep_tokens.json", type=str)
parser.add_argument("--model_config", default="./mt5-small/config.json", type=str)
parser.add_argument("--model", default="./mt5-small/mt5-small.bin", type=str)
parser.add_argument("--output_model", default="./mt5-small/mt5-small-cn.bin", type=str)
args = parser.parse_args()


# 该文件记录需要保留的token的id，
with open(args.keep_tokens) as f:
    keep_tokens = json.load(f)

# 创建模型
mt5_config = MT5Config.from_json_file(args.model_config)
mt5_model = MT5ForConditionalGeneration(mt5_config)

# resize模型的embedding维度
mt5_model.resize_token_embeddings(len(keep_tokens))

state_dict = torch.load("./mt5-small/mt5-small.bin")

# 获取需要修改的两个参数层,lm_head是输出的weight，
shared_weight = state_dict["shared.weight"]
lm_head = state_dict["lm_head.weight"]

# 新建两个空的层用来替换
new_embedding = torch.empty([len(keep_tokens), 512])
new_lm_head = torch.empty([len(keep_tokens), 512])

for i, token_id in enumerate(tqdm(keep_tokens)):
    # 选择需要保留的token_id对应的embedding，填入用于替换的矩阵中
    new_embedding[i] = shared_weight[token_id]
    new_lm_head[i] = lm_head[token_id]

# 将需要替换的层换掉，模型中encoder和decoder的weight是共享的，并记录在share.weight中，所以需要全部替换掉
state_dict["shared.weight"] = new_embedding
state_dict["encoder.embed_tokens.weight"] = new_embedding
state_dict["decoder.embed_tokens.weight"] = new_embedding
state_dict["lm_head.weight"] = new_lm_head

# 加载模型，如果没出错就保存
try:
    mt5_model.load_state_dict(state_dict)

except Exception as e:
    print(e)
    raise


torch.save(mt5_model.state_dict(), args.output_model)

print(f'new_embedding.shape: {new_embedding.shape}')