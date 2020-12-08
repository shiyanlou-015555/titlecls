# -*- coding: utf-8 -*-
# 必须添加头文件
from new_moudle import bertToken
import numpy as  np
import torch
import re
from Config import *
from new_moudle import bertExtractor
from new_moudle import GlobalEncoder
'''
默认定义长度:
1. 默认定义每个句子长度最长是256
2. 每个篇章最长是50个句子
3. batchsize = 1
'''
max_edu_num = 50
max_tok_len = 256
batch_size = 1
bertTokenition = bertToken.BertTokenHelper('bert-base-uncased')
# print(bertTokenition)
data = ["我国对二胎制定了一些政策，但随着人们生活压力越来越大，有些人会觉得带一个孩子已经很累了，不再打算生孩子。接下来为大家介绍再婚夫妻可以生几个孩子及女再婚男初婚能生几个。再婚夫妻可以生几个孩子如今已开放二胎政策，生二胎已成为人们关注的话题，包括再婚可以生几个孩子的问题，目前在二胎政策中有规定，再婚的男女要少于两个孩子的才能再生，而且要符合一定的条件[num]、对确定有病残的第一个小孩或第一胎双胞胎、多胞胎的，出现残疾没有正常劳动力，医学上认为可以再生育。[num]、再婚男女，如果有一方再婚前有两个小孩，而另一方未生育过，那可以再生育。[num]、再婚男女，如果婚前两人各生育一个小孩，离婚时小孩随前配偶，新组合家庭无子女的。[num]、再婚男女，如果婚前两人各生育一个小孩，新家庭有小孩但是残疾儿，这一类可以再生育。[num]、再婚男女，如果婚前都没有生育过小孩，结婚后鉴定出有无法生育的病症，可以依法收养子女后又怀孕的，可以再生育。女再婚男初婚能生几个根据相关法律规定，有下列情形的，可以由夫妻双方共同申请，经乡镇、街道人口、计划生育工作机构或县级以上直属审批，可再生育一胎子女[num]、经计划生育部门确定病残儿医学鉴定组织鉴定，第一个孩子有残疾或第一胎系双胞胎、多胞胎均有残疾没有正常劳动力，医学上认为可以再生育的。[num]、再婚夫妻，再婚前一方生育两个以内子女，另一方未生育过的。[num]、再婚夫妻，再婚前双方各生育一个小孩，离婚后确定孩子随前配偶，新组合家庭无子女。[num]、再婚夫妻，再婚前双方各生育一个小孩，新组合家庭只有一个孩子而且是残疾儿，没有劳动力，且医学上认为可以再生育的。[num]、夫妻双方婚前没有生育过子女，婚后经医疗机构鉴定患不孕症，依法收养孩子后又怀孕的。[num]、夫妻双方是独生小孩，而且只有一个小孩的。[num]、夫妻双方只生育一个小孩，而且是女孩的。[num]、夫妻双方在内地定居的港澳台居民，只有一个孩子在内地定居的。小编总结关于再婚夫妻可以生几个孩子及女再婚男初婚能生几个就介绍到这里了，希望对大家有所帮助，想了解更多相关知识，可以关注齐家网资讯。什么，装修还用自己的钱？！齐家装修分期，超低年利率[num].[num]起，最高可贷[num]万。立即申请享受优惠原文网址:再婚夫妻可以生几个孩子女再婚男初婚能生几个[url]"]
def sentence_split(text):
    sentences = re.split('(。|！|\!|\.|？|\?)', text)  # 保留分割符

    new_sents = []
    for i in range(int(len(sentences) / 2)):
        sent = sentences[2 * i] + sentences[2 * i + 1]
        new_sents.append(sent)
    return new_sents
def batch_bert_variable(onebatch,tokenizer,max_edu_num,max_tok_len):
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    for idx,edu in enumerate(onebatch):
        input_list = sentence_split(edu)
        input_ids,token_type_ids,attention_mask = bertTokenition.batch_bert_id(input_list)
        input_ids_list.append(input_ids)
        token_type_ids_list.append(token_type_ids)
        attention_mask_list.append(attention_mask)
    batch_size = len(onebatch)
    batch_input_ids = np.ones([batch_size, max_edu_num, max_tok_len], dtype=np.long) * tokenizer.pad_token_id()
    batch_token_type_ids = np.zeros([batch_size, max_edu_num, max_tok_len], dtype=np.long)
    batch_attention_mask = np.zeros([batch_size, max_edu_num, max_tok_len], dtype=np.long)
    token_lengths = np.ones([batch_size])
    for idx in range(batch_size):
        edu = len(input_ids_list[idx])
        token_lengths[idx] = edu
        for idy in range(edu):
            tok_len = len(input_ids_list[idx][idy])
            for idz in range(tok_len):
                batch_input_ids[idx,idy,idz] = input_ids_list[idx][idy][idz]
                batch_token_type_ids[idx,idy,idz] = token_type_ids_list[idx][idy][idz]
                batch_attention_mask[idx,idy,idz] = attention_mask_list[idx][idy][idz]
    batch_input_ids = torch.LongTensor(batch_input_ids)
    batch_token_type_ids = torch.LongTensor(batch_token_type_ids)
    batch_attention_mask = torch.LongTensor(batch_attention_mask)
    return batch_input_ids,batch_token_type_ids,batch_attention_mask,token_lengths


batch_input_ids,batch_token_type_ids,batch_attention_mask,token_lengths = batch_bert_variable(data,bertTokenition,max_edu_num,max_tok_len)
config = Configurable('default.cfg')
bert_extractor = bertExtractor.BertExtractor(config,bertTokenition)
globalencode = GlobalEncoder.GlobalEncoder(config,bert_extractor)
x,y = globalencode(batch_input_ids,batch_token_type_ids,batch_attention_mask,token_lengths)

# print(batch_input_ids.shape)torch.Size([1, 50, 256])
# print(token_lengths) 20句话