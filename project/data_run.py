# -*- coding: utf-8 -*-
# 必须添加头文件
from new_moudle import bertToken
import numpy as  np
import torch
import re
from Config import Configurable
from new_moudle import bertExtractor
from new_moudle import GlobalEncoder
from new_moudle import classfication
from new_moudle import decode
from data_load.batch_load import *
import time
import torch.nn.functional as F
from  sklearn import metrics
'''
默认定义长度:
1. 默认定义每个句子长度最长是256
2. 每个篇章最长是50个句子
'''

# print(bertTokenition)
data = ["我国对二胎制定了一些政策，但随着人们生活压力越来越大，有些人会觉得带一个孩子已经很累了，不再打算生孩子。接下来为大家介绍再婚夫妻可以生几个孩子及女再婚男初婚能生几个。再婚夫妻可以生几个孩子如今已开放二胎政策，生二胎已成为人们关注的话题，包括再婚可以生几个孩子的问题，目前在二胎政策中有规定，再婚的男女要少于两个孩子的才能再生，而且要符合一定的条件[num]、对确定有病残的第一个小孩或第一胎双胞胎、多胞胎的，出现残疾没有正常劳动力，医学上认为可以再生育。[num]、再婚男女，如果有一方再婚前有两个小孩，而另一方未生育过，那可以再生育。[num]、再婚男女，如果婚前两人各生育一个小孩，离婚时小孩随前配偶，新组合家庭无子女的。[num]、再婚男女，如果婚前两人各生育一个小孩，新家庭有小孩但是残疾儿，这一类可以再生育。[num]、再婚男女，如果婚前都没有生育过小孩，结婚后鉴定出有无法生育的病症，可以依法收养子女后又怀孕的，可以再生育。女再婚男初婚能生几个根据相关法律规定，有下列情形的，可以由夫妻双方共同申请，经乡镇、街道人口、计划生育工作机构或县级以上直属审批，可再生育一胎子女[num]、经计划生育部门确定病残儿医学鉴定组织鉴定，第一个孩子有残疾或第一胎系双胞胎、多胞胎均有残疾没有正常劳动力，医学上认为可以再生育的。[num]、再婚夫妻，再婚前一方生育两个以内子女，另一方未生育过的。[num]、再婚夫妻，再婚前双方各生育一个小孩，离婚后确定孩子随前配偶，新组合家庭无子女。[num]、再婚夫妻，再婚前双方各生育一个小孩，新组合家庭只有一个孩子而且是残疾儿，没有劳动力，且医学上认为可以再生育的。[num]、夫妻双方婚前没有生育过子女，婚后经医疗机构鉴定患不孕症，依法收养孩子后又怀孕的。[num]、夫妻双方是独生小孩，而且只有一个小孩的。[num]、夫妻双方只生育一个小孩，而且是女孩的。[num]、夫妻双方在内地定居的港澳台居民，只有一个孩子在内地定居的。小编总结关于再婚夫妻可以生几个孩子及女再婚男初婚能生几个就介绍到这里了，希望对大家有所帮助，想了解更多相关知识，可以关注齐家网资讯。什么，装修还用自己的钱？！齐家装修分期，超低年利率[num].[num]起，最高可贷[num]万。立即申请享受优惠原文网址:再婚夫妻可以生几个孩子女再婚男初婚能生几个[url]"]
def sentence_split(text):
    sentences = re.split('。|！|\!|\.|？|\?', text)  # 保留分割符

    # new_sents = []
    # for i in range(len(sentences)):
    #     new_sents.append(sent)
    # new_sents.append(sentences[-1])
    return sentences
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
        if edu>max_edu_num:
            edu = max_edu_num
        token_lengths[idx] = edu
        # print("edu is {}".format(edu))

        for idy in range(edu):
            tok_len = len(input_ids_list[idx][idy])
            # print("token_len is {}".format(tok_len))
            if tok_len>max_tok_len:
                tok_len = max_tok_len
            for idz in range(tok_len):
                batch_input_ids[idx,idy,idz] = input_ids_list[idx][idy][idz]
                batch_token_type_ids[idx,idy,idz] = token_type_ids_list[idx][idy][idz]
                batch_attention_mask[idx,idy,idz] = attention_mask_list[idx][idy][idz]
    batch_input_ids = torch.LongTensor(batch_input_ids)
    batch_token_type_ids = torch.LongTensor(batch_token_type_ids)
    batch_attention_mask = torch.LongTensor(batch_attention_mask)
    return batch_input_ids,batch_token_type_ids,batch_attention_mask,token_lengths

bertTokenition = bertToken.BertTokenHelper('bert-base-uncased')
config = Configurable('default.cfg')
bert_extractor = bertExtractor.BertExtractor(config,bertTokenition)
globalencode = GlobalEncoder.GlobalEncoder(config,bert_extractor)
decoder = decode.decode(config)
bert_extractor.cuda()
globalencode.cuda()
decoder.cuda()
'''
放入cuda的代码
'''
cls_1 = classfication.cls(globalencode,decoder)

# y_hat = cls_1.forward(batch_input_ids,batch_token_type_ids,batch_attention_mask,token_lengths)

train_data,dev_data,test_data = build_dataset(config)# 数据载入
def evaluate(dev_data, model, config, tokenhelper):
    model.eval()
    loss_total = 0
    predict_all = []
    labels_all = []
    with torch.no_grad():
        for onebatch in data_iter_load(dev_data[:20],batch_size=config.test_batch_size,shuffle=False):
            x = onebatch[0]
            y = torch.tensor(onebatch[1]).cuda()
            batch_input_ids, batch_token_type_ids, batch_attention_mask, token_lengths = batch_bert_variable(x,
                                                                                                             tokenhelper,
                                                                                                             config.max_edu_len,
                                                                                                             config.max_tok_len)
            # batch_input_ids,batch_token_type_ids,batch_attention_mask[batchsize*max_edu*max_tok]
            y_hat = model.forward(batch_input_ids, batch_token_type_ids, batch_attention_mask, token_lengths)
            loss = F.cross_entropy(y_hat, y)
            loss_total += loss
            labels = y.cpu().numpy().tolist()
            predic = torch.max(y_hat.data, 1)[1].cpu().numpy().tolist()
            labels_all.extend(labels)
            predict_all.extend(predic)
    model.train()
    acc = metrics.accuracy_score(labels_all, predict_all)
    f1 = metrics.f1_score(labels_all,predict_all,average='micro')
    return acc,f1
# train_iter = data_iter_load(dev_data,config.train_batch_size,True)
def train(train_data,dev_data,test_data,parser,config,tokenhelper):
    model_param = list(parser.globalencode.parameters())+list(parser.decode.parameters())
    optimizer = torch.optim.Adam(model_param,lr=config.learning_rate)
    loss = torch.nn.CrossEntropyLoss()


    for i in  range(config.train_iters):
        # print("theNo.{}".format(i))
        total_batch = 0
        loss_sum = 0
        train_acc_sum = 0
        start  = time.time()
        n = 0
        for onebatch in data_iter_load(train_data,batch_size=config.train_batch_size,shuffle=True):
            #print(onebatch)[content,label]
            x = onebatch[0]
            y = torch.tensor(onebatch[1]).cuda()
            batch_input_ids, batch_token_type_ids, batch_attention_mask, token_lengths = batch_bert_variable(x,
                                                                                                             tokenhelper,
                                                                                                             config.max_edu_len,
                                                                                                             config.max_tok_len)
            #batch_input_ids,batch_token_type_ids,batch_attention_mask[batchsize*max_edu*max_tok]
            y_hat = parser.forward(batch_input_ids,batch_token_type_ids,batch_attention_mask,token_lengths)
            # print(y_hat)
            # print(y)
            #
            # print(loss(y_hat,y))
            l = loss(y_hat,y)
            l.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_sum = loss_sum + l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n+=y.shape[0]
            if total_batch % config.validate_every == 0:
                dev_score,f1_score = evaluate(dev_data,parser,config,tokenhelper)
                print('epoch %d, loss %.4f, train acc %.3f, dev_f1 %.3f,time %.1f sec'
                      % (total_batch, loss_sum/ config.validate_every, train_acc_sum / n, f1_score, time.time() - start))
                loss_sum = 0
            total_batch+=1
train(train_data,dev_data,test_data,cls_1,config,bertTokenition)


print(len(train_data))
# print(y_hat.shape)
# print(batch_input_ids.shape)torch.Size([1, 50, 256])
# print(token_lengths) 20句话