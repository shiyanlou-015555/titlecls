import torch
from torch import nn
class cls(object):
    def __init__(self,globalencode,decode):
        # self.bert_tokenhelper = bert_tokenhelper
        self.globalencode = globalencode
        self.decode = decode
        self.use_cuda = next(filter(lambda p: p.requires_grad, self.decode.parameters())).is_cuda
        print(self.use_cuda)
    def train(self):
        # 训练模式
        self.globalencode.train()
        self.decode.train()
    def eval(self):
        #测试模式
        self.globalencode.eval()
        self.decode.eval()
    def forward(self,batch_input_ids,batch_token_type_ids,batch_attention_mask,token_lengths):
        if self.use_cuda:
            batch_input_ids = batch_input_ids.cuda()
            batch_token_type_ids = batch_token_type_ids.cuda()
            batch_attention_mask = batch_attention_mask.cuda()
        x_embed, hidden = self.globalencode(batch_input_ids,batch_token_type_ids,batch_attention_mask,token_lengths)
        out = self.decode(hidden)
        return out