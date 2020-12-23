from transformers import BertTokenizer
# bert的tokennizer，主要是将句子转化为id
class BertTokenHelper(object):
    def __init__(self,bert_dir):
        self.tokenizer = BertTokenizer.from_pretrained(bert_dir)
        special_tokens_dict = {'additional_special_tokens':['[url]','[num]','[word]']}#随机初始化词向量
        self.tokenizer.add_special_tokens(special_tokens_dict)
        print("Load bert vocabulary finished")
    def pad_token_id(self):
        return self.tokenizer.pad_token_id
    def batch_bert_id(self,instext):
        outputs = self.tokenizer.batch_encode_plus(instext,add_special_tokens=True)
        input_ids = outputs.data['input_ids']
        token_type_ids = outputs.data['token_type_ids']
        attention_mask = outputs.data['attention_mask']
        return input_ids,token_type_ids,attention_mask