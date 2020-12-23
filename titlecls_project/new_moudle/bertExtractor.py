from transformers.modeling_bert import BertModel
import  torch.nn as nn
# bert模型导入
class BertExtractor(nn.Module):
    def __init__(self, config, tok_helper):
        super(BertExtractor, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config.bert_dir)
        self.bert.resize_token_embeddings(len(tok_helper.tokenizer))# 导入的是token的
        print("Load bert model finished.")
# 导入bert
        self.bert_layers = self.bert.config.num_hidden_layers + 1
        self.start_layer = config.start_layer
        self.end_layer = config.end_layer
        if self.start_layer > self.bert_layers - 1: self.start_layer = self.bert_layers - 1
        self.layer_num = self.end_layer - self.start_layer
        for param in self.bert.parameters():
            param.requires_grad = True
    def forward(self, input_ids, token_type_ids, attention_mask):
        sequence_output, pooled_output, encoder_outputs = self.bert(input_ids=input_ids,
                                                                    attention_mask=attention_mask,
                                                                    token_type_ids=token_type_ids,
                                                                    output_hidden_states=True)
        return sequence_output, pooled_output, encoder_outputs
