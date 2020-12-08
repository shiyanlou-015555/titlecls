import torch.nn as nn
import torch
class GlobalEncoder(nn.Module):
    def __init__(self,config,bert_extractor):
        super(GlobalEncoder,self).__init__()
        self.bert_extractor = bert_extractor
        self.drop_emb = nn.Dropout(config.dropout_emb)

        self.start_layer = bert_extractor.start_layer
        self.bert_layers = bert_extractor.bert_layers
        self.layer_num = bert_extractor.layer_num
        config.bert_hidden_size = bert_extractor.bert.config.hidden_size
        # print(config.bert_hidden_size)768
        self.linear1 = nn.Sequential(nn.Linear(config.bert_hidden_size,config.word_dims),
                                     nn.Tanh()
                                     )
        self.gru = nn.GRU(input_size=config.word_dims,
                          hidden_size=config.gru_hiddens,
                          num_layers=config.gru_layers,
                          bidirectional=True,
                          batch_first=True)
        self.hidden_drop = nn.Dropout(config.dropout_gru_hidden)

    def forward(self, input_ids, token_type_ids, attention_mask, edu_lengths):
        batch_size, max_edu_num, max_tok_len = input_ids.size()
        input_ids = input_ids.view(-1, max_tok_len)
        #
        token_type_ids = token_type_ids.view(-1, max_tok_len)
        attention_mask = attention_mask.view(-1, max_tok_len)

        with torch.no_grad():
            _, _, encoder_outputs = \
                self.bert_extractor(input_ids, token_type_ids, attention_mask)
        # 取了后三层的输出
        #         print(self.start_layer) 10
        #         print(self.bert_layers) 13
        #         print(len(encoder_outputs)) 13
        #         print(encoder_outputs[0].shape) [52,14,768]
        bert_inputs = []
        for idx in range(self.start_layer, self.bert_layers):
            input = encoder_outputs[idx][:, 0]  # 取最前面一个单词的词向量
            bert_inputs.append(input)
        # print(bert_inputs[0].shape) bert_inputs:只使用了最后一层
        proj_hiddens = []
        for idx in range(self.layer_num):
            proj_hiddens.append(self.linear1(bert_inputs[idx]))
        # print(proj_hiddens[0].shape)50*100
        # x_embed = self.rescale(proj_hiddens)
        # x_embed = self.drop_emb(x_embed)
        x_embed = proj_hiddens[0]
        # print(x_embed.shape)[52*100]
        x_embed = x_embed.view(batch_size, max_edu_num, -1)
        # print(x_embed.shape)[4*13*100]
        gru_input = nn.utils.rnn.pack_padded_sequence(x_embed, edu_lengths, batch_first=True, enforce_sorted=False)
        outputs, _ = self.gru(gru_input)
        outputs,lengths = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        hidden = self.hidden_drop(outputs[0])
        # print(hidden.shape)20*500
        return x_embed, hidden