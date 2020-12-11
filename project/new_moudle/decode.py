import torch
from torch import nn
import torch.nn.functional as F
class decode(nn.Module):
    def __init__(self,config):
        super(decode,self).__init__()
        self.size1 = config.mlp_rel_size
        # print(self.size1)
        self.linear1 = nn.Linear(self.size1,self.size1//4)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(self.size1//4,config.get_label)
    def forward(self,inputs):
        temp = inputs.permute(0,2,1)
        inputs_pooled = F.max_pool1d(temp,temp.size(2))
        out = self.linear2(self.relu(self.linear1(inputs_pooled.reshape(inputs_pooled.size(0),inputs_pooled.size(1)))))
        return out
