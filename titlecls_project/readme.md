# 注意
- 本文采用bert-base-uncased[模型](https://huggingface.co/bert-base-uncased/tree/main)，请从网址中下载，至少需要config.json,pytorch_model_bin,vocab.text
- 本文是用的transformers---3.5.0
##目录结构
1. bert-base-uncased：bert目录
2. data 数据目录
3. dataload：数据导入文件
4. new_moudle:模型目录
5. 运行data_run即可
### 运行过程中截图
8408it [00:00, 67333.61it/s]
934it [00:00, 65142.59it/s]
20000it [00:00, 76314.05it/s]
train:8408,dev:934,test:20000
epoch 0, loss 0.0234, train acc 0.000, dev_f1 0.000,time 14.6 sec
Token indices sequence length is longer than the specified maximum sequence length for this model (639 > 512). Running this sequence through the model will result in indexing errors
epoch 100, loss 2.1572, train acc 0.169, dev_f1 0.350,time 755.4 sec
epoch 200, loss 1.9679, train acc 0.213, dev_f1 0.400,time 1503.6 sec
epoch 300, loss 1.8215, train acc 0.252, dev_f1 0.400,time 2250.1 sec
epoch 400, loss 1.5737, train acc 0.300, dev_f1 0.800,time 2996.0 sec
epoch 500, loss 1.5413, train acc 0.333, dev_f1 0.850,time 3745.2 sec
epoch 0, loss 0.0130, train acc 0.562, dev_f1 0.850,time 16.4 sec
epoch 100, loss 1.3083, train acc 0.556, dev_f1 0.800,time 765.3 sec
epoch 200, loss 1.2658, train acc 0.560, dev_f1 0.650,time 1510.2 sec
epoch 300, loss 1.2794, train acc 0.566, dev_f1 0.850,time 2256.7 sec
epoch 400, loss 1.1955, train acc 0.574, dev_f1 0.900,time 3004.0 sec
epoch 500, loss 1.1300, train acc 0.585, dev_f1 0.850,time 3752.4 sec
epoch 0, loss 0.0105, train acc 0.625, dev_f1 0.900,time 16.6 sec
epoch 100, loss 1.0415, train acc 0.658, dev_f1 0.900,time 765.3 sec
epoch 200, loss 1.0344, train acc 0.662, dev_f1 0.950,time 1514.6 sec
epoch 300, loss 1.0323, train acc 0.665, dev_f1 0.900,time 2263.5 sec
epoch 400, loss 1.0238, train acc 0.665, dev_f1 1.000,time 3011.8 sec
epoch 500, loss 0.9344, train acc 0.671, dev_f1 0.900,time 3758.3 sec
epoch 0, loss 0.0135, train acc 0.562, dev_f1 1.000,time 16.6 sec
epoch 100, loss 0.9113, train acc 0.700, dev_f1 1.000,time 762.7 sec
epoch 200, loss 0.9719, train acc 0.688, dev_f1 0.900,time 1507.2 sec
epoch 300, loss 0.8945, train acc 0.695, dev_f1 0.950,time 2251.4 sec
epoch 400, loss 0.9717, train acc 0.693, dev_f1 1.000,time 2996.3 sec
epoch 500, loss 0.9253, train acc 0.694, dev_f1 1.000,time 3741.9 sec
epoch 0, loss 0.0055, train acc 0.875, dev_f1 0.950,time 16.4 sec
epoch 100, loss 0.8766, train acc 0.719, dev_f1 0.900,time 762.5 sec
epoch 200, loss 0.8736, train acc 0.716, dev_f1 0.950,time 1508.0 sec
epoch 300, loss 0.8878, train acc 0.713, dev_f1 0.950,time 2257.3 sec
epoch 400, loss 0.9418, train acc 0.711, dev_f1 0.950,time 3003.0 sec