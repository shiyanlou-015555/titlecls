from tqdm import tqdm
import random
import math
def build_dataset(config):
    def load_dataset(path):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                # lin = line.strip()
                # if not lin:
                #     continue
                # content, label = lin.rsplit('\t',1)
                # if label==None:
                #   print('label is none')
                contents.append(line.strip())
        return contents
    train = load_dataset(config.train_file)
    dev = load_dataset(config.dev_file)
    test = load_dataset(config.test_file)
    print('train:{},dev:{},test:{}'.format(len(train),len(dev),len(test)))
    # return train, dev, test
    return train,dev,test
def batch_slice(data, batch_size):
    batch_num = int(math.ceil(len(data) // float(batch_size)))
    sentences = []
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        sentences.append([data[i * batch_size + b] for b in range(cur_batch_size)])

    return sentences


def data_iter_load(data, batch_size, shuffle=True):
    if shuffle: random.shuffle(data)
    batched_data  = batch_slice(data,batch_size)

    if shuffle: random.shuffle(batched_data)
    batch_list = []
    for batch in batched_data:
        content = []
        y_label = []
        for idx in  batch:
            temp = idx.split('\t')#content + y_label
            content.append(temp[0])
            y_label.append(int(temp[1]))
        temp1 = []
        temp1.append(content)
        temp1.append(y_label)
        batch_list.append(temp1)
    return batch_list