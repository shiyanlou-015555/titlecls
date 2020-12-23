# -*- coding: utf-8 -*-
# @Time    : 2020/11/14
# @Author  : YYWind
# @Email   : yywind@126.com
# @File    : test.py
import pandas
import numpy as np
import random
random.seed(88)
np.random.seed(88)
import re
from bs4 import  BeautifulSoup

label_values = {'财经':0, '房产':1, '家居':2, '教育':3, '科技':4, '时尚':5, '时政':6, '游戏':7, '娱乐':8, '体育':9}
id_label = {0:'财经', 1:'房产', 2:'家居', 3:'教育', 4:'科技', 5:'时尚', 6:'时政', 7:'游戏', 8:'娱乐', 9:'体育'}
def clean_chineses_text(text):
    """
    数据清洗
    1.url->"网址"
    2.数字->num
    3.删除除中英文字符及标点之外的符号
    :param text:
    :return:
    """
    # 清除除中英文字符，标点，数字之外的符号
    cop = re.compile("[^\u4e00-\u9fa5^.^a-z^A-Z^0-9.\W]")
    temp = cop.sub("", text)

    # 去掉html标签
    text_result = BeautifulSoup(temp, 'html.parser').get_text()

    # 清除url
    results = re.compile(r'http://[a-zA-Z0-9.?/&=\-:]*', re.S)
    text_result = results.sub("[网址]", text_result)
    results = re.compile(r'https://[a-zA-Z0-9.?/&=\-:]*', re.S)
    text_result = results.sub("[网址]", text_result)

    # 清除空格
    # 包含中文空格和英文空格
    text_result = text_result.replace(" ", "")
    text_result = text_result.replace("%", "")
    text_result = text_result.replace(" ", "")
    text_result = text_result.replace(" ", "")
    text_result = text_result.replace("　", "")
    text_result = text_result.replace("\t", "")
# 特殊中文符号
    #--------------------------------
    text_result = text_result.replace("：", "")
    text_result = text_result.replace("（", "")
    text_result = text_result.replace("）", "")
    text_result = text_result.replace("）", "")
    text_result = text_result.replace("\"","")
    text_result = text_result.replace('­',"")
    text_result = text_result.replace('【',"")
    text_result = text_result.replace('】',"")
    #--------------------------------
    # 将所有num换成"num"
    # 大数先替换，否则会出现替换不完全或者一个数字多个num
    # id不可替换，但是后面可能会出现与id相同的数字
    text_result = text_result.lower()
    # digits = re.findall("\[a-z]+", text_result)
    # digits.sort(key=lambda i: len(i), reverse=True)
    #
    # sen = text_result
    # for digit in digits:
    #     sen = sen.replace(digit, "[word]")
    # text_result = sen
    text_result = re.sub("[a-z]+",'[word]',text_result)
    digits = re.findall(r"\d+", text_result)
    digits.sort(key=lambda i: len(i), reverse=True)

    sen = text_result
    for digit in digits:
        sen = sen.replace(digit, "[num]")

    text_result = sen

    # 大小写统一
    text_result = text_result.replace('[网址]', "[url]")
    # results = re.compile(r'[a-zA-Z]*', re.S)
    # text_result = results.sub("word", text_result)
    return text_result
def pre_view1():
    # 查看 label 在 content 中的样本数量，按照标签分类统计
    label_data=pandas.read_csv('my_data/train/labeled_data.csv')
    in_list=[]
    for i in range(label_data.shape[0]):
        if label_data['class_label'][i] in label_data['content'][i]:
            in_list.append(1)
        else:
            in_list.append(0)
    label_data['in_list']=in_list
    res=label_data.groupby(['class_label'])['in_list'].sum()/1000
    # print(res)
def single(label_values,content):
    cnt=0
    res_label=''
    for label in label_values:
        if label in content:
            res_label=label
            cnt+=1
        if cnt>1:
            return ''
    if cnt==1:
        return res_label
def add_label_data_m1():
    #查看 label 在 content 中，且 content 中只包含该 label 的数据
    unlabel_data = pandas.read_csv('my_data/train/unlabeled_data.csv')
    label_data = pandas.read_csv('my_data/train/labeled_data.csv')
    # label_values=label_data['class_label'].unique()
    label_values=['财经','房产','家居','教育','科技','时尚','时政','游戏','娱乐','体育']
    labels=[]
    for content in unlabel_data['content']:
        labels.append(single(label_values,content))
    unlabel_data['class_label']=labels
    add_label_data=unlabel_data[unlabel_data['class_label']!='']
    add_label_data=add_label_data[['id','class_label','content']]
    games_data=add_label_data[add_label_data['class_label']=='游戏'].sample(n=1000)
    sports_data=add_label_data[add_label_data['class_label']=='体育']
    entertain_data=add_label_data[add_label_data['class_label']=='娱乐']
    # add_datas=[jiaju_data,fangchan_data,jiaoyu_data,shishang_data,keji_data,shizheng_data,caijing_data,games_data,sports_data,entertain_data]
    add_datas=[games_data,sports_data,entertain_data]
    new_labeled_data=label_data.append(add_datas,ignore_index=True)

    new_labeled_data.to_csv('my_data/train/m1_labeled_data.csv',index=False)
    print('methord 1:')
    print(new_labeled_data.groupby('class_label').count())

def gen_train_dev_data(path,sam_frac=0.9):
    with open('my_data/data/class.txt','w',encoding='utf-8') as f:
        for key in label_values.keys():
            f.write(key+'\n')
    data=pandas.read_csv(path)
    data['content'] = data['content'].apply(lambda x:clean_chineses_text(x))
    # 训练集和验证集合
    train_data=data.sample(frac=sam_frac)
    dev_data=data[~data.index.isin(train_data.index)]
    with open('my_data/data.txt','w',encoding='utf-8') as f1:
        for idx in data.index:
            content=str(data.loc[idx,'content'])
            content = content.replace('\r\n', '')
            content = content.replace('\n', '')
            content = content.replace('\r', '')
            label=data.loc[idx,'class_label']
            f1.write(content+'\t'+str(label_values.get(label))+'\n')
    with open('my_data/data/train.txt','w',encoding='utf-8') as f2:
        for idx in train_data.index:
            content=str(train_data.loc[idx,'content'])
            content = content.replace('\r\n', '')
            content = content.replace('\n', '')
            content = content.replace('\r', '')
            label = train_data.loc[idx, 'class_label']
            
            f2.write(content+'\t'+str(label_values.get(label))+'\n')
    with open('my_data/data/dev.txt','w',encoding='utf-8') as f3:
        for idx in dev_data.index:
            content=str(dev_data.loc[idx,'content'])
            content = content.replace('\r\n', '')
            content = content.replace('\n', '')
            content = content.replace('\r', '')
            label = dev_data.loc[idx, 'class_label']
            if label is None:
                print('class_label is none!')
            f3.write(content+'\t'+str(label_values.get(label))+'\n')
    print()

def gen_test_data(path):
    data = pandas.read_csv(path)
    data['content'] = data['content'].apply(lambda x:clean_chineses_text(x))
    with open('my_data/data/test.txt','w',encoding='utf-8') as f3:
        for idx in data.index:
            content=str(data.loc[idx,'content'])
            content = content.replace('\r\n', '')
            content = content.replace('\n', '')
            content = content.replace('\r', '')
            id = data.loc[idx, 'id']
            f3.write(content+'\t'+str(id)+'\n')

