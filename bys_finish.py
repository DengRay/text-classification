#coding: utf-8
import os
import sklearn
from sklearn.naive_bayes import MultinomialNB
import numpy as np

train_data_list = []
test_data_list = []
train_class_list = []
test_class_list = []
folder_path_train = r"C:\Users\pc\Desktop\text_cla\data\traindata.txt"
folder_path_test = r"C:\Users\pc\Desktop\text_cla\data\testdata.txt"
with open(folder_path_train, 'r') as fp:
    text_train = fp.readlines()
for i in text_train:
    temp = i.split()
    train_data_list.append(temp[1:])
    train_class_list.append(temp[0])
    
with open(folder_path_test, 'r') as fp:
    text_test = fp.readlines()
for i in text_test:
    temp = i.split()
    test_data_list.append(temp[1:])
    test_class_list.append(temp[0])

#print(len(data_list))
#print(len(class_list))

all_words_dict = {}
for word_list in train_data_list:
    for word in word_list:
        if word in all_words_dict:
            all_words_dict[word] += 1
        else:
            all_words_dict[word] = 1          
all_words_tuple_list = sorted(all_words_dict.items(), key=lambda f:f[1], reverse=True) # 内建函数sorted参数需为list
all_words_list = list(zip(*all_words_tuple_list))[0]

#print(all_words_tuple_list[:100])
#print(all_words_list[:100])

stopwords_file = r'C:\Users\pc\Desktop\text_cla\data\stopwords_cn.txt'
words_set = set()
with open(stopwords_file, 'r') as fp:
    for line in fp.readlines():
        word = line.strip()
        if len(word)>0 and word not in words_set: # 去重
            words_set.add(word)
stopwords_set = words_set

feature_words = []
n = 1
for t in range(0, len(all_words_list), 1):
    if n > 1000: # feature_words的维度1000
        break
    # print all_words_list[t]
    if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1<len(all_words_list[t])<5:
        feature_words.append(all_words_list[t])
        n += 1

train_feature_list = []
test_feature_list = []
for text in train_data_list:
    text_words = set(text)
    ## -----------------------------------------------------------------------------------
    #if flag == 'nltk':
        ## nltk特征 dict
        #features = {word:1 if word in text_words else 0 for word in feature_words}
    #elif flag == 'sklearn':
        ## sklearn特征 list
    temp = [1 if word in text_words else 0 for word in feature_words]
    train_feature_list.append(temp)
    #else:
        #features = []
for text in test_data_list:
    text_words = set(text)
    temp = [1 if word in text_words else 0 for word in feature_words]
    test_feature_list.append(temp)
    
classifier = MultinomialNB().fit(train_feature_list, train_class_list)
test_accuracy = classifier.score(test_feature_list, test_class_list)

print ("finished")
print("The accuracy of testdata is :",test_accuracy)
#print(len(features))
#print(feature_words)
#print(len(all_words_list))            
#print(words_set)           
#print(len(words_set))