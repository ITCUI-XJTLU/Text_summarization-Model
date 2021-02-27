from os import path
import jieba
import re


def open_text(location):
    with open(location,encoding='utf-8') as f:
        data=f.read()
    return data

#把text这一整个字符串切开
def get_sentecs(text):
    all_lines=text.split('\n')
    information=""
    for line in all_lines:
        information+=line
    sentences=re.split('[。？！]',information)
    return sentences

def get_stopwords(location):
    with open(location,encoding='utf-8') as f:
        words=f.read()
    stopwords=words.split("\n")

    return stopwords

#统计词频
def count_fre(text,stopwords):
    wordcount={}
    for word in jieba.cut(text):
        if word not in stopwords:
            if word not in wordcount:
                wordcount[word]=1
            else:
                wordcount[word]+=1

    for key in wordcount.keys():
        wordcount[key]=wordcount[key]/max(wordcount.values())

    return wordcount

#计算句子得分
def sore (sentence_list,wordcount):
    sentences_sore={}
    for sentence in sentence_list:
        if sentence not in sentences_sore:
            single_sore=0
            for word in jieba.cut(sentence):
                if word in wordcount:
                    single_sore+=wordcount[word]
            sentences_sore[sentence]=single_sore
    return sentences_sore

#排序，提取出概率最大的句子们
def order_sentences(sentence_sore,num):
    top_sentences=sorted(sentence_sore.items(),key=lambda item:item[1],reverse=True)
    top_sentences=top_sentences[:num]
    #top_list=()
    #for sentence in top_sentences:
        #top_list.add(sentence[0])

    return top_sentences

