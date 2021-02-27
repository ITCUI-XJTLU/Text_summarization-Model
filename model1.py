import re
import  jieba
import model1_function as fun

##预处理
text_location="D://Research//text-summariztaion//data1.txt"
stopwords_location="D://Research//text-summariztaion//stopwords-master//baidu_stopwords.txt"
summarization_num=3

text=fun.open_text(text_location)
setences_list=fun.get_sentecs(text)
stopwords=fun.get_stopwords(stopwords_location)

##开始统计分析：
word_frequence=fun.count_fre(text,stopwords)
sentence_sore=fun.sore(setences_list,word_frequence)
top_list=fun.order_sentences(sentence_sore,summarization_num)

print(top_list)


