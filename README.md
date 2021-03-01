# Text_summarization-Model  
### Document:   
As a beginner of artificial intelligence, I explored NLP field in 2020 winter holidy. This is the presentation of my project, which is used to automatically generate a text summary . Since I am new to text-summarization, there may be some mistakes in the project, so I appreciate any kind of advice and feedback. The project includes three different models: model 1 is simply based on word frequnce , model 2 is based on the Seq2Seq model, model 3 is only implemented by the attention method.

## Model 1:


## Model 2: Sequence to Sequence Learning with Neural Networks  
### Introdoction:  
In deep learning, the seq2eq model is actually a encoder-decoder model , which is used to transform a sequence to another by the recurrent neural network (RNN). Most public projects on Github are usually applied to the language translation models, however, seq2seq model is able to carry great value to text summarization. Based on the paper [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) and [some current translation models](https://github.com/ITCUI-XJTLU/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb), model 2 is developed.  
Model 2 can be divided into three parts: `Encoder` , `Decoder` , `Attention Mechanism`  
* **Encoer** :    Comprised by a series of RNN (in my model , classical RNN model is replaced by GRU, which is an advanced RNN model), the main work of the encoder part is to encode the input source into a sigle vector (the vector is called the context vector in the paper). Firstly, the input sentence would be embedded and then input into the encoder layers. In the stage of the encoder layers, for each step _t_, the input to the encoder RNN is both the embedding of the current word _e(x_t)_, as well as the hidden state from the previous time-step _h_(t-1)_ , and the encoder RNN outputs a new hidden state _h_t_. It can be presented as:   

* **Decoder** :    Comprised by a series of RNN (in my model , classical RNN model is replaced by GRU, which is an advanced RNN model), the main work of the decoder part is to decode the context vector into a target sequence (the summarization of the input). At each time-step, the input to the decoder RNN is the embedding of current word, _d(y_t)_, as well as the hidden state from the previous time-step, _s(t-1)_ , where the initial decoder hidden state, _s0_, is the context vector, _s0 = z = h_T_, i.e. the initial decoder hidden state is the final encoder hidden state. Thus, similar to the encoder, we can represent the decoder as:

* **Attention** :   Attention mechanism is a method to improve the performance of classical encoder-decoder framework, which usually has trouble with gradient disappearence and gradient blast. Instead of only getting information from the context vector, the attention mechanism allow the decoder part use all of token information in the final hidden layer by the diffrent weights. If you want view the detail of this technique , please take a look at the code of Attention class in model2.trainclass.py or a famous paper [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) .  We can visualize the method as the picture below: 

* **Note** : Apart from the parts above , there are also some important and common motheds to process data , such as using torchtext to automatically generate dataset we need , and loading english dictionaty from spacy library to embede the words of the input. **If you are confused about the codes , please read the explanation around the codes**.  

### Model Information :
#### Prerequisites:
* Python == 3.6
* numpy == 1.18.0
* pandas == 0.24.2
* torch == 1.0.0
* torchtext == 0.4.0
* spacy == 2.1.8
* GPU (optional)
> Warning: I suggest you to create a new virtual environment by anaconda for this project . Since if your packages can not match with each other , some strange and tough problems will appears. (I was stucked here for a time :sob: .)

#### Data: 
* [My Data in kaggle](https://www.kaggle.com/cuitengfeui/textsummarization-data)

