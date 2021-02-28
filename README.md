# Text_summarization-Model  
### Document:   
As a beginner of artificial intelligence, I explored NLP field in 2020 winter holidy. This is the presentation of my project, which is used to automatically generate a text summary . Since I am new to text-summarization, there may be some mistakes in the project, so I appreciate any kind of advice and feedback. The project includes three different models: model 1 is simply based on word frequnce , model 2 is based on the Seq2Seq model, model 3 is only implemented by the attention method.

## Model 1:


## Model 2: Sequence to Sequence Learning with Neural Networks  
### Introdoction:  
In deep learning, the seq2eq model is actually a encoder-decoder model , which is used to transform a sequence to another by the recurrent neural network (RNN). Most public projects on Github are usually applied to the language translation models, however, seq2seq model is able to carry great value to text summarization. Based on the paper [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) and some current translation models, model 2 is developed.  
Model 2 can be divided into three parts: `Encoder` , `Decoder` , `Attention Mechanism`  
* Encoer:    Comprised by a series of RNN , the main work of the encoder part is to encode the input source into a sigle vector(the vector is called the context vextor in the paper). Firstly, the input sentence would be embedded and then input into the encoder layers. In the stage of the encoder layers, for each step, the input to the encoder RNN is both the embedding, $e$, of the current word, $e(x_t)$, as well as the hidden state from the previous time-step, $h_{t-1}$, and the encoder RNN outputs a new hidden state $h_t$. It can be presented as:  
 "$$s_t = \\text{DecoderRNN}(d(y_t), s_{t-1})$$\n"

