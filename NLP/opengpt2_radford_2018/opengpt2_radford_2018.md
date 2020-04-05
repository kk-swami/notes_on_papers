# Citation  

Language models are unsupervised multitask learners
Radford et al 2018

# Tags  

OpenGPT2, Open AI Transformers, transfer learning

# Significance

Demonstrate language models that can perform downstream tasks in a zero shot setting, for a wide variety of tasks 


# Context and summary  

Current ML systems work well on narrow tasks, but generalize poorly across tasks. The goal is to create a general 
system which can work well across tasks. Benchmarks such as GLUE (Wang 2018) and decaNLP (Mccann 2018) help evaluate this. 

Current SOTA systems for transfer learning use a combination of pre-training and supervised fine tuning.
started with the philosophy of [w2vec](../w2vec1_mikolov_2013/w2vec1_mikolov_2013.md) and [glove](../glove_pennington_2014/glove_pennington_2014.ipynb)
, where  word embeddings are transferred between tasks , moved to transferring higher level representations 
of recurrent networks between tasks ([ELMO](../elmo_peters_2018/Elmo_peters_2018.md) and [ULMFIT](../ulmfit_howard_ruder_2018/ulmfit_howard_ruder_2018.md)) 
and finally using self-attention blocks with a [transformer](../transformers_vaswani_2017/transformers_attention_vaswani_2017.md) architecure ([OpenGPT](opengpt_radford_2018.md) and [BERT])


The above methods involve minimal (or more) supervised training for the specific task, here, the authors 
pursue development of Language models which can demonstrate zero-shot learning, and can perform well on a new task
without explicit supervised learning


Other earlier approaches to multi-task learning express a common model across tasks in a supervised setting as p(output|input, task) 
instead of the usual p(outout|input) . Such a task conditioning is implemented architecturally (Kaiser et al 2017) 
or algorithmically (Finn et al 2017). Mccann et al 2018 use language itself to capture tasks, inputs and outputs,
and train a single model, the MQAN to perform across a wide variety of tasks. 

The authors of this paper argue that implementing a language model in an unsupervised setting maybe harder to train, 
but theoretically has the same or lesser global minimum as using a supervised setting. So basically, a language model 
with enough capacity (large number of parameters :) should infer and perform well on different tasks on natural language sequences





# Method in more detail  

1) Input representation - use byte pair encoding (Sennrich et al 2015) with some customization, (since it avoids
usual steps of lower casing, tokenizing, removing OOV words, which restricts vocabulary , which goes against the goal of a
large diverse model )   . BPE basically interpolates between word level inputs for frequent symbol sequences and
character level input for infrequent symbol sequences. (some customization is done for BPE - prevention of merging across
character categories for any byte sequence -> for example, dog? dog. and dog should be merged)  
This method allows dealing with any input string. 

2) Model - Largely follows the [earlier GPT paper](../opengpt_radford_2018/opengpt_radford_2018.md)) with the following modifications
    a) Layer normalization (Ba 2016) moved to input of each sub block
    b) additional layer normalization after final self attention block
    c) modified initialization - scale weights of residual layers by a factor of 1/sqrt(N) where N is the  number of residual layers
    d) vocab expanded to 50257
    e) context size increased from 512 to 1024 tokens
    f) larger batch size of 512 used


# Experiments    
1) Dataset - Since the goal is to build a language model with as much capacity as possible , which can work on a wide variety of tasks,
a diverse data set is necessary . So instead of using standard datasets like news articles or wikipedia, web scrapes are better 
(Eg : Common crawl).
 advantages  - orders of magnitude larger and more diverse than standard data sets
 disadvantages - has significant data quality issues.  
 To solve the dq issues, only web pages which are curated/filtered by humans are scraped - how do we identify them  ?
 use all outbound links from reddit which have 3 "karma" .
 This results in a dataset called webtext, which contains text subset of the 45MM links. 
 A combination of dragnet (Peters and Lecocq 2013) and newspaper content extractors are used for this purpose. 
 -> eventually 8 MM documents, 40 GB of text. Removed wiki articles as they are used for testing. 
 
 2) architecture hyperparameters   
 
 Tested with 4 architectures for the LM - layer size 12, 24, 36, 48
 
 ![architecture hyperparameters](opengpt2_1.png "Table 2 from paper" )  
 
 Image credit - table 2 from paper    
 
 The smallest model is equivalent to opengpt1 in terms of parameter size, the second smallest 
 equivalent to BERT. 
 
 





## Results

1) Zero shot results on many data sets for LM  - Large improvements noted on several datasets 


![results_1](opengpt2_2.png "Table 3 from paper" )  

Image credit - table 3 from paper  


2) Children's book test (Hill et al 2015)  
Rather than perplexity, metric is predicting correct choice out of 10 possible choices for omitted word. 


![results_1](opengpt2_3.png "Figure 2 from paper" )    

Image credit - figure 2 from paper  

3) Lambada dataset (Paperno et al 2016) 
Task is to predict final word of sentences which require at least 50 tokens of context for a human to successfully predict 
This models improves SOTA from 99.8 (Graves et al 2016) to 8.8 and accuracy from 19 to 52.7% (Dehgelani)  

4) winograd schema challenge (levesque et al 2012) 
to resolve ambiguities in text. 

![results_1](opengpt2_4.png "Figure 3 from paper" )   
Image credit - figure 3 in paper. 


5) Reading comprehension (CoQA dataset Reddy 2018) 
55% F1 compared to near 89% for BERT which is a supervised model  (done by greedy decoding when conditioned
on document, history of associated conversation and final token A)  

6) Summarization (Nallapati 2016)

To induce summarization, add token TL;DR after article, generate 100 tokens with 
top K random sampling (k set to 2) . Use first three generated sentences in the 100 tokens as summary

Not great compared to supervised models   

![results_1](opengpt2_5.png "Figure 4 from paper" )   
Image credit - figure 4 in paper. 


7) Translation 
To allow the model to infer that the task is translation, 
condition on a sample of example pairs of format english sentence = french sentence,
and after final prompt of english sentence, sample from model with with greedy decoding ,
use first generated word as translation (?)  
Gets a BLEU of 5 on WMT-14 english french test set, and 11.5 on the french english test set  


8) Question answering  
Similar to translation, works by seeding context of LM with
example question answer pairs, which helps LM infer short answer style of dataset . 
GPT-2 answers 4.2% correctly, much lesser than 30-50% of supervised systems (Alberti et al 2019) 









