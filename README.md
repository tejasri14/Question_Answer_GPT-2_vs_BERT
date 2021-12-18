# Question Answer BERT and GPT-2

Crispin Lobo cvl2106 and Tejasri Kurapati tk2928

# Summary 

The goal of this project is to compare the performance of BERT and GPT-2  for Question Answer Generation tasks. 

Our objective is to find the model that gives the most intelligible and grammatically accurate answers. We compare the two models qualitatively (intelligibility, grammatically accurate) and quantitatively (exactness Score and F1 score).

The models are pre-trained on SQuAD 2 dataset and fine tuned on CoQA dataset. The performance of GPT-2 and BERT is compared on Factual, Biomedical, and Conversational question answering dataset distribution. 

The BERT folder contains four files
1. BERT_qa_finetuning.ipynb
2. BERT_question_answering_squad_2.ipynb
3. BERT_evaluation.ipynb
4. BERT_question_answering_perplexity_evaluation.ipynb


The GPT_2 folder contains two folder
1. Implementaion - Contains all code used in implementaion
2. Experiments -  Contains all performed experiments

# Motivation 

With the growing dependency on chatbots to make information retrieval easy, it is important that we make the experience user friendly. 

Also, with the growing amount of digitized information, having a machine interpret a question and give you answers for it in real-time has lots of applications in the business, legal, healthcare, and entertainment domain. 

In this project, we seek to compare two question-answering models to understand which one is better in a conversational question-answering setting.

# Background Work

The main caveat in Question Answering models is to make the answers more intelligible. We want to build  a model that answers a particular question as close to a human’s response. In many question answering models till date (phre-GANs, LSTMs), we have observed that the response answers the question but the response is not tailored to the question. 

### For example: Q - What is the capital of India?
 							
### Ans - In December 1911 King George V of Britain decreed that the capital of British India would be moved from Calcutta (now Kolkata) to Delhi.

A lot of work has been done in analyzing the efficiency of BERT in QA context and it has been proven to be one of the best transformer models for QA. This is also addressed in the official BERT paper. Very little is known about GPT-2 for QA. Though the official paper suggests that GPT-2 is not the best models to be used for QA, we try fine-tuning GPT-2 on CoQA dataset to see if this helps GPT-2 perform better in QA setting

# Model Architecture

## BERT:
1. BERT has 340 million parameters
2. BERT has 24 layers
3. The number of self-attention heads in BERT is 16 and the hidden size is 1024.


## GPT-2:
1. GPT-2 has 1.5 billion parameters which 10 times more than GPT-1(117M).
2. GPT-2 had 48 layers and used 1600 dimensional vectors for word embedding.
3. Larger batch size of 512 and larger context window of 1024 tokens were used.
4. Layer normalisation was moved to input of each sub-block and an additional layer normalisation was added after final self-attention block.
5. At initialisation, the weight of residual layers was scaled by 1/√N, where N was the number of residual layers.


# Datasets

### CoQA 
CoQA is a large-scale dataset for building Conversational Question Answering systems. CoQA contains 127,000+ questions with answers collected from 8000+ conversations. 

Each conversation is collected by pairing two crowdworkers to chat about a passage in the form of questions and answers. 

CoQA tests reading comprehension capabilities and also the ability of models to answer questions that depend on conversation history 

### SQuAD 2
Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.

### Conversational dataset
CoQA test dataset which contains context as if someone was explaining a topic to a listener. The dataset comes with multiple questions and answers to the questions.

### Factual dataset
We chose the WikiQACorpus by Microsoft which contains Wikipedia context and questions to the context. The dataset also has the answers and the exact score. 

### Biomedical dataset 
For  biomedical dataset, we chose PubMed abstracts with questions and answers

# Solution Diagram 

<img width="707" alt="Screen Shot 2021-12-17 at 11 08 34 PM" src="https://user-images.githubusercontent.com/40158216/146628394-3734ddf8-b676-4b02-97da-9f73dd18c72a.png">

# Implementation 
The NVIDIA Tesla K80 GPU was used for fine tuning and evaluation

### Fine tuning BERT with CoQA dataset 

<img width="582" alt="Screen Shot 2021-12-17 at 11 09 50 PM" src="https://user-images.githubusercontent.com/40158216/146628428-9430d348-12ad-40e6-b3b5-fb26b2a8d633.png">

We use HuggingFace bert-large-uncased-whole-word-masking-squad2 model for the project

### GPT-2 Implementation Details

GPT-2 is a model with absolute position embeddings so it’s usually advised to pad the inputs on the right rather than the left.

The model  was trained with a causal language modeling (CLM) objective and is therefore powerful at predicting the next token in a sequence. Leveraging this feature allows GPT-2 to generate syntactically coherent text.

We have fine-tuned GPT-2 with CoQA dataset.

We use the Huggingface’s Pytorch implementation of GPT-2.

<img width="446" alt="Screen Shot 2021-12-17 at 11 11 27 PM" src="https://user-images.githubusercontent.com/40158216/146628469-29e48011-9389-4844-9c16-3faf012ae385.png">

# Outputs

<img width="615" alt="Screen Shot 2021-12-17 at 11 18 28 PM" src="https://user-images.githubusercontent.com/40158216/146628658-94ff951a-0568-42c8-9b04-774d597f37e6.png">


<img width="769" alt="Screen Shot 2021-12-17 at 11 18 42 PM" src="https://user-images.githubusercontent.com/40158216/146628664-b91beb8c-1d1f-4123-8064-763531ba330e.png">

# Qualitative Evaluation 

BERT
The model is not able to give Human like response. It returns the statements that are present in the context. 
The model is able to identify questions whose answers are not in the context
If a question has a spelling mistake, the model is not able to answer the question.
The model is able to understand the relation between different sentences in the context and answer questions

GPT-2
From the observations, we can say that GPT-2 does not do a good job in understanding the context and cannot be used for QA.

# Quantitative Evaluation 

### BERT 

Exact score -  79.97

F1 score - 83.01

Perplexity - 2.7365

### GPT-2

Exact score -  4.1

F1 score - 15.03

Perplexity - 51.187






