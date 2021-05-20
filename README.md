# This repository contains the learnings of the members of team CryptoFutureX
 ## 1.BERT  
 [BERT](https://github.com/google-research/bert) has inspired great interest in the field of NLP, especially the application of the Transformer for NLP tasks. This has led to a experimenting with different aspects           of pre-training, transformers and fine-tuning.
Reasons for using BERT are as following:
* It’s easy to get that BERT stands for Bidirectional Encoder Representations from Transformers. Each word here has a meaning to it and we will encounter that one by                one in this article. For now, the key takeaway from this line is – BERT is based on the Transformer architecture.
* BERT is pre-trained on a large corpus of unlabelled text including the entire Wikipedia and Book Corpus.
* BERT is a “deeply bidirectional” model. Bidirectional means that BERT learns information from both the left and the right side of a token’s context during the training phase.
![MOdels](https://www.google.com/imgres?imgurl=https%3A%2F%2Fmiro.medium.com%2Fmax%2F2020%2F1*RLBWful7k50nV9zTJCuZ4Q.png&imgrefurl=https%3A%2F%2Fmedium.com%2Fhuggingface%2Fdistilbert-8cf3380435b5&tbnid=LjpY9pJNiV1s3M&vet=12ahUKEwi4ur7RzNjwAhVZXX0KHYguBtAQMygCegUIARCrAQ..i&docid=IUWAALBSiomanM&w=1010&h=202&q=bert%20and%20distilbert&ved=2ahUKEwi4ur7RzNjwAhVZXX0KHYguBtAQMygCegUIARCrAQ)

## 2.DistilBERT 
At some point the BERT model increases become harder due to GPU/TPU memory limitations and longer training times. To address these problems, we used Dilbert to                    lower memory consumption and increase the training
[DistilBERT](https://github.com/huggingface/transformers/blob/master/docs/source/model_doc/distilbert.rst) is a small, fast, cheap, and light Transformer model trained by distilling Bert base. It has 40% less parameters than bert-base-uncased, runs 60% faster while preserving over 95% of Bert’s performances.

* Accurate as much as Original BERT Model
* 60% faster
* 40% fewer parameters
* It can run on CPU
---
# Challenges Faced 
As we know Colab gives us 12 hours of continuous execution time. After that, the whole virtual machine is cleared and we have to start again. We can run multiple CPU, GPU, and TPU instances simultaneously, but our resources are shared between these instances and in our scenario its takes almost 8 hours to train one epoch.Hence due to unavailability of high computational power we couldnt get the results.
