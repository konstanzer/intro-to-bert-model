# Intro to BERT

---
- [Summary](#summary)
- [Data](#data)
- [Planning pieline](#planning-pipeline)
- [Results](#results)
- [Appendix](#appendix)

## Summary

In 2018, Google researchers released the source code for a massive language model called BERT (Bidirectional Encoder Representations from Transformers.) BERT, in turn, was built on top of a new deep learning architecture called a transformer as well as deep contextual word embeddings. BERT smashed many of the General Language Understanding Evaluation (GLUE) benchmarks and thereby marked a watershed moment in NLP. In this project, I tried to understnd this architecture and then fine-tuned the language model ona sentiment classification task using the transformer library available at ![HuggingFace](https://huggingface.co/bert-base-uncased).

## Data

### Overview: Large Movie Review Dataset v1.0

This dataset contains Internet Movie Database (IMDb) movie reviews along with their associated binary sentiment polarity labels.
It is intended to serve as a benchmark for sentiment classification.

The dataset is described and ![linked to here](https://pitmonticone.github.io/BertSentimentClassification/).

### Description

Description provided by ![Pietro Monticone](https://github.com/pitmonticone).

The core dataset contains 50,000 reviews split evenly into 25k train and 25k test sets.
The overall distribution of labels is balanced (25k pos and 25k neg). In the entire collection, no more than 30 reviews are allowed for any given movie because reviews for the same movie tend to have correlated ratings.
Further, the train and test sets contain a disjoint set of movies, so no significant performance is obtained by memorizing movie-unique terms and their associated with observed labels.
In the labeled train/test sets, a negative review has a score <= 4 out of 10, and a positive review has a score >= 7 out of 10.
Thus reviews with more neutral ratings are not included in the train/test sets.

There are two top-level directories [train/, test/] corresponding to the training and test sets.
Each contains [pos/, neg/] directories for the reviews with binary labels positive and negative.
Within these directories, reviews are stored in text files named following the convention [[id]_[rating].txt] where [id] is a unique id and [rating] is the star rating for that review on a 1-10 scale.
For example, the file [test/pos/200_8.txt] is the text for a positive-labeled test set example with unique id 200 and star rating 8/10 from IMDb.
The [train/unsup/] directory has 0 for all ratings because the ratings are omitted for this portion of the dataset.

In addition to the review text files, we include already-tokenized bag of words (BoW) features that were used in our experiments.
These  are stored in .feat files in the train/test directories. Each .feat file is in LIBSVM format, an ascii sparse-vector format for labeled data. 
The feature indices in these files start from 0, and the text tokens corresponding to a feature index is found in [imdb.vocab].
So a line with 0:7 in a .feat file means the first word in [imdb.vocab] (the) appears 7 times in that review.

Reference: Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).

### Dictionary

| Feature                        | Description                                                                                                            |
|--------------------------------|------------------------------------------------------------------------------------------------------------------------|
| 'review'                       | raw text of user review (string)                              |
| 'label'                        | positive or negative (integer, target)                        |

## Planning pipeline

Step 1: Acquire

* Data is available ![on Kaggle](https://www.kaggle.com/macespinoza/large-movie-review-dataset-10) or at link above.

Step 3: Prepare Data

* Data is pre-split 50-50 (train, labeled / test, unlabeled) 

Step 4: Explore & Preprocess

Step 5: Model

* Logistic TFIDF baseline
* BERT-base, fine-tuned 

## Results

| model | classification accuracy |
| --- | --- |
| Logistic TFIDF baseline | .67 |
| BERT-base, fine-tuned | .89 |


Beyond the metrics, I looked at the misclassified reviews and was able to conclude some interesting things about state-of-the-art language models. 

* They can't detect sarcasm/trolling ("I loved it. 1 star.")
* Ambiguous reviews make it hard for any machine or person to predict the review.
* They get confused when a reviewer likes something other than the movie but not the movie itself. ("The book was great. The movie was not the book.")
* They predict based on any positive sentiment  ("I am great. I didn't watch the movie.")

## Appendix

### A technical overview of the transformer architecture

A transformer is a model architecture that eschews recurrence (RNN/LSTM/GRU) and instead relies entirely on an attention mechanism to draw global dependencies between input and output. Before Transformers, the dominant sequence-to-sequence models were based on recurrent or convolutional neural networks. The original Transformer employed an encoder and decoder, although modern Transformers do not, and also removed recurrence in favor of attention mechanisms allowing for more parallelization than methods like RNNs and CNNs, resulting in both faster training and improved performance on industry benchmarks.

 *Attention is All You Need* (Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin. (2017))[1] brought the attention-only architecture to the machine learning world. In the following years, modernized transformers, including GPT-2 or BERT, achieved state of the art results on natural language proceessing benchmarks. Further research is expanding the use of Transformers to computer vision.


### References

[1] Vaswani, Ashish & Shazeer, Noam & Parmar, Niki & Uszkoreit, Jakob & Jones, Llion & Gomez, Aidan & Kaiser, Lukasz & Polosukhin, Illia, “Attention is all you need” , 2017.

[2] Peter Bloem, “Transformers from scratch” blog post, 2019.

[3] Jay Alammar, “The Ilustrated Transformer” blog post, 2018.

[4] Lilian Weng, “Attention? Attention!!” blog post, 2018.

[5] Ricardo Faúndez-Carrasco, “Attention is all you need’s review” blog post, 2017

[6] Alexander Rush, “The Annotated Transformer”, 2018, Harvard NLP group.

[7] Eduardo Munoz, “Discovering the Transformer paper”, blog post, 2020.

