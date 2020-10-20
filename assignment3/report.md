# Assignment 3 - Language Modeling

*I pledge my honor that I have abided by the Stevens Honor System.* - Joshua Schmidt

Name: Joshua Schmidt, Date: 10/20/2020

## Introduction

In this assignment, there were two main techniques that were used for language modeling --- N-grams and RNN. A language model predicts the probability of a sequence of words, and can be used for text generation, machine translation, and auto-correct applications. In this assignment, language models were used to predict the next word given a sequence of words. The perplexity of each model was reported.

## Code structure

The source code for this project can be found in the `src` folder. Dependencies are managed with anaconda and can be installed through initializing the environment in the `environment.yml` file in the base directory. The entry point to the program is `src/main.py`. This script first cleans the datasets (which need to be placed in `data/raw_data`), and then runs the training and validation for all of the models. Working files are stored in the `data` folder, with clean csv files (of pandas dataframes) in the `data/clean_data` folder, and trained models in the `data/models` folder. The models are saved to disk based off of the training data file name. Outputs are logged using the `loguru` package, and the complete logs are saved in `output/logs.txt`. The `outputs` folder also contains graphs of loss and accuracy for the RNN training.

## N-gram Results

The complete predictions for the models with different N-grams smoothing algorithms can be seen in the log file, with samples below. The model was trained as a bigram, but can support any arbitrary n-grams. The testing was done on the `test.txt` data.

### Basic N-gram

- `2020-10-20 14:08:42.860 | INFO     | ngrams:n_grams_predict_next:451 - 1. consumers may want to move their telephones a little closer to the tv set [[is]]`
- `2020-10-20 14:08:42.868 | INFO     | ngrams:n_grams_predict_next:451 - 54. fox broadcasting [UNK] with this concept last year when viewers of married with children voted on whether al should say i love you to [UNK] on [UNK]'s day [[promoting]]`
- `2020-10-20 14:08:43.044 | INFO     | ngrams:n_grams_predict_next:451 - 801. the gathering is expected to focus on curbing the [UNK] of [UNK] and [UNK] limiting damage from industrial [UNK] and improving the handling of harmful chemicals [[from]]`

These are three samples from the basic N-gram model, with the predicted words in double brackets. These three randomly chosen outputs illustrate that this model can produce coherent sentences that maintain grammar and context. The perplexity was 3.5483.

### Good Turing

- `2020-10-20 14:08:45.062 | INFO     | ngrams:n_grams_predict_next:451 - 1. consumers may want to move their telephones a little closer to the tv set [[is]]`
- `2020-10-20 14:08:45.127 | INFO     | ngrams:n_grams_predict_next:451 - 362. no more [UNK] [UNK] or [UNK] [UNK] promises [UNK] corp. of [UNK] ind. the designer of a bed support to replace traditional [UNK] [[activities]]`
- `2020-10-20 14:08:45.137 | INFO     | ngrams:n_grams_predict_next:451 - 432. the metal and marble lobby of centrust bank's headquarters is [UNK] than your average savings and loan [[guarantees]]`

For the Good Turing N-gram implementation, the output was overall marginally more accurate than the basic N-gram model. Most of the outputs were the same, but there were fewer unknown predictions. The perplexity was 2.5942.

### Kneser Ney

- `2020-10-20 14:08:47.114 | INFO     | ngrams:n_grams_predict_next:451 - 1. consumers may want to move their telephones a little closer to the tv set [[is]]`
- `2020-10-20 14:08:47.294 | INFO     | ngrams:n_grams_predict_next:451 - 556. [UNK] [UNK] [UNK] a [UNK] [UNK] at [UNK] associates in san francisco considers it [UNK] conflict of interest for an auction house to both advise a client on purchases and to set price estimates on the paintings to be purchased [[for]]`
- `2020-10-20 14:08:47.655 | INFO     | ngrams:n_grams_predict_next:451 - 1957. a n n drop for [UNK] ag and dresdner bank ag's n n decline were especially [UNK] for their respective boards whose plans for major rights issues in november could now be in jeopardy [[because]]`

In the Kneser Ney implementation, there seemed to be an increase in confidence in the best predicted value, but a decrease in the overall quality of the outputs. The perplexity of the model was 1.9225.

### Next Word Predict

The first thirty lines predicted by the Kneser Ney N-grams implementation over the `input.txt` data are shown below (can also be found in the logs file):

- `2020-10-20 14:08:50.923 | INFO     | ngrams:n_grams_predict_next:451 - 1. but while the new york stock exchange didn't fall [[far]]`
- `2020-10-20 14:08:50.923 | INFO     | ngrams:n_grams_predict_next:451 - 2. some circuit breakers installed after the october n crash failed [[because]]`
- `2020-10-20 14:08:50.923 | INFO     | ngrams:n_grams_predict_next:451 - 3. the n stock specialist firms on the big board floor [[at]]`
- `2020-10-20 14:08:50.923 | INFO     | ngrams:n_grams_predict_next:451 - 4. big investment banks refused to step up to the plate [[and]]`
- `2020-10-20 14:08:50.923 | INFO     | ngrams:n_grams_predict_next:451 - 5. heavy selling of standard & poor's 500-stock index futures [[and]]`
- `2020-10-20 14:08:50.923 | INFO     | ngrams:n_grams_predict_next:451 - 6. seven big board stocks ual amr bankamerica walt disney capital [[[UNK]]]`
- `2020-10-20 14:08:50.923 | INFO     | ngrams:n_grams_predict_next:451 - 7. once again the specialists were not able to handle the [[[UNK]]]`
- `2020-10-20 14:08:50.924 | INFO     | ngrams:n_grams_predict_next:451 - 8. [UNK] james [UNK] chairman of specialists henderson brothers inc. it [[was]]`
- `2020-10-20 14:08:50.924 | INFO     | ngrams:n_grams_predict_next:451 - 9. when the dollar is in a [UNK] even central banks [[of]]`
- `2020-10-20 14:08:50.924 | INFO     | ngrams:n_grams_predict_next:451 - 10. speculators are calling for a degree of liquidity that is [[the]]`
- `2020-10-20 14:08:50.924 | INFO     | ngrams:n_grams_predict_next:451 - 11. many money managers and some traders had already left their [[[UNK]]]`
- `2020-10-20 14:08:50.924 | INFO     | ngrams:n_grams_predict_next:451 - 12. then in a [UNK] plunge the dow jones industrials in [[the]]`
- `2020-10-20 14:08:50.924 | INFO     | ngrams:n_grams_predict_next:451 - 13. [UNK] trading accelerated to n million shares a record for [[the]]`
- `2020-10-20 14:08:50.925 | INFO     | ngrams:n_grams_predict_next:451 - 14. at the end of the day n million shares were [[major]]`
- `2020-10-20 14:08:50.925 | INFO     | ngrams:n_grams_predict_next:451 - 15. the dow's decline was second in point terms only [[to]]`
- `2020-10-20 14:08:50.925 | INFO     | ngrams:n_grams_predict_next:451 - 16. in percentage terms however the dow's dive was the [[[UNK]]]`
- `2020-10-20 14:08:50.925 | INFO     | ngrams:n_grams_predict_next:451 - 17. shares of ual the parent of united airlines were extremely [[unlikely]]`
- `2020-10-20 14:08:50.925 | INFO     | ngrams:n_grams_predict_next:451 - 18. wall street's takeover-stock speculators or risk arbitragers had placed [[some]]`
- `2020-10-20 14:08:50.925 | INFO     | ngrams:n_grams_predict_next:451 - 19. at n p.m. edt came the [UNK] news the big [[[UNK]]]`
- `2020-10-20 14:08:50.926 | INFO     | ngrams:n_grams_predict_next:451 - 20. on the exchange floor as soon as ual stopped trading [[altogether]]`
- `2020-10-20 14:08:50.926 | INFO     | ngrams:n_grams_predict_next:451 - 21. several traders could be seen shaking their heads when the [[[UNK]]]`
- `2020-10-20 14:08:50.926 | INFO     | ngrams:n_grams_predict_next:451 - 22. for weeks the market had been nervous about takeovers after [[the]]`
- `2020-10-20 14:08:50.926 | INFO     | ngrams:n_grams_predict_next:451 - 23. and n minutes after the ual trading halt came news [[of]]`
- `2020-10-20 14:08:50.926 | INFO     | ngrams:n_grams_predict_next:451 - 24. arbitragers couldn't dump their ual stock but they rid [[[UNK]]]`
- `2020-10-20 14:08:50.926 | INFO     | ngrams:n_grams_predict_next:451 - 25. for example their selling caused trading halts to be declared [[a]]`
- `2020-10-20 14:08:50.926 | INFO     | ngrams:n_grams_predict_next:451 - 26. but as panic spread speculators began to sell blue-chip stocks [[[UNK]]]`
- `2020-10-20 14:08:50.927 | INFO     | ngrams:n_grams_predict_next:451 - 27. when trading was halted in philip morris the stock was [[a]]`
- `2020-10-20 14:08:50.927 | INFO     | ngrams:n_grams_predict_next:451 - 28. selling [UNK] because of waves of automatic stop-loss orders which [[is]]`
- `2020-10-20 14:08:50.927 | INFO     | ngrams:n_grams_predict_next:451 - 29. most of the stock selling pressure came from wall street [[the]]`
- `2020-10-20 14:08:50.929 | INFO     | ngrams:n_grams_predict_next:451 - 30. traders said most of their major institutional investors on the [[[UNK]]]`

Again, the predictions are in double brackets. Most of the predictions make grammatical sense, but they are not always the best prediction for the given sentence. The problem is these predictions are only based off the last words of the sentence, as opposed to the entire sentence with its context.

## RNN Results

The RNN overall had better performance than the N-grams models shown above. In the RNN, a text vectorizer was used to convert each word to a number. A tensorflow sequential model, with an embedding layer, a GRU layer and a Dense layer was used for the predictions.

### Training

The loss of the Adam optimizer after training for 10 epochs was 2.5858, with an accuracy of 0.4745. The training curve is shown below:

![Training Curve](./output/rnn_train_train_epoch_9.png)

\newpage

### test.txt

Below are prediction samples from the `test.txt` dataset. All of the outputs can be found in the log file.

- `2020-10-20 14:19:10.070 | INFO     | rnn:rnn_predict_next:292 - 1. consumers may want to move their telephones a little closer to the tv set [[for]]`
- `2020-10-20 14:19:22.581 | INFO     | rnn:rnn_predict_next:292 - 224. honeywell said it is negotiating the sale of a second stake in [UNK] but indicated it intends to hold at least n n of the joint venture's stock long term [[bonds]]`
- `2020-10-20 14:19:50.306 | INFO     | rnn:rnn_predict_next:292 - 724. but the hurdle of financing still has to be resolved [[quickly]]`

There are fewer unknown outputs in the predictions (compared to the n-gram models), and the predictions for the most part make sense. With more epochs of training, the model should get even better. The perplexity was 0.0841.

## Proof

Proof that perplexity is $\exp{\frac{total loss}{number of predictions}}$:

![Perplexity Proof](./proof/perplexity.jpg)

\newpage

### input.txt

Below are the next word predictions for the `input.txt` dataset. Again, the outputs can be found in the log file.

- `2020-10-20 14:22:18.620 | INFO     | rnn:rnn_predict_next:292 - 1. but while the new york stock exchange didn't fall [[the]]`
- `2020-10-20 14:22:18.872 | INFO     | rnn:rnn_predict_next:292 - 2. some circuit breakers installed after the october n crash failed [[to]]`
- `2020-10-20 14:22:18.927 | INFO     | rnn:rnn_predict_next:292 - 3. the n stock specialist firms on the big board floor [[traders]]`
- `2020-10-20 14:22:18.978 | INFO     | rnn:rnn_predict_next:292 - 4. big investment banks refused to step up to the plate [[and]]`
- `2020-10-20 14:22:19.038 | INFO     | rnn:rnn_predict_next:292 - 5. heavy selling of standard & poor's 500-stock index futures [[contract]]`
- `2020-10-20 14:22:19.090 | INFO     | rnn:rnn_predict_next:292 - 6. seven big board stocks ual amr bankamerica walt disney capital [[ltd]]`
- `2020-10-20 14:22:19.147 | INFO     | rnn:rnn_predict_next:292 - 7. once again the specialists were not able to handle the [[price]]`
- `2020-10-20 14:22:19.200 | INFO     | rnn:rnn_predict_next:292 - 8. [UNK] james [UNK] chairman of specialists henderson brothers inc. it [[is]]`
- `2020-10-20 14:22:19.252 | INFO     | rnn:rnn_predict_next:292 - 9. when the dollar is in a [UNK] even central banks [[biggest]]`
- `2020-10-20 14:22:19.305 | INFO     | rnn:rnn_predict_next:292 - 10. speculators are calling for a degree of liquidity that is [[unk]]`
- `2020-10-20 14:22:19.361 | INFO     | rnn:rnn_predict_next:292 - 11. many money managers and some traders had already left their [[ual]]`
- `2020-10-20 14:22:19.414 | INFO     | rnn:rnn_predict_next:292 - 12. then in a [UNK] plunge the dow jones industrials in [[the]]`
- `2020-10-20 14:22:19.467 | INFO     | rnn:rnn_predict_next:292 - 13. [UNK] trading accelerated to n million shares a record for [[the]]`
- `2020-10-20 14:22:19.519 | INFO     | rnn:rnn_predict_next:292 - 14. at the end of the day n million shares were [[traded]]`
- `2020-10-20 14:22:19.571 | INFO     | rnn:rnn_predict_next:292 - 15. the dow's decline was second in point terms only [[for]]`
- `2020-10-20 14:22:19.628 | INFO     | rnn:rnn_predict_next:292 - 16. in percentage terms however the dow's dive was the [[stock]]`
- `2020-10-20 14:22:19.681 | INFO     | rnn:rnn_predict_next:292 - 17. shares of ual the parent of united airlines were extremely [[profitable]]`
- `2020-10-20 14:22:19.733 | INFO     | rnn:rnn_predict_next:292 - 18. wall street's takeover-stock speculators or risk arbitragers had placed [[the]]`
- `2020-10-20 14:22:19.785 | INFO     | rnn:rnn_predict_next:292 - 19. at n p.m. edt came the [UNK] news the big [[board]]`
- `2020-10-20 14:22:19.841 | INFO     | rnn:rnn_predict_next:292 - 20. on the exchange floor as soon as ual stopped trading [[the]]`
- `2020-10-20 14:22:19.894 | INFO     | rnn:rnn_predict_next:292 - 21. several traders could be seen shaking their heads when the [[government]]`
- `2020-10-20 14:22:19.947 | INFO     | rnn:rnn_predict_next:292 - 22. for weeks the market had been nervous about takeovers after [[the]]`
- `2020-10-20 14:22:20.000 | INFO     | rnn:rnn_predict_next:292 - 23. and n minutes after the ual trading halt came news [[on]]`
- `2020-10-20 14:22:20.056 | INFO     | rnn:rnn_predict_next:292 - 24. arbitragers couldn't dump their ual stock but they rid [[of]]`
- `2020-10-20 14:22:20.110 | INFO     | rnn:rnn_predict_next:292 - 25. for example their selling caused trading halts to be declared [[in]]`
- `2020-10-20 14:22:20.162 | INFO     | rnn:rnn_predict_next:292 - 26. but as panic spread speculators began to sell blue-chip stocks [[in]]`
- `2020-10-20 14:22:20.215 | INFO     | rnn:rnn_predict_next:292 - 27. when trading was halted in philip morris the stock was [[created]]`
- `2020-10-20 14:22:20.267 | INFO     | rnn:rnn_predict_next:292 - 28. selling [UNK] because of waves of automatic stop-loss orders which [[traded]]`
- `2020-10-20 14:22:20.325 | INFO     | rnn:rnn_predict_next:292 - 29. most of the stock selling pressure came from wall street [[william]]`
- `2020-10-20 14:22:20.378 | INFO     | rnn:rnn_predict_next:292 - 30. traders said most of their major institutional investors on the [[cboe]]`

These predictions make more sense in the context of the given sentence. Sentences that have to do with trading and finance have predictions that are also in the realm of trading. One that sticks out is number 21, where traders are disappointed in the government's actions (this is fairly common).
