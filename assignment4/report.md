# Assignment 4 - Language Modeling

*I pledge my honor that I have abided by the Stevens Honor System.* - Joshua Schmidt

Name: Joshua Schmidt, Date: 11/10/2020

## Introduction

In this assignment, the goal was to create two deep learning models --- a CNN and an RNN (LSTM). These models were trained and tested on two datasets --- the book dataset, used in previous assignments, and a new review dataset. The first task was a simple classification problem, and the second was a sentiment analysis problem. Tensorflow was used to create the models, and the results from the models were compared to those created in previous assignments, namely the logistic regression and multilayer perceptron models.

## Model Descriptions

The four models all used the same training / validation / test split, with 80% of the whole dataset used for training, and 20% used for testing. 20% of the training data was used for validation, leaving 64% used for training data. These numbers were chosen somewhat arbitrarily, with the goal of using a larger percentage of the data for training than the other two categories.

### LSTM

The first model created was the LSTM model. The loss function used was sparse categorical cross-entropy, because it was recommended to use a cross-entropy loss function, and the data is categorical. The model has five sequential layers for both the book classification and sentiment analysis tasks. The summary for one of the LSTM models for books is shown below:

```txt
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
text_vectorization (TextVect (None, 250)               0         
_________________________________________________________________
embedding (Embedding)        (None, 250, 16)           160000    
_________________________________________________________________
lstm (LSTM)                  (None, 128)               74240     
_________________________________________________________________
dense_1 (Dense)              (None, 64)                8256      
_________________________________________________________________
dense (Dense)                (None, 3)                 195       
=================================================================
Total params: 242,691
Trainable params: 242,691
Non-trainable params: 0
_________________________________________________________________
```

The first layer is text vectorization, for converting text to a vector of integers. The output sequence length was set to 250, because that is a size that would encapsulate the majority of the paragraphs (the paragraphs are all <= 250 words in length, or are truncated). The next layer is an embedding layer, which was set at 16 dimensions to reduce the dimensionality of the text vectors to a reasonable size. Next is the actual LSTM layer, with a variable hidden state size. In this run the hidden state size was set at 128. Next is a dense layer, for dimensional reduction, using ReLU as the activation function. Finally, a dense layer is used to further reduce the dimensions into the three outputs, for the three authors of the books. The learning rate was set at 1e-3, the recommended one for this assignment. An Adam optimizer was used, since it was simple to implement and is relatively fast on deep learning models.

### CNN

The next model that was created was the CNN. The loss function used was binary cross-entropy, based on the instructions in the assignment, because the labels are binary (0 = negative, 1 = positive). The model has ten layers, which are shown below:

```txt
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
text_vectorization (TextVect (None, 250)               0         
_________________________________________________________________
embedding (Embedding)        (None, 250, 16)           160000    
_________________________________________________________________
zero_padding1d (ZeroPadding1 (None, 252, 16)           0         
_________________________________________________________________
conv1d (Conv1D)              (None, 250, 32)           1568      
_________________________________________________________________
max_pooling1d (MaxPooling1D) (None, 83, 32)            0         
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 82, 32)            2080      
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 41, 32)            0         
_________________________________________________________________
flatten (Flatten)            (None, 1312)              0         
_________________________________________________________________
dense_2 (Dense)              (None, 64)                84032     
_________________________________________________________________
dense (Dense)                (None, 3)                 195       
=================================================================
Total params: 247,875
Trainable params: 247,875
Non-trainable params: 0
_________________________________________________________________
```

This model has many similarities to the LSTM model. The first layer is text vectorization, and is the same as in the LSTM. The next layer is an embedding layer, which is also the same as before. Next there is a zero padding layer, which pad zeros to the beginning and end of each of the input data. Next there are two sets of two layers, a one-dimensional convolutional layer, and a one-dimensional max pooling layer. The filter size used on these two convolutional layers varied, but in this run they were both 32 in size. After the convolutional and max pooling is a dense layer, for dimensional reduction, using ReLU as the activation function. Finally, a dense layer is used to further reduce the dimensions into the three classes, for the authors of the paragraphs. The learning rate was set at 1e-3, the recommended one for this assignment. An Adam optimizer was used, since it was simple to implement and is relatively fast on deep learning models.

### Sentiment Analysis

In the sentiment analysis models, most of the layers are the same. They only needed to be slightly modified to have binary data input as opposed to the three-label classification. Binary cross-entropy was used for the loss because there are only two labels - 0 and 1. A sigmoid activation function was used for the output layer because that is what was recommended by the assignment.

## Plots

![Training and Validation Loss Books LSTM](./output/books_lstm.png)

![Training and Validation Loss Books CNN](./output/books_cnn.png)

![Training and Validation Loss Reviews LSTM](./output/reviews_lstm.png)

![Training and Validation Loss Reviews CNN](./output/reviews_cnn.png)

\newpage

## Model Statistics and Comparison

### Author Classification

|                         | precision | recall | f1-score | support |
|-------------------------|-----------|--------|----------|---------|
| SGD Logistic Regression | -         | -      | -        | -       |
| Jane Austen             | 0.97      | 0.94   | 0.95     | 496     |
| Fyodor Dostoyevsky      | 0.96      | 0.99   | 0.97     | 1186    |
| Arthur Conan Doyle      | 0.96      | 0.78   | 0.86     | 102     |
| Multilayer Perceptron   | -         | -      | -        | -       |
| Jane Austen             | 0.97      | 0.95   | 0.96     | 496     |
| Fyodor Dostoyevsky      | 0.96      | 0.99   | 0.97     | 1186    |
| Arthur Conan Doyle      | 0.90      | 0.77   | 0.83     | 102     |
| LSTM                    | -         | -      | -        | -       |
| Jane Austen             | 0.89      | 0.81   | 0.92     | 506     |
| Fyodor Dostoyevsky      | 0.80      | 0.99   | 0.89     | 1160    |
| Arthur Conan Doyle      | 0.81      | 0.86   | 0.91     | 114     |
| CNN                     | -         | -      | -        | -       |
| Jane Austen             | 0.98      | 0.90   | 0.94     | 504     |
| Fyodor Dostoyevsky      | 0.98      | 0.95   | 0.96     | 1162    |
| Arthur Conan Doyle      | 0.86      | 0.76   | 0.87     | 114     |

### Review Sentiment Analysis

|      | accuracy |
|------|----------|
| LSTM | 0.72     |
| CNN  | 0.85     |

### Analysis

After running the models in six different trial iterations, tweaking the hyper-parameters each time, I have arrived at a configuration that outputs moderately good results for the LSTM and CNN models. For the paragraph author classification, a hidden layer size of 128 for the LSTM seemed to work best, as going higher would result in over-fitting on the training data. Increasing the epoch size beyond 12 also caused over-fitting for the model, with the hidden layer sizes tested. For the CNN, a first filter size of 32 and second of 24 seemed to work best, trained over 14 epochs. The idea was to have a larger convolution followed by a smaller one to increase granularity, and this seemed to work well. In comparison to the Logistic Regression and MLP models, the Logistic Regression seemed to beat the LSTM and CNN in most categories, with the CNN outperforming the LSTM. This is surprising, because the more complicated models generally output better results. The discrepancy most likely stems from the limited training data. If a pre-trained embedding layer was used, it would most likely increase the LSTM and CNN metrics.

The review sentiment analysis also showed interesting results. I expected that the accuracy scores for the reviews would be similar to the book paragraphs, at around 80-90%, but instead it was a bit lower, at 0.72% and 0.85%. This is most likely because the writing styles of the authors of the reviews are somewhat different, and sentiment analysis is a completely different problem than paragraph classification. After testing several different combinations of epochs and hidden layer sizes, a hidden layer size of 256 trained over 12 epochs seemed to work best for the LSTM. Any larger and the model would be overfitted. In the CNN model, different combinations of filter sizes and epochs were tested, with a combination of 32 and 64 for the filters and 16 epochs for the training being used. The filter combination followed the same idea as the paragraph classification model.

In conclusion, this project was fairly successful. While the models did not perform as well as the logistic regression created for the original assignment, it was not that far off, and with the addition of a pre-trained embedding layer the performance should increase. The review sentiment models also performed well, but not as well as it was first postulated. Again, pre-training may help here as well.
