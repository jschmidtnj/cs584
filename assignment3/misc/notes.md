# notes

- https://github.com/maxbane/simplegoodturing
- https://github.com/smilli/kneser-ney
- https://stackabuse.com/python-for-nlp-developing-an-automatic-text-filler-using-n-grams/
- more slides: https://www.cs.cornell.edu/courses/cs4740/2014sp/lectures/smoothing+backoff.pdf
- `./src/main.py 2>&1 | grep "[UNK]" | wc -l`
- `./src/main.py > output/logs.txt 2>&1`
- kneser-ney: https://medium.com/@dennyc/a-simple-numerical-example-for-kneser-ney-smoothing-nlp-4600addf38b8
- following this for perplexity:
- http://qpleple.com/perplexity-to-evaluate-topic-models/
- https://stackoverflow.com/questions/54941966/how-can-i-calculate-perplexity-using-nltk/55043954

## rnn

- https://github.com/seyedsaeidmasoumzadeh/Predict-next-word
- https://stackoverflow.com/questions/46125018/use-lstm-tutorial-code-to-predict-next-word-in-a-sentence
- https://towardsdatascience.com/building-a-next-word-predictor-in-tensorflow-e7e681d4f03f
- https://www.tensorflow.org/tutorials/text/text_generation
- https://medium.com/@curiousily/making-a-predictive-keyboard-using-recurrent-neural-networks-tensorflow-for-hackers-part-v-3f238d824218
- https://github.com/priya-dwivedi/Deep-Learning/blob/master/RNN_text_generation/RNN_project.ipynb
- proof: https://towardsdatascience.com/perplexity-intuition-and-derivation-105dd481c8f3

- text vectorization: https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/TextVectorization
- model: https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/text/text_generation.ipynb#scrollTo=JIPcXllKjkdr
- save and load: https://www.tensorflow.org/tutorials/keras/save_and_load
- adam optimizer: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
- sequence loss: https://www.tensorflow.org/addons/api_docs/python/tfa/seq2seq/sequence_loss
