# Assignment 2 - Word Vectors

*I pledge my honor that I have abided by the Stevens Honor System.* - Joshua Schmidt

Name: Joshua Schmidt, Date: 10/5/2020

## Written

a. **Softmax (5pts)** Prove that softmax is invariant to constant offset in the input, that is, for any input vector $x$ and any constant $c$,\
$softmax(x) = softmax(x + c)$\
where $x+c$ means adding the constant $c$ to every dimension of $x$.\
*Note: In practice, we make use of this property and choose $c = −max_{i}x_{i}$ when computing softmaxprobabilities for numerical stability (i.e., subtracting its maximum element from all elements of x).*

b. **Sigmoid (5pts)** Derive the gradients of the sigmoid function and show that it can be rewritten as a function of the function value (i.e., in some expression where only $\sigma (x)$, but not $x$, is present). Assume that the input $x$ is a scalar for this question.\
\
![1a, 1b](proofs/1a_b.jpg)
\newpage
c. **Word2vec**
  1. Assume you are given a predicted word vector $v_{c}$ corresponding to the center word $c$ for skipgram, and the word prediction is made with the $softmax$ function where $o$ is the expected word, $w$ denotes the $w$-th word and $u_{w}(w = 1, ..., W)$ are the "output" word vectors for all words in the vocabulary. Assume cross entropy cost is applied to this prediction, derive the gradients with respect to $v_{c}$.

  2. As in the previous part, derive gradients for the "output" word vector $u_{w}$ (including $u_{o}$).\
  \
  ![1c_1, 1c_2](proofs/1c_12.jpg)
  \newpage
  3. Repeat a and b assuming we are using the negative sampling loss for the predicted vector $v_{c}$. Assume that $K$ negative samples (words) are drawn and they are $1,...,K$ respectively. For simplicity of notation, assume $(o \notin {1,...,K})$. Again for a given word $o$, use $u_{o}$ to denote its output vector.

  4. Derive gradients for all of the word vectors for skip-gram given the previous parts and given a set of context words $[word_{c − m},...,word_{c},...,word_{c + m}]$ where $m$ is the context size. Denote the "input" and "output" word vectors for word $k$ as $v_{k}$ and $u_{k}$ respectively. *Hint: feel free to use $F(o,v_{c})$ (where $o$ is the expected word) as a placeholder for the $J_{CE}(o,v_{c}...)$ or $J_{neg-sample}(o,v_{c}...)$ cost functions in this part – you’ll see that this is a useful abstraction for the coding part.*\
  \
  ![1c_3, 1c_4](proofs/1c_34.jpg)
  \newpage

## Results and Analysis

After the program finished training, the following diagram was created:

![Word Vector Visualization](output/word_vectors.png)
\newpage

It shows the relative similarities between words, based on the trained vector representations of the words, and illustrated by the distance between words. Most of the relationships between the words make sense. "enjoyable" is close to "annoying", and "sweet". "sweet" is near "tea", and "coffee" is near "tea". All of these words are related, as when people discuss tea it is either sweet or bitter, and it is an alternative to coffee. Another cluster that makes sense is "woman", "female", and "man", which are all close together, and are also close to adjectives ("brilliant", "cool", "bad"). These adjectives could be used to describe the given nouns.

There are some strange artifacts in the diagram, which may exist due to the limited training data provided. "hail" is on the opposite side of the diagram as "king", when they should be close together. "queen" is relatively far away from "king", when they would logically be close together. "male" is also relatively far away from "female", "woman" and "man" cluster. Other than those few misplaced words, the diagram does show a good similarity between different random words in the corpus.

Below is the list of the first 10 knn outputs tested:

```log
1. running knn for "great"
closest indices: [10641 10515  7522  8586]
closest words: ['kenneth', 'toast', 'juan', 'ringside']
2. running knn for "cool"
closest indices: [ 7387  1622 11409 12521]
closest words: ['rollicking', 'schmaltzy', 'fast-edit', 'hospitals']
3. running knn for "brilliant"
closest indices: [12465  1369  6602 13340]
closest words: ['undermines', 'stephen', 'liberating', 'ugly-duckling']
4. running knn for "wonderful"
closest indices: [ 8062  5045  1401 13618]
closest words: ['inclination', 'artifice', 'certain', 'repulsive']
5. running knn for "well"
closest indices: [13943 14406  8893 10455]
closest words: ['labored', 'countenance', 'hades', 'available']
6. running knn for "amazing"
closest indices: [1999 2929  626 1158]
closest words: ['trouble', 'challenging', 'madness', 'him']
7. running knn for "worth"
closest indices: [ 5926  7886  7638 15321]
closest words: ['creatively', 'fascinate', 'painting', 'dilbert']
8. running knn for "sweet"
closest indices: [ 7634 15451   299  5965]
closest words: ['self-mutilation', 'improvisations', 'helps', 'wrap']
9. running knn for "enjoyable"
closest indices: [11261  1369 16461  9428]
closest words: ['playfully', 'stephen', 'spit', 'aimlessness']
10. running knn for "boring"
closest indices: [  117 15050  8942 17147]
closest words: ['road', 'xerox', 'alexandre', 'reports']
```

Some of the knn outputs make sense, while others do not. For example, there are some names in the corpus, such as "kenneth" and "stephen", who are apparently "great" and "brilliant". I suspect that the corpus had a sentence or two that called people named "kenneth" and "stephen" great and brilliant, and the names did not show up again. Ignoring the names, most of the output makes sense. "xerox", "reports", and "road"s are indeed "boring", "brilliant" is similar to "liberating", and "rollicking" is synonymous with "cool".

Because the algorithms used passed the test cases and are theoretically correct, it can be safe to assume that the word2vec algorithm is implemented correctly. The small issues seen here can be fixed through increasing the amount and quality of data provided and adding better data cleaning (to remove names or other words that do not belong).

Below is the output for running `word2vec`:

```logs
==== Gradient check for skip-gram with naiveSoftmaxLossAndGradient ====
Gradient check passed!
==== Gradient check for skip-gram with negSamplingLossAndGradient ====
Gradient check passed!

=== Results ===
Skip-Gram with naiveSoftmaxLossAndGradient
Your Result:
Loss: 11.16610900153398
Gradient wrt Center Vectors (dJ/dV):
 [[ 0.          0.          0.        ]
 [ 0.          0.          0.        ]
 [-1.26947339 -1.36873189  2.45158957]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]]
Gradient wrt Outside Vectors (dJ/dU):
 [[-0.41045956  0.18834851  1.43272264]
 [ 0.38202831 -0.17530219 -1.33348241]
 [ 0.07009355 -0.03216399 -0.24466386]
 [ 0.09472154 -0.04346509 -0.33062865]
 [-0.13638384  0.06258276  0.47605228]]

Expected Result: Value should approximate these:
Loss: 11.16610900153398
Gradient wrt Center Vectors (dJ/dV):
 [[ 0.          0.          0.        ]
 [ 0.          0.          0.        ]
 [-1.26947339 -1.36873189  2.45158957]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]]
Gradient wrt Outside Vectors (dJ/dU):
 [[-0.41045956  0.18834851  1.43272264]
 [ 0.38202831 -0.17530219 -1.33348241]
 [ 0.07009355 -0.03216399 -0.24466386]
 [ 0.09472154 -0.04346509 -0.33062865]
 [-0.13638384  0.06258276  0.47605228]]
    
Skip-Gram with negSamplingLossAndGradient
Your Result:
Loss: 16.15119285363322
Gradient wrt Center Vectors (dJ/dV):
 [[ 0.          0.          0.        ]
 [ 0.          0.          0.        ]
 [-4.54650789 -1.85942252  0.76397441]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]]
 Gradient wrt Outside Vectors (dJ/dU):
 [[-0.69148188  0.31730185  2.41364029]
 [-0.22716495  0.10423969  0.79292674]
 [-0.45528438  0.20891737  1.58918512]
 [-0.31602611  0.14501561  1.10309954]
 [-0.80620296  0.36994417  2.81407799]]

Expected Result: Value should approximate these:
Loss: 16.15119285363322
Gradient wrt Center Vectors (dJ/dV):
 [[ 0.          0.          0.        ]
 [ 0.          0.          0.        ]
 [-4.54650789 -1.85942252  0.76397441]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]]
 Gradient wrt Outside Vectors (dJ/dU):
 [[-0.69148188  0.31730185  2.41364029]
 [-0.22716495  0.10423969  0.79292674]
 [-0.45528438  0.20891737  1.58918512]
 [-0.31602611  0.14501561  1.10309954]
 [-0.80620296  0.36994417  2.81407799]]
```
