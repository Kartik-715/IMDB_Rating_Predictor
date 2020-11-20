# Import required libraries
from collections import Counter
from scipy import sparse
import numpy as np
from math import log
from tqdm import tqdm
from random import shuffle, uniform
import pickle
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize

# Builds a vocabulary of words mapped to word ID and word frequency in the corpus
def build_vocabulary(corpus):
  vocabulary = Counter()
  # Tokenize corpus into sentences
  sentences = sent_tokenize(corpus)
  for sentence in sentences:
    # Tokenize each sentence into words and update the vocabulary
    words = word_tokenize(sentence)
    vocabulary.update(words)

  # Return vocabulary and list of sentences
  return {word: (id, freq) for id, (word, freq) in enumerate(vocabulary.items())}, sentences


# Builds the word cooccurrence matrix X for the given vocabulary. The final
# result is a list containing tuples of (center_ID, context_ID, Xij)
def build_cooccurrences(vocabulary, sentences, contextWindow):
  # Let V be the size of the vocabulary
  vocabSize = len(vocabulary)
  result = []

  # Initialize a sparse matrix for cooccurrences of size V x V
  cooccurrences = sparse.lil_matrix((vocabSize, vocabSize), dtype=np.float64)

  # Iterate over all sentences in the corpus
  for i, sentence in enumerate(sentences):
    # Iterate over all words in the sentence and fetch their word IDs in a list
    words = word_tokenize(sentence)
    wordIds = [vocabulary[word][0] for word in words]

    for i, centerId in enumerate(wordIds):
      # Collect all context word IDs in left window of center word
      contextIds = wordIds[max(0, i - contextWindow) : i]
      contextLen = len(contextIds)

      # Iterate over each context word in left window
      for j, contextId in enumerate(contextIds):
        # Distance from center word
        distance = contextLen - j

        # Weight by inverse of distance between words
        increment = 1.0 / float(distance)

        # Update cooccurrences of context word and center word symmetrically to
        # handle both left window and right window cooccurrences
        cooccurrences[centerId, contextId] += increment
        cooccurrences[contextId, centerId] += increment

  # Build the output result from the sparse matrix
  for i, (row, data) in enumerate(zip(cooccurrences.rows, cooccurrences.data)):
    for data_idx, j in enumerate(row):
      result.append((i, j, data[data_idx]))

  return result


# Runs a single iteration of adaptive gradient descent (AdaGrad) while training
# the GloVe word embeddings. Takes input cooccurrence data, weight vectors,
# biases and gradient histories. Returns the cost associated with
# the given weights and updates the weights by AdaGrad
def run_iteration(vocabulary, data, learningRate, xMax, alpha):
  globalCost = 0

  # Shuffle the data to avoid biasing of the word vectors
  shuffle(data)

  for (vMain, vContext, bMain, bContext, gradWMain, gradWContext,
       gradBMain, gradBContext, cooccurrence) in data:

    weight = (cooccurrence / xMax) ** alpha if cooccurrence < xMax else 1

    # Compute inner component of cost function
    costInner = (vMain.dot(vContext) + bMain[0] + bContext[0] - log(cooccurrence))

    # Compute cost function
    cost = weight * (costInner ** 2)

    # Add weighted cost to the global cost
    globalCost += 0.5 * cost

    # Compute gradients for word vectors
    gradMain = weight * costInner * vContext
    gradContext = weight * costInner * vMain

    # Compute gradients for bias terms
    gradBiasMain = weight * costInner
    gradBiasContext = weight * costInner

    # Perform adaptive gradient descent
    vMain -= (learningRate * gradMain / np.sqrt(gradWMain))
    vContext -= (learningRate * gradContext / np.sqrt(gradWContext))

    bMain -= (learningRate * gradBiasMain / np.sqrt(gradBMain))
    bContext -= (learningRate * gradBiasContext / np.sqrt(gradBContext))

    # Update squared gradient sums
    gradWMain += np.square(gradMain)
    gradWContext += np.square(gradContext)
    gradBMain += gradBiasMain ** 2
    gradBContext += gradBiasContext ** 2

  return globalCost


# Train GloVe vectors given cooccurrences and vocabulary. Takes input other
# parameters such as dimSize, iterations, learningRate, xMax and alpha. Returns
# the computed word vector matrix W of size 2V * d
def train_glove(vocabulary, cooccurrences, dimSize, iterations, learningRate, xMax, alpha):

  vocabSize = len(vocabulary)

  # Word vector matrix of size 2V * d initialized by random values in range
  # (-0.5, 0.5]
  W = (np.random.rand(vocabSize * 2, dimSize) - 0.5) / float(dimSize + 1)

  # Bias terms associated with each single vector initialized by random values
  # in range (-0.5, 0.5]
  biases = (np.random.rand(vocabSize * 2) - 0.5) / float(dimSize + 1)

  # Sum of squares of all previous gradients for adaptive gradient descent
  # (AdaGrad) initialized to 1 so that initial adaptive learning rate is equal
  # to global learning rate
  gradient = np.ones((vocabSize * 2, dimSize), dtype=np.float64)

  # Sum of squared gradients for the bias terms
  gradientBiases = np.ones(vocabSize * 2, dtype=np.float64)

  data = [ (W[iMain], W[iContext + vocabSize],
            biases[iMain : iMain + 1],
            biases[iContext + vocabSize : iContext + vocabSize + 1],
            gradient[iMain], gradient[iContext + vocabSize],
            gradientBiases[iMain : iMain + 1],
            gradientBiases[iContext + vocabSize : iContext + vocabSize + 1],
            cooccurrence )
            for iMain, iContext, cooccurrence in cooccurrences]

  # Train the word vector matrix for specific number of iterations
  for i in tqdm(range(iterations)):
    cost = run_iteration(vocabulary, data, learningRate, xMax, alpha)
    print('Iteration', i, '- Cost:', cost)

  # Return the word vector matrix
  return W


def glove_embeddings(corpus, contextWindow, dimSize, iterations, learningRate, xMax, alpha):
  vocabulary, sentences = build_vocabulary(corpus)
  cooccurrences = build_cooccurrences(vocabulary, sentences, contextWindow)
  W = train_glove(vocabulary, cooccurrences, dimSize, iterations, learningRate, xMax, alpha)
  return W, vocabulary

def custom_glove_embeddings(corpus):
    CONTEXT_WINDOW = 10
    DIM_SIZE = 100
    ITERATIONS = 50
    LEARNING_RATE = 0.05
    X_MAX = 100
    ALPHA = 0.75
    SMALL_RATIO = 0.15

    W, vocabulary = glove_embeddings(corpus, contextWindow=CONTEXT_WINDOW, dimSize=DIM_SIZE,
                                iterations=ITERATIONS, learningRate=LEARNING_RATE,
                                xMax=X_MAX, alpha=ALPHA)

    # Final GloVe word embedding is obtained by adding the center word embedding
    # and context word embedding for each word
    print(W.shape)
    embeddings = {}
    vocabSize = len(vocabulary)
    for word, (id, _) in vocabulary.items():
        embeddings[word] = W[id]+W[id+vocabSize]

    return embeddings, vocabulary
