import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import time

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore')

from main import read_raw_data, read_clean_data

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

def generate_text(model, char2idx, idx2char, start_string):
    num_generate = 1000
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    temperature = 1.0
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])

    return start_string + ''.join(text_generated)

def main():
    reviews, rating = read_clean_data()
    text = ' '.join(reviews)
    vocab = sorted(set(text))
    print("Unique Characters: ", len(vocab))

    # Creating a mapping from unique characters to indices
    char2idx = {u:i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)
    text_as_int = np.array([char2idx[c] for c in text])

    seq_length = 100
    examples_per_epoch = len(text)//(seq_length+1)
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

    sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
    dataset = sequences.map(split_input_target)

    BATCH_SIZE = 1
    BUFFER_SIZE = 10000
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    print(dataset)

    vocab_size = len(vocab)
    embedding_dim = 256
    rnn_units = 1024

    model = build_model(vocab_size=len(vocab), embedding_dim=embedding_dim, rnn_units=rnn_units,batch_size=BATCH_SIZE)

    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
        sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
        sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
        print(sampled_indices)
        print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
        print()
        print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices ])))

        example_batch_loss = loss(target_example_batch, example_batch_predictions)
        print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
        print("scalar_loss:      ", example_batch_loss.numpy().mean())

    model.summary()
    model.compile(optimizer='adam', loss=loss)

    EPOCHS = 2
    history = model.fit(dataset, epochs=EPOCHS)

    input_rating = []
    output_reviews = []

    while True:
        print('Input rating:')
        rating = int(input())
        if rating == -1:
            break

        input_rating.append(rating)

        print('Input movie name:')
        movie_name = input()

        output = generate_text(model,char2idx, idx2char, start_string=movie_name )
        output_reviews.append(output)
        print(output)

if __name__ == '__main__':
    main()
