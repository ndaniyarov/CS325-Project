# RNN (LSTM)

from matplotlib import pyplot
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from read_file import *
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

(x_train, y_train) = read_csv()

max_words = 20
# one hot vector
words = set(w for sent in x_train for w in sent.split())
word_map = {w: i+1 for (i, w) in enumerate(words)}
# -Changed the below line the inner for loop sent to sent.split()
sent_ints = [[word_map[w] for w in sent.split()] for sent in x_train]
# print(sent_ints)
vocab_size = len(words)
# -changed the below line - the outer for loop sentences to sent_ints
sequences = pad_sequences(sent_ints)
vectorized = to_categorical(sequences)
x_train = np.array(vectorized)

# define base model


def baseline_model():
    # create model
    # input_dim=len(x_train),
    model = Sequential()
    model.add(LSTM(128, dropout=0.2))
    model.add(Embedding(500, 128))
    model.add(Dense(13, input_dim=vocab_size+1,
              kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam',
                  metrics=['mse', 'mae'])
    return model


# evaluate model
estimator = KerasRegressor(build_fn=baseline_model,
                           epochs=50, batch_size=5, verbose=0)

history = estimator.fit(x_train, y_train)
pyplot.style.use('seaborn')
pyplot.plot(history.history['mae'])
print(history.history['mae'])
pyplot.title('RNN (LSTM): Mean Absolute Error')
pyplot.xlabel('Epoch')
pyplot.ylabel('MAE')
pyplot.savefig('mae-rnn-embeddings.png')

"""
# take 1/5 th of data
small_x_train = x_train[::5]
small_y_train = y_train[::5]

kfold = KFold(n_splits=10)

results = cross_val_score(estimator, small_x_train, small_y_train, cv=kfold)
print(results)
print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))
"""

