# RNN (LSTM)

# linear regression
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from read_file import *
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from keras.preprocessing.sequence import pad_sequences

(x_train, y_train) = read_csv()

max_words = 20
# one hot vector
words = set(w for sent in x_train for w in sent.split())
word_map = {w: i+1 for (i, w) in enumerate(words)}
# -Changed the below line the inner for loop sent to sent.split()
sent_ints = [[word_map[w] for w in sent.split()] for sent in x_train]
vocab_size = len(words)
print(vocab_size)
# -changed the below line - the outer for loop sentences to sent_ints
x_train = np.array([to_categorical(pad_sequences((sent,), max_words),
                                   vocab_size+1) for sent in sent_ints])
# define base model


def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(13, input_dim=len(x_train),
                    kernel_initializer='normal', activation='relu'))
    model.add(Embedding(500, 128))
    model.add(LSTM(128, dropout=0.2))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# evaluate model
estimator = KerasRegressor(build_fn=baseline_model,
                           epochs=100, batch_size=5, verbose=0)
kfold = KFold(n_splits=10)

print(x_train.shape)  # (501, 1, 20, 3846)
print(y_train.shape)  # (501,)

results = cross_val_score(estimator, x_train, y_train, cv=kfold)
# print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))
