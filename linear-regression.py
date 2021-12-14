# linear regression
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from read_file import *
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.sequence import pad_sequences

(x_train, y_train) = read_csv()
# enc = OneHotEncoder()
# enc.fit(x_train)

max_words = 20
# one hot vector
words = set(w for sent in x_train for w in sent.split())
word_map = {w: i+1 for (i, w) in enumerate(words)}
# -Changed the below line the inner for loop sent to sent.split()
sent_ints = [[word_map[w] for w in sent.split()] for sent in x_train]
# print(sent_ints)
vocab_size = len(words)
print(vocab_size)
# -changed the below line - the outer for loop sentences to sent_ints
sequences = pad_sequences(sent_ints)
vectorized = to_categorical(sequences)
x_train = np.array(vectorized)
# x_train = np.array([to_categorical(pad_sequences((sent,), max_words),
#                                  vocab_size+1) for sent in sent_ints])

# x_train = [to_categorical(pad_sequences((sent,), max_words),
#       vocab_size+1) for sent in sent_ints]
# define base model


def baseline_model():
    # create model
    model = Sequential()
    # model.add(Dense(13, input_dim=len(x_train),
    #         kernel_initializer='normal', activation='relu'))
    model.add(Dense(13, input_dim=vocab_size+1,
              kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# evaluate model
estimator = KerasRegressor(build_fn=baseline_model,
                           epochs=100, batch_size=5, verbose=0)
#kfold = KFold(n_splits=10)
history = estimator.fit(x_train, y_train)
# print(history['mean_absolute_error'])
# print(x_train.shape)  # (501, 1, 20, 3846)
# print(y_train.shape)  # (501,)
# print(x_train)
# print(y_train)
"""
x_train = []
for x in range(0, 501):
    x_train.append([1, 0])
x_train = np.array(x_train)
print(x_train)

model = baseline_model()
history = model.fit(
    x_train,
    y_train,
    epochs=100,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split=0.2)
"""
# results = cross_val_score(estimator, x_train, y_train, cv=kfold)
# print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))
