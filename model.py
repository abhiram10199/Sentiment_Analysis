from tokeniser import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


# Getting the data
vocab = getter()
training_matrix, ytrain, testing_matrix, ytest = getter2(vocab)
no_of_words = training_matrix.shape[1]


def model() -> None:
    # define network
    model = Sequential()
    model.add(Dense(50, input_shape=(no_of_words,), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile network
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit network
    model.fit(training_matrix, ytrain, epochs=50, verbose=2)

    # evaluate
    loss, acc = model.evaluate(testing_matrix, ytest, verbose=0)
    print('Test Accuracy: %f' % (acc*100))



def main():
    model()


if __name__ == '__main__':
    main()

