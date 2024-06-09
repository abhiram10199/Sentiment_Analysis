from tokeniser import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from pandas import DataFrame
from matplotlib import pyplot


print("Running model.py")

# Getting the data
vocab = getter()    
training_matrix, ytrain, train_docs, testing_matrix, ytest, test_docs = getter2(vocab)
no_of_words = training_matrix.shape[1]


# Preperaring the data
def prepare_data(train_docs, test_docs, mode) -> tuple:
    tokeniser = Tokenizer()
    tokeniser.fit_on_texts(train_docs)
    training_matrix = tokeniser.texts_to_matrix(train_docs, mode=mode)
    testing_matrix = tokeniser.texts_to_matrix(test_docs, mode=mode)
    return training_matrix, testing_matrix


# Evaluate the model
def evaluate_model(training_matrix, ytrain, testing_matrix, ytest) -> list:
    scores = []
    no_of_words = training_matrix.shape[1]
    for i in range(30):
        model = Sequential()
        model.add(Dense(50, input_shape=(no_of_words,), activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(training_matrix, ytrain, epochs=50, verbose=0)
        loss, acc = model.evaluate(testing_matrix, ytest, verbose=0)
        scores.append(acc)
        print('%d accuracy: %s' % ((i+1), acc))
        return scores


# Evaluates a sentiment analysis model using different tokenization modes
def model(train_docs, test_docs) -> None:
    modes = ['binary', 'count', 'tfidf', 'freq']
    results = DataFrame()
    for mode in modes:
        training_matrix, testing_matrix = prepare_data(train_docs, test_docs, mode)
        results[mode] = evaluate_model(training_matrix, ytrain, testing_matrix, ytest)
    print(results.describe())
    results.boxplot()
    pyplot.show()
'''
“binary” Where words are marked as present (1) or absent (0).
“count” Where the occurrence count for each word is marked as an integer.
“tfidf” Where each word is scored based on their frequency, where words that are common across all documents are penalized.
“freq” Where words are scored based on their frequency of occurrence within the document.
'''

# Predict the sentiment of a review
def predict_sentiment(review, vocab, tokenizer, model) -> str:
    tokens = clean_doc(review)
    tokens = [w for w in tokens if w in vocab]
    line = ' '.join(tokens)
    encode = tokenizer.texts_to_matrix([line], mode='binary')
    yhat = model.predict(encode, verbose=0)
    return round(yhat[0,0])


# MAIN FUNCTION
def main(train_docs, test_docs) -> None:
    test_review = "love this movie"
    tokenizer = Tokenizer()
    print(predict_sentiment(test_review, vocab, tokenizer, model))



if __name__ == '__main__':
    main(train_docs, test_docs)