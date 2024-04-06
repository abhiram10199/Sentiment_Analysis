from vocab import *
from numpy import array
from tensorflow.keras.preprocessing.text import Tokenizer  #type: ignore


# Converts a document to a line of tokens and fits the tokeniser on the text
def fit_tokeniser(tokeniser, docs) -> None:
    tokeniser.fit_on_texts(docs)


# Returns a list of tokens from the training data
def train_list_of_docs(vocab) -> list[str]:
    positive_lines = process_docs('txt_sentoken/pos', vocab, True)
    negative_lines = process_docs('txt_sentoken/neg', vocab, True)
    docs = positive_lines + negative_lines
    #print("Docs =", len(docs))
    return docs


# Returns a list of tokens from the testing data
def test_list_of_docs(vocab) -> list[str]:
    positive_lines = process_docs('txt_sentoken/pos', vocab, False)
    negative_lines = process_docs('txt_sentoken/neg', vocab, False)
    docs = positive_lines + negative_lines
    #print("Docs =", len(docs))
    return docs


# Encodes the training data using the tokeniser & their frequencies
def training_data_matrix(tokeniser, vocab) -> list:
    docs = train_list_of_docs(vocab)
    fit_tokeniser(tokeniser, docs)
    training_matrix = tokeniser.texts_to_matrix(docs, mode='freq')
    print("Training Matrix =", training_matrix.shape)
    ytrain = array([0 for _ in range(900)] + [1 for _ in range(900)])
    return training_matrix, ytrain


# Encodes the testing data using the tokeniser & their frequencies
def testing_data_matrix(tokeniser, vocab) -> None:   
    docs = test_list_of_docs(vocab)
    fit_tokeniser(tokeniser, docs)
    testing_matrix = tokeniser.texts_to_matrix(docs, mode='freq')
    print("Testing Matrix =", testing_matrix.shape)
    ytest = array([0 for _ in range(100)] + [1 for _ in range(100)])
    return testing_matrix, ytest


# return the matrices so i can get it in model.py
def getter2(vocab) -> tuple:
    tokeniser = Tokenizer()
    
    training_matrix, ytrain = training_data_matrix(tokeniser, vocab)
    testing_matrix, ytest = testing_data_matrix(tokeniser, vocab)

    return training_matrix, ytrain, testing_matrix, ytest


def main():
    tokeniser = Tokenizer()
    vocab = getter()

    # Fit on texts
    fit_tokeniser(tokeniser, vocab)

    # Get the matrices
    training_matrix, ytrain = training_data_matrix(tokeniser, vocab)
    testing_matrix, ytest = testing_data_matrix(tokeniser, vocab)


if __name__ == '__main__':
    main()