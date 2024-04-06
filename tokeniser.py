from vocab import *
from tensorflow.keras.preprocessing.text import Tokenizer


# Converts a document to a line of tokens and fits the tokeniser on the text
def fit_tokeniser(tokeniser, docs) -> None:
    tokeniser.fit_on_texts(docs)


# Encodes the documents using the tokeniser & their frequencies
def encode_docs(tokeniser, docs) -> list:
    matrix = tokeniser.texts_to_matrix(docs, mode='freq')
    print("Matrix Shape =", matrix.shape)
    return matrix


'''
# Testing the data matrix
def testing_data_matrix() -> None:
    positive_lines = process_docs('txt_sentoken/pos', vocab, False)
    negative_lines = process_docs('txt_sentoken/neg', vocab, False)
    docs = negative_lines + positive_lines
    # encode training data set
    matrix = tokenizer.texts_to_matrix(docs, mode='freq')
    print(matrix.shape)
'''

def main():
    tokeniser = Tokenizer()
    vocab, docs = get_docs_vocab()

    # Fit on texts
    fit_tokeniser(tokeniser, docs)

    # Encode the documents
    encoded_docs = encode_docs(tokeniser, docs)