'''
vocab.py is a script that creates a vocabulary from the movie review dataset. 
It uses the cleaner.py script to clean the text and remove punctuation. 
The vocabulary is saved to a file called vocab.txt. 
The vocabulary is created by processing all the documents in the dataset 
and counting the frequency of each word. The words that appear at least 
twice are added to the vocabulary. The vocabulary is then loaded from 
the file and converted to a set.
'''


from cleaner import *
from string import punctuation
from os import listdir
from collections import Counter


# Adds the words from a document to the vocabulary
def add_doc_to_vocab(filename, vocab) -> None:
    doc = load_doc(filename)
    tokens = clean_doc(doc)
    vocab.update(tokens)


# Processes the documents in a directory
def process_docs(directory, vocab, is_train) -> list[str]:
    lines = []
    for filename in listdir(directory):
        if is_train and filename.startswith('cv9'):
            continue
        if not is_train and not filename.startswith('cv9'):
            continue
        path = directory + '/' + filename
        '''add_doc_to_vocab(path, vocab)     !!!Remove for new vocab creation!!!'''
        line = doc_to_line(path, vocab)
        lines.append(line)
    return lines


# Saves the vocabulary to a file
def save_list(lines, filename) -> None:
    tokens = '\n'.join(lines)
    with open(filename, 'w') as file:
        file.write(tokens)


# Creates a new vocabulary with the words that appear at least twice
def new_vocab() -> list[str]:
    vocab = Counter()
    process_docs('txt_sentoken/pos', vocab)
    process_docs('txt_sentoken/neg', vocab)
    # k = word, c = frequency
    return [k for k, c in vocab.items() if c >= 2] 


# Loads the vocabulary from the vocab file and makes a set
def vocab_set() -> None:
    vocab = load_doc('vocab.txt')
    vocab = set(vocab.split())
    return vocab


# Get the vocab to other files
def getter() -> tuple:
    vocab = vocab_set()
    #docs = list_of_docs(vocab)
    return vocab


def main():
    # Create a new vocabulary
    vocab = vocab_set()
    

if __name__ == '__main__':
    main()
