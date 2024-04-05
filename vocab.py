'''
vocab.py is a script that creates a vocabulary from the movie review dataset. 
It uses the cleaner.py script to clean the text and remove punctuation. 
The vocabulary is saved to a file called vocab.txt. 
The vocabulary is created by processing all the documents in the dataset 
and counting the frequency of each word. The words that appear at least 
twice are added to the vocabulary. The vocabulary is then loaded from the 
file and converted to a set for fast membership testing.
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
def process_docs(directory, vocab) -> list[str]:
    lines = []
    for filename in listdir(directory):
        if not filename.endswith('.txt') or filename.startswith('cv9'):
            continue
        path = directory + '/' + filename
        #add_doc_to_vocab(path, vocab)
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
    tokens = [k for k, c in vocab.items() if c >= 2]
    return tokens


# Loads the vocabulary from a file and makes a set
def vocab_set() -> None:
    vocab_filename = 'vocab.txt'
    vocab = load_doc(vocab_filename)
    vocab = vocab.split()
    vocab = set(vocab)
    return vocab


def main():
    vocab = vocab_set()
    positive_lines = process_docs('txt_sentoken/pos', vocab)
    negative_lines = process_docs('txt_sentoken/neg', vocab)
    print(len(positive_lines), len(negative_lines))

if __name__ == '__main__':
    main()