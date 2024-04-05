'''
cleaner.py is a script that defines functions for loading and cleaning text data.
'''

import string
try:
    from nltk.corpus import stopwords
except ImportError:
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords


# Reads all the text in a file and returns it
def load_doc(filename) -> str:
    with open(filename, 'r') as file:
        text = file.read()
    return text


# Splits the text into words and removes punctuation and stopwords
def clean_doc(doc) -> list[str]:
    tokens = doc.split()
    
    # Remove punctuation
    table = str.maketrans('', '', string.punctuation)
    tokens: list[str] = [w.translate(table) for w in tokens]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens: list[str] = [word for word in tokens 
                         if word.isalpha() 
                         and word not in stop_words 
                         and len(word) > 1]
    
    return [word for word in tokens if len(word) > 1]


# Converting the reviews to lines
def doc_to_line(filename, vocab) -> str:
    doc = load_doc(filename)
    tokens = clean_doc(doc)
    tokens = [w for w in tokens if w in vocab]
    return ' '.join(tokens)


def main():
    text: str = load_doc('txt_sentoken\pos\cv100_11528.txt')
    print(clean_doc(text))


if __name__ == '__main__':
    main()