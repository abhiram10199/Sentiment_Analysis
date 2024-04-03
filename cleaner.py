import string
try:
    from nltk.corpus import stopwords
except ImportError:
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords


# Reading the reviews/file
def load_doc(filename) -> str:
    with open(filename, 'r') as file:
        text = file.read()
    return text


# Cleaning the reviews
def clean_doc(doc) -> list[str]:
    tokens = doc.split()
    table = str.maketrans('', '', string.punctuation)
    tokens: list[str] = [w.translate(table) for w in tokens]
    tokens: list[str] = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens: list[str] = [w for w in tokens if not w in stop_words]
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