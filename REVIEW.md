# IMDb Sentiment Analysis Using Bag Of Words Model

**Goal:** Classify text as positive or negative using a bag of words (BoW) model for movie review sentiment classification from IMDb. 

**Intro:** I decided to follow a [tutorial](https://machinelearningmastery.com/deep-learning-bag-of-words-model-sentiment-analysis/), by Jason Brownlee, to build a BoW sentiment analysis model as a way to practically apply a bit of COMP111, Intro to AI, and to understand how the bits of theory I learnt fit in the bigger picture.<br>
While the process was insightful, the results weren't what I expected. Despite using a vocabulary of around 25,767 words, the model consistently misclassified reviews, often labeling negative reviews as positive.

### Data and Preprocessing

**Dataset:** The Movie Review Dataset is a collection of 1000 positive & negative reviews drawn from an archive of the rec.arts.movies.reviews newsgroup hosted at [imdb.com](https://reviews.imdb.com/Reviews), by Bo Pang and Lillian Lee.<br>The data has been cleaned up somewhat, for example:
-   The dataset is comprised of only English reviews.
-   All text has been conerted to lowercase.
-   There is white space around punctuation like periods, commas, and brackets.
-   Text has been split into one sentence per line.

**Preprocessing:** Seperated the reviews into training and testing data, 1800 and 200 reviews respectively. 90%-10% split for both positive and negative. The file names were also numbered 000-999 making it easy to classify 000-899 as training data, 900 onwards testing data.<br>
In cleaner.py the functions for extracting the data using basic python file-handling and cleaning based on the rules below exist:
- The tokens are split based on whitespace.
- Remove all punctuation.
- Remove all words that are not purely consisted of alphabetical characters.
- Remove all known stop words. Used stopwords from Natural Language Toolkit -> `from nltk.corpus import stopwords`.
- Remove all words less than 2 characters ('a', 'I', etc).<br>

```
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
```

**Vocabulary:** To create a foundation for the sentiment analysis model, a vocabulary of unique words was constructed, in vocab.py. This involved creating a dictionary mapping of the words to their frequency and removing any words than occus less than 3 times. This was done to constrain the words that might be predictive. This resulted in a vocabulary of approximately 25,767 words, which served as the basis for representing text as numerical features. The words are then stored in a vocab.txt file.

## Model Architecture
This section explores how to convert reviews into numerical representations suitable for training a Multilayer Perceptron (MLP) model for sentiment analysis. The process involves transforming text reviews into "bag-of-words" vectors. 

**Converting reviews to lines of tokens.** 
Before reviews are converted to vectors, they are cleaned, removing words not included in the vocab.txt file, and then making the leftover words into a single string/line. Used previous functions in cleaner.py as helper functions. Also load the vocabulary and turn it into a set for use in cleaning reviews.

**Encoding reviews with a bag-of-words model representation.**
The Tokenize() class provided in the Keras API is used to convert reviews to encoded document vectors.
`from  tensorflow.keras.preprocessing.text import Tokenizer`
After the tokeniser is created and fit onto the text documents, the documents are encoded using `texts_to_matrix(List[doc], mode)` function. Repeated for the testing data.
```
# Encodes the training data using the tokeniser & their frequencies
def training_data_matrix(tokeniser, vocab) -> list:
    train_docs = train_list_of_docs(vocab)
    fit_tokeniser(tokeniser, train_docs)
    training_matrix = tokeniser.texts_to_matrix(train_docs, mode='freq')
    #print("Training Matrix =", training_matrix.shape)
    ytrain = array([0 for _ in range(900)] + [1 for _ in range(900)])
    return training_matrix, ytrain, train_docs

# Encodes the testing data using the tokeniser & their frequencies
def testing_data_matrix(tokeniser, vocab) -> None:   
    test_docs = test_list_of_docs(vocab)
    fit_tokeniser(tokeniser, test_docs)
    testing_matrix = tokeniser.texts_to_matrix(test_docs, mode='freq')
    #print("Testing Matrix =", testing_matrix.shape)
    ytest = array([0 for _ in range(100)] + [1 for _ in range(100)])
    return testing_matrix, ytest, test_docs
```

### Sentiment Analysis Model

A Multilayer Perceptron (MLP) models to classify encoded documents as either positive or negative is used. The models will be simple feedforward network models with fully connected layers called  _Dense_  in the Keras deep learning library.

#### First Sentiment Analysis Model

Built a simple MLP model to predict the sentiment of encoded reviews.

-   **Input Layer:** The number of neurons in the input layer is equal to the number of words in the vocabulary (length of the input documents). This value is stored in `n_words`.
    
-   **Labels:** We need class labels for both training and testing data. Since we loaded and encoded reviews deterministically (negative then positive), we can directly set the labels.
    
-   **Network Definition:**
    -   A single hidden layer with 50 neurons and a rectified linear activation function (ReLU) [Neural Network easier to optimise if linear?].
    -   An output layer with a single neuron and a sigmoid activation function to predict 0 for negative and 1 for positive reviews.

-   **Training:**    
    -   The network is trained using the efficient Adam optimizer and the binary cross-entropy loss function, suited for binary classification problems.
    -   We monitor accuracy during training and evaluation.

-   **Evaluation:**  The model's performance is evaluated by making predictions on the test dataset and printing the accuracy.

#### Comparing Word Scoring Methods

The `texts_to_matrix()` function from Keras Tokenizer offers four different methods for scoring words:

-   **binary:** Words are marked as present (1) or absent (0).
-   **count:** The occurrence count for each word is marked as an integer.
-   **tfidf:** Where each word is scored based on their frequency, where words that are common across all documents are penalized.
-   **freq:** Scores words based on their frequency of occurrence within the document.

We can evaluate the model performance with each scoring mode:

1.  **Function `prepare_data()`**: Creates an encoding of loaded documents based on a chosen scoring model.

2.  **Function `evaluate_mode()`**:
    -   Takes encoded documents.
    -   Trains the MLP model 10 times on the train set and estimates accuracy on the test set.
    -   Returns a list of accuracy scores across all runs.
```
# Evaluate the model
def evaluate_model(training_matrix, ytrain, testing_matrix, ytest) -> list:
    scores = []
    no_of_words = training_matrix.shape[1]
    n = 10                   # Number of times to run the model
    for i in range(n):
        model = Sequential()
        model.add(Dense(50, input_shape=(no_of_words,), activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(training_matrix, ytrain, epochs=50, verbose=0)
        loss, acc = model.evaluate(testing_matrix, ytest, verbose=0)
        scores.append(acc)
        # print('%d accuracy: %s' % ((i+1), acc))
    return model

```

#### **Evaluation Process**:
- Prepare encoded documents for each scoring mode using `prepare_data()`.
 -   Evaluate the model on each encoded data using `evaluate_mode()`.
 -   Summarize and visualize the results.

### Model Evaluation:
The mean score of both "freq" and "binary" methods appear to be better than "count" and "tfidf". The summary statistics on each word scoring method are below. 
```
          binary      count      tfidf       freq
count  10.000000  10.000000  10.000000  10.000000
mean    0.917500   0.888000   0.857500   0.907500
std     0.011118   0.006325   0.009501   0.003536
min     0.900000   0.875000   0.845000   0.900000
25%     0.910000   0.885000   0.851250   0.905000
50%     0.920000   0.887500   0.857500   0.910000
75%     0.923750   0.893750   0.863750   0.910000
max     0.935000   0.895000   0.875000   0.910000
```
!(Figure 1 - Box & Whisker Plot)[https://github.com/abhiram10199/Sentiment_Analysis/blob/main/Figure_1.png]


### Experimentation & Results
To evaluate the impact of different word scoring methods, a simple test was conducted using the following example reviews and their scores below:

_Review 1:_ "love movie"
_Review 2:_ "terrible movie"

| Method|      Review 1      |  Review 2 |
|----------|:-------------:|------:|
| Binary   |	    1	   |   1   |
| Frequency |    0   |   1 |
| TF-IDF| 1 |    1 |
| Count | 1 | 1|

**Expected Results:** 
1 for a positive review 
0 for a negative review
### Analysis:
The "binary" and "frequency" methods demonstrated slightly better performance compared to "count" and "tfidf". However, the results are not what was expected. The discrepancy between the expected and actual results in the table might stem from several reasons:

- Model Complexity: Simple MLP might not be sufficient
- Data Quality: Dataset (vocab.py) contains noise that could be filtered out like actor names.
- BoW Model ignores context and meaning
- Overtuning or Undertuning 

**Improvements:**

- Better data cleaning
- n-grams for better text representation
- Different model architecture

***
