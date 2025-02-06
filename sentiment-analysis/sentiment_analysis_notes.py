# file pulled from https://www.youtube.com/watch?v=HIvQWdqvl7o repo
# with comments by me for breaking down code into what they do 

import nltk
import pandas
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import movie_reviews

def main():
    # adapted the file to use the main fuction so making these global to use in other function while conforming to my setup
    global vectorizer
    global model

    nltk.download("movie_reviews") # pulls presetup data set from nlk called movie-reviews

    documents = [
    (" ".join(movie_reviews.words(fileid)), category) # grabs the words from each of the files and catergorizes them base on if they are pos or neg
    for category in movie_reviews.categories() # pulls the categories or the names of the files in this case ['pos','neg']
    for fileid in movie_reviews.fileids(category) # takes the category and grabs all the files that have the matching category
    ]

    # data frame: A table from the pandas library that is a labled datastructure that consists of rows and collumns to seperate data like a csv
    data_frame = pandas.DataFrame(documents, columns=["review", "sentiment"]) # creates a dataframe that will hold the words a.k.a review and categories a.k.a pos or neg 

    # Document Term Matrix (DTM): a matrix that shows how often words appear in a specified document
    vectorizer = CountVectorizer(max_features=2000) # Converts text into a document-term matrix which will store the top 2,000 most frequent words from the text
    X = vectorizer.fit_transform(data_frame["review"]) # fit() finds frequency of words in reviews; transform() takes the fitted data and builds a DTM
    y = data_frame["sentiment"] # extracts whether it was pos or neg

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # splits the vectorized data into trained and test data with 80% training and 20% test
    )

    # Naive Bayes: based on the bayes theorem that is a mathematical formula that helps determine the conditional probablility of an event based on prior and new knowledge
    model = MultinomialNB() # runs a Mulinomial Naive Bayes classifier
    model.fit(X_train, y_train) # fits the data based on the Naive Bayes Model and train the model using the training data


    y_pred = model.predict(X_test) # tests the trained model using the trained data
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

def predict_sentiment(text):
    text_vector = vectorizer.transform([text]) # converts provided text into a DTM using the top 2,000 most frequent words
    prediction = model.predict(text_vector) # does a prediction on the trained MulinomialNB model
    return prediction[0]


if __name__ == "__main__":
    main()
    print(predict_sentiment("I absolutely loved this movie! It was fantastic."))
    print(predict_sentiment("It was a terrible film. I hated it."))
    print(predict_sentiment("The movie was okay, nothing special."))
