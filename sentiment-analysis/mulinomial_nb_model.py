import nltk
import pandas
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import movie_reviews
from additional_movie_reviews import additional_movie_reviews # my own file created so that not all the reviews are within this file alone


def train_model():
    global vectorizer
    global model

    nltk.download("movie_reviews") 

    documents = [
    (" ".join(movie_reviews.words(fileid)), category) 
    for category in movie_reviews.categories() 
    for fileid in movie_reviews.fileids(category) 
    ] + additional_movie_reviews # adding my generated reviews to the test 
    
    data_frame = pandas.DataFrame(documents, columns=["review", "sentiment"])

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data_frame["review"]) 
    y = data_frame["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42  
    )
    
    model = MultinomialNB() 
    model.fit(X_train, y_train) 


def get_sentiment(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)

    if prediction[0] == "pos":
        prediction = "positive"
    else:
        prediction = "negative"

    return prediction


def predict_sentiment_of_review():
    while True:
        text =  ""
        while text == "": 
            text = input("\nProvide a review to see its sentiment:\n")
        
        prediction = get_sentiment(text)

        
        print(f"\nThe review \"{text}\" was deemed a {prediction} review.")

        choice = input("\nEnter y in order to check another review. Enter any other key to end program.\n").lower()

        if choice != "y":
            break
    

if __name__ == "__main__":
    train_model()
    predict_sentiment_of_review()
