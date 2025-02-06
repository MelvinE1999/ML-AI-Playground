import mulinomial_nb_model as NaiveBayes
from textblob import TextBlob


model_usage_flag = {
    "Naive Bayes" : False,
    "TextBlob": False    
}

def set_model_selection_flags() -> None:
    model = ""
    choices = ["nb", "naivebayes", "tb", "textblob", "both"]

    while model not in choices:
        model = input("\nSelect if you want to use Naive Bayes, Textblob, or both models for this review.\n").replace(" ","").lower()
    
    if model in ["nb", "naivebayes", "both"]:
        model_usage_flag["Naive Bayes"] = True
    
    if model in ["tb", "textblob", "both"]:
        model_usage_flag["TextBlob"] = True

def get_sentiment_based_on_polarity(polarity:float) -> str:
    if polarity < 0:
        return "negative"
    else:
        return "positive"
    
def print_results(text:str, sentiment_naive_bayes:str, sentiment_textblob:str) -> None:
    for model, flag in model_usage_flag.items():
        if not flag:
            continue

        sentiment = ""
        if model == "Naive Bayes":
            sentiment = sentiment_naive_bayes
        elif model == "TextBlob":
            sentiment = sentiment_textblob
        
        print(f"{model} Output: The review \"{text}\" was deemed a {sentiment} review.\n")

def main() -> None:
    sentiment_textblob = ""
    sentiment_naive_bayes = ""
    text = input("\nEnter your review and I'll tell you if it is positive or negative:\n")

    set_model_selection_flags()
    print("") # seperator for readability

    if model_usage_flag["Naive Bayes"] == True:
        NaiveBayes.train_model()
        print("") # seperator as it prints the download in the terminal 
        sentiment_naive_bayes = NaiveBayes.get_sentiment(text)
        
    if model_usage_flag["TextBlob"] == True: 
        blob = TextBlob(text)
        text_sentiment_polarity = blob.sentiment.polarity
        sentiment_textblob = get_sentiment_based_on_polarity(text_sentiment_polarity)
    
    print_results(text, sentiment_naive_bayes, sentiment_textblob)

    

if __name__ == "__main__":
    main()