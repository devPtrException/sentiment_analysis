from django.shortcuts import render
import numpy as np
from . import *
import joblib
from joblib import load
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


model = load_model(
    "/home/mrmauler/DRIVE/projects/dl/sentiment_analysis/core/model/model.keras"
)
tokenizer = joblib.load(
    "/home/mrmauler/DRIVE/projects/dl/sentiment_analysis/core/model/tokenizer.pkl"
)

# model = load("/home/mrmauler/DRIVE/projects/dl/sentiment_analysis/core/model/model.keras")


def predict_sentiment(review):
    sequence = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequence, maxlen=200)
    # padded_sequence = pad_sequences(tokenizer.texts_to_sequences(review), maxlen=200)
    prediction = model.predict(padded_sequence)

    sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
    return sentiment


def predictor(request):
    if request.method == "POST":

        com_text = request.POST.get("com_text")
        
        sentiment = predict_sentiment(com_text)
        # result_df = pd.DataFrame(result, columns=data.columns[1:])

        print(sentiment)

        context = {
            "sentiment": str(sentiment),
            # "hate_speech_count": float(pred[0][1]),
            # "offensive_language_count": float(pred[0][2]),
            # "neither_count": float(pred[0][3]),
            # "class": float(pred[0][4]),
        }

        return render(request, "result.html", {"result": context})
    return render(request, "index.html")


# print((model) + "hello")
def predict_sentiment(request):
    if request.method == "POST":
        text = request.POST.get("text", "")
        
        sequence = tokenizer.texts_to_sequences([text])  # List input
        return HttpResponse(f"Sequence: {sequence}")
    return HttpResponse("Send a POST request with 'text' field")
