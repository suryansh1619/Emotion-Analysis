import tensorflow as tf
import numpy as np
import re
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.data.path.append('/usr/share/nltk_data')
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('omw-1.4')
model = tf.keras.models.load_model('emotion_analysis.keras')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

label_mapping = {1:'enthusiasm',
    2: 'fun',
    0: 'anger', 
    6: 'neutral', 
    5: 'love', 
    3: 'happiness', 
    8: 'sadness', 
    7: 'relief', 
    4: 'hate'
}



def text_preprocess(text):
    text = contractions.fix(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\d+", "", text)
    text = text.lower()

    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)

tokenizer = Tokenizer()

MAXLEN = 78  

def predict_emotion(text):
    processed_text = text_preprocess(text)
    tokenizer.fit_on_texts([processed_text])  

    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=MAXLEN)

    predictions = model.predict(padded_sequence)
    predicted_label = np.argmax(predictions)
    predicted_emotion = label_mapping.get(predicted_label, "Unknown")

    return predicted_emotion

