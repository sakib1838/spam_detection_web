from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

app = Flask(__name__)

# Load the saved model
model = load_model('gap.h5')

# Load tokenizer (you should have saved this during training)
tokenizer = Tokenizer()
# Load tokenizer from a file or some other way

# Function to preprocess text
def preprocess_text(text):
    msg_list =[]
    msg_list.append(text)
    # Tokenize and pad the input text
    print(msg_list)
    text = tokenizer.texts_to_sequences(msg_list)
    padded_seq = pad_sequences(text,maxlen=162)  
    return padded_seq

# Function to predict if the text is spam or not
def predict_spam(text):
    preprocessed_text = preprocess_text(text)
    prediction = model.predict(preprocessed_text)
    print("prediction:")
    print(prediction)
    return prediction[0][0]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        message = request.form['message']
        print("Running")
        prediction = predict_spam(message)
        if prediction > 0.45:
            result = 'Spam'
        else:
            result = 'Not Spam'
        
        
        
        return render_template('index.html', result=result, message=message)
    return render_template('index.html', result=None, message=None)

if __name__ == '__main__':
    app.run(debug=True)
