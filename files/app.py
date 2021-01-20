import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
app = Flask(__name__)
app.static_folder = './static'
app.template_folder ='./templates'
model = load_model('fake_nlp.h5')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #int_features = [int(x) for x in request.form.values()]
    text = [request.form['news']]
    oh = [one_hot(words, 5000) for words in text]
    sent_length = 20
    ed = pad_sequences(oh, padding='pre', maxlen=sent_length)
    x = np.array(ed)
    print(x)
    mp = model.predict(x)

    if (mp[0][0] >= 0.5):
        output = "True"
    else:
        output = "False"

    print(output)
    print(mp[0][0])

    return render_template('index.html', prediction_text=output)




if __name__ == "__main__":
    app.run(debug=True)
