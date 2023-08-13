import os
import pickle
import numpy as np
#from tqdm.notebook import tqdm
from flask import Flask, render_template, request
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
#from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
#from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from PIL import Image
#import io
#import matplotlib.pyplot as plt
#from nltk.translate.bleu_score import corpus_bleu

app = Flask(__name__)

BASE_DIR = 'archive'
WORKING_DIR = 'cap'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = None
tokenizer = None
vgg_model = None
max_length = None
features = None
mapping = None

def load_pretrained_models():
    global model, tokenizer, vgg_model, max_length, features, mapping
    model = load_model(os.path.join(WORKING_DIR, 'best_model.h5'))
    with open(os.path.join(WORKING_DIR, 'tokenizer.pkl'), 'rb') as f:
        tokenizer = pickle.load(f)
    vgg_model = VGG16()
    vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
    max_length = 35  # Set your max_length here
    '''with open(os.path.join(WORKING_DIR, 'features.pkl'), 'rb') as f:
        features = pickle.load(f)
    with open(os.path.join(WORKING_DIR, 'mapping.pkl'), 'rb') as f:
        mapping = pickle.load(f)'''

def preprocess_image(image):
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    return in_text

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            image_path = os.path.join('uploads', uploaded_file.filename)
            uploaded_file.save(image_path)
            image = Image.open(uploaded_file)
            preprocessed_image = preprocess_image(image)
            feature = vgg_model.predict(preprocessed_image, verbose=0)
            caption = predict_caption(model, feature, tokenizer, max_length)
            return render_template('index.html', caption=caption, image_path=image_path)
    return render_template('index.html')

'''@app.route('/generate_caption/<image_name>')
def generate_caption(image_name):
    image_id = image_name.split('.')[0]
    img_path = os.path.join(BASE_DIR, "Images", image_name)
    image = Image.open(img_path)
    captions = mapping[image_id]
    actual_captions = "\n".join(captions)
    feature = features[image_id]
    predicted_caption = predict_caption(model, feature, tokenizer, max_length)
    return render_template('generate_caption.html', actual_captions=actual_captions, predicted_caption=predicted_caption, image_path=img_path)'''

if __name__ == '__main__':
    load_pretrained_models()
    app.run(debug=True)
