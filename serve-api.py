# Code adapted from George-Bogdan Ivanov's "Natural Language Processing For Hackers"
# and "Real World Machine Learning", by Brink, Richards and Fetherolf 


import os
import re, string
import multiprocessing as mp
import numpy as np
from flask import Flask
from flask import request, jsonify
from sklearn.externals import joblib
from gensim.models.word2vec import Word2Vec


data_dir = os.environ['DATA_DIR'] + '/'

stop_words = set(['all', "she'll", "don't", 'being', 'over', 'through', 'yourselves', 'its', 'before', "he's", "when's", "we've", 'had', 'should', "he'd", 'to', 'only', "there's", 'those', 'under', 'ours', 'has', "haven't", 'do', 'them', 'his', "they'll", 'very', "who's", "they'd", 'cannot', "you've", 'they', 'not', 'during', 'yourself', 'him', 'nor', "we'll", 'did', "they've", 'this', 'she', 'each', "won't", 'where', "mustn't", "isn't", "i'll", "why's", 'because', "you'd", 'doing', 'some', 'up', 'are', 'further', 'ourselves', 'out', 'what', 'for', 'while', "wasn't", 'does', "shouldn't", 'above', 'between', 'be', 'we', 'who', "you're", 'were', 'here', 'hers', "aren't", 'by', 'both', 'about', 'would', 'of', 'could', 'against', "i'd", "weren't", "i'm", 'or', "can't", 'own', 'into', 'whom', 'down', "hadn't", "couldn't", 'your', "doesn't", 'from', "how's", 'her', 'their', "it's", 'there', 'been', 'why', 'few', 'too', 'themselves', 'was', 'until', 'more', 'himself', "where's", "i've", 'with', "didn't", "what's", 'but', 'herself', 'than', "here's", 'he', 'me', "they're", 'myself', 'these', "hasn't", 'below', 'ought', 'theirs', 'my', "wouldn't", "we'd", 'and', 'then', 'is', 'am', 'it', 'an', 'as', 'itself', 'at', 'have', 'in', 'any', 'if', 'again', 'no', 'that', 'when', 'same', 'how', 'other', 'which', 'you', "shan't", 'our', 'after', "let's", 'most', 'such', 'on', "he'll", 'a', 'off', 'i', "she'd", 'yours', "you'll", 'so', "we're", "she's", 'the', "that's", 'having', 'once'])

def tokenize_one(d):
    pattern = re.compile('[\W_]+', re.UNICODE)
    sentence = d.lower().split(" ") 
    sentence = [pattern.sub('', w) for w in sentence]
    result = [w for w in sentence if w not in stop_words]
    # Some sentences are empty, putting NaNs in features, which is bad
    return ['read'] if len(result) == 0 else result

def tokenize(docs):
    pool = mp.Pool(mp.cpu_count())
    sentences = pool.map(tokenize_one, [d for d in docs])
    pool.close()
    
    return sentences    

def featurize_w2v(model, sentences):
    f = np.zeros((len(sentences), model.vector_size))
    for i,s in enumerate(sentences):
        for w in s:
            try:
                vec = model[w]
            except KeyError:
                continue
            f[i,:] = f[i,:] + vec
        f[i,:] = f[i,:] / len(s)
    return f    

model = Word2Vec.load(data_dir + 'word2vec.model')
forest_model = joblib.load(data_dir + 'forest_model.sav')

def predict_one_review(one_review_text):
    one_review = [one_review_text]
    one_sentences = tokenize(one_review)
    one_features_w2v = featurize_w2v(model, one_sentences)
    one_prediction = forest_model.predict_proba(one_features_w2v)[0]
    return np.argmax(one_prediction) + 1, one_prediction

app = Flask(__name__)

@app.route("/review", methods=['POST'])
def review():
    prediction, prediction_array = predict_one_review(str(request.data))
    return jsonify(review=str(prediction), array=str(prediction_array))

