from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np
from vectorizer import vect

class ReviewForm(Form):
    moviereview = TextAreaField('', [validators.DataRequired(), validators.length(min=15)])

def classify(document):
    label = {0: 'negative', 1: 'positive'}
    X = vect.transform([document])
    y = clf.predict(X)[0]
    proba = np.max(clf.predict_proba(X))
    return label[y], proba

def train(document, y):
    X = vect.transform([document])
    clf.partial_fit(X, [y])

def sqlite_entry(path, document, y):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("INSERT INTO review_db (review, sentiment, date)"\
    " VALUES (?, ?, DATETIME('now'))", (document, y))
    conn.commit()
    conn.close()

# Create an instance of Flask micro web framework
# "__name__" is needed so that Flask knows where to look for resources such as templates and static files.
app = Flask(__name__)

# Get current directory
cur_dir = os.path.dirname(__file__)

# Read in the pickled classifier.  Copy the pickled object file from "pickle-test"
clf = pickle.load(open(os.path.join(cur_dir,
                 'pkl_objects',
                 'classifier.pkl'), 'rb')) # rb: read in binary mode

# Get the review database path
db = os.path.join(cur_dir, 'reviews.sqlite')

# Specify the URL that should trigger the execution of the index() function
@app.route('/') # this is a decorator and we are passing the function below
def index():
    form = ReviewForm(request.form)
    return render_template('reviewform.html', form=form)

# Specify the URL that should trigger the execution of the results() function
@app.route('/results', methods=['POST']) # this is a decorator and we are passing the function below
def results():
    form = ReviewForm(request.form)
    if request.method == 'POST' and form.validate():
        review = request.form['moviereview']
        y, proba = classify(review)
        return render_template('results.html',
                                content=review,
                                prediction=y,
                                probability=round(proba*100, 2))
    return render_template('reviewform.html', form=form)

# Specify the URL that should trigger the execution of the feedback() function
@app.route('/thanks', methods=['POST'])
def feedback():
    feedback = request.form['feedback_button']
    review = request.form['review']
    prediction = request.form['prediction']
    inv_label = {'negative': 0, 'positive': 1}
    y = inv_label[prediction]
    if feedback == 'Incorrect':
        y = int(not(y))
    train(review, y)
    # Store reviews in a database.  When the app restarts, the classifier will forget about these reviews.
    sqlite_entry(db, review, y)
    return render_template('thanks.html')

# Run the Flask web server
# Don't run the app if this app is hosted on a web hosting service such as Pythonanywhere.com because the web hosting service will take care of it
if __name__ == '__main__':
    app.run(debug=True)
