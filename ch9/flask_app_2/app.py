from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators

class HelloForm(Form):
    sayhello = TextAreaField('',[validators.DataRequired()])

# Create an instance of Flask micro web framework
# "__name__" is needed so that Flask knows where to look for resources such as templates and static files.
app = Flask(__name__)

# Specify the URL that should trigger the execution of the index() function
@app.route('/') # this is a decorator and we are passing the index() function
def index():
    form = HelloForm(request.form)
    return render_template('first_app.html', form=form)

# Specify the URL that should trigger the execution of the hello() function
@app.route('/hello', methods=['POST']) # this is a decorator and we are passing the hello() function
def hello():
    form = HelloForm(request.form)
    if request.method == 'POST' and form.validate():
        name = request.form['sayhello']
        return render_template('hello.html', name=name)
    return render_template('first_app.html', form=form)

# Run the Flask web server
app.run(debug=True)