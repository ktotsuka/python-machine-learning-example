from flask import Flask, render_template

# Create an instance of Flask micro web framework
# "__name__" is needed so that Flask knows where to look for resources such as templates and static files.
app = Flask(__name__)

# Specify the URL that should trigger the execution of the index() function
@app.route('/') # this is a decorator and we are passing the index() function
def index():
    return render_template('first_app.html')

# Run the Flask web server
app.run(debug=True)
