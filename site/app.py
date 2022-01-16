from flask import Flask, render_template, request
from model.model import generate

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results', methods=['POST'])
def results():
    if request.method == 'POST':
       inp = request.form.get('input')
       result = generate(inp)
       return render_template('index.html',result=result, prompt=inp)

if __name__ == '__main__':
    app.run()