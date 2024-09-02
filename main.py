from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['text']
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        return render_template('index.html', prediction=prediction, text=user_input)
    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)
