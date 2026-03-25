from flask import Flask, render_template, request
import numpy as np
import cv2
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model("model.h5")

labels = ["Cat", "Dog", "Lion", "Tiger"]

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    path = "static/" + file.filename
    file.save(path)

    img = cv2.imread(path)
    img = cv2.resize(img, (128,128))
    img = img / 255.0
    img = np.reshape(img, (1,128,128,3))

    pred = model.predict(img)
    result = labels[np.argmax(pred)]

    return render_template("index.html", prediction=result, image=path)

if __name__ == "__main__":
    app.run(debug=True)
