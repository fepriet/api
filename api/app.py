from flask import Flask, jsonify
import joblib
import sklearn

app = Flask(__name__)

@app.route("/")
def home():
    return app.send_static_file('index.html')

@app.route("/api")
def api():
    medidas = [[2, 1, 123, 456, 1, 1, 1, 1, 3, 2, 1.0, 0.0, 0.0, 0.0, 1, 5, 2, 3, 1, 5000, 10000, 2, 4, 6]]
    rf = joblib.load('modelo_entrenado.pkl')
    prediccion = rf.predict(medidas)
    return jsonify(prediccion.tolist())

if __name__ == '__main__':
    app.run()


