from flask import Flask, jsonify, request
import joblib
import sklearn
#curl -d "{\"Medidas\":[[2, 1, 123, 456, 1, 1, 1, 1, 3, 2, 1.0, 0.0, 0.0, 0.0, 1, 5, 2, 3, 1, 5000, 10000, 2, 4, 6]]}" -H "Content-Type: application/json" -X POST http://127.0.0.1:5000/predecir
app = Flask(__name__)

@app.route("/")
def home():
    return 'La pagina esta funcionando bien'

@app.route("/predecir", methods=["POST"])
def predecir():
    json=request.get_json(force=True)
    medidas=json['Medidas']
    rf=joblib.load('modelo_entrenado.pkl')
    prediccion=rf.predict(medidas)
    return'Las medidas que diste corresponden a la clase {0}\n\n'.format(prediccion)

if __name__ == '__main__':
    app.run

