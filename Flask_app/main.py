
import pickle
from flask import Flask, request, jsonify
from model_files.ml_model import predict_loan

app = Flask("Loan_prediction")


@app.route('/', methods=['GET', 'POST'])
def predict():
    data = request.get_json()
    
    with open("classifier.bin","rb") as fin:
        model = pickle.load(fin)
        fin.close()
        
    pred = predict_loan(data,model)
    
    response = {
        'Loan_Status': list(pred)
    }
    
    return jsonify(response)

'''
@app.route('/', methods=['GET'])
def ping():
    return "Pinging Model Application!"
'''

if __name__ == '__main__':
    app.run(debug=True)
    


