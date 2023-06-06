#Install Libraries
from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
application = Flask(__name__)

@application.route('/prediction', methods=['POST'])
#define function
def predict():
    if lr:
        try:
            scaler = StandardScaler()
            json_ = request.json
            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=rnd_columns, fill_value=0)
            query['Amount'] = scaler.fit_transform(query[['Amount']])
            predict = list(lr.predict(query))
            return jsonify({'prediction': str(predict)})
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Model not good')
        return ('Model is not good')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except:
        port = 12346 
        lr = joblib.load("log_smote.pkl") 
        print ('Model loaded')
        rnd_columns = joblib.load("log_smote_columns.pkl") # Load “rnd_columns.pkl”
        print ('Model columns loaded')
        application.run(port=port, debug=True)