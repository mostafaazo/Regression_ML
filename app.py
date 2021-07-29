from flask import Flask
from flask_restful import Api, Resource, reqparse
import joblib
import pandas as pd
import string

APP = Flask(__name__)
API = Api(APP)

MODEL = joblib.load('models/final_prediction_model.mdl')


class Predict(Resource):

    @staticmethod
    def post():
        parser = reqparse.RequestParser()
        parser.add_argument('est_num1')
        parser.add_argument('est_num2')
        parser.add_argument('est_letter')


        args = parser.parse_args()  # creates dict

        df = pd.DataFrame()
        df = df.append(args, ignore_index=True)
        df["est_letter"] = df["est_letter"].astype(pd.CategoricalDtype(categories=list(string.ascii_lowercase)))
        df = pd.get_dummies(df)
        X_new = df.values

        out = {'Prediction': MODEL.predict(X_new)[0]}

        return out, 200



API.add_resource(Predict, '/predict')

if __name__ == '__main__':
    APP.run(debug=True, port='1080')