'''
API to serve the model
'''
import pickle as p
import json
import os
import pandas as pd
from flask import Flask, request

APP = Flask(__name__)

def clea_data(df):
    cols_to_drop = ['id','debt_requests_count','housing_base_cost',
            'a_mal_count','a_mal_active_amount',
            'e_mal_count','e_mal_active_amount',
            'contact_channel', 'blanco_amount',
            'inquiries_count','credit_card_amount', 'credit_used','income_employment',
            'income_tax','creditors_count','salary_surplus', 'capital_deduction',
            'credit_count','income_gross',
            'loan_type','customer_postal']

    df.drop(cols_to_drop, inplace=True, axis=1)
    bc_dict = {'f': 0, 't': 1}
    df.big_city = df.big_city.map(bc_dict) 
@APP.route('/', methods=['POST'])
def pred():
    json_data = request.get_json()
    data = pd.DataFrame(json_data, index=[0])

    cwd = os.getcwd()
    loaded_model = p.load(open(cwd + '/models/model.pickle', 'rb'))


    result = loaded_model.predict_proba(data)
    val = result.tolist()[0][1]
    return json.dumps({'pd': val})


if __name__ == '__main__':
    APP.run(debug=True)
