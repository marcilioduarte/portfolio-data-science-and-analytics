## IMPORTING LIBS

import pandas as pd
import plotly.express as px

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

import pickle
import gradio as gr

## CREATING FUNCTION

def predict_credit_worthiness(name, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22):
    path =  'german_credit_risk/model/model.pickle'
    greet = 'Hey, ' + name + '!'
    with open(path, 'rb') as file:
        model = pickle.load(file)
        inputs = {'Account Balance_1': int(x1),
                  'Account Balance_2': int(x2),
                  'Account Balance_3': int(x3),
                  'Payment Status of Previous Credit_1': int(x4),
                  'Payment Status of Previous Credit_3': int(x5),
                  'Purpose_1': int(x6),
                  'Purpose_4': int(x7),
                  'Value Savings/Stocks_1': int(x8),
                  'Value Savings/Stocks_3': int(x9),
                  'Value Savings/Stocks_5': int(x10),
                  'Length of current employment_1': int(x11),
                  'Length of current employment_4': int(x12),
                  'Instalment per cent_4': int(x13),
                  'Guarantors_1': int(x14),
                  'Duration in Current address_1': int(x15),
                  'Duration in Current address_2': int(x16),
                  'Most valuable available asset_1': int(x17),
                  'Most valuable available asset_4': int(x18),
                  'Concurrent Credits_3': int(x19),
                  'Type of apartment_1': int(x20),
                  'No of Credits at this Bank_1': int(x21),
                  'Occupation_1': int(x22)
                  }
        prediction = model.predict([list(inputs.values())])
        
    y_test = pd.read_parquet('german_credit_risk/data/processed/y_test.parquet')
    y_test = y_test.squeeze()

    yhat = pd.read_parquet('german_credit_risk/data/processed/yhat.parquet')
    yhat = yhat.squeeze()
    
    precision = precision_score(y_test, yhat).round(2)
    recall = recall_score(y_test, yhat).round(2)
    f1 = f1_score(y_test, yhat).round(2)

    features_names =  ['No account', 'No balance', 'Some balance', 'No credit problems', 
                       'Some credit problems', 'New car', 'Other purpose', 'No savings', 
                       'DM betwenn [100, 1000]', 'DM >= 1000', 'Employment: <1 year (or unemployed)', 'Employment: 4<x<7 years', 
                       'Installment smaller than 20%', 'No guarantors', 'Less than a year in same address', '1<x<4 years in address', 
                       'Not available / no assets', 'Ownership of house or land',  'No further running credits', 'Free ap', 
                       'One credit at thins bank','Unemployed or unskilled']
    importance = model.feature_importances_
    data = pd.DataFrame()
    data['Feature Importance'] = importance
    data['Feature'] = features_names
    p = px.bar(data, y='Feature Importance', x='Feature', width=1200, height=500)
    
    cfm = confusion_matrix(y_test, yhat)
    cfm_plot = px.imshow(cfm,
                x=['Predicted 0', 'Predicted 1'],
                y=['Actual 0', 'Actual 1'],
                color_continuous_scale='Blues',
                labels=dict(x="Predicted", y="Actual", color="Count"),
                text_auto=True)    
    
    if prediction == 1:
        return (greet + ' According to our model, your client is eligible for the loan.', 
                'Precision: '+ str(precision), 
                'Recall: '+ str(recall), 
                'F1 Score: '+ str(f1), 
                p, 
                cfm_plot)
    else:
        return (greet + ' Unfortunately, according to our model, your client is not eligible for the loan for now :(.', 
                'Precision: '+ str(precision), 
                'Recall: '+ str(recall), 
                'F1 Score: '+ str(f1), 
                p, 
                cfm_plot)
    
## creating the interface

with gr.Blocks() as demo:
    gr.Markdown('# Credit Worthiness Prediction')
    gr.Markdown("""
                To predict our clients' creditworthiness, please use this application as follows:
                
                1. Enter your name and navigate through the client's information tabs. Select the boxes that best match your client's characteristics. Leave blank if none apply.

                2. Once completed, click 'Predict' to determine if the client is creditworthy.
                """)
    with gr.Accordion('Name'):
        name = gr.Textbox(lines=1, label='Your name')
    with gr.Accordion("Enter your client's information"):
        with gr.Tab('Account Balance'):
            gr.Markdown('Select only one option. Leave all boxes blank if none of the options fits the client.')
            x1 = gr.Checkbox(1, label='No account')
            x2 = gr.Checkbox(0, label='No balance')
            x3 = gr.Checkbox(0, label='Some balance')
        with gr.Tab('Payment status of previous credit'):
            gr.Markdown('Select only one option. Leave all boxes blank if none of the options fits the client.')
            x4 = gr.Checkbox(1, label='Some problems')
            x5 = gr.Checkbox(0, label='No problems in this bank')
        with gr.Tab('Purpose'):
            gr.Markdown('Select only one option. Leave all boxes blank if none of the options fits the client.')
            x6 = gr.Checkbox(1, label='New car')
            x7 = gr.Checkbox(0, label='Other')
        with gr.Tab('Value savings/stocks'):
            gr.Markdown('Select only one option. Leave all boxes blank if none of the options fits the client.')
            x8 = gr.Checkbox(1, label='No savings')
            x9 = gr.Checkbox(0, label='DM betwenn [100, 1000]')
            x10 = gr.Checkbox(0, label='DM >= 1000')
        with gr.Tab('Length of current employment'):
            gr.Markdown('Select only one option. Leave all boxes blank if none of the options fits the client.')
            x11 = gr.Checkbox(1, label='Below 1 year (or unemployed)')
            x12 = gr.Checkbox(0, label='Between 4 and 7 years')
        with gr.Tab('Instalment per cent'):
            gr.Markdown('Select only one option. Leave all boxes blank if none of the options fits the client.')
            x13 = gr.Checkbox(0, label='Smaller than 20%')
        with gr.Tab('Guarantors'):
            gr.Markdown('Select only one option. Leave all boxes blank if none of the options fits the client.')
            x14 = gr.Checkbox(0, label='No guarantors')
        with gr.Tab('Duration in current address'):
            gr.Markdown('Select only one option. Leave all boxes blank if none of the options fits the client.')
            x15 = gr.Checkbox(1, label='Less than a year')
            x16 = gr.Checkbox(0, label='Between 1 and 4 years')
        with gr.Tab('Most valuable available asset'):
            gr.Markdown('Select only one option. Leave all boxes blank if none of the options fits the client.')
            x17 = gr.Checkbox(1, label='Not available / no assets')
            x18 = gr.Checkbox(0, label='Ownership of house or land')
        with gr.Tab('Concurrent credits'):
            gr.Markdown('Select only one option. Leave all boxes blank if none of the options fits the client.')
            x19 = gr.Checkbox(0, label='No further running credits')
        with gr.Tab('Type of apartment'):
            gr.Markdown('Select only one option. Leave all boxes blank if none of the options fits the client.')
            x20 = gr.Checkbox(0, label='Free apartment')
        with gr.Tab('Number of credits at this Bank'):
            gr.Markdown('Select only one option. Leave all boxes blank if none of the options fits the client.')
            x21 = gr.Checkbox(0, label='One credit')        
        with gr.Tab('Occupation'):
            gr.Markdown('Select only one option. Leave all boxes blank if none of the options fits the client.')
            x22 = gr.Checkbox(0, label='Unemployed or unskilled with no permanent') 
    predict_button = gr.Button('Predict')
    prediction_output = gr.Label(num_top_classes=2)
    with gr.Accordion('Metrics and plots'):
        with gr.Tab('Metrics'):
            with gr.Row():
                precision_output = gr.Label()
            with gr.Row():
                recall_output = gr.Label()
            with gr.Row():
                f1_output = gr.Label()
        with gr.Tab('Feature Importances'):
            fimp_output = gr.Plot()
        with gr.Tab('Confusion Matrix'):
            cfm_output = gr.Plot()
    predict_button.click(fn=predict_credit_worthiness,
                         inputs=[name, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22],
                         outputs=[prediction_output,precision_output, recall_output, f1_output, fimp_output, cfm_output])
    gr.Markdown('''
                Want to work in a project together or have interest in my services? Reach me:
                [Linkedin](https://www.linkedin.com/in/marcilioduarte98/)
                [Github](https://github.com/marcilioduarte)
                @marcilioduarte | Economics and Data Science
                ''')
demo.launch()