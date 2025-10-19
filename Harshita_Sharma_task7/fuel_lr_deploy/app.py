from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load all saved files
try:
    with open('numeric_cols.pkl', 'rb') as f:
        numeric_cols = pickle.load(f)
    
    with open('categorical_cols.pkl', 'rb') as f:
        categorical_cols = pickle.load(f)
    
    with open('encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    
    with open('final_feature_names.pkl', 'rb') as f:
        final_feature_names = pickle.load(f)
    
    with open('linear_regression_fuel_data.pkl', 'rb') as f:
        model = pickle.load(f)
    
    print("All files loaded successfully")
    print("Model expects", len(final_feature_names), "features")
except Exception as e:
    print("Error loading files:", str(e))
    exit(1)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', 
                         prediction=None, 
                         error=None, 
                         input_data=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form = request.form
        
        # Collect numeric values
        numeric_data = {
            'ENGINE SIZE': float(form.get('ENGINE SIZE')),
            'CYLINDERS': float(form.get('CYLINDERS')),
            'FUEL CONSUMPTION': float(form.get('FUEL CONSUMPTION')),
            'ENGINE_CYLINDER_RATIO': float(form.get('ENGINE_CYLINDER_RATIO')),
            'POWER_INDICATOR': float(form.get('POWER_INDICATOR'))
        }
        numeric_df = pd.DataFrame([numeric_data])
        
        # Collect categorical values
        categorical_data = {
            'MAKE': str(form.get('MAKE')),
            'MODEL': str(form.get('MODEL')),
            'VEHICLE CLASS': str(form.get('VEHICLE CLASS')),
            'TRANSMISSION': str(form.get('TRANSMISSION')),
            'FUEL': str(form.get('FUEL'))
        }
        categorical_df = pd.DataFrame([categorical_data])
        
        # Encode categorical columns
        encoded_array = encoder.transform(categorical_df)
        encoded_df = pd.DataFrame(
            encoded_array, 
            columns=encoder.get_feature_names_out(categorical_cols)
        )
        
        # Combine numeric and encoded
        numeric_df.reset_index(drop=True, inplace=True)
        encoded_df.reset_index(drop=True, inplace=True)
        combined = pd.concat([numeric_df, encoded_df], axis=1)
        
        # Align to match training features
        for col in final_feature_names:
            if col not in combined.columns:
                combined[col] = 0.0
        combined = combined[final_feature_names]
        
        # Predict
        prediction = model.predict(combined)[0]
        
        return render_template('index.html', 
                             prediction=round(float(prediction), 2), 
                             error=None, 
                             input_data=form)
    
    except ValueError as ve:
        return render_template('index.html', 
                             prediction=None, 
                             error=f"Invalid input: {str(ve)}", 
                             input_data=request.form)
    except Exception as e:
        return render_template('index.html', 
                             prediction=None, 
                             error=f"Error: {str(e)}", 
                             input_data=request.form)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("Starting server on port", port)
    app.run(host='0.0.0.0', port=port, debug=False)
