import pandas as pd
import joblib

# Load the trained model, label encoder, and columns
model = joblib.load('/Users/eesharamkumar/Downloads/repurpose.ai/final_model.pkl')
label_encoder = joblib.load('/Users/eesharamkumar/Downloads/repurpose.ai/label_encoder.pkl')
model_columns = joblib.load('/Users/eesharamkumar/Downloads/repurpose.ai/model_columns.pkl')

# Load the processed data
processed_data_path = '/Users/eesharamkumar/Downloads/repurpose.ai/processed_data.csv'
data = pd.read_csv(processed_data_path)

# Select a sample input from the processed data
sample_input = data.iloc[0]  # You can change the index to select a different sample

# Convert sample input to DataFrame
sample_df = pd.DataFrame([sample_input])

# Preprocess sample input to match training data format
def preprocess_sample(sample_df, model_columns):
    sample_df = pd.get_dummies(sample_df, columns=['MOA', 'Target', 'Indication', 'Vendor', 'Phase', 'Deprecated ID'])
    sample_df = sample_df.reindex(columns=model_columns, fill_value=0)
    return sample_df

sample_preprocessed = preprocess_sample(sample_df, model_columns)

# Predict the disease area
predicted_disease_encoded = model.predict(sample_preprocessed)
predicted_disease = label_encoder.inverse_transform(predicted_disease_encoded)

print(f'Predicted Disease Area: {predicted_disease[0]}')

