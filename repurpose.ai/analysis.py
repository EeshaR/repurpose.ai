import pandas as pd
import openai
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Set up OpenAI API key

# Load the processed data
processed_data_path = '/Users/eesharamkumar/Downloads/repurpose.ai/processed_data.csv'
data = pd.read_csv(processed_data_path)

# Prepare data for prediction
def prepare_data(data):
    # Label encode the 'Disease Area' column as the target variable
    le = LabelEncoder()
    data['Disease Area'] = le.fit_transform(data['Disease Area'])
    
    # Convert categorical data to numeric using one-hot encoding
    data = pd.get_dummies(data, columns=['MOA', 'Target', 'Indication', 'Vendor', 'Phase', 'Deprecated ID'])
    
    # Separate features and target
    X = data.drop(columns=['Name', 'SMILES', 'InChIKey', 'Id'])
    y = data['Disease Area']
    
    return X, y, le

X, y, label_encoder = prepare_data(data)

# Generate synthetic data with GPT-4
def generate_synthetic_data(sample, n_samples=10):
    prompt = f"Generate {n_samples} synthetic data samples similar to: {sample}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a data augmentation assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500  # Reduce max tokens to avoid rate limit
    )
    synthetic_data = response.choices[0].message['content'].strip()
    return synthetic_data

# Example sample data for generating synthetic data
example_sample = {
    'MOA': 'dopamine receptor agonist',
    'Target': 'DRD1',
    'Disease Area': 'neurology/psychiatry',
    'Indication': 'Parkinson\'s Disease',
    'Vendor': 'Tocris',
    'Purity': 98.9,
    'Phase': 'Launched',
    'Deprecated ID': 'BRD-K76022557-003-28-9'
}

# Function to parse GPT-4 response and integrate into the dataset
def parse_and_integrate_synthetic_data(synthetic_data, X, y, label_encoder):
    # This function should parse the GPT-4 response and convert it to a DataFrame
    # For simplicity, we'll assume synthetic_data is a list of dictionaries
    new_data = pd.DataFrame([example_sample for _ in range(10)])  # Mocking the synthetic data
    new_data['Disease Area'] = label_encoder.transform(['neurology/psychiatry']*10)  # Mocking the target variable
    new_data = pd.get_dummies(new_data, columns=['MOA', 'Target', 'Indication', 'Vendor', 'Phase', 'Deprecated ID'])
    
    # Align columns with the original dataset
    new_data = new_data.reindex(columns=X.columns, fill_value=0)
    
    X_augmented = pd.concat([X, new_data], ignore_index=True)
    y_augmented = np.concatenate([y, new_data['Disease Area']])
    
    return X_augmented, y_augmented

# Generate synthetic data and augment the dataset
synthetic_data = generate_synthetic_data(example_sample)
X_augmented, y_augmented = parse_and_integrate_synthetic_data(synthetic_data, X, y, label_encoder)

# Hyperparameter Tuning using GridSearchCV
def hyperparameter_tuning(X, y):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X, y)
    
    print(f'Best parameters: {grid_search.best_params_}')
    return grid_search.best_estimator_

best_model = hyperparameter_tuning(X_augmented, y_augmented)

# Cross-Validation
def cross_validation(model, X, y):
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f'Cross-validation scores: {cv_scores}')
    print(f'Mean cross-validation score: {cv_scores.mean() * 100:.2f}%')

cross_validation(best_model, X_augmented, y_augmented)

# Feature Importance
def feature_importance(model, X):
    importances = model.feature_importances_
    feature_names = X.columns
    feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
    print(feature_importances.head(20))  # Display top 20 features

    return feature_importances

important_features = feature_importance(best_model, X_augmented)

# Retrain the model with the best parameters and evaluate
def train_model(X, y, model):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    
    return model

# Optionally, you could filter features based on importance
# For example, using only top 50% important features
top_features = important_features['Feature'][:len(important_features)//2]
X_top = X_augmented[top_features]

# Train the final model with top features
final_model = train_model(X_top, y_augmented, best_model)

# Save the trained model and label encoder
joblib.dump(final_model, 'final_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
joblib.dump(X.columns, 'model_columns.pkl')
