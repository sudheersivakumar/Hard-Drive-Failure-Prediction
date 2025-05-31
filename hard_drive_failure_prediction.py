import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib

# Load the dataset
# Replace 'hard_drive_data.csv' with your actual dataset filename
df = pd.read_csv('C:/Users/sudhe/OneDrive/Documents/ML projects/Supervised learning/harddrive.csv/harddrive.csv')

# Display basic information about the dataset
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nSummary statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Assuming 'failure' is the target column (1 for failed drives, 0 for healthy)
# Check class distribution
if 'failure' in df.columns:
    print("\nClass distribution:")
    print(df['failure'].value_counts())
    print(f"Failure rate: {df['failure'].mean() * 100:.2f}%")

# Data preprocessing
def preprocess_data(df, target_col='failure'):
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Remove any non-numeric columns (or encode them if needed)
    non_numeric_cols = X.select_dtypes(exclude=['number']).columns
    if len(non_numeric_cols) > 0:
        print(f"Removing non-numeric columns: {list(non_numeric_cols)}")
        X = X.drop(columns=non_numeric_cols)
    
    # Handle missing values
    print("\nHandling missing values...")
    
    # Calculate the percentage of missing values in each column
    missing_percentage = X.isnull().mean() * 100
    
    # Drop columns with too many missing values (e.g., more than 80%)
    columns_to_drop = missing_percentage[missing_percentage > 80].index.tolist()
    if columns_to_drop:
        print(f"Dropping columns with >80% missing values: {len(columns_to_drop)} columns")
        X = X.drop(columns=columns_to_drop)
    
    # For remaining columns with missing values, impute with median
    print("Imputing remaining missing values with median...")
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    print("Imputation complete.")
    
    # Convert back to DataFrame to keep column names
    X = pd.DataFrame(X_imputed, columns=X.columns)
    
    # Split the data
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale the features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Handle class imbalance using SMOTE
    print("Applying SMOTE to handle class imbalance...")
    print(f"Before SMOTE - Class distribution: {np.bincount(y_train)}")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    print(f"After SMOTE - Class distribution: {np.bincount(y_train_resampled)}")
    print("SMOTE complete.")
    
    return X_train_resampled, X_test_scaled, y_train_resampled, y_test, scaler, X.columns

# Train and evaluate models
def train_and_evaluate():
    # Preprocess the data
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)
    
    # Train Random Forest model
    print("Training Random Forest model...")
    import time
    start_time = time.time()
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_time = time.time() - start_time
    print(f"Random Forest training completed in {rf_time:.2f} seconds")
    
    # Train Gradient Boosting model
    print("Training Gradient Boosting model...")
    start_time = time.time()
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    gb_time = time.time() - start_time
    print(f"Gradient Boosting training completed in {gb_time:.2f} seconds")
    
    # Evaluate models
    models = {
        'Random Forest': rf_model,
        'Gradient Boosting': gb_model
    }
    
    for name, model in models.items():
        print(f"\n{name} Model Evaluation:")
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        print("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        auc = roc_auc_score(y_test, y_prob)
        print(f"ROC AUC Score: {auc:.4f}")
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            print("\nTop 10 important features:")
            for i in range(min(10, len(feature_names))):
                print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    # Save the best model (you can change this based on performance)
    best_model = models['Gradient Boosting']  # Default choice, adjust based on results
    joblib.dump(best_model, 'c:\\Users\\sudhe\\OneDrive\\Documents\\ML projects\\hard_drive_failure_model.pkl')
    joblib.dump(scaler, 'c:\\Users\\sudhe\\OneDrive\\Documents\\ML projects\\hard_drive_scaler.pkl')
    
    return best_model, scaler, feature_names

# Run the training and evaluation
if __name__ == "__main__":
    model, scaler, feature_names = train_and_evaluate()
    
    print("\nModel training complete. The model has been saved to 'hard_drive_failure_model.pkl'")
    print("You can now use this model to predict hard drive failures.")