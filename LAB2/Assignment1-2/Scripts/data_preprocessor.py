import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
file_path = "/Users/binalpatel/Desktop/Machine learning/LAB2/Assignment1-2/Data/messy_data.csv"
data = pd.read_csv(file_path)
print(data.head())  # Shows the first few rows of the dataframe


# 1. Impute Missing Values
def impute_missing_values(data, strategy='mean'):
    """
    Fill missing values in the dataset.
    :param data: pandas DataFrame
    :param strategy: str, imputation method ('mean', 'median', 'mode')
    :return: pandas DataFrame
    """
    numeric_data = data.select_dtypes(include=['number'])

    if strategy == 'mean':
        data = data.fillna(numeric_data.mean())
    elif strategy == 'median':
        data = data.fillna(numeric_data.median())
    elif strategy == 'mode':
        data = data.fillna(numeric_data.mode().iloc[0])
    
    print("Data after imputation:")
    print(data.head())
    return data

data = impute_missing_values(data)  # Fill missing values using default 'mean' strategy

# 2. Remove Duplicates
def remove_duplicates(data):
    """
    Remove duplicate rows from the dataset.
    :param data: pandas DataFrame
    :return: pandas DataFrame
    """
    print("Data after removing duplicates:")
    data = data.drop_duplicates()
    print(data.head())
    return data

data = remove_duplicates(data)  # Remove duplicates

# 3. Normalize Numerical Data
def normalize_data(data, method='minmax'):
    """Apply normalization to numerical features.
    :param data: pandas DataFrame
    :param method: str, normalization method ('minmax' (default) or 'standard')
    """
    scaler = MinMaxScaler() if method == 'minmax' else StandardScaler()
    num_cols = data.select_dtypes(include=['number']).columns
    data[num_cols] = scaler.fit_transform(data[num_cols])
    
    print("Data after normalization:")
    print(data.head())
    return data

data = normalize_data(data)  # Normalize data using 'minmax' method

# 4. Remove Redundant Features   
def remove_redundant_features(data, threshold=0.9):
    """Remove redundant or duplicate columns based on correlation.
    :param data: pandas DataFrame
    :param threshold: float, correlation threshold
    :return: pandas DataFrame
    """
    # Select only numeric columns to avoid the string columns causing issues
    numeric_data = data.select_dtypes(include=['number'])
    
    # Compute the correlation matrix for numeric columns only
    corr_matrix = numeric_data.corr().abs()
    
    # Get the upper triangle of the correlation matrix (to avoid double counting)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find columns that have a correlation greater than the threshold
    drop_cols = [col for col in upper.columns if any(upper[col] > threshold)]
    
    # Drop the redundant columns from the original dataset
    data_cleaned = data.drop(columns=drop_cols)
    
    return data_cleaned



# ---------------------------------------------------

def simple_model(input_data, split_data=True, scale_data=False, print_report=False):
    """
    A simple logistic regression model for target classification.
    Parameters:
    input_data (pd.DataFrame): The input data containing features and the target variable 'target' (assume 'target' is the first column).
    split_data (bool): Whether to split the data into training and testing sets. Default is True.
    scale_data (bool): Whether to scale the features using StandardScaler. Default is False.
    print_report (bool): Whether to print the classification report. Default is False.
    Returns:
    None
    The function performs the following steps:
    1. Removes columns with missing data.
    2. Splits the input data into features and target.
    3. Encodes categorical features using one-hot encoding.
    4. Splits the data into training and testing sets (if split_data is True).
    5. Scales the features using StandardScaler (if scale_data is True).
    6. Instantiates and fits a logistic regression model.
    7. Makes predictions on the test set.
    8. Evaluates the model using accuracy score and classification report.
    9. Prints the accuracy and classification report (if print_report is True).
    """

    # if there's any missing data, remove the columns
    input_data.dropna(inplace=True)

    # split the data into features and target
    target = input_data.iloc[:, 0]  # First column as target
    features = input_data.iloc[:, 1:]
    
    # Encode categorical features using one-hot encoding
    features = pd.get_dummies(features)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, stratify=target, random_state=42)

    # Scale the data if scale_data is True
    if scale_data:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
    # Instantiate and fit the model
    model = LogisticRegression(max_iter=100, solver='liblinear', penalty='l2', C=1.0)
    model.fit(X_train, y_train)

    # Make predictions and evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy:.2f}')
    
    # if specified, print the classification report
    if print_report:
        print('Classification Report:')
        print(report)
        print('Read more about the classification report: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html')

    return None
