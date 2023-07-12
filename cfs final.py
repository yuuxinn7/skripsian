import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn_genetic import GAFeatureSelectionCV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import metrics
from imblearn.over_sampling import SMOTE
import pickle


def correlation_feature_selection(data, target_column, threshold):
    # Separate features and target variable
    features = data.drop(target_column, axis=1)
    target = data[target_column]

    # Calculate the correlation matrix
    corr_matrix = features.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)
    )

    # Find features with correlation above threshold
    correlated_features = [
        column for column in upper.columns if any(upper[column] > threshold)
    ]

    return correlated_features


# Fungsi untuk menyimpan akurasi ke file pickle
def save_accuracy(accuracy):
    with open("best_accuracy.pkl", "wb") as file:
        pickle.dump(accuracy, file)


# Fungsi untuk memuat akurasi dari file pickle
def load_accuracy():
    try:
        with open("best_accuracy.pkl", "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        return 0.0


# Membaca data dari file csv
df = pd.read_csv("diabetes.csv")

# Identify the 0 values in the independent variables
X = df.iloc[:, :-1]
X[X == 0] = pd.NA

# Initialize an imputer object with SimpleImputer to fill missing values with the mean value
imputer = SimpleImputer(strategy="mean")

# Apply imputation to the independent variables by calling the fit_transform method of the imputer object
X_imputed = imputer.fit_transform(X)

# Transform the independent variables using StandardScaler
scaler = StandardScaler()
X_transformed = scaler.fit_transform(X_imputed)

# Combine the imputed and transformed independent variables with the dependent variable
data_transformed = pd.concat(
    [pd.DataFrame(X_transformed, columns=X.columns), df.iloc[:, -1]], axis=1
)

# Mengambil data fitur dan label
X = np.array(data_transformed.drop(columns=["Outcome"]))
y = np.array(data_transformed["Outcome"])

# Set the target column name
target_column = "Outcome"

# Set the correlation threshold
threshold = 0.1

# Perform correlation-based feature selection
selected_features = correlation_feature_selection(
    data_transformed, target_column, threshold
)

print("\nSelected features by correlation: \n", selected_features)

# Create a new dataframe with selected features
df_new = data_transformed[selected_features]

# Adding Label
df_new.loc[:, "Outcome"] = y

# Use df_new for feature selection
X_i = df_new.drop(columns=["Outcome"])
y_i = df_new["Outcome"]

# Lakukan oversampling menggunakan SMOTE
oversampler = SMOTE(random_state=0)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_i, y_i)

noise = np.random.uniform(0, 2, size=(X_train_resampled.shape[0], 5))
X_train_resampled = np.hstack((X_train_resampled, noise))

# Perform feature selection using genetic algorithm
clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=15)
evolved_estimator = GAFeatureSelectionCV(
    estimator=clf,
    scoring="accuracy",
    population_size=10,
    generations=10,
    n_jobs=-1,
)

# Train and select the features
evolved_estimator.fit(X_train_resampled, y_train_resampled)

# Features selected by the algorithm
features = evolved_estimator.best_features_

# Split data after feature selection
X_train, X_test, y_train, y_test = train_test_split(
    X_train_resampled[:, np.array(features)],
    y_train_resampled,
    test_size=0.3,
    random_state=25,
)

# Train the Random Forest classifier on the selected features
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)

# Load the best accuracy from pickle file
best_accuracy = load_accuracy()

# Compare the accuracy with the best accuracy
if accuracy > best_accuracy:
    best_accuracy = accuracy
    save_accuracy(best_accuracy)
    print("Best accuracy updated!")

# Print the original accuracy and the best accuracy
print("Original Accuracy:", accuracy)
print("Best Accuracy:", best_accuracy)

