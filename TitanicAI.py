import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the training data
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
data = pd.read_csv(url)

# Drop columns that won't be used
data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Convert categorical variables into dummy/indicator variables
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

# Split the data into features and target variable
X = data.drop('Survived', axis=1)  # Features
y = data['Survived']  # Target variable

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Drop rows with any missing values in the training and test sets
X_train = X_train.dropna()
y_train = y_train[X_train.index]  # Keep y_train consistent with X_train
X_test = X_test.dropna()
y_test = y_test[X_test.index]  # Keep y_test consistent with X_test

# Initialize the model
model = LogisticRegression(max_iter=200)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Function for user input to predict survival
def predict_survival():
    print("\n--- Enter your details to check if you would have survived the Titanic ---")
    pclass = int(input("Enter your passenger class (1 = First, 2 = Second, 3 = Third): "))
    age = float(input("Enter your age: "))
    sibsp = int(input("Enter the number of siblings/spouses aboard: "))
    parch = int(input("Enter the number of parents/children aboard: "))
    fare = float(input("Enter the fare amount you paid: "))
    sex = input("Enter your sex (male/female): ").strip().lower()
    embarked = input("Enter your port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton): ").strip().upper()

    # Convert input to match the feature columns
    sex_male = 1 if sex == 'male' else 0
    embarked_Q = 1 if embarked == 'Q' else 0
    embarked_S = 1 if embarked == 'S' else 0

    # Create a DataFrame for the user's input
    user_data = pd.DataFrame([{
        'Pclass': pclass,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Sex_male': sex_male,
        'Embarked_Q': embarked_Q,
        'Embarked_S': embarked_S
    }])

    # Predict survival
    prediction = model.predict(user_data)
    prediction_prob = model.predict_proba(user_data)[0]

    # Output result
    if prediction[0] == 1:
        print("\nYou would have survived the Titanic!")
    else:
        print("\nUnfortunately, you would not have survived the Titanic.")
    
    print(f"Prediction confidence: Survived: {prediction_prob[1]:.2f}, Not survived: {prediction_prob[0]:.2f}")

# Call the function to allow user input
predict_survival()
