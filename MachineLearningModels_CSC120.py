from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from waitress import serve


 ## to ignore waarning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load your dataset
dataset = pd.read_csv('drugs_datasets.csv')  # Replace 'drugs_datasets.csv' with your actual dataset file

# Define the reinforcement learning agent
class DrugRecommendationAgent:
    def __init__(self, dataset):
        self.dataset = dataset
    
    def recommend_drug(self, condition):
        condition = condition.lower()
        matching_drugs = self.dataset.loc[self.dataset['condition'].str.lower().eq(condition), 'drugName'].unique()
        
        if matching_drugs.any():
            return matching_drugs.tolist()
        else:
            return ['No drugs found for the given condition.']
        
# Instantiate the recommendation agent
agent = DrugRecommendationAgent(dataset)


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/linear_regression')
def linear_regression():
    return render_template('linear_regression.html')

@app.route('/linear_regression_results')
def linear_regression_results():
    return render_template('linear_regression_results.html')

@app.route('/linear_regression_ui', methods=['POST', 'Get'])
def linear_regression_ui():
    # Read the data file
    file_name = 'advertising_datasets.csv'  # Replace with your file name
    data = pd.read_csv(file_name)

    #Initializing the variables

    X = data[['Tv', 'Radio', 'Newspaper']]
    y = data['Sales'].values.reshape(-1,1)

    # Drop rows with NaN values
    data.dropna(axis=0, inplace=True)

    #Splitting our dataset to Training and Testing dataset
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Fitting Linear Regression to the training set
    from sklearn.linear_model import LinearRegression
    multiple_reg = LinearRegression()
    multiple_reg.fit(X_train, y_train)

    #predicting the Test set result
    y_pred = multiple_reg.predict(X_test)

    #Calculating the Coefficients
    multiple_reg.coef_

    #Calculating the Intercept
    multiple_reg.intercept_

    #Calculating the R squared value
    from sklearn.metrics import r2_score
    r2_score(y_test, y_pred)

    # Calculating the Coefficients
    coefficients = multiple_reg.coef_

    # Calculating the Intercept
    intercept = multiple_reg.intercept_

    # Calculating the R squared value
    r2 = r2_score(y_test, y_pred)

    if request.method == 'POST':
        tv_budget = float(request.form['tv_budget'])
        radio_budget= float(request.form['radio_budget'])
        newspapers_budget = float(request.form['newspapers_budget'])   

        # Create a feature vector for prediction
        #input_data = np.array([[tv_budget, radio_budget, newspapers_budget]])
        # Make a prediction
        prediction = multiple_reg.predict([[tv_budget, radio_budget, newspapers_budget]])[0]

        return render_template('linear_regression_ui.html', prediction=prediction, tv_budget=tv_budget, radio_budget=radio_budget,
                                   newspapers_budget=newspapers_budget, )
    else:
        # Handle the GET request for rendering the form
        return render_template('linear_regression_ui.html')

@app.route('/logistic_regression')
def logistic_regression():
    return render_template('logistic_regression.html')

#Read the data file and preprocess the data
file_name = 'LA.csv'  # Replace with your file name
df = pd.read_csv(file_name)
df = df.dropna()

le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'].astype(str))
df['Married'] = le.fit_transform(df['Married'].astype(str))
df['Dependents'] = le.fit_transform(df['Dependents'].astype(str))
df['Education'] = le.fit_transform(df['Education'].astype(str))
df['Self_Employed'] = le.fit_transform(df['Self_Employed'].astype(str))
df['Credit_History'] = le.fit_transform(df['Credit_History'].astype(str))
df['Property_Area'] = le.fit_transform(df['Property_Area'].astype(str))
df['Loan_Status'] = le.fit_transform(df['Loan_Status'].astype(str))

# Calculate loan amount ratio and total income ratio for each row
df['LoanAmountRatio'] = 1 - df['LoanAmount'] / df['Total_Income']
df['TotalIncomeRatio'] = (df['Total_Income'] - df['LoanAmount']) / df['Total_Income']


X = df[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'TotalIncomeRatio',
        'LoanAmountRatio', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']].values
y = df['Loan_Status'].values

X = np.nan_to_num(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = SGDClassifier(loss='log', penalty='elasticnet', alpha=0.0001, l1_ratio=0.15, max_iter=1000, tol=1e-3)
clf.fit(X_train, y_train)

@app.route('/logistic_regression_results')
def logistic_regression_results():
    return render_template('logistic_regression_results.html')

@app.route('/logistic_regression_ui', methods=['POST', 'GET'])
def logistic_regression_ui():
    if request.method == 'POST':
        gender = request.form['gender']
        married = request.form['married']
        dependents = request.form['dependents']
        education = request.form['education']
        self_employed = request.form['self_employed']
        total_income = float(request.form['total_income'])
        loan_amount = float(request.form['loan_amount'])
        loan_term = float(request.form['loan_term'])
        credit_history = request.form['credit_history']
        property_area = request.form['property_area']

        loan_amount_ratio =  (1 - loan_amount / total_income) *1.1
        total_income_ratio = ((total_income - loan_amount) / total_income)*1.1

        input_data = np.array([[gender, married, dependents, education,
                                self_employed, total_income_ratio, loan_amount_ratio, loan_term,
                                credit_history, property_area]])

        input_data = scaler.transform(input_data)

        prediction_proba = clf.predict_proba(input_data)[0]
        formatted_proba = ["{:.5f}".format(prob) for prob in prediction_proba]
        prediction = "Loan Application Approved" if prediction_proba[1] >= 0.80 else "Loan Application Rejected"

        return render_template('logistic_regression_ui.html', gender=gender, married=married, dependents=dependents,
                               education=education, self_employed=self_employed, total_income=total_income,
                               loan_amount=loan_amount, loan_term=loan_term, credit_history=credit_history,property_area=property_area,
                               formatted_proba=formatted_proba, prediction=prediction)
    else:
        # Handle the GET request for rendering the form
        return render_template('logistic_regression_ui.html')

        
@app.route('/neural_network')
def neural_network():
    return render_template('neural_network.html')

@app.route('/neural_network_results')
def neural_network_results():
    return render_template('neural_network_results.html')

@app.route('/neural_network_ui', methods=['POST', 'Get'])
def neural_network_ui():
    # Load the dataset
    file_name = 'PCOS_datasets.csv'  # Replace with your file name
    data = pd.read_csv(file_name)

    X = data[['Age', 'Weight', 'Height', 'BMI', 'BloodGroup', 'PulseRate', 'RR', 'HB', 'Cycle', 'CycleLength', 'MarraigeStatus', 'Pregnant', 'Noofabortions', 'IbetaHCG', 'FSH', 'LH', 'FSHLH', 'Hip', 'Waist', 'WaistHipRatio', 'Weightgain', 'hairgrowth', 'Skindarkening', 'Hairloss', 'Pimples', 'Fastfood', 'RegExercise', 'BP _Systolic', 'BP _Diastolic', 'TSH', 'AMH', 'PRL', 'VitD3', 'PRG']]
    Y = data['PCOS'].values

    # Handle missing values
    X = np.nan_to_num(X)

    # Scale data using RobustScaler
    scaler = RobustScaler()
    X = scaler.fit_transform(X)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Define the feedforward neural network model
    model = Sequential([
        Dense(32, activation='relu', input_shape=(X.shape[1],)),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

    # Evaluate the model on test data
    y_pred_probs = model.predict(X_test)
    y_pred = np.round(y_pred_probs).flatten()

    if request.method == 'POST':
        age = float(request.form['age']) 
        weight =float(request.form['weight'])
        height = float(request.form['height'])
        bmi = float(request.form['bmi'])
        bg = float(request.form['bg'])
        pr = float(request.form['pr'])
        rr = float(request.form['rr'])
        hb = float(request.form['hb'])
        cc = float(request.form['cc'])
        cl = float(request.form['cl'])
        ms = float(request.form['ms'])
        pregnant = float(request.form['pregnant'])
        noa = float(request.form['noa'])
        ib = float(request.form['ib'])
        fsh = float(request.form['fsh'])
        lh = float(request.form['lh'])
        fshlh = float(request.form['fshlh'])
        hip = float(request.form['hip'])
        w = float(request.form['w'])
        whr = float(request.form['whr'])
        wg = float(request.form['wg'])
        hg = float(request.form['hg'])
        sd = float(request.form['sd'])
        hlss = float(request.form['hlss'])
        pim = float(request.form['pim'])
        ff = float(request.form['ff'])
        ex = float(request.form['ex'])
        bps = float(request.form['bps'])
        bpd = float(request.form['bpd'])
        tsh = float(request.form['tsh'])
        amh = float(request.form['amh'])
        prl = float(request.form['prl'])
        vit =  float(request.form['vit'])
        prg = float(request.form['prg'])

        # Create a feature vector for prediction
        input_data = np.array([[age,weight , height, bmi,
                                bg, pr, rr, hb,cc, cl, ms, pregnant, noa, ib, fsh, lh,  fshlh, hip, w, whr, wg, hg, sd,hlss,
                                pim, ff, ex, bps,  bpd, tsh , amh, prl, vit, prg]])

        # Scale the input data
        input_data = scaler.transform(input_data)

        # Make a prediction
        prediction_proba = model.predict(input_data)[0]
        formatted_proba = "{:.5f}".format(prediction_proba[0])
        prediction = "Positive for PCOS" if prediction_proba[0] >= 0.5 else "Negative for PCOS"


        return render_template('neural_network_ui.html', prediction=prediction, formatted_proba=formatted_proba)
    else:
        # Handle the GET request for rendering the form
        return render_template('neural_network_ui.html')

@app.route('/deep_learning')
def deep_learning():
    return render_template('deep_learning.html')

@app.route('/deep_learning_ui', methods=['POST', 'GET'])
def deep_learning_ui():
    file_name = 'DeepLearningDatasetsFraudDetection.csv'  # Replace with your file name
    df = pd.read_csv(file_name)

    # Drop rows with NaN values
    df.dropna(axis=0, inplace=True)

    df.replace(to_replace=['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'], value=[1, 4, 2, 5, 3], inplace=True)

    df['nbo_amount'] = df['obo'] - df['amount']

    # Identify nbo as fraud when nbo_amount is not equal to nbo
    df['isFraud'] = np.where(df['nbo_amount'] != df['nbo'], 1, 0)

    # Split the dataset into features (X) and target (y)
    X = df[['typ', 'amount', 'obo', 'nbo', 'obd', 'nbd', 'isFraud']]
    y = df['isFraud']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # Normalize the input features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define the DNN model
    model = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=1, batch_size=32)

    if request.method == 'POST':
        typ = request.form['typ']
        amount = float(request.form['amount'])
        obo = float(request.form['obo'])
        nbo = float(request.form['nbo'])
        obd = float(request.form['obd'])
        nbd = float(request.form['nbd'])

        nbo_amount = obo - amount

        # Identify nbo as fraud when nbo_amount is not equal to nbo
        is_fraud = int(nbo_amount != nbo)

        # Create a feature vector for prediction
        input_data = np.array([[typ, amount, obo, nbo, obd, nbd, is_fraud]])

        # Scale the input data
        input_data = scaler.transform(input_data)

        # Make a prediction
        prediction_proba = model.predict(input_data)[0]
        formatted_proba = "{:.5f}".format(prediction_proba[0])
        prediction = "Fraud Detected" if prediction_proba[0] >= 0.5 else "Not Fraud"

        return render_template('deep_learning_ui.html', formatted_proba=formatted_proba, prediction=prediction, typ=typ, amount=amount, obo=obo, nbo=nbo, obd=obd, nbd=nbd)
    else:
        # Handle the GET request for rendering the form
        return render_template('deep_learning_ui.html')

@app.route('/rl')
def rl():
    return render_template('rl.html')


import random

# Load your dataset
dataset = pd.read_csv('sample10.csv')  # Replace 'sample10.csv' with your actual dataset file

# Define the reinforcement learning agent
class DrugRecommendationAgentRL:
    def __init__(self, dataset):
        self.dataset = dataset
        self.q_table = {}  # Add a Q-table attribute for the agent
        self.learning_rate = 0.1  # Set the learning rate
        self.discount_factor = 0.9  # Set the discount factor
        self.epsilon = 0.2  # Set the exploration rate
        self.previous_recommendations = []  # Store previous recommendations
        self.high_rated_drug = None  # Store the highest rated drug from the previous round
        self.rating_threshold = 5  # Set the rating threshold for drug recommendation
        self.avoided_drugs = []  # Store drugs to be avoided due to low ratings
        self.cumulative_reward = 0  # Store cumulative reward for regret computation
        self.regret = 0
        
    def recommend_drug(self, condition, user_rating=None):
        condition = condition.lower()

        if random.random() < self.epsilon:
            # Explore: Select a random drug
            matching_drugs = self.get_drugs_for_condition(condition)
            if not matching_drugs.empty:
                # Remove drugs with a rating below the threshold that were recommended in the previous round
                matching_drugs = matching_drugs[~matching_drugs['drugName'].isin(self.previous_recommendations)]
                
                # Remove drugs that have been avoided due to low ratings
                matching_drugs = matching_drugs[~matching_drugs['drugName'].isin(self.avoided_drugs)]
                
                # Add the highest rated drug from the previous round if available
                if self.high_rated_drug is not None and self.high_rated_drug not in matching_drugs['drugName'].tolist():
                    matching_drugs = matching_drugs.append({'drugName': self.high_rated_drug}, ignore_index=True)
                
                sample_size = min(3, len(matching_drugs))  # Ensure sample size doesn't exceed the available drugs
                
                # Filter out blank recommendations
                recommendations = random.sample(matching_drugs['drugName'].tolist(), k=sample_size)
                recommendations = [r for r in recommendations if r.strip()]
                
                if recommendations:
                    return recommendations
                else:
                    return ['No drugs found for the given condition.']
            else:
                return ['No drugs found for the given condition.']
        else:
            # Exploit: Select drugs randomly without considering ratings
            matching_drugs = self.get_drugs_for_condition(condition)
            if not matching_drugs.empty:
                # Remove drugs with a rating below the threshold that were recommended in the previous round
                matching_drugs = matching_drugs[~matching_drugs['drugName'].isin(self.previous_recommendations)]
                
                # Remove drugs that have been avoided due to low ratings
                matching_drugs = matching_drugs[~matching_drugs['drugName'].isin(self.avoided_drugs)]
                
                # Add the highest rated drug from the previous round if available
                if self.high_rated_drug is not None and self.high_rated_drug not in matching_drugs['drugName'].tolist():
                    matching_drugs = matching_drugs.append({'drugName': self.high_rated_drug}, ignore_index=True)
                
                sample_size = min(3, len(matching_drugs))  # Ensure sample size doesn't exceed the available drugs
                
                # Filter out blank recommendations
                recommendations = random.sample(matching_drugs['drugName'].tolist(), k=sample_size)
                recommendations = [r for r in recommendations if r.strip()]
                
                if recommendations:
                    return recommendations
                else:
                    return ['No drugs found for the given condition.']
            else:
                return ['No drugs found for the given condition.']
    
    def update(self, state, action, reward, next_state):
        # Update Q-value based on the observed experience
        if state not in self.q_table:
            self.q_table[state] = {}

        if isinstance(action, list):
            actions = action
        else:
            actions = [action]

        for action in actions:
            action = tuple(action)  # Convert action to a tuple
            if action not in self.q_table[state]:
                self.q_table[state][action] = 0

            current_q_value = self.q_table[state][action]

            next_state_str = str(next_state)  # Convert the next_state list to a string
            if next_state_str not in self.q_table:
                self.q_table[next_state_str] = {}

            next_q_values = self.q_table[next_state_str]
            max_next_q_value = max(next_q_values.values()) if next_q_values else 0

            # Adjust the learning rate based on the reward
            if reward < 0:
                # Apply a higher learning rate for negative rewards
                adjusted_learning_rate = self.learning_rate * 2.0
            else:
                adjusted_learning_rate = self.learning_rate

            new_q_value = (1 - adjusted_learning_rate) * current_q_value + adjusted_learning_rate * (
                    reward + self.discount_factor * max_next_q_value)

            self.q_table[state][action] = new_q_value
            
            # Update cumulative reward
            self.cumulative_reward += reward
           
    def get_drugs_for_condition(self, condition):
        return self.dataset.loc[self.dataset['condition'].str.lower().eq(condition)]

# Instantiate the recommendation agent with reinforcement learning
agent = DrugRecommendationAgentRL(dataset)

regret_list = []  # Store the regret at each step

@app.route('/deep_learning_results')
def deep_learning_results():
    return render_template('deep_learning_results.html')

@app.route('/rl_ui', methods=['POST', 'Get'])
def rl_ui():
    condition = ""
    recommended_drugs = []
    regret = 0  # Initialize regret to 0
    
    if request.method == 'POST':
        condition = request.form['condition']
        
        if 'user_rating' in request.form:
            user_rating = int(request.form['user_rating'])
        else:
            user_rating = None
        
        recommended_drugs = agent.recommend_drug(condition, user_rating)

        if 'selected_drug' in request.form and 'rating' in request.form:
            selected_drug = request.form['selected_drug']
            rating = int(request.form['rating'])
            
            # Calculate regret
            optimal_reward = agent.cumulative_reward + max(agent.q_table.get(str(condition), {}).values(), default=0)
            selected_reward = agent.cumulative_reward - rating
            regret = optimal_reward - selected_reward
            regret_list.append(regret)
            agent.regret += regret 
            
            # Update the agent's state based on the rating
            if rating < agent.rating_threshold:
                agent.avoided_drugs.append(selected_drug)

            agent.update(condition, selected_drug, rating, recommended_drugs)
        else:
            # User hasn't provided a rating yet, display the recommend_drugs template
            return render_template('recommend_drug.html', recommended_drugs=recommended_drugs, condition=condition)
    return render_template('rate_drug.html', recommended_drugs=recommended_drugs, condition=condition, regret=regret)

@app.route('/rl_index')
def rl_index():
    return render_template('rl_index.html')

@app.route('/rl_reset')
def rl_reset():
    global agent, regret_list
    agent = DrugRecommendationAgentRL(dataset)
    regret_list = []
    return render_template('rl_index.html')

@app.route('/rl_results')
def rl_results():
    return render_template('rl_results.html', regret_list=regret_list)

# Run the Flask application
if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8080)


