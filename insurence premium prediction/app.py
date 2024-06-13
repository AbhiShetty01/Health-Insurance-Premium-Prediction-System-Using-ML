import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import random

# Load the model
model = joblib.load('model_joblib_test')

# Function to preprocess data
def preprocess_data(data):
    data['sex'] = data['sex'].map({'female': 0, 'male': 1})
    data['smoker'] = data['smoker'].map({'yes': 1, 'no': 0})
    data['region'] = data['region'].map({'southwest': 1, 'southeast': 2, 'northwest': 3, 'northeast': 4})
    return data

# Load dataset
data = pd.read_csv('insurance.csv')

# Preprocess dataset
data = preprocess_data(data)

X = data.drop(['charges'],axis=1)
y = data['charges']


# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
lr = LinearRegression()
lr.fit(X_train, y_train)

svm = SVR()
svm.fit(X_train, y_train)

rf = RandomForestRegressor()
rf.fit(X_train, y_train)

# Model names
models = {'Linear Regression': lr, 'Support Vector Machine': svm, 'Random Forest': rf}

# Function to calculate BMI
def calculate_bmi(weight, height):
    return weight / ((height / 100) ** 2)  # Convert height to meters

# Indian insurance provider names and cost ranges
insurance_providers = {
    'ICICI Lombard General Insurance': (1005, 20000),
    'HDFC ERGO General Insurance': (18000, 25000),
    'Bajaj Allianz General Insurance': (18000, 30000),
    'Reliance General Insurance': (12000, 22000),
    'Star Health and Allied Insurance': (9000, 18000),
    'SBI General Insurance': (19000, 38000),
    'United India Insurance Company': (12000, 20000),
    'Oriental Insurance Company': (13000, 23000),
    'New India Assurance Company': (14000, 25000),
    'National Insurance Company': (17000, 27000)
}

# Page configuration
st.set_page_config(page_title='Health Insurance Premium Prediction', page_icon=':hospital:', layout='centered')

# Sidebar
st.sidebar.image(r"helath.png", use_column_width=True)
st.sidebar.write("Health Insurance Premium Prediction")
st.sidebar.title('Navigation')
page = st.sidebar.radio("Go to", ('Home', 'Predict', 'BMI Calculator', 'Recommendation', 'About'))

# Home Page
if page == 'Home':
    st.markdown("<h1 style='color: #6495ED;'>Health Insurance Premium Prediction App</h1>", unsafe_allow_html=True)
    st.write("This app allows you to predict health insurance premiums based on various factors such as age, BMI, smoker status, and region.")
    st.write("Key Features:")
    st.write("- Predict insurance premiums using machine learning models.")
    st.write("- Explore insights from the dataset.")
    st.write("- Easy-to-use interface.")
    st.write("To get started, select an option from the sidebar.")

# Prediction Page
elif page == 'Predict':
    st.title("Insurance Cost Prediction")
    st.subheader("Choose Algorithm")
    model_name = st.selectbox("Select Model", list(models.keys()))

    model = models[model_name]

    age = st.number_input("Age", min_value=0, max_value=100, value=25)
    sex = st.selectbox("Sex", ["Male", "Female"])
    bmi = st.number_input("BMI (If unknown, please calculate BMI on the BMI Calculator page)", min_value=0, max_value=100, value=25)
    children = st.number_input("Children", min_value=0, max_value=10, value=0)
    smoker = st.selectbox("Smoker", ["Yes", "No"])
    region = st.selectbox("Region", ["Southwest", "Southeast", "Northwest", "Northeast"])

    if st.button("Predict"):
        sex_num = 1 if sex == "Male" else 0
        smoker_num = 1 if smoker == "Yes" else 0
        region_num = ['Southwest', 'Southeast', 'Northwest', 'Northeast'].index(region) + 1

        input_data = [[age, sex_num, bmi, children, smoker_num, region_num]]
        result = model.predict(input_data)
        st.session_state['result'] = result
        st.write("Insurance Cost: ₹", result[0])

    # Show accuracy
    st.subheader("Model Accuracy")
    st.write(f"{model_name} R^2 Score: {metrics.r2_score(y_test, model.predict(X_test))}")

# BMI Calculation Page
elif page == 'BMI Calculator':
    st.title("BMI Calculation")
    st.write("Calculate your Body Mass Index (BMI) :")
    st.latex(r'BMI = \frac{{\text{{weight in kilograms}}}}{{\left(\frac{{\text{{height in centimeters}}}}{{100}}\right)^2}}')

    # Input fields
    weight = st.number_input("Enter weight (kg)", min_value=0.0, step=0.1)
    height = st.number_input("Enter height (cm)", min_value=0.0, step=0.01)

    if st.button("Calculate"):
        bmi = calculate_bmi(weight, height)
        st.write(f"Your BMI is: {bmi:.2f}")

# Recommendation Page
elif page == 'Recommendation':
    st.title("Insurance Provider Recommendations")
    st.write("Based on your predicted insurance cost, here are some recommended insurance providers/plans:")

    # Get the predicted insurance cost from the prediction page
    result = None
    if 'result' in st.session_state:
        result = st.session_state['result']

    if result is not None:
        predicted_cost = result[0]

        # Filter insurance providers based on the predicted cost
        filtered_providers = {provider: cost_range for provider, cost_range in insurance_providers.items() if cost_range[0] <= predicted_cost <= cost_range[1]}

        # Display filtered insurance providers
        if filtered_providers:
            for provider, cost_range in filtered_providers.items():
                st.write(f"{provider}: ₹{cost_range[0]} to ₹{cost_range[1]}")
        else:
            st.write("No insurance providers found within the predicted cost range.")
    else:
        st.write("Please navigate to the 'Predict' page and click the 'Predict' button first.")

# About Page
elif page == 'About':
    st.title("About Health Insurance Premium Prediction App")
    
    st.header("Datasets")
    st.write("The dataset used for this project contains information about individuals and their respective health insurance premiums. It includes features such as age, sex, BMI, number of children, smoker status, and region.")
    st.write("You can download the dataset from the following link:")
    st.write("[Insurance Premium Prediction Dataset](https://www.kaggle.com/datasets/noordeen/insurance-premium-prediction)")
    
    st.header("Requirements")
    st.write("The following packages and software are required to run this Streamlit app:")
    st.write("- Python 3")
    st.write("- Jupyter Notebook (optional, for development)")
    st.write("- Pandas")
    st.write("- NumPy")
    st.write("- Matplotlib")
    st.write("- Seaborn")
    st.write("- Scikit-learn")
    st.write("- Streamlit")
    st.write("You can install these packages using pip:\n`pip install pandas numpy matplotlib seaborn scikit-learn streamlit`")
    
    
    st.header("Methodology")
    st.write("The methodology involves the following steps:")
    st.write("1. Data Preprocessing: The dataset is preprocessed to handle categorical variables, null values, and feature scaling.")
    st.write("2. Model Training: Three machine learning models (Linear Regression, Support Vector Machine, Random Forest) are trained on the preproces data.")
    st.write("3. Model Evaluation: The trained models are evaluated using performance metrics such as R^2 score and mean absolute error.")
    
    st.header("Results")
    st.write("The results of the model evaluation are as follows:")
    st.write("- Linear Regression R^2 Score: 0.783")
    st.write("- Support Vector Machine R^2 Score: -0.072")
    st.write("- Random Forest R^2 Score: 0.867")