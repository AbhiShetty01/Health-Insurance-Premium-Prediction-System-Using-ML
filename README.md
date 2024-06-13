# Insurance-Premium-Prediction
This project aims to predict the insurance premium of individuals based on various factors such as age, gender, BMI, smoking habit, region, and number of children.

# Dataset           
The dataset used in this project is the "Medical Cost Personal Datasets" available on Kaggle. It contains information about 1338 individuals including their age, sex, BMI, children, smoker status, region, and medical charges.              

# Requirements          
Python 3              
Jupyter Notebook            
Pandas                   
NumPy                            
Matplotlib                        
Seaborn                            
Scikit-learn                               

# Methodology                                                      
The project uses a machine learning model to predict the insurance premium of individuals. The following steps were taken:                             

Data Preprocessing: The dataset was cleaned and preprocessed by handling missing values, encoding categorical variables, and scaling numerical variables.       

Exploratory Data Analysis: The dataset was explored using visualizations to gain insights into the relationships between variables and their impact on insurance charges.

Feature Selection: The most important features were selected using various feature selection techniques such as correlation matrix, recursive feature elimination, and feature importance.                                        

Model Selection: Several machine learning algorithms were evaluated and compared to select the best model for the dataset.                            

Model Tuning: The hyperparameters of the selected model were tuned using grid search and cross-validation to improve its performance.                     

Model Evaluation: The performance of the final model was evaluated using various metrics such as mean absolute error, mean squared error, and R-squared.   

# Results                   
The final model achieved an R-squared value of 0.85, indicating that it can explain 85% of the variation in the insurance charges. The most important features for predicting insurance charges were age, BMI, and smoking habit. The model can be used to predict the insurance premium of individuals with a reasonable degree of accuracy.   


