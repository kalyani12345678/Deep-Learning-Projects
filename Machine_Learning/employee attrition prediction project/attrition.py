import pandas as pd
import numpy as np
import pickle
import streamlit as st

# Define mappings for categorical features
education_mapping = {'Life Sciences': 0, 'Other': 1, 'Medical': 2, 'Marketing': 3, 'Technical Degree': 4, 'Human Resources': 5}
marital_status_mapping = {'Divorced': 0, 'Married': 1, 'Single': 2}
over_time_mapping = {'No': 0, 'Yes': 1}
job_role_mapping = {'Healthcare Representative': 0, 'Human Resources': 1, 'Laboratory Technician': 2, 
                    'Manager': 3, 'Manufacturing Director': 4, 'Research Director': 5, 'Research Scientist': 6,
                    'Sales Executive': 7, 'Sales Representative': 8}

# loading the saved model
loaded_model = pickle.load(open('/content/xgboost_model.pkl', 'rb'))

# creating a function for Prediction
def attrition_prediction(input_data):
    # Convert input_data to DataFrame with correct feature names
    input_df = pd.DataFrame([input_data], columns=['YearsWithCurrManager', 'EducationField', 'Age', 'YearsSinceLastPromotion', 'TotalWorkingYears',
                    'YearsAtCompany', 'MaritalStatus', 'StockOptionLevel', 'MonthlyIncome', 'JobSatisfaction',
                    'JobInvolvement', 'OverTime', 'DistanceFromHome', 'JobRole', 'DailyRate', 'TrainingTimesLastYear',
                    'EnvironmentSatisfaction', 'YearsInCurrentRole', 'NumCompaniesWorked', 'JobLevel', 'WorkLifeBalance'])

    # Convert string columns to numeric data types
    input_df = input_df.apply(pd.to_numeric)

    # Make predictions using the loaded model
    prediction = loaded_model.predict(input_df)
    if prediction[0] == 0:
        return 'The employee is likely to stay.'
    else:
        return 'The employee is likely to leave.'

def main():
    # giving a title
    st.title('Employee Attrition Prediction')
    # getting the input data from the user
    YearsWithCurrManager = st.text_input('Years With Current Manager')
    EducationField = st.selectbox('Education Field', options=list(education_mapping.keys()))
    Age = st.text_input('Age')
    YearsSinceLastPromotion = st.text_input('Years Since Last Promotion')
    TotalWorkingYears = st.text_input('Total Working Years')
    YearsAtCompany = st.text_input('Years at Company')
    MaritalStatus = st.selectbox('Marital Status', options=list(marital_status_mapping.keys()))
    StockOptionLevel = st.text_input('Stock Option Level')
    MonthlyIncome = st.text_input('Monthly Income')
    JobSatisfaction = st.text_input('Job Satisfaction')
    JobInvolvement = st.text_input('Job Involvement')
    OverTime = st.selectbox('Over Time', options=list(over_time_mapping.keys()))
    DistanceFromHome = st.text_input('Distance From Home')
    JobRole = st.selectbox('Job Role', options=list(job_role_mapping.keys()))
    DailyRate = st.text_input('Daily Rate')
    TrainingTimesLastYear = st.text_input('Training Times Last Year')
    EnvironmentSatisfaction = st.text_input('Environment Satisfaction')
    YearsInCurrentRole = st.text_input('Years In Current Role')
    NumCompaniesWorked = st.text_input('Number of Companies Worked')
    JobLevel = st.text_input('Job Level')
    WorkLifeBalance = st.text_input('Work Life Balance')

    # code for Prediction
    result = ''
    
    # creating a button for Prediction
    if st.button('Predict Attrition'):
        # Check if any field is empty
        if (not YearsWithCurrManager or not Age or not YearsSinceLastPromotion or not TotalWorkingYears or
            not YearsAtCompany or not StockOptionLevel or not MonthlyIncome or not JobSatisfaction or
            not JobInvolvement or not DistanceFromHome or not DailyRate or not TrainingTimesLastYear or
            not EnvironmentSatisfaction or not YearsInCurrentRole or not NumCompaniesWorked or not JobLevel or not WorkLifeBalance):
            result = "Please fill in all the required fields."
        else:
            # Prepare input data as a list
            input_data = [YearsWithCurrManager, education_mapping[EducationField], Age, YearsSinceLastPromotion, TotalWorkingYears,
                          YearsAtCompany, marital_status_mapping[MaritalStatus], StockOptionLevel, MonthlyIncome, JobSatisfaction,
                          JobInvolvement, over_time_mapping[OverTime], DistanceFromHome, job_role_mapping[JobRole], DailyRate, TrainingTimesLastYear,
                          EnvironmentSatisfaction, YearsInCurrentRole, NumCompaniesWorked, JobLevel, WorkLifeBalance]

            # Make prediction
            result = attrition_prediction(input_data)
        
    st.success(result)
    

if __name__ == '__main__':
    main()
