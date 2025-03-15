import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import datetime
import io
import os

def load_and_preprocess_data(file_path):
    try:
        df = pd.read_excel(file_path, engine='openpyxl')  # For modern Excel formats

        for col in df.columns:
            if df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col].fillna(df[col].mean(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)

        df['Login_Hour'] = pd.to_datetime(df['Login Time']).dt.hour
        df['Logout_Hour'] = pd.to_datetime(df['Logout Time']).dt.hour
        # df['Time_Spent_Hours'] = df['Time spent'] / 3600.0
        # df['Login Time'] = pd.to_datetime(df['Login Time'])
        # df['Logout Time'] = pd.to_datetime(df['Logout Time'])

        df['Time_spent'] = (df['Logout Time'] - df['Login Time']).dt.total_seconds() / 60  # Convert to minutes


        le_speciality = LabelEncoder()
        le_region = LabelEncoder()
        df['Speciality_Encoded'] = le_speciality.fit_transform(df['Speciality'])
        df['Region_Encoded'] = le_region.fit_transform(df['Region'])

        median_time = df['Time_spent'].median()
        median_attempts = df['Count of Survey Attempts'].median()
        df['Likely_Survey'] = ((df['Time_spent'] > median_time) | (df['Count of Survey Attempts'] > median_attempts)).astype(int)

        features = ['Speciality_Encoded', 'Region_Encoded', 'Login_Hour', 'Logout_Hour', 'Time_spent', 'Count of Survey Attempts']
        target = 'Likely_Survey'

        return df, features, target

    except Exception as e:
        st.error(f"Error loading/preprocessing data: {e}")
        return None, None, None

def train_model(df, features, target):
    try:
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        return model
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None

def predict_likely_doctors(model, df, features, input_hour):
    try:
        df['Input_Hour'] = input_hour
        df['Hour_Difference'] = abs(df['Login_Hour'] - df['Input_Hour'])
        df['Hour_Difference'] = df['Hour_Difference'].apply(lambda x: min(x, 24-x))
        df_filtered = df[df['Hour_Difference'] <= 2]

        X_predict = df_filtered[features]
        predictions = model.predict(X_predict)
        likely_doctors = df_filtered[predictions == 1]['NPI'].tolist()
        return likely_doctors

    except Exception as e:
        st.error(f"Error predicting likely doctors: {e}")
        return []

def main():
    st.title("Doctor Survey Campaign")

    file_path = r"D:\cognify\alpha\model\dummy_npi_data.xlsx"
    
    try:
        df = pd.read_excel(file_path, engine='openpyxl')  # Using openpyxl engine
        print("File loaded successfully!")
    except Exception as e:
        print(f"Error loading data: {e}")
# Replace with your file path.
    if not os.path.exists(file_path):
        st.error("Dataset file not found. Please ensure the file exists at the specified path.")
        return

    df, features, target = load_and_preprocess_data(file_path)

    if df is None or features is None or target is None:
        return

    model = train_model(df, features, target)

    if model is None:
        return

    input_time = st.time_input("Enter Time (HH:MM)")
    input_hour = input_time.hour

    if st.button("Get Likely Doctors"):
        likely_doctors = predict_likely_doctors(model, df, features, input_hour)

        if not likely_doctors:
            st.warning("No doctors found for the given time.")
            return

        result_df = pd.DataFrame({'NPI': likely_doctors})
        csv_buffer = io.StringIO()
        result_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        csv_data = csv_buffer.getvalue().encode()

        st.download_button(
            label="Download Likely Doctors (CSV)",
            data=csv_data,
            file_name="likely_doctors.csv",
            mime="text/csv",
        )
        st.dataframe(result_df)

if __name__ == "__main__":
    main()