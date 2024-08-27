import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import LabelEncoder

# Load data
data = pd.read_csv('flip_data_cleaned.csv')  # Update with the correct path

# Load trained model
model = pickle.load(open('flip_model.pkl', 'rb'))  # Update with the correct path

st.title('Mobile Phone Price Prediction Dashboard')

st.sidebar.header('User Input Features')
st.write("### Data Overview")
st.dataframe(data.head())

# Exclude non-numeric columns for pairplot and correlation
numeric_data = data.select_dtypes(include=[float, int])

st.write("### Data Distribution")
pairplot_fig = sns.pairplot(numeric_data)
st.pyplot(pairplot_fig)

st.write("### Feature Correlation")
corr = numeric_data.corr()

fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

def user_input_features():
    # User input
    brand = st.sidebar.selectbox('Brand', options=['Samsung', 'Apple', 'Realme', 'OnePlus', 'Xiaomi', 'Others'])
    ram = st.sidebar.slider('RAM (in GB)', min_value=2, max_value=12, step=1)
    storage = st.sidebar.slider('Storage (in GB)', min_value=32, max_value=512, step=32)
    display_size = st.sidebar.slider('Display Size (in inches)', min_value=4.0, max_value=7.0, step=0.1)
    display_type = st.sidebar.selectbox('Display Type', options=['LCD', 'AMOLED', 'OLED'])
    rear_camera = st.sidebar.slider('Rear Camera (in MP)', min_value=5, max_value=108, step=5)
    front_camera = st.sidebar.slider('Front Camera (in MP)', min_value=2, max_value=40, step=2)

    # Create DataFrame
    data = {'brand': [brand],
            'ram': [ram],
            'storage': [storage],
            'display size': [display_size],
            'display type': [display_type],
            'rear camera': [rear_camera],
            'front camera': [front_camera]}
    features = pd.DataFrame(data)

    # Label encoding for categorical variables
    label_encoders = {}
    for column in ['brand', 'display type']:
        le = LabelEncoder()
        le.fit(data[column])
        features[column] = le.transform(features[column])
        label_encoders[column] = le

    return features

input_df = user_input_features()
st.write('### User Input Features')
st.write(input_df)

if st.button('Predict Price'):
    prediction = model.predict(input_df)
    st.write(f'### Predicted Price: ${prediction[0]:,.2f}')
