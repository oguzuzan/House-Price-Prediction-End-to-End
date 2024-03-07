import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import skew
from sklearn.preprocessing import LabelEncoder
import joblib

feature_names = [
    'OverallQual',
    'GarageType_Attchd',
    'BsmtQual',
    'TotalSF',
    'GarageCars',
    'FullBath',
    'GrLivArea',
    'PavedDrive',
    'ExterQual',
    'Fireplaces',
    'YearRemodAdd',
    'CentralAir',
    'KitchenQual',
    'MSZoning_RL'
]

# Kaydettigim Joblib dosyalarini Cagirmak icin
def load_model_and_features():
    # Modeli cagiriyorum.
    model = joblib.load('new_model_xgb.joblib')

    # Sutun isimlerini cagiriyorum.
    feature_names = joblib.load('feature_names.joblib')

    return model, feature_names

def predict_sale_price(model, input_data):
    label_encoder = LabelEncoder()

    for feature in feature_names:
        if input_data[feature].dtype == 'object':
            input_data[feature] = label_encoder.fit_transform(input_data[feature].astype(str))

    prediction = model.predict(input_data)
    return np.expm1(prediction)  # Reverse the log transformation

# Sutunlarin Aciklama Kismi icin
feature_descriptions = {
    'OverallQual': 'Genel malzeme ve kaplama kalitesi',
    'GarageType_Attchd': 'Ek garaj Var ise 1 Yok ise 0',
    'BsmtQual': 'Bodrum katının yüksekliği (Yok ise 0)',
    'TotalSF': 'Evin Toplam Metrekaresi',
    'GarageCars': 'Garaja sığan araba sayısı',
    'FullBath': 'Banyo Sayısı',
    'GrLivArea': 'Ust kat var ise Metrekaresi Yok ise 0',
    'PavedDrive': 'Asfaltlanmış araba yolu var mi? Yoksa 0',
    'ExterQual': 'Dış malzeme kalitesi',
    'Fireplaces': 'Şömine sayısı',
    'YearRemodAdd': 'Tadilat tarihi (tadilat veya ekleme yoksa inşaat tarihiyle aynı)',
    'CentralAir': 'Klima Var mi?',
    'KitchenQual': 'Mutfak kalitesi',
    'MSZoning_RL': 'Ev Az nüfuslu Bir Yerde mi?'
}

# House App
def main():
    st.title("House Sale Price Prediction App")

    # Modeli Cagiriyoruz.
    trained_model, feature_names = load_model_and_features()

    # 2.Baslik
    st.header("User Input Features")

    # Sutunlarin Ne oldugunu aciklayan SideBar
    st.sidebar.title("Feature Descriptions")
    for feature, description in feature_descriptions.items():
        st.sidebar.markdown(f"**{feature}**: {description}")

    # Inputlari girdigimiz yerin Duzenlemesi ve Ayarlanmasi
    input_features = {}

    input_methods = {
    'OverallQual': lambda: st.slider(f"Select Overall Quality", 1, 10, value=5),
    'GarageType_Attchd': lambda: st.selectbox("Select GarageType_Attchd", ['0', '1']),
    'BsmtQual': lambda: st.selectbox("Select BsmtQual", ['0', '1', '2', '3', '4']),
    'GarageCars': lambda: st.number_input("Enter GarageCars", value=0, step=1),
    'FullBath': lambda: st.number_input("Enter FullBath", value=0, step=1),
    'Fireplaces': lambda: st.number_input("Enter Fireplaces", value=0, step=1),
    'PavedDrive': lambda: st.selectbox("Select Paved Drive", [0, 1]),
    'ExterQual': lambda: st.slider("Select Exter Quality", 1, 3, value=5),
    'YearRemodAdd': lambda: st.number_input("Enter YearRemodAdd", value=0, step=1),
    'CentralAir': lambda: st.selectbox("Select CentralAir", [0, 1]),
    'KitchenQual': lambda: st.slider("Select KitchenQual", 1, 3, value=5),
    'MSZoning_RL': lambda: st.selectbox("Select MSZoning_RL", [0, 1]),
}

    for feature in feature_names:
        input_features[feature] = input_methods.get(feature, lambda: st.number_input(f"Enter {feature}", value=0.0))()

    # Girdigimiz inputlari kaydeden bir DataFrame Olusturmak icin (Opsiyonel)
    input_df = pd.DataFrame([input_features])

    # Inputlari Tek bir sirada goruntulemek icin
    st.subheader("User Input Data")
    show_input_data = st.checkbox("Show Input Data", value=False)
    if show_input_data:
        st.write(input_df)

    # Predict tusu ve Predict
    if st.button("Predict Sale Price"):
        prediction = predict_sale_price(trained_model, input_df)
        st.success(f"The predicted sale price is ${prediction[0]:,.2f}")

if __name__ == "__main__":
    main()