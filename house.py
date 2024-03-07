import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import skew
from sklearn.preprocessing import LabelEncoder
import joblib
import streamlit as st

#-----------------------------------------------------------------------------------------------
# Kullanisli Fonksiyonlar Tanimladim.

def read_data(file_path):
    return pd.read_csv(file_path)

def drop_rows(data_frame, indices):
    data_frame.drop(indices, axis=0, inplace=True)

def apply_log_transform(data_frame, column):
    data_frame[column] = np.log1p(data_frame[column])

def fill_na_with_value(data_frame, column, value):
    data_frame[column] = data_frame[column].fillna(value)

def fill_na_with_mode(data_frame, column):
    mode_value = data_frame[column].mode().iloc[0]
    data_frame[column] = data_frame[column].fillna(mode_value)

def label_encode_categorical_columns(data_frame, columns):
    for col in columns:
        lbl = LabelEncoder() 
        data_frame[col] = lbl.fit_transform(data_frame[col].astype(str))

def create_dummy_variables(data_frame):
    return pd.get_dummies(data_frame)

def train_xgboost_model(X, y, params):
    model = xgb.XGBRegressor(**params)
    model.fit(X, y)
    return model

def save_model(model, file_name):
    joblib.dump(model, file_name)

def save_feature_names(feature_names, file_name):
    joblib.dump(feature_names, open(file_name, "wb"))


#----------------------------------------------------------------------------------
# Dosyayi Yukleme kismi
file_path = "train.csv"
train_data = read_data(file_path)

# Outlier Dropladim
drop_rows(train_data, [524, 1299])

# 'SalePrice' sutununa Log transform
apply_log_transform(train_data, "SalePrice")

# Bos Deger Doldurma kismi
columns_to_fill_with_none = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
for col in columns_to_fill_with_none:
    fill_na_with_value(train_data, col, "None")

for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    fill_na_with_value(train_data, col, 'None')
    
for col in ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
            'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']:
    fill_na_with_value(train_data, col, 0)
    
for col in ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']:
    fill_na_with_value(train_data, col, 'None')
    
columns_with_value_fill = ['MasVnrType', 'MasVnrArea', 'MSZoning', 'Functional', 'Electrical',
                            'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 'MSSubClass']
for col in columns_with_value_fill:
    fill_na_with_value(train_data, col, train_data[col].mode().iloc[0])

# Yeni sutun olusturma kismi
train_data['TotalSF'] = train_data['TotalBsmtSF'] + train_data['1stFlrSF'] + train_data['2ndFlrSF']

# Once Label Encoder Uyguladim
categorical_columns = ['FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 'ExterQual',
                        'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 'BsmtFinType2',
                        'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope', 'LotShape',
                        'PavedDrive', 'Street', 'Alley', 'CentralAir']
label_encode_categorical_columns(train_data, categorical_columns)

# Sonrasinda Onehot Encoder Uyguladim.
train_data = create_dummy_variables(train_data)

# Katsayisi en fazla olan 14 Sutunu tanimladim.
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

# Veriyi X_train ve y_train olarak boldum.
X_train = train_data[feature_names]
y_train = train_data["SalePrice"]

# En iyi performans veren modelim
xgboost_params = {
    'colsample_bytree': 0.4603,
    'gamma': 0.0468,
    'learning_rate': 0.1,
    'max_depth': 3,
    'min_child_weight': 1.7817,
    'n_estimators': 2200,
    'reg_alpha': 0.4640,
    'reg_lambda': 0.8571,
    'subsample': 0.5213,
    'random_state': 7,
    'nthread': -1
}

# Fit ediyoruz.
trained_model = train_xgboost_model(X_train, y_train, xgboost_params)

# Joblib Dosyasi Kaydetme Kismi
save_model(trained_model, 'new_model_xgb.joblib')
save_feature_names(feature_names, 'feature_names.joblib')

# ---------------------------------------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import skew
from sklearn.preprocessing import LabelEncoder
import joblib

# Kaydettigim Joblib dosyalarini Cagirmak icin
# Önbellek kullanımı
@st.cache_data(persist=True)
def load_model_and_features():
    model = joblib.load('new_model_xgb.joblib')
    feature_names = joblib.load('feature_names.joblib')
    
    return model, feature_names

@st.cache_data(hash_funcs={xgb.sklearn.XGBRegressor: id})
def predict_sale_price(model, input_data):
    label_encoder = LabelEncoder()

    for feature in feature_names:
        if input_data[feature].dtype == 'object':
            input_data[feature] = label_encoder.fit_transform(input_data[feature].astype(str))

    prediction = model.predict(input_data)
    return np.expm1(prediction)  # Reverse the log transformation

# Sutunlarin Aciklama Kismi icin
@st.cache_data(persist=True)
def get_feature_descriptions():
    return {
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
    feature_descriptions = get_feature_descriptions()
    for feature, description in feature_descriptions.items():
        st.sidebar.markdown(f"**{feature}**: {description}")

    # Inputlari girdigimiz yerin Duzenlemesi ve Ayarlanmasi
    input_features = {}
    for feature in feature_names:
        if feature == 'OverallQual':
            input_features[feature] = st.slider(f"Select {feature}", 1, 10, value=5)
        elif feature == 'GarageType_Attchd':
            input_features[feature] = st.selectbox(f"Select {feature}", ['0', '1'])
        elif feature == 'BsmtQual':
            input_features[feature] = st.selectbox(f"Select {feature}", ['0', '1', '2', '3', '4'])
        elif feature == 'GarageCars':
            input_features[feature] = st.number_input(f"Enter {feature}", value=0, step=1)
        elif feature in ["FullBath", "Fireplaces"]:
            input_features[feature] = st.number_input(f"Enter {feature}", value=0, step=1)
        elif feature == 'PavedDrive':
            input_features[feature] = st.selectbox(f"Select {feature}", [0, 1])
        elif feature == 'ExterQual':
            input_features[feature] = st.slider(f"Select {feature}", 1, 3, value=5)
        elif feature == 'YearRemodAdd':
            input_features[feature] = st.number_input(f"Enter {feature}", value=0, step=1)
        elif feature == 'CentralAir':
            input_features[feature] = st.selectbox(f"Select {feature}", [0, 1])
        elif feature == 'KitchenQual':
            input_features[feature] = st.slider(f"Select {feature}", 1, 3, value=5)
        elif feature == 'MSZoning_RL':
            input_features[feature] = st.selectbox(f"Select {feature}", [0, 1])
        else:
            input_features[feature] = st.number_input(f"Enter {feature}", value=0.0)

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
