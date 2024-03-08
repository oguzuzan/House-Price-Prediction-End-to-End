import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import skew
from sklearn.preprocessing import LabelEncoder
import joblib


model = joblib.load('new_model_xgb.joblib')
feature_names = joblib.load('feature_names.joblib')

def predict_sale_price(model, input_data):
    label_encoder = LabelEncoder()

    for feature in feature_names:
        if input_data[feature].dtype == 'object':
            input_data[feature] = label_encoder.fit_transform(input_data[feature].astype(str))

    prediction = model.predict(input_data)
    return np.expm1(prediction)  # Reverse the log transformation


# House App
def main():
    st.title("House Sale Price Prediction App")

    # Modeli Cagiriyoruz.
    trained_model = model
    feature_names = joblib.load('feature_names.joblib') 

    # Resim eklemek için st.image fonksiyonunu kullanın
    st.image("house.png", use_column_width=True)
    # 2.Baslik
    st.header("Evin Bilgilerini Giriniz.")

    # Inputlari girdigimiz yerin Duzenlemesi ve Ayarlanmasi
    input_features = {}
    for feature in feature_names:
        if feature == 'OverallQual':
            input_features[feature] = st.slider("Genel malzeme ve kaplama kalitesi: (1-10)", 1, 10, value=10)
            puanlar_metin = {
            10: "Very Excellent",
            9: "Excellent",
            8: "Very Good",
            7: "Good",
            6: "Above Average",
            5: "Average",
            4: "Below Average",
            3: "Fair",
            2: "Poor",
            1: "Very Poor"
            }
            secilen_metin = puanlar_metin.get(input_features[feature], "Bilinmeyen Puan")
            input_features[feature] = input_features[feature]
            st.write(f"Seçilen Puan: {input_features[feature]} - {secilen_metin}")

        elif feature == 'GarageType_Attchd':
            input_features[feature] = st.selectbox("Ek garaj Var mı?", {'Hayir': 0, 'Evet': 1}, key="garaj_selectbox")
        elif feature == 'BsmtQual':
            bsmt_qual_siniflandirma = {
                '5': "Excellent (100+ inches)",
                '4': "Good (90-99 inches)",
                '3': "Typical (80-89 inches)",
                '2': "Fair (70-79 inches)",
                '1': "Poor (<70 inches)",
                '0': "No Basement",
            }
            # Benzersiz bir key değeri oluşturunuz (örneğin, feature adını kullanabilirsiniz)
            selectbox_key = f"{feature}_selectbox"
            # Dış malzeme kalitesi için Streamlit selectbox'ını kullanarak değeri alınız
            input_features[feature] = st.selectbox("Bodrum Katının Kalitesini Seçin:", list(bsmt_qual_siniflandirma.keys()), key=selectbox_key)

            # Kullanıcının seçtiği değere göre sınıflandırmayı belirleyiniz
            sınıflandırma = bsmt_qual_siniflandirma[input_features[feature]]

            # Sınıflandırmayı ve seçilen değeri ekrana yazdırınız
            st.write(f"Seçilen: {input_features[feature]} - Sınıflandırma: {sınıflandırma}")
        elif feature == 'GarageCars':
            input_features[feature] = st.number_input("Garaja sığan araba sayısı", value=0, step=1)
        elif feature in ["FullBath"]:
            input_features[feature] = st.number_input("Banyo Sayısı", value=0, step=1)
        elif feature == 'Fireplaces':
            input_features[feature] = st.number_input("Şömine sayısı", value=0, step=1)
        elif feature == 'PavedDrive':
            paved_road_description = {
                'Asfaltlı Yol': '2',
                'Yarı Asfaltlı': '1',
                'Toprak Yol': '0'
            }
            input_features[feature] = st.selectbox("Asfaltlanmış araba yolu var mi?", list(paved_road_description.keys()))
        
        elif feature == 'ExterQual':
            puanlar = {
                    5: "Ex",
                    4: "Gd",
                    3: "TA",
                    2: "Fa",
                    1: "Po",
            }
            # Sınıflandırma sözlüğü
            siniflandirma = {
                5: "Excellent",
                4: "Good",
                3: "Average/Typical",
                2: "Fair",
                1: "Poor",
            }
            # Benzersiz bir key değeri oluşturunuz (örneğin, feature adını kullanabilirsiniz)
            slider_key = f"{feature}_slider"
            # Dış malzeme kalitesi için Streamlit slider'ını kullanarak değeri alınız
            input_features[feature] = st.slider("Dış malzeme kalitesi", 1, 5, value=5, key=slider_key)

            # Kullanıcının seçtiği değere göre sınıflandırmayı belirleyiniz
            sınıflandırma = siniflandirma[input_features[feature]]

            # Sınıflandırmayı ekrana yazdırınız
            st.write(f"Seçilen {input_features[feature]} :", sınıflandırma)

        elif feature == 'YearRemodAdd':
            input_features[feature] = st.number_input("Tadilat tarihi (tadilat veya ekleme yoksa inşaat tarihiyle aynı)", value=0, step=1)
        elif feature == 'CentralAir':
            input_features[feature] = st.selectbox("Klima Var mi?", {'Hayır': 0, 'Evet': 1})
        elif feature == 'KitchenQual':
            # Sınıflandırma sözlüğü
            siniflandirma_kitchen = {
                5: "Excellent",
                4: "Good",
                3: "Average/Typical",
                2: "Fair",
                1: "Poor",
            }
            # Benzersiz bir key değeri oluşturunuz (örneğin, feature adını kullanabilirsiniz)
            slider_key = f"{feature}_slider"
            # Dış malzeme kalitesi için Streamlit slider'ını kullanarak değeri alınız
            input_features[feature] = st.slider("Mutfak Kalitesi", 1, 5, value=5, key=slider_key)

            # Kullanıcının seçtiği değere göre sınıflandırmayı belirleyiniz
            siniflandirma_kitchen = siniflandirma_kitchen[input_features[feature]]

            # Sınıflandırmayı ekrana yazdırınız
            st.write(f"Seçilen {input_features[feature]} :", siniflandirma_kitchen)
        elif feature == 'MSZoning_RL':
            input_features[feature] = st.selectbox("Ev Az nüfuslu Bir Yerde mi?", {'Hayir': 0, 'Evet': 1})
        elif feature == 'TotalSF':
            input_features[feature] = st.number_input("Evin Toplam Metrekaresi", value=0.0)
        elif feature == 'GrLivArea':
            input_features[feature] = st.number_input("Zemin Katın Metrekaresi", value=0.0)
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
