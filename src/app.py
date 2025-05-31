from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Flask uygulaması
app = Flask(__name__)

# Veri yükleme ve ön işleme
df = pd.read_csv('src/heart.csv')

label_encoder_sex = LabelEncoder()
label_encoder_cp = LabelEncoder()
df['Sex'] = label_encoder_sex.fit_transform(df['Sex'])
df['ChestPainType'] = label_encoder_cp.fit_transform(df['ChestPainType'])
df['RestingECG'] = LabelEncoder().fit_transform(df['RestingECG'])
df['ExerciseAngina'] = LabelEncoder().fit_transform(df['ExerciseAngina'])
df['ST_Slope'] = LabelEncoder().fit_transform(df['ST_Slope'])

X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

# Yapay Sinir Ağı Modeli
model = Sequential()
model.add(Dense(16, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dense(1, activation='sigmoid'))
optimizer = Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.fit(X_train_scaled, y_train_smote, epochs=100, verbose=0, validation_data=(X_test_scaled, y_test))

# Ana sayfa
@app.route('/')
def home():
    return render_template('index.html')

# Tahmin API'si
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Kullanıcıdan gelen veriler
        age = float(request.form['age'])
        if age < 0 or age > 120:  # Mantıklı bir yaş aralığı kontrolü
            return jsonify({'error': 'Yaş 0 ile 120 arasında olmalıdır.'})

        sex = request.form['sex']
        chest_pain = request.form['chest_pain']
        resting_bp = float(request.form['resting_bp'])
        cholesterol = float(request.form['cholesterol'])

        # Veriyi dönüştürme
        user_data = pd.DataFrame({
            'Age': [age],
            'Sex': [label_encoder_sex.transform([sex])[0]],
            'ChestPainType': [label_encoder_cp.transform([chest_pain])[0]],
            'RestingBP': [resting_bp],
            'Cholesterol': [cholesterol],
            'FastingBS': [0],
            'RestingECG': [0],
            'MaxHR': [150],
            'ExerciseAngina': [0],
            'Oldpeak': [0.0],
            'ST_Slope': [1]
        })

        # Veriyi ölçeklendirme
        user_data_scaled = scaler.transform(user_data)

        # Tahmin yapma
        prediction = model.predict(user_data_scaled)
        risk = prediction[0][0] * 100

        return jsonify({'risk': f'{risk:.2f}%'})
    except Exception as e:
        return jsonify({'error': str(e)})

# Flask uygulamasını çalıştır
if __name__ == '__main__':
    app.run(debug=True)