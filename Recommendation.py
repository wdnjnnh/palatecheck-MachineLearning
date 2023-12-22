# jadi inti model ini adalah mempelajari kombinasi preferensi user dengan kategori kafe.
# ketika user suka atau cocok terhadap cafe user akan memberikan vote 1 jika tidak cocok user memberikan 0, 
# nilai vote tersebut jadi y (target), y yang akan selanjutnya dipredict nilainya dengan
# model yang dibuat. dengan hasil rangenya 0-1. fungsi yang ada yang akan dijadikan API.
# karena ini data likenya pakai dummy, hasil model akan ampas, 
# maka penilaian akan lebih besar dari jarak geografis dulu untuk hyperparameter.

# import data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv('PP_merged.csv')

# drop columns
df_model = df.drop(columns=['like_id', 'user_id', 'cafe_id'])
print(df_model.columns)

# Split the data into features (X) and target (y)
X = df_model.drop(columns=['vote'])  # Features
y = df_model['vote']  # Target
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# modelling
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy}')

def palateCheck(user_id_to_predict, cafe_id_to_predict):
  user_cafe_data = df_model[(df['user_id'] == user_id_to_predict) & (df['cafe_id'] == cafe_id_to_predict)].drop(columns=['vote'])
  user_cafe_data_scaled = scaler.transform(user_cafe_data)
  prediction = model.predict(user_cafe_data_scaled)
  return prediction
  #return value of cocoklogi

def palateSearch(user_id_to_predict, geolocation):
  return #array of cafe_ids that pass the cocoklogi thereshold

def palateFilterSearch(Kpopers=0, JapanLovers=0, AnimalLovers=0, Quite=0, MusicLovers=0, BookLovers=0, ArtLovers=0, ViewsLovers=0, CoffeeLovers=0, NonCoffeeLovers=0, groupsComer=0, geolocation=0):
  return #array of cafe_ids that pass the cocoklogi thereshold

# Implementation
print(palateCheck(478,'ChIJ0aVF46v2aS4Rnh8lZ3d5tUQ'))

# Example: Making predictions for a specific user (replace 'user_id_to_predict' with the actual user ID)
user_id_to_predict = 478
user_data = df_model[df['user_id'] == user_id_to_predict].drop(columns=['vote'])
user_data_scaled = scaler.transform(user_data)
recommendation = (model.predict(user_data_scaled) > 0.5).astype(int)
# Display the recommendation for the user
print(f'Recommendation for User {user_id_to_predict}: {recommendation}')


# Example: Making predictions for a specific user and cafe combination
user_id_to_predict = 478
cafe_id_to_predict = 'ChIJ0aVF46v2aS4Rnh8lZ3d5tUQ'  # Replace with the actual cafe ID
user_cafe_data = df_model[(df['user_id'] == user_id_to_predict) & (df['cafe_id'] == cafe_id_to_predict)].drop(columns=['vote'])
user_cafe_data_scaled = scaler.transform(user_cafe_data)
prediction = model.predict(user_cafe_data_scaled)
# Display the predicted likelihood of the user liking the cafe
print(f'Predicted Likelihood for User {user_id_to_predict} liking Cafe {cafe_id_to_predict}: {prediction}')


model.save("PP_model_1.h5", save_format='h5')