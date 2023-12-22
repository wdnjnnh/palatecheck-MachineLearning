from flask import Flask,jsonify,request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model

app = Flask(__name__)

df = pd.read_csv('PP_merged.csv')

# drop columns
df_model = df.drop(columns=['like_id', 'user_id', 'cafe_id'])

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

# Save the model
model.save("PP_model_2.h5", save_format='h5')


model = load_model("PP_model_2.h5")



@app.route('/predict/palatecheck/<int:user_id>/<string:cafe_id>', methods=['GET'])
def palateCheck(user_id, cafe_id):
    user_cafe_data = df_model[(df['user_id'] == user_id) & (df['cafe_id'] == cafe_id)].drop(columns=['vote'])
    
    if user_cafe_data.empty:
        return jsonify({'error': 'Data not found for the given user_id and cafe_id'}), 404
    
    user_cafe_data_scaled = scaler.transform(user_cafe_data)
    prediction = model.predict(user_cafe_data_scaled)
    prediction = float(prediction)
    
    response = {
        'user_id':user_id,
        'cafe_id':cafe_id,
        'data_prediction': prediction[0][0] if isinstance(prediction, np.ndarray) else prediction
    }
    return jsonify(response)

@app.route('/predict/recommendation',methods=['POST'])
def recomendation():
    user_id = int(request.args.get('user_id'))
    user_data = df_model[df['user_id'] == user_id].drop(columns=['vote'])
    
    if user_data.empty:
        return jsonify({'error': 'Data not found for the given user_id'}), 404
    
    user_data_scaled = scaler.transform(user_data)
    recommendation = (model.predict(user_data_scaled) > 0.5).astype(int)
    
    response = {
        'user_id':user_id,
        'recommendation for User': recommendation.tolist() if isinstance(recommendation, np.ndarray) else recommendation
    }
    return jsonify(response)

@app.route('/predict/likelihood',methods=['POST'])
def likelihood():

    user_id = int(request.args.get('user_id'))
    cafe_id = request.args.get('cafe_id')

    user_cafe_data = df_model[(df['user_id'] == user_id) & (df['cafe_id'] == cafe_id)].drop(columns=['vote'])

    if user_cafe_data.empty:
        return jsonify({'error': 'Data not found for the given user_id and cafe_id'}), 404

    user_cafe_data_scaled = scaler.transform(user_cafe_data)

    prediction = model.predict(user_cafe_data_scaled)
    prediction_likeihood = prediction.tolist() if isinstance(prediction, np.ndarray) else prediction

    response = {
        'user_id': user_id,
        'cafe_id': cafe_id,
        'Predicted Likelihood': prediction_likeihood[0][0]
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)