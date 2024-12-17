from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from keras.models import load_model
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load and preprocess data
data = pd.read_csv('https://raw.githubusercontent.com/adil200/Medical-Diagnoser/main/medical_data.csv')

tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(data['Patient_Problem'])

sequences = tokenizer.texts_to_sequences(data['Patient_Problem'])
max_length = max(len(x) for x in sequences)

label_encoder_disease = LabelEncoder()
label_encoder_prescription = LabelEncoder()

disease_labels = label_encoder_disease.fit_transform(data['Disease'])
prescription_labels = label_encoder_prescription.fit_transform(data['Prescription'])

# Load the trained model
model = load_model('disease_prescription_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'Patient_Problem' not in data:
        return jsonify({'error': 'Missing patient_problem in request'}), 400
    
    patient_problem = data['Patient_Problem']
    
    # Preprocess input
    sequence = tokenizer.texts_to_sequences([patient_problem])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
    
    # Make prediction
    prediction = model.predict(padded_sequence)
    
    # Decode prediction
    disease_index = np.argmax(prediction[0], axis=1)[0]
    prescription_index = np.argmax(prediction[1], axis=1)[0]
    
    disease_predicted = label_encoder_disease.inverse_transform([disease_index])[0]
    prescription_predicted = label_encoder_prescription.inverse_transform([prescription_index])[0]
    
    return jsonify({
        'predicted_disease': disease_predicted,
        'suggested_prescription': prescription_predicted
    })

if __name__ == '__main__':
    app.run(debug=True)
