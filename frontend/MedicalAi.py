import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from keras.models import load_model

data = pd.read_csv('https://raw.githubusercontent.com/adil200/Medical-Diagnoser/main/medical_data.csv')
data.head()

# A Tokenizer is created to convert the textual data into a sequence of integers. If the model
# the comes into contact with any unknown words then it will replace it with the <oov> token
tokenizer = Tokenizer(num_words = 5000, oov_token="<OOV>")
tokenizer.fit_on_texts(data['Patient_Problem'])

sequences = tokenizer.texts_to_sequences(data['Patient_Problem'])

# In order to make input sequences have the same length, the code will first find the longest
# sequence and pads all the other sequences with zeros at the end known as post padding
max_length = max(len(x) for x in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Know we are going to stack the binary class metrics together in order to form a single multi-label
# target variable called 'y' This will allow the model to predict both Disease and Prescription from
# the patients probelem

label_encoder_disease = LabelEncoder()
label_encoder_prescription = LabelEncoder()

disease_labels = label_encoder_disease.fit_transform(data['Disease'])
prescription_labels = label_encoder_prescription.fit_transform(data['Prescription'])

#Converting the lavels to categorical
disease_labels_categorical = to_categorical(disease_labels)
prescription_labels_categorical = to_categorical(prescription_labels)

Y = np.hstack((disease_labels_categorical, prescription_labels_categorical))

#Now we will build the model using the lSTM and Sequential Algorithm from TensorFlow.
#The Model and Input will be used in order to define the model archetechture and embedding to conver
#The integer sequences into dense vectors of fixed size

input_layer = Input(shape = (max_length,))

embedding = Embedding(input_dim=5000, output_dim = 64)(input_layer)
lstm_layer = LSTM(64)(embedding)

disease_ouput = Dense(len(label_encoder_disease.classes_),
activation='softmax',name='disease_output')(lstm_layer)

prescription_output = Dense(len(label_encoder_prescription.classes_), activation='softmax', name='prescription_output')(lstm_layer)


model = Model(input_layer, outputs=[disease_ouput, prescription_output])

model.compile(
    loss={'disease_output' : 'categorical_crossentropy',
          'prescription_output': 'categorical_crossentropy'},
    optimizer='adam',
    metrics={'disease_output': ['accuracy'], 'prescription_output':['accuracy']}
)

model.summary()

#Training the model
model.fit(padded_sequences, {'disease_output' : disease_labels_categorical, 'prescription_output' : prescription_labels_categorical}, epochs=100, batch_size=32)

model.save('disease_prescription_model.h5')

#Making predictions functin

def make_predictions(Patient_Problem):

    #Loading the trained model
    model = load_model('disease_prescription_model.h5')

    # Recompile the model (necessary for training or evaluation)
    model.compile(
        optimizer='adam',
        loss={
            'disease_output': 'categorical_crossentropy',
            'prescription_output': 'categorical_crossentropy'
        },
        metrics={
            'disease_output': 'accuracy',
            'prescription_output': 'accuracy'
        }
    )

    #Preprocessing the input
    sequence = tokenizer.texts_to_sequences(['Patient_Problem'])
    padded_sequences = pad_sequences(sequence, maxlen=max_length, padding='post')

    #Making the predictions
    prediction = model.predict(padded_sequences)

    #Decoding the prediction
    disease_index = np.argmax(prediction[0], axis=1)[0]
    prescription_index = np.argmax(prediction[1], axis=1)[0]

    disease_predicted = label_encoder_disease.inverse_transform([disease_index])[0]
    prescription_predicted = label_encoder_prescription.inverse_transform([prescription_index])[0]

    print(f"Predicted Disease: {disease_predicted}")
    print(f"Suggested Prescription: {prescription_predicted}")


# Train and save the model
model.fit(
    padded_sequences,
    {'disease_output': disease_labels_categorical, 'prescription_output': prescription_labels_categorical},
    epochs=100,
    batch_size=32,
    verbose=0
)
model.save('disease_prescription_model.h5')


patient_input = "I have been peeing blood"
make_predictions(patient_input)

