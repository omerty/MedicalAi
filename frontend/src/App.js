import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [patientProblem, setPatientProblem] = useState('');
  const [predictedDisease, setPredictedDisease] = useState('');
  const [suggestedPrescription, setSuggestedPrescription] = useState('');
  
  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      const response = await axios.post('http://localhost:5000/predict', {
        Patient_Problem: patientProblem
      });
      console.log(response);
      setPredictedDisease(response.data['predicted_disease']);
      setSuggestedPrescription(response.data['suggested_prescription']);
    } catch (error) {
      console.error('Error making prediction', error);
    }
  };

  return (
    <div className="App">
      <h1>Medical Diagnoser</h1>
      <form onSubmit={handleSubmit}>
        <label>
          Patient Problem:
          <input
            type="text"
            value={patientProblem}
            onChange={(e) => setPatientProblem(e.target.value)}
          />
        </label>
        <button type="submit">Get Prediction</button>
      </form>

      {predictedDisease && (
        <div>
          <h2>Predicted Disease: {predictedDisease}</h2>
          <h3>Suggested Prescription: {suggestedPrescription}</h3>
        </div>
      )}
    </div>
  );
}

export default App;
