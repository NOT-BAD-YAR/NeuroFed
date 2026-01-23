import numpy as np
import tensorflow as tf
import os

def evaluate_evolution(model_path, round_name):
    if not os.path.exists(model_path):
        print(f"❌ Error: {model_path} not found. Run the server/client first!")
        return

    try:
        # 1. Load the trained model
        model = tf.keras.models.load_model(model_path)
        print(f"\n--- Testing Model Evolution: {round_name} ---")

        # 2. Define Test Cases (Matching your 3 columns: age, heart_rate, blood_pressure)
        # Case A: 75 years old, 110 HR, 180 BP -> High Risk (Should be SICK/1)
        # Case B: 25 years old, 72 HR, 120 BP -> Healthy (Should be HEALTHY/0)
        test_cases = np.array([
            [75, 110, 180], 
            [25, 72, 120]
        ])

        # 3. Get Predictions
        predictions = model.predict(test_cases, verbose=0)

        for i, pred in enumerate(predictions):
            val = pred[0]
            status = "SICK" if val > 0.5 else "HEALTHY"
            confidence = val if val > 0.5 else (1 - val)
            
            patient_type = "High Risk" if i == 0 else "Normal"
            print(f"Patient {i+1} ({patient_type}): {status} ({confidence*100:.2f}% confidence)")
            
            # Logic check for the "High Risk" patient
            if i == 0:
                if status == "SICK":
                    print("✅ SUCCESS: Model correctly identified high-risk symptoms.")
                else:
                    print("⚠️  CRITICAL ERROR: Model failed to detect high blood pressure/heart rate!")

    except Exception as e:
        print(f"❌ Error during evaluation: {e}")

if __name__ == "__main__":
    # Ensure this points to where your strategy.py saves the model
    evaluate_evolution('global_model.h5', 'Audit Round')