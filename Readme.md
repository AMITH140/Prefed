# Machine Failure Prediction 🛠️

Predict machine failures using sensor data! This project uses machine learning to analyze sensor readings and predict whether a machine is likely to fail. The dataset includes features like temperature, air quality, and rotational position, with a binary target indicating failure (`1`) or no failure (`0`).

---

## Dataset Overview 📊

The dataset contains the following columns:

| Column        | Description                                                                 |
|---------------|-----------------------------------------------------------------------------|
| **footfall**  | Number of people or objects passing by the machine.                         |
| **temp Mode** | Temperature mode or setting of the machine.                                |
| **AQ**        | Air quality index near the machine.                                         |
| **USS**       | Ultrasonic sensor data (proximity measurements).                            |
| **CS**        | Current sensor readings (electrical current usage).                         |
| **VOC**       | Volatile organic compounds level detected near the machine.                 |
| **RP**        | Rotational position or RPM (revolutions per minute) of machine parts.       |
| **IP**        | Input pressure to the machine.                                              |
| **Temperature** | Operating temperature of the machine.                                      |
| **fail**      | Binary indicator of machine failure (`1` for failure, `0` for no failure). |

---

## How It Works 🛠️

1. **Load Data**: The script reads the `data.csv` file and checks for missing values.
2. **Preprocess Data**: Missing values are filled with the mean of the column, and the data is split into features (`X`) and target (`y`).
3. **Train-Test Split**: The data is divided into training (80%) and testing (20%) sets.
4. **Standardize Features**: Features are scaled to ensure they're on the same scale for better model performance.
5. **Train Model**: A Random Forest Classifier is trained on the training data.
6. **Evaluate Model**: The model's accuracy and classification report are printed, along with feature importances.

---

## How to Run 🚀

1. **Install Dependencies**:
   ```bash
   pip install pandas scikit-learn
2. **Make sure the dataset data.csv is in the same folder**
3. **Run the file Prefed.py**