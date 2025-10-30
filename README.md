# Magic Wand Gesture Recognition

A gesture recognition system using accelerometer and gyroscope data to classify 3 magic gestures:  
**Stone **, **Paper **, and **Scissors ** — implemented with a Random Forest classifier.

---

##  Goal

Build a machine learning pipeline that:
- Records wand gestures via motion sensors
- Preprocesses and extracts features
- Trains a classifier (Random Forest)
- Predicts the gesture class in real-time

---

## Features

-  Records motion sensor data (acceleration + gyroscope)
-  Visualizes sensor data over time
-  Preprocesses and standardizes features
-  Trains and evaluates a Random Forest model
-  Live prediction using trained model

---

##  Tech Stack

| Layer     | Tool / Library              |
|-----------|-----------------------------|
| Data Viz  | `matplotlib`, `pandas`      |
| ML Model  | `scikit-learn`              |
| Preproc   | `pandas`, `joblib`, `os`    |
| Input     | CSV recordings (6-axis data) |

---

##  File Overview
├── recordings/ # Raw gesture data (CSV)
├── features/ # Extracted feature vectors
├── graphs/ # Saved plots of motion data
├── preprocess.py # Feature extraction + labeling
├── train_classifier.py # Train + evaluate + save model
├── yourcode.py # Predict spell from new gesture
├── plot_graphs.py # Plot gesture data
├── README.md # Project description

---


---

##  Getting Started

### 1.  Install Dependencies

```bash
pip install pandas matplotlib scikit-learn joblib
```

2.  Collect and Save Gesture Recordings
Each recording is saved as a CSV with columns:
time, accX, accY, accZ, gyroX, gyroY, gyroZ

Save recordings to the recordings/ folder. Make sure filenames contain "Stone", "Paper", or "Scissors" to label them correctly.

3.  Preprocess Data

```bash
python preprocess.py
```
Extracts mean, std, min, max from all 6 channels

Adds gesture labels

Saves result to features/features.csv

4.  Train the Classifier
```bash
python train_classifier.py
```
Loads features

Splits data (80% training, 20% test)

Trains Random Forest model

Evaluates performance

Saves random_forest_model.pkl

5.  Predict on New Gestures
```bash
python yourcode.py
```
Loads a new CSV from recordings/

Extracts features

Predicts the gesture with the trained model


Outputs predicted class (stone/paper/scissors)

