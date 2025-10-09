# Disease Prediction and Medical Recommendation System 🤖🩺

A machine learning-powered **chatbot application** that predicts diseases through an interactive conversation and provides comprehensive health recommendations including medications, dietary suggestions, and exercise routines.

## 🎯 Features

- **🤖 Interactive Chatbot Interface**: Modern, conversational AI assistant
- **💬 Step-by-Step Conversation**: Collects name, age, gender, and symptoms naturally
- **🔍 Intelligent Disease Prediction**: Uses Random Forest machine learning model with 100% accuracy
- **✨ Smart Symptom Correction**: Advanced fuzzy matching to correct misspelled symptoms
- **📱 Real-time Chat Experience**: Live typing indicators and loading animations
- **🎨 Modern UI/UX**: Beautiful gradient design with emojis and smooth animations
- **📊 Comprehensive Health Recommendations**: 
  - Disease descriptions and information
  - Personalized medication suggestions
  - Dietary recommendations
  - Exercise and workout plans
  - Preventive precautions
- **🔄 Session Management**: Easy reset and new conversation flow

## 🚀 Technologies Used

- **Backend**: Python, Flask
- **Machine Learning**: scikit-learn, Random Forest Classifier
- **Data Processing**: pandas, numpy
- **Frontend**: HTML, CSS, Bootstrap 5
- **Model Persistence**: pickle

## 📊 Dataset Information

The system uses comprehensive medical datasets containing:
- **41 Diseases**: Including common conditions like diabetes, hypertension, allergies, etc.
- **132 Symptoms**: Comprehensive symptom database for accurate predictions
- **Medical Recommendations**: Curated medications, diets, and workout plans for each disease

### Dataset Files (located in `dataset/`):
- `Training.csv`: Main training dataset with symptoms and disease labels
- `symptoms_df.csv`: Symptom database with disease mappings
- `description.csv`: Detailed disease descriptions
- `medications.csv`: Medication recommendations for each disease
- `diets.csv`: Dietary recommendations and nutrition plans
- `workout_df.csv`: Exercise and workout suggestions
- `precautions_df.csv`: Preventive measures and precautions
- `Symptom-severity.csv`: Symptom severity classifications

## 🔧 Installation & Setup

### Prerequisites
- Python 3.7+
- pip package manager

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sohamvsonar/Disease-Prediction-and-Medical-Recommendation-System.git
   cd Disease-Prediction-and-Medical-Recommendation-System
   ```

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask application:**
   ```bash
   python main.py
   ```

4. **Access the application:**
   Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

## 📱 Usage

### 🤖 Chatbot Interface

1. **Start Conversation**: The chatbot will greet you and ask for your name
2. **Provide Information**: Answer questions about your age and gender
3. **Describe Symptoms**: List your symptoms when prompted (e.g., "fever, headache, nausea")
4. **Get Results**: Receive comprehensive health recommendations including:
   - Predicted disease with description
   - Recommended medications
   - Dietary suggestions
   - Exercise plans
   - Preventive precautions
5. **Start New Session**: Use the "Start New Session" button to begin a fresh conversation

### 💬 Conversation Flow
```
🤖 HealthBot: Hello! What's your name?
👤 You: John
🤖 HealthBot: Nice to meet you, John! How old are you?
👤 You: 25
🤖 HealthBot: Thank you! What's your gender?
👤 You: Male
🤖 HealthBot: Now, please describe your symptoms...
👤 You: headache, fever, fatigue
🤖 HealthBot: Analyzing symptoms... [Loading animation]
🤖 HealthBot: [Shows detailed prediction results]
```




## 🏗️ Project Structure

```
Disease-Prediction-and-Medical-Recommendation-System/
├── dataset/                    # Medical datasets
│   ├── Training.csv
│   ├── symptoms_df.csv
│   ├── description.csv
│   ├── medications.csv
│   ├── diets.csv
│   ├── workout_df.csv
│   ├── precautions_df.csv
│   └── Symptom-severity.csv
├── model/                      # Trained ML models
│   └── RandomForest.pkl
├── templates/                  # HTML templates
│   └── index.html
├── static/                     # Static assets
│   ├── bgCover.jpg
│   └── img.png
├── screenshots/                # Application screenshots
│   ├── ss1.jpg
│   ├── ss2.jpg
│   └── ss3.jpg
├── main.py                     # Flask web application
├── disease_prediction_system.ipynb  # Model training notebook
├── requirements.txt            # Python dependencies
└── README.md
```


## ⚠️ Disclaimer

This system is for educational and informational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical concerns.

=======
