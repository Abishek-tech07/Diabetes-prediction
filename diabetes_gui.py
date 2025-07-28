from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

app = Flask(__name__)


data = pd.read_csv("diabetes.csv")
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)


accuracy = accuracy_score(y_test, model.predict(X_test))
accuracy_percent = round(accuracy * 100, 2)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['preg']),
            float(request.form['gluc']),
            float(request.form['bp']),
            float(request.form['skin']),
            float(request.form['insulin']),
            float(request.form['bmi']),
            float(request.form['dpf']),
            float(request.form['age'])
        ]

        max_values = [10, 200, 122, 99, 846, 67.1, 2.5, 100]
        field_names = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness',
                       'Insulin', 'BMI', 'Diabetes Pedigree Function', 'Age']

        for i in range(len(features)):
            if features[i] < 0 or features[i] > max_values[i]:
                return render_template('index.html',
                    prediction=f"❌ {field_names[i]} should be between 0 and {max_values[i]}.",
                    accuracy=accuracy_percent)

        scaled = scaler.transform([features])
        prediction = model.predict(scaled)[0]
        result = "✅ Positive (Diabetic)" if prediction == 1 else "✅ Negative (Not Diabetic)"
    except:
        result = "❌ Invalid input. Enter valid numbers."

    return render_template('index.html', prediction=result, accuracy=accuracy_percent)

if __name__ == '__main__':
    app.run(debug=True)
