
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained Machine Learning model
model = joblib.load("placement_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        cgpa = float(request.form["cgpa"])
        aptitude = float(request.form["aptitude_score"])
        communication = float(request.form["communication_skill"])
        internship = int(request.form["internship"])
        projects = int(request.form["projects"])

        # Validate student details
        if not 0 <= cgpa <= 10:
            return "CGPA must be between 0 and 10.", 400

        if not 0 <= aptitude <= 100:
            return "Aptitude score must be between 0 and 100.", 400

        if not 1 <= communication <= 10:
            return "Communication skill must be between 1 and 10.", 400

        if internship not in (0, 1):
            return "Invalid internship selection.", 400

        if not 0 <= projects <= 20:
            return "Projects must be between 0 and 20.", 400

        # Prepare student data
        student = pd.DataFrame([{
            "cgpa": cgpa,
            "aptitude_score": aptitude,
            "communication_skill": communication,
            "internship": internship,
            "projects": projects
        }])

        # Predict placement status
        prediction = int(model.predict(student)[0])

        # Calculate model-estimated placement score
        probabilities = model.predict_proba(student)[0]
        class_index = list(model.classes_).index(1)

        placement_chance = round(
            float(probabilities[class_index]) * 100, 1
        )

        if prediction == 1:
            result = "Placement Likely"
        else:
            result = "Placement Unlikely"

        return render_template(
            "index.html",
            prediction_text=result,
            placement_chance=placement_chance,
            cgpa=cgpa,
            aptitude=aptitude,
            communication=communication,
            internship=internship,
            projects=projects
        )

    except (ValueError, KeyError):
        return "Please enter valid student details.", 400


if __name__ == "__main__":
    app.run(debug=True)