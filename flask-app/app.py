from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the saved model
model_filename = 'xgboost_regression_model.pkl'
loaded_model = joblib.load(model_filename)

# Load the label encoders
label_encoder_month = joblib.load('label_encoder_month.pkl')
label_encoder_material = joblib.load('label_encoder_material.pkl')

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        year_month = request.form["year_month"]
        material_code = request.form["material_code"]

        # Encode the input values
        year_month_encoded = label_encoder_month.transform([year_month])[0]
        material_code_encoded = label_encoder_material.transform([material_code])[0]

        # Prepare data for prediction
        input_data = pd.DataFrame({
            'YEAR_MONTH': [year_month_encoded],
            'MATERIAL_CODE': [material_code_encoded]
        })

        # Make a prediction using the loaded model
        prediction = loaded_model.predict(input_data)[0]

        return render_template("index.html", prediction=prediction)

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
