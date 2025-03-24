import joblib
import pandas as pd
import numpy as np
import shap
from flask import Flask, request, render_template, jsonify
import io

final_model = joblib.load("loan_model.pkl")
preprocessor_final = joblib.load("preprocessor.pkl")

try:
    features = preprocessor_final.get_feature_names_out()
except AttributeError:
    features = preprocessor_final.feature_names_in_

app = Flask(__name__)

def rule_based_fraud_detection(row):
    ratio = row["LoanAmount"] / (row["Total_Income"] + 1e-6)
    income_to_loan = row["Income_to_Loan_Ratio"]
    flags = []
    if ratio > 0.6 and row["Credit_History"] == 0:
        flags.append("High Loan/Income Ratio with No Credit History")
    if income_to_loan < 0.3:
        flags.append("Low Income to Loan Ratio")
    if row["ApplicantIncome"] < 1500 and row["LoanAmount"] > 200:
        flags.append("Low Applicant Income with High Loan Amount")
    if row["CoapplicantIncome"] == 0 and row["Self_Employed"] == "No" and row["LoanAmount"] > 150:
        flags.append("No Coapplicant Income and Not Self Employed")
    if row["Property_Area"] == "Rural" and row["ApplicantIncome"] < 2500:
        flags.append("Low Income in Rural Area")
    if flags:
        return "Potential Fraudulent Case: " + "; ".join(flags)
    return "Normal"

@app.route("/", methods=["POST", "GET"])
def index():
    if request.method == "GET":
        return render_template("index_loan_pred.html")
    if request.method == "POST":
        try:
            app.logger.info(f"Form data: {request.form}")
            app.logger.info(f"Files: {request.files}")
            input_type = request.form.get("inputType", "manual")
            if input_type == "csv":
                file = request.files["file"]
                if file.filename.endswith(".csv"):
                    file_content = file.read()
                    df = pd.read_csv(io.BytesIO(file_content))
                    app.logger.info(f"CSV file read, shape: {df.shape}")
                    results = process_csv(df)
                    return render_template("index_loan_pred.html", results=results)
                else:
                    raise ValueError("Please upload a CSV file")
            else:
                input_data = {
                    "Credit_History": float(request.form.get("Credit_History", 0)),
                    "CoapplicantIncome": float(request.form.get("CoapplicantIncome", 0)),
                    "ApplicantIncome": float(request.form.get("ApplicantIncome", 0)),
                    "LoanAmount": float(request.form.get("LoanAmount", 1)),
                    "Total_Income": float(request.form.get("ApplicantIncome", 0)) + float(request.form.get("CoapplicantIncome", 0)),
                    "Income_to_Loan_Ratio": (float(request.form.get("ApplicantIncome", 0)) + float(request.form.get("CoapplicantIncome", 0))) / (float(request.form.get("LoanAmount", 1)) + 1e-6),
                    "Married": request.form.get("Married", "No"),
                    "Dependents": request.form.get("Dependents", "0"),
                    "Self_Employed": request.form.get("Self_Employed", "No"),
                    "Education": request.form.get("Education", "Graduate"),
                    "Property_Area": request.form.get("Property_Area", "Urban"),
                }
                input_df = pd.DataFrame([input_data])
                results = process_prediction(input_df)
                return render_template("index_loan_pred.html", results=results)
        except Exception as e:
            app.logger.error(f"Error processing request: {str(e)}")
            error_message = str(e)
            return render_template("index_loan_pred.html", error=error_message)

def process_csv(df):
    try:
        required_columns = ["Credit_History", "CoapplicantIncome", "ApplicantIncome", "LoanAmount", "Married", "Dependents", "Self_Employed", "Education", "Property_Area"]
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in CSV: {missing_cols}")
        df["Total_Income"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
        df["Income_to_Loan_Ratio"] = df["Total_Income"] / (df["LoanAmount"] + 1e-6)
        return process_prediction(df)
    except Exception as e:
        raise ValueError(f"Error processing CSV: {str(e)}")

def process_prediction(df):
    input_processed = preprocessor_final.transform(df)
    final_preds = final_model.predict(input_processed)
    loan_statuses = ["Approved" if pred == 1 else "Rejected" for pred in final_preds]
    explainer = shap.Explainer(final_model)
    shap_values = explainer(input_processed)
    explanations = []
    suggestions_all = []
    for idx, feature_contribution in enumerate(shap_values.values):
        top_features = np.argsort(abs(feature_contribution))[-3:][::-1]
        explanation_text = f'Loan application {idx + 1} was {loan_statuses[idx]} due to:\n'
        for i in top_features:
            sign = "increased" if feature_contribution[i] > 0 else "decreased"
            explanation_text += f"- {features[i]} {sign} the chances.\n"
        suggestions = []
        if loan_statuses[idx] == "Rejected":
            thresh = 0.2
            for i in top_features:
                if feature_contribution[i] < 0:
                    diff = abs(feature_contribution[i]) * thresh
                    if diff > 0:
                        suggestions.append(f'Increasing the feature {features[i]} by {diff:.2f} might be useful')
        explanations.append(explanation_text)
        suggestions_all.append(suggestions)
    fraud_results = []
    for i in range(len(df)):
        fraud_results.append(rule_based_fraud_detection(df.iloc[i]))
    results = []
    for i in range(len(df)):
        results.append({
            "Result": loan_statuses[i],
            "Explanation": explanations[i],
            "Suggestions": suggestions_all[i],
            "Anomaly_Check": fraud_results[i]
        })
    return results

if __name__ == "__main__":
    app.run(debug=True)

