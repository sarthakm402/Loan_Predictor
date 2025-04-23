import gradio as gr
import pandas as pd
import numpy as np
import joblib
import shap
import re 

# Load model and preprocessor
final_model = joblib.load("loan_model.pkl")
preprocessor_final = joblib.load("preprocessor.pkl")

try:
    features = preprocessor_final.get_feature_names_out()
except AttributeError:
    features = preprocessor_final.feature_names_in_

cleaned_features = [re.sub(r"_\d+$", "", f.replace("num__", "").replace("cat__", "")) for f in features]
features = cleaned_features

# Rule-based fraud detection
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
    return "Potential Fraudulent Case: " + "; ".join(flags) if flags else "Normal"

# Prediction processor
def process_prediction(df):
    df["Total_Income"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["Income_to_Loan_Ratio"] = df["Total_Income"] / (df["LoanAmount"] + 1e-6)
    
    input_processed = preprocessor_final.transform(df)
    final_preds = final_model.predict(input_processed)
    loan_statuses = ["✅ Approved" if pred == 1 else "❌ Rejected" for pred in final_preds]

    explainer = shap.Explainer(final_model)
    shap_values = explainer(input_processed)

    results = []
    for idx, feature_contribution in enumerate(shap_values.values):
        top_features = np.argsort(abs(feature_contribution))[-3:][::-1]
        explanation_text = f'Loan Application {idx + 1} was {loan_statuses[idx]} due to:\n'
        suggestions = []

        for i in top_features:
            sign = "increased" if feature_contribution[i] > 0 else "decreased"
            explanation_text += f"- {features[i]} {sign} the chances.\n"
            if loan_statuses[idx] == "❌ Rejected" and feature_contribution[i] < 0:
                diff = abs(feature_contribution[i]) * 0.2
                suggestions.append(f'➕ Increasing {features[i]} by {diff:.2f} might help')

        fraud_check = rule_based_fraud_detection(df.iloc[idx])
        result = f"Result: {loan_statuses[idx]}\n\n" \
                 f"Loan Application {idx + 1} was {loan_statuses[idx]} due to:\n" \
                 f"{explanation_text.strip()}"
        if suggestions:
            result += "\n\n💡 Suggestions:\n" + "\n".join(suggestions)
        result += f"\n\n🕵️‍♂️ Anomaly Check: {fraud_check}"
        results.append(result)

    return "\n\n".join(results)

# Gradio functions
def manual_input(Credit_History, CoapplicantIncome, ApplicantIncome, LoanAmount, Married, Dependents, Self_Employed, Education, Property_Area):
    data = pd.DataFrame([{
        "Credit_History": Credit_History,
        "CoapplicantIncome": CoapplicantIncome,
        "ApplicantIncome": ApplicantIncome,
        "LoanAmount": LoanAmount,
        "Married": Married,
        "Dependents": Dependents,
        "Self_Employed": Self_Employed,
        "Education": Education,
        "Property_Area": Property_Area,
    }])
    return process_prediction(data)

def from_csv(file):
    df = pd.read_csv(file.name)
    required_columns = [
        "Credit_History", "CoapplicantIncome", "ApplicantIncome", "LoanAmount", 
        "Married", "Dependents", "Self_Employed", "Education", "Property_Area"
    ]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return [f"❌ Missing columns: {missing_cols}"]
    return process_prediction(df)

# Gradio interface
manual_form = gr.Interface(
    fn=manual_input,
    inputs=[ 
        gr.Number(label="Credit History (0 or 1)"),
        gr.Number(label="Coapplicant Income"),
        gr.Number(label="Applicant Income"),
        gr.Number(label="Loan Amount"),
        gr.Radio(["Yes", "No"], label="Married"),
        gr.Radio(["0", "1", "2", "3+"], label="Dependents"),
        gr.Radio(["Yes", "No"], label="Self Employed"),
        gr.Radio(["Graduate", "Not Graduate"], label="Education"),
        gr.Radio(["Urban", "Rural", "Semiurban"], label="Property Area"),
    ],
    outputs=gr.Textbox(label="Prediction Result"),
    title="🧠 Loan Approval Predictor",
    description="Enter details manually to get loan prediction with explanation and fraud detection."
)

csv_form = gr.Interface(
    fn=from_csv,
    inputs=gr.File(label="Upload CSV File"),
    outputs=gr.Textbox(label="Prediction Results"),
    title="📄 Batch Prediction",
    description="Upload a CSV with loan data to get predictions for multiple applicants."
)

app = gr.TabbedInterface([manual_form, csv_form], ["🔢 Manual Input", "📁 CSV Upload"])

if __name__ == "__main__":
    app.launch()
