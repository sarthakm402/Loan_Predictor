# Loan Predictor

A web-based application designed to predict loan approval status based on applicant information using machine learning techniques.

## 🧠 Overview

This project leverages machine learning to assess the eligibility of loan applicants. By inputting various personal and financial details, the system predicts whether a loan should be approved or not. The application is built using Python and Flask for the backend, with HTML for the frontend interface.

## 🚀 Features

- **User-Friendly Interface**: Simple HTML forms for input and result display.
- **ML-Powered Predictions**: Predicts loan approval based on real-world features.
- **Flask Backend**: Processes form data and displays results.

## 📁 Project Structure

```
Loan_Predictor/
├── index_loan_pred.html                # Input form page
├── index_result_loan_pred.html        # Result display page
├── internship_loan_predictor.py       # ML model and logic
└── internship_loan_predictore_flask.py # Flask application
```

## ⚙️ Getting Started

### Prerequisites

- Python 3.x
- Flask
- pandas
- scikit-learn

You can install required packages via:

```bash
pip install flask pandas scikit-learn
```

### Running the App

1. **Clone the repository**:

```bash
git clone https://github.com/sarthakm402/Loan_Predictor.git
cd Loan_Predictor
```

2. **Run the Flask app**:

```bash
python internship_loan_predictore_flask.py
```

3. **Open in browser**:

Go to `http://localhost:5000` and start predicting!

## 🧪 How It Works

- The user fills in loan applicant details via the HTML form.
- Flask routes the input to a Python function that loads the ML model.
- The model makes a prediction (loan approved or not).
- The result is rendered on a new HTML page.

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repo and open a pull request with improvements or bug fixes.

## 📄 License

This project is licensed under the MIT License.

---

> Created by [Sarthak Mohapatra](https://github.com/sarthakm402)
