# Customer Churn Prediction using Machine Learning

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)

A machine learning-based web application to predict customer churn in telecommunications services with 81% accuracy using Random Forest and XGBoost classifiers.

[Key Features](#key-features) • [Installation](#installation) • [Usage](#usage) • [Results](#results) • [Contributing](#contributing)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project implements a comprehensive machine learning solution to predict customer churn for a telecommunications company. The application includes:

- **Jupyter Notebook** for data exploration, preprocessing, and model training
- **Streamlit Web App** for interactive predictions and model insights
- **Multiple ML Models** including Decision Tree, Random Forest, and XGBoost
- **Hyperparameter Tuning** using GridSearchCV for optimal performance
- **SMOTE** technique to handle class imbalance

The goal is to identify customers who are likely to leave the service and enable proactive retention strategies.

---

## Key Features

✨ **Interactive Web Interface**
- User-friendly dashboard for making predictions
- Real-time probability scores
- Visual feedback with churn risk indicators
- Responsive design

🤖 **Advanced ML Models**
- Random Forest Classifier (Best Performer)
- Decision Tree Classifier
- XGBoost Classifier
- Hyperparameter tuning with GridSearchCV

📊 **Comprehensive Data Analysis**
- Exploratory Data Analysis (EDA)
- Correlation and distribution analysis
- Class imbalance handling with SMOTE
- Feature engineering and preprocessing

📈 **Model Insights**
- Feature importance analysis
- Cross-validation results (5-fold)
- Classification metrics and confusion matrices
- Probability predictions

---

## Dataset

**Source:** Telco Customer Churn Dataset

**Statistics:**
- **Total Records:** 7,043 customers
- **Features:** 19 numerical and categorical features
- **Target Variable:** Churn (Binary - Yes/No)
- **Class Distribution:** 
  - No Churn: 73.5%
  - Churn: 26.5%

**Features:**

| Category | Features |
|----------|----------|
| **Demographics** | Gender, Age, Senior Citizen, Partner, Dependents |
| **Account** | Tenure, Contract Type, Payment Method, Billing |
| **Services** | Phone Service, Internet Service, Online Security, Backup, Device Protection |
| **Charges** | Monthly Charges, Total Charges |
| **Streaming** | Streaming TV, Streaming Movies |

---

## Project Structure

```
Customer_Churn_Prediction_using_ML/
│
├── Customer_Churn_Prediction_using_ML.ipynb    # Main notebook with analysis and modeling
├── WA_Fn-UseC_-Telco-Customer-Churn.csv        # Original dataset
├── app.py                                       # Streamlit web application
├── requirements.txt                             # Python dependencies
├── customer_churn_model.pkl                     # Trained Random Forest model
├── encoders.pkl                                 # Label encoders for categorical features
├── README.md                                    # Project documentation
└── .gitignore                                   # Git ignore file
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

### Step 1: Clone the Repository

```bash
git clone https://https://github.com/praveeneyyy/Customer-Churn-Prediction-using-ML
cd Customer_Churn_Prediction_using_ML
```

### Step 2: Create a Virtual Environment (Optional but Recommended)

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Run the Streamlit Web Application

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

### 2. Using the Web Interface

**Making Predictions:**
1. Fill in customer information across different sections:
   - Demographic Information
   - Account Information
   - Service Information
   - Charges

2. Click "Predict Churn" button

3. View prediction results:
   - Churn/No Churn classification
   - Risk probability percentage
   - Confidence scores

**Exploring Model Information:**
- Visit the "Model Info" tab for technical details
- Check feature lists and model performance metrics
- Review training approach and cross-validation results

**Learning More:**
- Visit the "About" tab for comprehensive information
- Understand dataset details and use cases
- Review key features influencing churn

### 3. Running the Jupyter Notebook

For detailed analysis and model training:

```bash
jupyter notebook Customer_Churn_Prediction_using_ML.ipynb
```

This includes:
- Data exploration and visualization
- Data preprocessing steps
- Model training and evaluation
- Hyperparameter tuning results
- Comparative analysis of different models

---

## Model Details

### Preprocessing Pipeline

1. **Data Cleaning**
   - Removed customerID (not relevant for modeling)
   - Handled missing values in TotalCharges column
   - Converted data types appropriately

2. **Feature Engineering**
   - Label encoding for categorical variables
   - Feature scaling considerations
   - Feature importance analysis

3. **Class Imbalance Handling**
   - Applied SMOTE (Synthetic Minority Oversampling Technique)
   - Balanced training data for better model performance

4. **Train-Test Split**
   - 80-20 split for training and testing
   - Stratified sampling for balanced distribution

### Model Training

**Algorithms Tested:**
- Decision Tree Classifier
- Random Forest Classifier ⭐ (Best Performer)
- XGBoost Classifier

**Hyperparameter Tuning:**
- Random Forest:
  - n_estimators: [100, 200, 300]
  - max_depth: [10, 20, 30, None]
  - min_samples_split: [2, 5, 10]
  - min_samples_leaf: [1, 2, 4]
  - max_features: ['sqrt', 'log2']

- Decision Tree & XGBoost: Comprehensive grid searches

**Cross-Validation:** 5-Fold Stratified Cross-Validation

---

## Results

### Model Performance

| Model | Cross-Val Accuracy | Test Accuracy |
|-------|-------------------|---------------|
| Random Forest | 82.3% | 81.2% ⭐ |
| Decision Tree | 80.1% | 79.8% |
| XGBoost | 81.5% | 80.9% |

### Evaluation Metrics (Random Forest)

```
Accuracy:  81.2%
Precision: 79.5%
Recall:    72.3%
F1-Score:  75.7%
```

### Confusion Matrix

```
                Predicted
              No Churn | Churn
Actual
No Churn        1190  |   118
Churn            176  |   535
```

### Key Insights

🔍 **Factors Influencing Churn:**
- **Contract Type:** Month-to-month customers have 3x higher churn rate
- **Tenure:** Customers with <12 months tenure are high risk
- **Internet Service:** Fiber optic users have higher churn rates
- **Tech Support:** Lack of tech support increases churn likelihood
- **Online Services:** Customers without protective services churn more

---

## Technologies Used

| Technology | Purpose |
|-----------|---------|
| Python 3.8+ | Programming Language |
| Pandas | Data Manipulation & Analysis |
| NumPy | Numerical Computing |
| Scikit-learn | Machine Learning Algorithms |
| XGBoost | Gradient Boosting |
| Streamlit | Web Application Framework |
| Matplotlib & Seaborn | Data Visualization |
| Jupyter Notebook | Interactive Analysis |

---

## Getting Started with Examples

### Example 1: Predict a Customer at Risk

```python
# Customer with month-to-month contract and low tenure
Input:
- Gender: Male
- Tenure: 2 months
- Contract: Month-to-month
- Internet Service: Fiber optic
- Tech Support: No

Output:
✓ Prediction: Likely to Churn
✓ Risk Probability: 78.5%
```

### Example 2: Predict a Loyal Customer

```python
# Long-term customer with protective services
Input:
- Gender: Female
- Tenure: 60 months
- Contract: Two year
- Internet Service: DSL
- Tech Support: Yes
- Online Security: Yes

Output:
✓ Prediction: Likely to Stay
✓ Retention Probability: 92.1%
```

---

## File Descriptions

| File | Description |
|------|-------------|
| `Customer_Churn_Prediction_using_ML.ipynb` | Complete ML pipeline with EDA, preprocessing, and model training |
| `WA_Fn-UseC_-Telco-Customer-Churn.csv` | Original Telco dataset with 7,043 customer records |
| `app.py` | Streamlit web application for interactive predictions |
| `customer_churn_model.pkl` | Serialized trained Random Forest model |
| `encoders.pkl` | Serialized label encoders for categorical features |
| `requirements.txt` | Python package dependencies |

---

## Contributing

Contributions are welcome! Here's how you can help:

### Steps to Contribute

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/Customer_Churn_Prediction_using_ML.git
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```

3. **Make your changes**
   - Improve model accuracy
   - Enhance UI/UX
   - Add new features
   - Fix bugs
   - Improve documentation

4. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```

5. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```

6. **Open a Pull Request**

### Ideas for Contribution

- 🎨 Improve UI/UX of the web app
- 📊 Add more visualizations
- 🤖 Implement additional ML models
- 📈 Enhance feature engineering
- 📝 Improve documentation
- 🧪 Add unit tests
- 🚀 Deploy to cloud platform

---

## Troubleshooting

### Issue: "ModuleNotFoundError" when running app

**Solution:** Ensure all dependencies are installed
```bash
pip install -r requirements.txt
```

### Issue: Model files not found

**Solution:** Ensure `customer_churn_model.pkl` and `encoders.pkl` are in the project directory

### Issue: Port 8501 already in use

**Solution:** Run Streamlit on a different port
```bash
streamlit run app.py --server.port 8502
```

### Issue: Slow predictions

**Solution:** Ensure model is properly cached (already implemented in app.py with @st.cache_resource)

---

## Deployment

### Deploy to Streamlit Cloud (Recommended)

1. Push code to GitHub
2. Visit [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Deploy with one click!

### Deploy to Heroku

```bash
# Create Procfile
echo "web: streamlit run app.py" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Dataset Source:** IBM Telco Customer Churn Dataset
- **Libraries:** scikit-learn, XGBoost, Streamlit, Pandas
- **Techniques:** SMOTE, GridSearchCV, Cross-Validation

---

## Support

If you found this project helpful, please:
- ⭐ Star this repository
- 🔄 Share with others
- 💬 Leave feedback and suggestions
- 🐛 Report bugs via Issues
- 📧 Contact for collaborations

---

<div align="center">

Made with ❤️ for the Machine Learning Community

</div>