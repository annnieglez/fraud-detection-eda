# Detecting Fraud Patterns in Credit Card Transactions: A Data Science Approach

## Project Overview  

This project analyzes **credit card transactions** to detect **fraud patterns** using **Exploratory Data Analysis (EDA)**. The goal is to identify key trends, anomalies, and insights in fraudulent vs. non-fraudulent transactions.  

### **Dataset:**

- **Source**: Kaggle - [Credit Card Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection?select=fraudTrain.csv)
- **Transactions**: 555,719
- **Fraud Cases**: 2,145 (0.39%)

### **Key Objectives:**

- Understand the **distribution of fraud vs. non-fraud transactions**  
- Analyze **time-based patterns** in fraudulent activity  
- Identify **high-risk transaction types**  
- Create **interactive visualizations** with Tableau  

### **Key Findings:**

- **Fraud transactions** often involve higher amounts.
- **Fraud is more frequent at night**.
- Fraud transaction values **increase during the day**.

### **Key Steps:**

1. **Data Preprocessing:**
   - Cleaned and processed the raw dataset by removing unnecessary columns and handling missing values.
   - Creating new features based on existing variables

2. **Exploratory Data Analysis (EDA):**
   - Explored the distribution of fraud vs. non-fraud transactions.
   - Analyzed transaction amounts and frequencies to identify patterns.
   - Investigated time-based trends in fraudulent transactions.
   - Univariate and bivariate analyses were performed to identify potential fraud indicators.

3. **Tableau Visualization:**
   - Built interactive dashboards in Tableau to showcase the findings visually, focusing on transaction patterns, fraud detection over time, and high-risk transaction types.
   - Added **filters for customer history** to allow deeper insights:
     - **Age**: Analyze trends in fraud across different age groups.
     - **Sex**: Explore the distribution of fraud incidents by gender.
     - **Location**: Filter by geographic regions to identify regional fraud patterns.
     - These filters enable users to explore how **age**, **sex**, and **location** impact fraudulent behavior.

## Project Structure

```bash
.
├── data                              # Folder containing the raw and processed data
│   ├── fraudTest.csv                 # The dataset used for fraud detection analysis
│   └── README.md                     # Data description and usage guide
├── images                            # Folder for storing any visual assets
├── reports                           # Folder to store project reports and analyses
│   └── Data Analysis & Insights.pdf  # Report on data analysis and insights gained from the project
├── scripts                           # Folder containing Python scripts used for preprocessing, analysis and visualization
│    ├── fonts                        # Folder for storing the custom font files used in the visualizations
│    │    └── Montserrat-Regular.ttf  # Custom font Montserrat-Regular
│    ├── __init__.py                  # Marks the 'scripts' directory as a Python package
│    ├── data_check.py                # Script for checking the dataset for inconsistencies or issues
│    ├── data_cleaning.py             # Script for cleaning and preprocessing the dataset (e.g., handling missing data)
│    └── eda_functions.py             # Script containing functions used in exploratory data analysis (EDA) and visualization
├── tableau                           # Folder to store Tableau workbooks and dashboards
│    └── Fraud_detection_insights.twb # Tableau workbook containing fraud detection insights and interactive dashboards
├── .gitattributes                    # Defines Git attributes like line endings, diff settings, etc.
│    └── data/*.csv                   # Use Git LFS for CSV files in the data folder
├── .gitignore                        # Specifies files/folders for Git to ignore (e.g., temporary files, credentials)
│    └── node_modules/                # Ignore node_modules directory
│    └── *.log                        # Ignore log files
│    └── __pycache__/                 # Ignore Python bytecode cache
│    └── *.pyc                        # Ignore Python bytecode files
├── LICENSE                           # License file containing the full license terms
├── README.md                         # The readme file with project overview, objectives, and instructions
├── main.ipynb                        # Main script to run the project
└── requirements.txt                  # List of Python dependencies required for the project (e.g., pandas, matplotlib, etc.)

```

## How to Run the Project

### 1. Install Dependencies

Before starting, make sure you have all the necessary dependencies installed. You can install them by running the following command in your terminal:

```bash
pip install -r requirements.txt 
```

### 2. **Run the Jupyter Notebook:**

Open `main.ipynb` in Visual Code and execute the cells sequentially:

```bash
main.ipynb 
```

## Technologies Used

- Python (Pandas, Matplotlib, Seaborn)
- Visual Studio Code
- Tableau (for interactive visualizations) - [Dashboard](https://public.tableau.com/app/profile/annie.meneses.gonzalez/viz/Book1_17412944537210/Dashboard1)
- Trello Board (for project management) - [Data Analytics Project](https://trello.com/b/IeT9tkSq/data-analytics-project)

## Next Steps

- The next phase of the project will involve using machine learning algorithms to build predictive models for detecting fraud.
- After selecting the best-performing model, the next step is to deploy it for real-time or batch predictions on new transactions.

## Author

Developed by Annie Meneses Gonzalez.

Feel free to connect via [LinkedIn](www.linkedin.com/in/annie-meneses-gonzalez-57bb9b145) or check out more projects on [GitHub](https://github.com/annnieglez) and [Tableau](https://public.tableau.com/app/profile/annie.meneses.gonzalez/vizzes)

 This is an open-source project and contributions are welcome. Please ensure that you follow the MIT license guidelines while contributing to this project.

## License

This project is licensed under the [MIT License](LICENSE).
