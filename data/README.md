# Credit Card Transactions Fraud Detection Dataset

This is a simulated credit card transaction dataset containing legitimate and fraud transactions from the duration 1st Jan 2019 - 31st Dec 2020. It covers credit cards of 1000 customers doing transactions with a pool of 800 merchants.

This dataset contains information about credit card transactions and is used to detect fraudulent transactions. It includes multiple features such as transaction details, customer demographics, merchant information, and geographic data. The dataset is designed to aid in building fraud detection models and performing exploratory data analysis (EDA).

## Dataset Columns Description

1. **index**  
   - **Description**: Unique Identifier for each row.  
   - **Type**: Integer

2. **trans_date_trans_time**  
   - **Description**: The date and time when the transaction took place.  
   - **Type**: Datetime

3. **cc_num**  
   - **Description**: The credit card number of the customer making the transaction.  
   - **Type**: String

4. **merchant**  
   - **Description**: The name of the merchant where the transaction took place.  
   - **Type**: String

5. **category**  
   - **Description**: The category of the merchant where the transaction took place (e.g., Electronics, Grocery).  
   - **Type**: String

6. **amt**  
   - **Description**: The amount of money for the transaction.  
   - **Type**: Float

7. **first**  
   - **Description**: The first name of the credit card holder.  
   - **Type**: String

8. **last**  
   - **Description**: The last name of the credit card holder.  
   - **Type**: String

9. **gender**  
   - **Description**: The gender of the credit card holder.  
   - **Type**: String (Male, Female)

10. **street**  
    - **Description**: The street address of the credit card holder.  
    - **Type**: String

11. **city**  
    - **Description**: The city of the credit card holder.  
    - **Type**: String

12. **state**  
    - **Description**: The state of the credit card holder.  
    - **Type**: String

13. **zip**  
    - **Description**: The zip code of the credit card holder's address.  
    - **Type**: String

14. **lat**  
    - **Description**: The latitude location of the credit card holder.  
    - **Type**: Float

15. **long**  
    - **Description**: The longitude location of the credit card holder.  
    - **Type**: Float

16. **city_pop**  
    - **Description**: The population of the city where the credit card holder resides.  
    - **Type**: Integer

17. **job**  
    - **Description**: The job or occupation of the credit card holder.  
    - **Type**: String

18. **dob**  
    - **Description**: The date of birth of the credit card holder.  
    - **Type**: Datetime

19. **trans_num**  
    - **Description**: Unique transaction number for the specific transaction.  
    - **Type**: Integer

20. **unix_time**  
    - **Description**: The UNIX timestamp of the transaction date and time.  
    - **Type**: Integer (seconds since January 1, 1970)

21. **merch_lat**  
    - **Description**: The latitude location of the merchant where the transaction took place.  
    - **Type**: Float

22. **merch_long**  
    - **Description**: The longitude location of the merchant where the transaction took place.  
    - **Type**: Float

23. **is_fraud**  
    - **Description**: Flag indicating whether the transaction is fraudulent.  
    - **Type**: Integer (0 = Non-Fraudulent, 1 = Fraudulent)  
    - **Target Class**

## Additional Notes

- **Fraud Detection**: The target class `is_fraud` is the primary label used for fraud detection, where `1` represents fraudulent transactions, and `0` represents non-fraudulent transactions.
- The dataset includes both personal information of cardholders and merchant-specific data, as well as transaction details and geographic coordinates, allowing for a comprehensive analysis of fraud patterns.
  
## Data Source

This dataset is sourced from [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection?select=fraudTrain.csv).
