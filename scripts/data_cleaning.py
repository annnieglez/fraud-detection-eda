'''This file groups functions for data cleaning, such as 
    formatting columns to a consistent format.'''

import pandas as pd
from datetime import datetime

def snake(data_frame):
    '''
    Converts column names to snake_case (lowercase with underscores).
    
    Parameters:
        - data_frame (pd.DataFrame): The input DataFrame whose columns need to be formatted.

    Returns:
        - pd.DataFrame: DataFrame with column names in snake_case.
    '''

    data_frame.columns = [column.lower().replace(" ", "_") for column in data_frame.columns]

    return data_frame

def drop_col(data_frame, columns):
    '''
    Drops specified columns from a DataFrame.
    
    Parameters:
        - data_frame (pd.DataFrame): The input DataFrame from which columns will be dropped.
        - columns (list or str): A list of column names or a single column name to be dropped.
    
    Returns:
        - pd.DataFrame: The DataFrame with the specified columns removed.
    '''

    # Check for columns that do not exist in the DataFrame
    missing_cols = [col for col in columns if col not in data_frame.columns]

    # If there are missing columns, print a message and exclude them from the drop list
    if missing_cols:
        print(f"Warning: The following columns were not found and will be skipped: {', '.join(missing_cols)}")
        columns = [col for col in columns if col in data_frame.columns]  # Keep only existing columns
    
    # Drop the existing columns
    data_frame = data_frame.drop(columns, axis=1)

    return data_frame

def remove_prefix_from_column(data_frame, column, word):
    '''
    Removes a specified word from the beginning of each entry in a column.

    Parameters:
        - data_frame (pd.DataFrame): The input DataFrame.
        - column (str): The column from which to remove the prefix.
        - word (str): The word to remove from the beginning of each entry.

    Returns:
        - pd.DataFrame: The DataFrame with the updated column.
    '''

    data_frame[column] = data_frame[column].apply(lambda x: x[len(word):] if isinstance(x, str) and x.startswith(word) else x)

    return data_frame

def clean_category_column(data_frame, column):
    '''
    Cleans the specified column by removing underscores and capitalizing the first letter of each word
    for each row in the column.
    
    Parameters:
        - data_frame (pd.DataFrame): The input DataFrame.
        - column (str): The name of the column to clean.
    
    Returns:
        - pd.DataFrame: The DataFrame with the cleaned column.
    '''

    data_frame[column] = data_frame[column].apply(lambda x: x.replace('_', ' ').title() if isinstance(x, str) else x)
    
    return data_frame


def clean_gender_column(data_frame):
    '''
    Cleans the gender column by replacing 'F' with 'Female' and 'M' with 'Male'.
    
    Parameters:
        - data_frame (pd.DataFrame): The input DataFrame.
    
    Returns:
        - pd.DataFrame: The DataFrame with the cleaned gender column.
    '''

    data_frame['gender'] = data_frame['gender'].replace({'M': 'Male', 'F': 'Female'})

    return data_frame

def replace_state_abbreviations(data_frame):
    '''
    Replaces U.S. state abbreviations with their full state names.

    Parameters:
        - data_frame (pd.DataFrame): The DataFrame containing the state abbreviations.
    
    Returns:
        - pd.DataFrame: The DataFrame with state abbreviations replaced by full state names.
    '''

    # Dictionary of state abbreviations and their full names
    state_dict = {
        'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California', 
        'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia', 
        'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 
        'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland', 
        'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri', 
        'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey', 
        'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 
        'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina', 
        'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont', 
        'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'
    }
    
    # Replace state abbreviations with full names
    data_frame['state'] = data_frame['state'].replace(state_dict)

    return data_frame

def convert_to_datetime(data_frame, columns):
    '''
    Converts specified columns to datetime format.
    
    Parameters:
        - data_frame (pd.DataFrame): The input DataFrame.
        - columns (str or list): Column name or list of column names to convert to datetime.
    
    Returns:
        - pd.DataFrame: The DataFrame with specified columns converted to datetime.
    '''

    # If a single column is provided, convert it to a list
    if isinstance(columns, str):
        columns = [columns]
    
    # Loop through each specified column and convert to datetime
    for col in columns:
        if not pd.api.types.is_datetime64_any_dtype(data_frame[col]):
            data_frame[col] = pd.to_datetime(data_frame[col], errors='coerce')
    
    return data_frame

def age_calculator(data_frame):
    '''
    Calculates the age of each person based on their date of birth.
    
    Parameters:
        - data_frame (pd.DataFrame): The DataFrame containing the date of birth.
    
    Returns:
        - pd.DataFrame: The DataFrame with the new 'age' column in years.
    '''

    current_date = datetime.now()
    data_frame['age'] = data_frame['dob'].apply(lambda x: (current_date - x).days // 365)

    return data_frame

def preprocess_datetime(data_frame):
    '''
    Extracts time-based features from 'trans_date_trans_time'.
    
    Parameters:
        - data_frame (pd.DataFrame): The dataset containing the transaction date/time column.
    
    Returns:
        - pd.DataFrame: The modified dataframe with new time-based columns.
    '''

    data_frame['hour'] = data_frame['trans_date_trans_time'].dt.hour
    data_frame['day'] = data_frame['trans_date_trans_time'].dt.day
    data_frame['month'] = data_frame['trans_date_trans_time'].dt.month
    data_frame['day_of_week'] = data_frame['trans_date_trans_time'].dt.day_name()

    preprocess_datetime_day_of_week(data_frame)

    return data_frame

def preprocess_datetime_day_of_week(data_frame):
    '''
    Preprocess the 'day_of_week' column in the provided DataFrame, converting it to a 
    categorical variable with an ordered list of days starting from Sunday to Saturday.
    
    Parameters:
        data_frame (pd.DataFrame): The dataset containing the 'day_of_week' column, 
                                    which will be converted to a categorical type.
    
    Returns:
        pd.DataFrame: The modified DataFrame with 'day_of_week' column as a categorical 
                      variable with specified order.
    '''

    data_frame['day_of_week'] = pd.Categorical(
        data_frame['day_of_week'], 
        categories=['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'], 
        ordered=True
    )

    return data_frame