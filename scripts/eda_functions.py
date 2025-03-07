'''This module contains functions and configurations for visualizing data, 
    including setting up a clean and professional styling for plots, defining 
    color palettes, and handling font customization. 
'''

# Standard Libraries
import os

# Data Handling & Computation
import pandas as pd
import numpy as np

# Visualization Libraries
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as patches

# ==============================
# Directory Setup
# ==============================

# Define the directory name for saving images
OUTPUT_DIR = "images"

# Check if the directory exists, if not, create it
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==============================
# Plot Styling & Customization
# ==============================

# Set a Minimalist Style
sns.set_style("whitegrid")
# Customize Matplotlib settings for a modern look
mpl.rcParams.update({
    'axes.edgecolor': 'grey',       
    'axes.labelcolor': 'black',     
    'xtick.color': 'black',         
    'ytick.color': 'black',         
    'text.color': 'black'           
})

# General color palette for plots
custom_colors = ["#1F4E79", "#8F2C78"]

# Colors for fraud and non-fraud visualization
fraud_color = "#8F2C78"  
non_fraud_color = "#1F4E79"

# Define a custom colormap from light to dark shades of purple
custom_cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "custom_purple", ["#F5A7C4", "#8F2C78", "#5C0E2F"]
)

# ==============================
# Font Configuration
# ==============================

# Path to the custom font file
FONT_PATH = './scripts/fonts/Montserrat-Regular.ttf'

# Add the font to matplotlib's font manager
font_manager.fontManager.addfont(FONT_PATH)

# Set the font family to Montserrat
plt.rcParams['font.family'] = 'Montserrat'

# ==============================
# Custom Formatter Functions
# ==============================

def currency_formatter(x, pos):
    '''
    Custom formatter function to display y-axis values,
    formatted as currency with comma separators.

    Parameters:
    - x (float): The numerical value to format.
    - pos (int): The tick position (required for matplotlib formatters).

    Returns:
    - str: Formatted string representation of the value.
    '''
    return f'${x:,.2f}'

def remove_axes():
    '''
    Removes the axes from the current plot by hiding the spines.

    Usage:
        Call this function after plotting your data to remove the axes from the figure.
    '''

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

def safe_mode(series):
    '''Returns the mode if available, otherwise returns NaN'''
    mode_values = series.mode()
    return mode_values.iloc[0] if not mode_values.empty else np.nan


# ==============================
# Vizualization Functions
# ==============================

def calculate_fraud_percentage(data_frame):
    '''
    Calculates the percentage of fraud vs. non-fraud transactions.

    It then creates two tables: one with a breakdown of the counts and percentages, 
    and another with a total row summarizing the overall counts and percentages.

    Parameters:
        - data_frame (pd.DataFrame): The dataset containing the 'is_fraud' column,
                                      which indicates whether a transaction is fraudulent (1) or not (0).
    
    Returns:
        - df (pd.DataFrame): A DataFrame containing the absolute and relative frequencies 
                              of fraud vs. non-fraud transactions.
        - df_total (pd.DataFrame): A DataFrame containing a summary row with total counts 
                               and percentages for fraud vs. non-fraud transactions.
    '''

    # Compute the absolute and relative frequencies of fraud occurrences
    fraud_count = data_frame['is_fraud'].value_counts()
    fraud_count_n = data_frame['is_fraud'].value_counts(normalize=True).round(4) * 100

    # Combine absolute and relative frequencies into a single DataFrame
    df = pd.concat([fraud_count, fraud_count_n], axis=1)
    df.columns = ["Absolute frequency", "Relative frequency"]

    # Set the 'is_fraud' values (0, 1) as the index of the table
    df.index = df.index.map({1: 'Fraud', 0: 'Non-Fraud'})  

    # Create a row for the total counts and percentages and append it to the DataFrame
    total_absolute = fraud_count.sum()
    total_relative = fraud_count_n.sum()
    total_row = pd.DataFrame({
        'Absolute frequency': [total_absolute],
        'Relative frequency': [total_relative]
    }, index=['Total']) 

    # Concatenate the totals row to the existing DataFrame
    df_total = pd.concat([df,total_row])
    
    return (df, df_total)

def plot_fraud_percentage(data_frame):
    '''
    Creates a pie chart and bar chart for fraud vs. non-fraud transactions.
    
    Parameters:
        - data_frame (pd.DataFrame): The dataset containing the fraud column.

    Returns:
        - None: This function generates and saves the plots.
    '''

    fraud_percentage, fraud_percentage_total = calculate_fraud_percentage(data_frame)

    # Bar chart
    fig_bar, ax_bar = plt.subplots(figsize=(7, 5))
    sns.barplot(x='is_fraud', 
                y='Absolute frequency', 
                hue='is_fraud', 
                data=fraud_percentage, 
                ax=ax_bar, 
                legend=False, 
                palette=custom_colors)
    ax_bar.set_title('Non-Fraud vs. Fraud Absolute Frequency', 
                    fontsize=14, 
                    fontweight='regular', 
                    color='black')
    ax_bar.set_xlabel('Fraud Status', fontsize=12)
    ax_bar.set_ylabel(f'Absolute Frequency', fontsize=12, labelpad=10) 
    ax_bar.tick_params(axis='x')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    fig_bar.savefig(os.path.join(OUTPUT_DIR, f"Bar_plot_fraud_vs_non-fraud.png"), 
                    bbox_inches='tight', 
                    facecolor='none', 
                    transparent=True)

    # Log Scale Bar chart
    fig_bar, ax_bar_log = plt.subplots(figsize=(7, 5))
    bars = sns.barplot(x='is_fraud', 
                       y='Absolute frequency', 
                       hue='is_fraud', 
                       data=fraud_percentage, 
                       ax=ax_bar_log, 
                       legend=False, 
                       palette=custom_colors)
    ax_bar_log.set_title('Non-Fraud vs. Fraud Absolute Frequency (Log Scale)', 
                        fontsize=14, 
                        fontweight='regular', 
                        color='black')
    ax_bar_log.set_xlabel('Fraud Status', fontsize=12)
    ax_bar_log.set_ylabel('Absolute Frequency (log scale)', fontsize=12, labelpad=10)
    ax_bar_log.set_yscale('log')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Generate logarithmic ticks
    major_ticks = [10**4,  10**5,  10**6]  # Added 10^6
    ax_bar_log.set_yticks(major_ticks)

    # Adding total number of transaction on top of the bars
    for bar in bars.patches:
        height = bar.get_height()
        ax_bar_log.text(bar.get_x() + bar.get_width()/2, height, 
                    f'{int(height)}', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')

    fig_bar.savefig(os.path.join(OUTPUT_DIR, f"Bar_plot_fraud_vs_non-fraud_log.png"), 
                    bbox_inches='tight', 
                    facecolor='none', 
                    transparent=True)

    # Pie chart
    fig_pie, ax_pie = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax_pie.pie(fraud_percentage['Absolute frequency'], 
                                          labels=fraud_percentage.index, 
                                          autopct='%1.2f%%', 
                                          startangle=90, 
                                          colors=custom_colors, 
                                          pctdistance=0.6, 
                                          explode=(0, 0.1))
    ax_pie.set_title('Non-Fraud vs. Fraud Relative Frequency', fontsize=14, fontweight='regular', color='black')
    ax_pie.set_ylabel('')

    for i, (label, text) in enumerate(zip(texts, autotexts)):
        if i == 0: 
            label.set_position((-0.15, -0.5))  
            text.set_position((-0.4, -0.35))  
            label.set_color('white')
            text.set_color('white') 
            label.set_fontweight('black') 
            label.set_fontsize(12)
        else:
            label.set_position((0.1, 1.1))   
            text.set_position((0.6, 1.1))
            label.set_color(fraud_color)
            text.set_color(fraud_color)   
            label.set_fontweight('black') 
            label.set_fontsize(12)

    fig_pie.savefig(os.path.join(OUTPUT_DIR, f"Pie_plot_relative_frequency.png"), 
                    bbox_inches='tight', 
                    facecolor='none', 
                    transparent=True)

    # Donut chart
    fig_do, ax_do = plt.subplots(figsize=(7, 5))
    
    # Plot the Donut chart
    wedges, texts, autotexts = ax_do.pie(fraud_percentage['Absolute frequency'], 
                                        labels=fraud_percentage['Absolute frequency'].index, 
                                        autopct='%1.2f%%', 
                                        startangle=90, 
                                        colors=custom_colors, 
                                        pctdistance=0.6, 
                                        explode=(0, 0.1), 
                                        wedgeprops={'width': 0.3})
    
    ax_do.set_title('Non-Fraud vs. Fraud Relative Frequency', fontsize=14, fontweight='regular', color='black')
    ax_do.set_ylabel('')

    for i, (label, text) in enumerate(zip(texts, autotexts)):
        if i == 0:  
            label.set_position((-0.65, -0.9))  
            text.set_position((-1.0, -0.75))  
            label.set_color('#1F4E79')
            text.set_color('#1F4E79') 
            label.set_fontweight('black') 
            label.set_fontsize(12)
        else:  
            label.set_position((0.1, 1.1))   
            text.set_position((0.6, 1.1))
            label.set_color(fraud_color)
            text.set_color(fraud_color)  
            label.set_fontweight('black') 
            label.set_fontsize(12)

    fig_pie.savefig(os.path.join(OUTPUT_DIR, f"Donut_plot_relative_frequency.png"), 
                    bbox_inches='tight', 
                    facecolor='none', 
                    transparent=True)

def plot_fraud_frequency_by_card(data_frame):
    '''
    Creates a histogram showing the frequency of fraud transactions per credit card.
    
    Parameters:
        - data_frame (pd.DataFrame): The dataset containing 'cc_num' and 'is_fraud' columns.
    
    Returns:
        - None: This function generates and saves the plots.
    '''
    # Let's filter fraud transactions and count fraud occurrences for each card
    fraud_data = data_frame[data_frame['is_fraud'] == 1]
    fraud_counts = fraud_data['cc_num'].value_counts()

    # Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(fraud_counts, bins=17, kde=False, color=fraud_color)
    plt.xlabel('Number of Fraud Transactions per Credit Card', fontsize=12)
    plt.ylabel('Frequency of Fraud', fontsize=12)
    plt.title('Fraud Transaction Frequency by Credit Card', fontsize=14)

    # Format the x-axis ticks as integers
    plt.xticks(ticks=range(int(fraud_counts.min()), 
                           int(fraud_counts.max()) + 1), 
                           labels=[str(i) for i in range(int(fraud_counts.min()), 
                                                         int(fraud_counts.max()) + 1)])

    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.grid(axis='x', visible=False)
    plt.savefig(os.path.join(OUTPUT_DIR, "fraud_per_cc.png"), 
                bbox_inches='tight', 
                facecolor='none', 
                transparent=True) 

def analyze_transaction_amounts(data_frame):
    """
    This function analyzes and compares the average transaction amounts
    for fraud vs. non-fraud transactions.

    Parameters::
        - data_frame (pd.DataFrame): DataFrame containing the transactions data.

    Returns:
        - avg_transaction_amounts (pd.DataFrame): A table with average transaction amounts for fraud and non-fraud categories.
    """
    
    # Group by 'is_fraud' and calculate the mean of the 'amt' column for each group
    avg_transaction_amounts = data_frame.groupby('is_fraud', observed=False)['amt'].mean().reset_index()
    avg_transaction_amounts.index = avg_transaction_amounts.index.map({0: 'Non-Fraud', 1: 'Fraud'})  
    avg_transaction_amounts = avg_transaction_amounts.drop("is_fraud", axis=1)
    avg_transaction_amounts.columns = ["Average transactions value"]

    return avg_transaction_amounts

def fraud_statistics(data_frame, filter = False):
    '''
    Generates a statistical description of the data, divided by fraud and non-fraud.
    If filter set to True it grouped by transactions between the hours 00:00 and 12:00 PM.

    Parameters:
        - data_frame (pd.DataFrame): The dataset containing time-based features and fraud flag.

    Returns:
        - pd.DataFrame: A DataFrame containing the statistical summary for fraud and non-fraud.
    '''

    if filter:
        # Filter transactions for hours between 0 and 12
        filtered_data = data_frame[(data_frame['hour'] >= 0) & (data_frame['hour'] <= 11)]
    else: 
        filtered_data = data_frame

    # Grouping by 'is_fraud' and calculating statistics
    grouped = filtered_data.groupby('is_fraud')['amt'].agg([
        'count', 'mean', 'median',  ('mode', safe_mode), 'std', 'min', 'max',
        ('25%', lambda x: x.quantile(0.25)), 
        ('75%', lambda x: x.quantile(0.75))  
    ]).reset_index()

    # Adding a new column for IQR
    grouped['IQR'] = grouped['75%'] - grouped['25%']
    # Sort the data by fraud status (Non-Fraud = 0 first)
    grouped = grouped.loc[grouped['is_fraud'].argsort()]
    grouped.index = grouped.index.map({0: 'Non-Fraud', 1: 'Fraud'})  
    grouped = grouped.drop("is_fraud", axis=1)

    grouped.columns = [col.title() if col != 'IQR' else col for col in grouped.columns]

    return grouped

def plot_fraud_transaction_amounts(data_frame, zoom):
    '''
    Creates a boxplot and violin plot to compare transaction amounts for fraud vs. non-fraud transactions.

    Parameters:
        - data_frame (pd.DataFrame): The dataset containing the transaction amounts and fraud status.
        - zoom (bool): If True y axes get resize to remove outliers

    Returns:
        - None: This function generates and saves the plots.
    '''
    # Replace 0 with 'Non-Fraud' and 1 with 'Fraud' in the 'is_fraud' column
    data_frame['is_fraud'] = data_frame['is_fraud'].replace({0: 'Non-Fraud', 1: 'Fraud'})

    # Histogram
    fig, ax1 = plt.subplots(figsize=(9, 8))

    # Fraud transactions on the left axis
    fraud_data = data_frame[data_frame['is_fraud'] == 'Non-Fraud']['amt']
    sns.histplot(fraud_data, bins=2000, color=non_fraud_color, label='Non-Fraud', ax=ax1)
    ax1.set_title('Histogram of Transaction Amounts Non-Fraud vs. Fraud', 
                  fontsize=14, 
                  fontweight='regular', 
                  color='black')
    ax1.set_xlabel('Transaction Amount', fontsize=12)
    ax1.set_ylabel(f'Frequency Non-Fraud', fontsize=12, labelpad=10, color=non_fraud_color)
    ax1.tick_params(axis='y', labelcolor=non_fraud_color)
    ax1.set_xlim(left=0)

    def currency_formatter(x, pos):
        return f'${x:,.2f}'  

    ax1.xaxis.set_major_formatter(FuncFormatter(currency_formatter))
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
   
    # Non-Fraud transactions on the right axis
    ax2 = ax1.twinx()  
    non_fraud_data = data_frame[data_frame['is_fraud'] == 'Fraud']['amt']
    sns.histplot(non_fraud_data, bins=200, color=fraud_color, label='Fraud', ax=ax2)
    ax2.set_ylabel(f'Frequency Fraud', color=fraud_color, fontsize=12, labelpad=10)
    ax2.set_xlim(left=0)
    ax2.xaxis.set_major_formatter(FuncFormatter(currency_formatter))
    ax2.tick_params(axis='y', labelcolor=fraud_color)
    ax2.grid(False)

    # Combine legends from both axes into one and place it on the right side
    handles, labels = ax1.get_legend_handles_labels() 
    handles2, labels2 = ax2.get_legend_handles_labels()  
    handles.extend(handles2) 
    labels.extend(labels2) 
    ax1.legend(handles=handles, labels=labels, loc='upper left', fontsize=12, bbox_to_anchor=(1.05, 1))

    if zoom:
        Q1 = data_frame[data_frame['is_fraud'] == 'Fraud']['amt'].quantile(0.25)
        Q3 = data_frame[data_frame['is_fraud'] == 'Fraud']['amt'].quantile(0.75)
        IQR = Q3 - Q1
        max_value = Q3 + 1.5 * IQR

    if zoom:
        plt.xlim(0, max_value)
        plt.savefig(os.path.join(OUTPUT_DIR, "hist_amt_zoom.png"), 
                    bbox_inches='tight', 
                    facecolor='none', 
                    transparent=True)
    else:
        plt.savefig(os.path.join(OUTPUT_DIR, "hist_amt.png"), 
                    bbox_inches='tight', 
                    facecolor='none', 
                    transparent=True)

    # Boxplot of Transaction Amounts for Fraud vs Non-Fraud
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='is_fraud', 
                y='amt', 
                data=data_frame, 
                hue='is_fraud', 
                legend=False, 
                palette=custom_colors, 
                showfliers= not zoom)
    
    plt.title('Transaction Amounts for Non-Fraud vs. Fraud', fontsize=14, fontweight='regular', color='black')
    plt.xlabel('Fraud Status', fontsize=12)
    plt.ylabel('Transaction Amount', fontsize=12, labelpad=10)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.yscale('log')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(currency_formatter))

    if zoom:
        plt.savefig(os.path.join(OUTPUT_DIR, "Boxplot_fraud_vs_non-fraud_zoom.png"), 
                    bbox_inches='tight', 
                    facecolor='none', 
                    transparent=True)
    else:
        plt.savefig(os.path.join(OUTPUT_DIR, "Boxplot_fraud_vs_non-fraud.png"), 
                    bbox_inches='tight', 
                    facecolor='none', 
                    transparent=True)

    # Violin Plot of Transaction Amounts for Fraud vs Non-Fraud
    plt.figure(figsize=(8, 6))
    sns.violinplot(x='is_fraud', 
                   y='amt', 
                   data=data_frame, 
                   hue='is_fraud', 
                   legend=False, 
                   palette=custom_colors)
    
    plt.title('Transaction Amounts for Non-Fraud vs. Fraud', fontsize=14, fontweight='regular', color='black')
    plt.xlabel('Fraud Status', fontsize=12)
    plt.ylabel('Transaction Amount', fontsize=12, labelpad=10)
    plt.ylim(bottom=0)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(currency_formatter))

    if zoom:
        plt.ylim(0, max_value)
        plt.savefig(os.path.join(OUTPUT_DIR, "Violin_plot_fraud_vs_non-fraud_zoom.png"), 
                    bbox_inches='tight', 
                    facecolor='none', 
                    transparent=True)
    else:
        plt.savefig(os.path.join(OUTPUT_DIR, "Violin_plot_fraud_vs_non-fraud.png"), 
                    bbox_inches='tight', 
                    facecolor='none', 
                    transparent=True)

def fraud_trends(data_frame, time_unit='month', aggregation = 'sum'):
    '''
    This function analyzes and compares the fraud vs. non-fraud trend over time.
    
    Parameters:
        - data_frame (pd.DataFrame): The dataset containing time-based features and fraud flag.
        - time_unit (str): The time unit to group data by ('hour', 'day', or 'month').
        - aggregation (str): The aggregation method for the 'amt' column ('sum', 'mean', 'median', 'mode', 'max', 'min').
    
    Returns:
        - fraud_counts_pivot (pd.DataFrame): Data in wide format showing fraud and non-fraud trends over time.
    '''

    # Aggregate based on the selected aggregation method
    if aggregation == 'sum':
        operation = 'Count'
        aggregated_data = data_frame.groupby([time_unit, 'is_fraud'], observed=False).size().reset_index(name='Count')
    elif aggregation == 'mean':
        operation = 'Mean'
        aggregated_data = data_frame.groupby([time_unit, 'is_fraud'], observed=False)['amt'].mean().round(2).reset_index(name='Mean')
    elif aggregation == 'median':
        operation = 'Median'
        aggregated_data = data_frame.groupby([time_unit, 'is_fraud'], observed=False)['amt'].median().round(2).reset_index(name='Medisan')
    elif aggregation == 'mode':
        operation = 'Mode'
        aggregated_data = data_frame.groupby([time_unit, 'is_fraud'], observed=False)['amt'].apply(lambda x: x.mode()[0]).round(2).reset_index(name='Mode')
    elif aggregation == 'max':
        operation = 'Max'
        aggregated_data = data_frame.groupby([time_unit, 'is_fraud'], observed=False)['amt'].max().round(2).reset_index(name='Max')
    elif aggregation == 'min':
        operation = 'Min'
        aggregated_data = data_frame.groupby([time_unit, 'is_fraud'], observed=False)['amt'].min().round(2).reset_index(name='Min')

    # Pivot the data so we have separate columns for fraud and non-fraud
    fraud_counts_pivot = aggregated_data.pivot(index='is_fraud', columns=time_unit, values=operation)
    fraud_counts_pivot.index = ['Fraud', 'Non-Fraud']
    fraud_counts_pivot.columns = [str(col).capitalize() if isinstance(col, str) else col for col in fraud_counts_pivot.columns]

    return fraud_counts_pivot

def fraud_statistics(data_frame, filter = False):
    '''
    Generates a statistical description of the data, divided by fraud and non-fraud,
    grouped by the specified time unit (hour, day, or month).

    Parameters:
        - data_frame (pd.DataFrame): The dataset containing time-based features and fraud flag.
        - filter (bool): Whether to filter transactions for hours between 0 and 12.

    Returns:
        - pd.DataFrame: A DataFrame containing the statistical summary for fraud and non-fraud.
    '''

    if filter:
        # Filter transactions for hours between 0 and 12
        filtered_data = data_frame[(data_frame['hour'] >= 0) & (data_frame['hour'] <= 11)]
    else: 
        filtered_data = data_frame

    # Grouping by 'is_fraud' and calculating statistics
    grouped = filtered_data.groupby('is_fraud')['amt'].agg([
        'count', 'mean', 'median',  ('mode', safe_mode), 'std', 'min', 'max',
        ('25%', lambda x: x.quantile(0.25)),  # Q1
        ('75%', lambda x: x.quantile(0.75))  # Q3
    ]).reset_index()
    grouped['IQR'] = grouped['75%'] - grouped['25%']
    grouped = grouped.loc[grouped['is_fraud'].argsort()]
    grouped.index = grouped.index.map({0: 'Non-Fraud', 1: 'Fraud'})  
    grouped = grouped.drop("is_fraud", axis=1)
    grouped.columns = [col.title() if col != 'IQR' else col for col in grouped.columns]

    return grouped

def plot_fraud_trends(data_frame, time_unit='month', aggregation='sum'):
    '''
    Plots fraud trends over time using bar plots

    Parameters:
        - data_frame (pd.DataFrame): The dataset containing time-based features and fraud flag.
        - time_unit (str): The time unit to group data by ('hour', 'day', 'month', or 'day_of_week').
        - aggregation (str): The aggregation method ('sum', 'mean', 'median', 'mode', 'max', 'min').
       
    Returns:
        - None: This function generates the plot and saves it.
    '''

    data_frame['is_fraud'] = data_frame['is_fraud'].replace({'Non-Fraud': 0, 'Fraud': 1}).astype(int).infer_objects()

    # Aggregate the counts based on the selected aggregation method
    if aggregation == 'sum':
        operation = 'Count'
        aggregated_data = data_frame.groupby([time_unit, 'is_fraud'], observed=False).size().reset_index(name='Count')
    elif aggregation == 'mean':
        operation = 'Mean'
        aggregated_data = data_frame.groupby([time_unit, 'is_fraud'], observed=False)['amt'].mean().round(2).reset_index(name='Count')
    elif aggregation == 'median':
        operation = 'Median'
        aggregated_data = data_frame.groupby([time_unit, 'is_fraud'], observed=False)['amt'].median().round(2).reset_index(name='Count')
    elif aggregation == 'mode':
        operation = 'Mode'
        aggregated_data = data_frame.groupby([time_unit, 'is_fraud'], observed=False)['amt'].apply(lambda x: x.mode()[0]).round(2).reset_index(name='Count')
    elif aggregation == 'max':
        operation = 'Max'
        aggregated_data = data_frame.groupby([time_unit, 'is_fraud'], observed=False)['amt'].max().round(2).reset_index(name='Count')
    elif aggregation == 'min':
        operation = 'Min'
        aggregated_data = data_frame.groupby([time_unit, 'is_fraud'], observed=False)['amt'].min().round(2).reset_index(name='Count')

    # Pivot the data so we have separate columns for fraud and non-fraud
    fraud_pivot = aggregated_data.pivot(index='is_fraud', columns=time_unit, values='Count').fillna(0)

    # Bar plot
    fig2, ax3 = plt.subplots(figsize=(10, 5))
    bar_width = 0.4
   
    # Generate positions for fraud and non-fraud bars
    if time_unit == 'month':
        x_positions = range(6, 6+len(fraud_pivot.columns))
    else:
        x_positions = range(len(fraud_pivot.columns))

    # Plot bars for fraud and non-fraud transactions (side-by-side)
    bars_non_fraud = ax3.bar([x - bar_width/2 for x in x_positions], fraud_pivot.loc[0], width=bar_width, color=non_fraud_color, label='Non-Fraud', zorder=3)
    ax4 = ax3.twinx()
    bars_fraud = ax4.bar([x + bar_width/2 for x in x_positions], fraud_pivot.loc[1], width=bar_width, color=fraud_color, label='Fraud', zorder=3)
    ax3.grid(True, axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax4.grid(False)

    # Adjust axes limits based on time_unit
    if time_unit == 'hour':
        ax3.set_xlim(-0.5, 23.5)
        ax3.set_xticks(range(24))
    elif time_unit == 'day':
        ax3.set_xlim(-0.5, 30.5)
        ax4.set_xlim(-0.5, 30.5)
    elif time_unit == 'month':
        ax3.set_xlim(5.5, 12.5)

    # Labels and title
    if time_unit == 'day_of_week':
        x_label = 'Day of Week'
        label = 'Day of Week'
    else:
        x_label = time_unit.title()
        label = time_unit.title() 

    ax3.set_xlabel(x_label, fontsize=12)
    ax3.set_ylabel(f"{operation.title()} Non-Fraud Transactions", fontsize=12, color = non_fraud_color)
    ax4.set_ylabel(f"{operation.title()} Fraud Transactions", fontsize=12, color = fraud_color)
    plt.title(f'Non-Fraud vs. Fraud Transactions {operation} ({label})')

    ax3.tick_params(axis='y', labelcolor=non_fraud_color)
    ax4.tick_params(axis='y', labelcolor=fraud_color)

    handles, labels = ax3.get_legend_handles_labels() 
    handles2, labels2 = ax4.get_legend_handles_labels()  
    handles.extend(handles2) 
    labels.extend(labels2) 
    ax3.legend(handles=handles, labels=labels, loc='upper left', fontsize=12, bbox_to_anchor=(1.1, 1))

    plt.tight_layout()  # Ensure the layout doesn't overlap

    if (operation == "Mean") or (operation == "Median") or (operation == "Mode") or (operation == "Min") or (operation == "Max"): 
        ax3.yaxis.set_major_formatter(FuncFormatter(currency_formatter))
        ax4.yaxis.set_major_formatter(FuncFormatter(currency_formatter))

    # Saving the barplot 
    plt.savefig(os.path.join(OUTPUT_DIR, f"Bar_plot_fraud_vs_non-fraud_{operation}.png"), 
                bbox_inches='tight', 
                facecolor='none', 
                transparent=True) 
    

def plot_fraud_rate_by_time(data_frame, time_category):
    '''
    Plots the fraud rate (fraud transactions / total transactions) per time unit (day, hour, or month),
    comparing with total transactions.

    Parameters:
        - data_frame: Pandas DataFrame containing the data.
        - time_category: Column name of the time unit (e.g., "transaction_day", "transaction_hour", "transaction_month").

    Returns:
        - None: Displays the barplot.
    '''

    # Group by time unit and compute total and fraud transactions
    time_counts = data_frame.groupby(time_category, observed=False)['is_fraud'].agg(['count', 'sum']).reset_index()
    time_counts.columns = [time_category, 'total_transactions', 'fraud_transactions']

    # Compute fraud rate
    time_counts['fraud_rate'] = time_counts['fraud_transactions'] / time_counts['total_transactions']
   
    if time_category == 'month':
        # Ensure time_category is treated as an integer
        time_counts[time_category] = time_counts[time_category].astype(int)

        # Sort values to ensure proper order
        time_counts = time_counts.sort_values(by=time_category)

        # Convert to string to ensure correct categorical ordering
        time_counts[time_category] = time_counts[time_category].astype(str)

    # Bar plot

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=time_counts[time_category], y=time_counts['fraud_rate'], color=fraud_color, label='Fraud Rate')
    ax2 = ax.twinx()

    # Line plot
    sns.lineplot(x=time_counts[time_category], y=time_counts['total_transactions'], ax=ax2, color='black', marker='o', label='Total Transactions', legend=False)

    # Titles and labels
    plt.title(f'Fraud Rate vs. Total Transactions by {time_category.replace("_", " ").title()}')
    ax.set_xlabel(time_category.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel('Fraud Rate', fontsize=12)
    ax2.set_ylabel('Total Transactions', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    #ax.grid(False)
    ax2.grid(False)
  
    # Display legend
    if time_category == 'hour':
        ax.set_xlim(-0.5, 23.5)
    elif time_category == 'day':
        ax.set_xlim(-0.5, 30.5)
    #elif time_category == 'month':
    #    ax.set_xlim(0,13)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1.07, 1), borderaxespad=0)

    plt.savefig(os.path.join(OUTPUT_DIR, f"fraud_rate_by_{time_category}.png"), 
                bbox_inches='tight', 
                facecolor='none', 
                transparent=True)

def plot_fraud_trends_same_axes(data_frame, time_unit='month', aggregation='sum'):
    '''
    Plots fraud trends over time using bar plots on the same axes

    Parameters:
        - data_frame (pd.DataFrame): The dataset containing time-based features and fraud flag.
        - time_unit (str): The time unit to group data by ('hour', 'day', 'month', or 'day_of_week').
        - aggregation (str): The aggregation method ('sum', 'mean', 'median', 'mode', 'max', 'min').
       
    Returns:
        - None: This function generates the plot and saves it.
    '''

    data_frame['is_fraud'] = data_frame['is_fraud'].replace({'Non-Fraud': 0, 'Fraud': 1}).astype(int)

    # Aggregate the counts based on the selected aggregation method
    if aggregation == 'sum':
        operation = 'Count'
        aggregated_data = data_frame.groupby([time_unit, 'is_fraud'], observed=False).size().reset_index(name='Count')
    elif aggregation == 'mean':
        operation = 'Mean'
        aggregated_data = data_frame.groupby([time_unit, 'is_fraud'], observed=False)['amt'].mean().round(2).reset_index(name='Count')
    elif aggregation == 'median':
        operation = 'Median'
        aggregated_data = data_frame.groupby([time_unit, 'is_fraud'], observed=False)['amt'].median().round(2).reset_index(name='Count')
    elif aggregation == 'mode':
        operation = 'Mode'
        aggregated_data = data_frame.groupby([time_unit, 'is_fraud'], observed=False)['amt'].apply(lambda x: x.mode()[0]).round(2).reset_index(name='Count')
    elif aggregation == 'max':
        operation = 'Max'
        aggregated_data = data_frame.groupby([time_unit, 'is_fraud'], observed=False)['amt'].max().round(2).reset_index(name='Count')
    elif aggregation == 'min':
        operation = 'Min'
        aggregated_data = data_frame.groupby([time_unit, 'is_fraud'], observed=False)['amt'].min().round(2).reset_index(name='Count')

    # Pivot the data so we have separate columns for fraud and non-fraud
    fraud_pivot = aggregated_data.pivot(index='is_fraud', columns=time_unit, values='Count').fillna(0)

    # Bar plot
    fig2, ax3 = plt.subplots(figsize=(10, 5))

    # Define width for the bars
    bar_width = 0.4

    if time_unit == 'month':
        x_positions = range(6, 6+len(fraud_pivot.columns))
    else:
        x_positions = range(len(fraud_pivot.columns))

    # Plot bars for fraud and non-fraud transactions (side-by-side)
    bars_non_fraud = ax3.bar([x - bar_width/2 for x in x_positions], fraud_pivot.loc[0], width=bar_width, color=non_fraud_color, label='Non-Fraud', zorder=3)
    bars_fraud = ax3.bar([x + bar_width/2 for x in x_positions], fraud_pivot.loc[1], width=bar_width, color=fraud_color, label='Fraud', zorder=3)

    plt.grid(axis='y', linestyle='--', alpha=0.5)

    if time_unit == 'hour':
        ax3.set_xlim(-0.5, 23.5)
        ax3.set_xticks(range(24))
    elif time_unit == 'day':
        ax3.set_xlim(-0.5, 30.5)
    elif time_unit == 'month':
        ax3.set_xlim(5.5, 12.5)

    # Labels and title
    if time_unit == 'day_of_week':
        x_label = 'Day of Week'
        label = 'Day of Week'
    else:
        x_label = time_unit.title()
        label = time_unit.title() 
    
    # Labels and title for the bar chart
    ax3.set_xlabel(x_label, fontsize=12)
    ax3.set_ylabel(f"{operation} Transactions", fontsize=12)
    plt.title(f'Non-Fraud vs. Fraud Transactions {operation} ({label})')

    # Set tick colors to match the bar colors
    ax3.tick_params(axis='y')
    ax3.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)

    plt.tight_layout()  

    if (operation == "Mean") or (operation == "Median") or (operation == "Mode") or (operation == "Min") or (operation == "Max"): 
        ax3.yaxis.set_major_formatter(FuncFormatter(currency_formatter))

    # Saving the barplot
    plt.savefig(os.path.join(OUTPUT_DIR, f"Bar_plot_fraud_vs_non-fraud_{operation}_same_axes.png"), 
                bbox_inches='tight', 
                facecolor='none', 
                transparent=True)

def plot_fraud_boxplots(data_frame):
    '''
    Plots side-by-side box plots comparing transaction amounts for fraud and non-fraud cases.
    Each fraud category (fraud, non-fraud) has two box plots:
        - Transactions between 00:00 and 11:59
        - Transactions between 12:00 and 23:59

    Parameters:
        - data_frame (pd.DataFrame): The dataset containing 'amt', 'is_fraud', and 'hour' columns.

    Returns:
        - None: This function generates the plot and saves it.
    '''
    # Creating a new column for the time category
    data_frame = data_frame.copy()
    data_frame['time_period'] = data_frame['hour'].apply(lambda x: '00:00-11:00' if x < 12 else '12:00-23:00')


    custom_palette = {'00:00-11:00': '#4BA5F0', '12:00-23:00': '#6EE5D9'}

    # Set up the plot
    plt.figure(figsize=(10, 6))
    sns.boxplot (
        x='is_fraud', 
        y='amt', 
        hue='time_period', 
        data=data_frame, 
        showfliers=False, 
        palette=custom_palette,
        hue_order=['00:00-11:00', '12:00-23:00']
        )

    # Formating to currency
    plt.gca().yaxis.set_major_formatter(FuncFormatter(currency_formatter))

    # Adjust labels
    plt.xlabel('Fraud Category')
    plt.ylabel('Transaction Amount')
    plt.xticks(ticks=[0, 1], labels=['Non-Fraud', 'Fraud'])
    plt.title('Transaction Amount Distribution by Fraud Category and Time Period')
    plt.legend(title='Time Period', loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.savefig(os.path.join(OUTPUT_DIR, "fraud_per_hour_group_box_plot.png"), 
                bbox_inches='tight', 
                facecolor='none', 
                transparent=True) 

def plot_fraud_boxplots_time(data_frame, column):
    '''
    Plots side-by-side box plots comparing transaction amounts for fraud and non-fraud cases.
    Each category have two box plots for:
        - Fraud transactions
        - Non-fraud transactions

    Parameters:
        - data_frame (pd.DataFrame): The dataset containing 'amt', 'is_fraud', 'hour', and 'date' columns.
    '''

    # Set up the plot
    plt.figure(figsize=(12, 8))
    sns.boxplot(
        x=column, 
        y='amt', 
        hue='is_fraud', 
        data=data_frame, 
        showfliers=False, 
        palette={0: '#1F4E79', 1: '#8F2C78'},  
        hue_order=[0, 1],  
        dodge=True, 
        hue_norm=None
    )

    # Apply currency format
    plt.gca().yaxis.set_major_formatter(FuncFormatter(currency_formatter))

    # Adjust labels and title
    plt.xlabel(f'{column.replace("_"," ").title()}')
    plt.ylabel('Transaction Amount')
    plt.title(f'Transaction Amount Distribution by {column.replace("_"," ").title()} and Fraud Category')
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=[handles[0], handles[1]], labels=['Non-Fraud','Fraud'], fontsize=12, loc= 'upper left', bbox_to_anchor=(1.01, 1))
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Add grid for better visualization
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Saving the plot
    plt.savefig(os.path.join(OUTPUT_DIR, f"Bar_plot_fraud_vs_non-fraud_per_{column}.png"), 
                bbox_inches='tight', 
                facecolor='none', 
                transparent=True)

def plot_fraud_heatmap(data_frame, period_1, period_2):
    '''
    Creates a heatmap to visualize transactions occurrence by time.
    
    Parameters:
        - data_frame (pd.DataFrame): The dataset containing time-based features and fraud flag.
        - period 1: hour, day, month or day_of_week 
        - period 2: hour, day, month or day_of_week
    
    Returns:
        - None: Displays the heatmap.
    '''


    # **Heatmap 31: Total Transactions**
    
    heatmap_data = data_frame.groupby([period_1, period_2], observed=False)['is_fraud'].size().unstack()

    plt.figure(figsize=(17, 12))
    sns.heatmap(heatmap_data, cmap="Blues", linewidths=0.5, annot=True, fmt=".0f")
    plt.title(f"Transaction Heatmap")

    # Saving the heatmap
    plt.savefig(os.path.join(OUTPUT_DIR, f"Transaction_Occurrences_Heatmap_{period_1}_vs_{period_2}.png"), 
                bbox_inches='tight', 
                facecolor='none', 
                transparent=True)

    # **Heatmap 3: Fraud Transactions**

    if period_1 == 'day_of_week':
        label_1 = 'Day of Week'
    elif period_2 == 'day_of_week':
        label_2 = 'Day of Week'

    if period_1 != 'day_of_week':
        label_1 = period_1.title()
    if period_2 != 'day_of_week':
        label_2 = period_2.title()

    plt.xlabel(f"{label_2}")
    plt.ylabel(f"{label_1}")

    # Filter for fraud transactions
    fraud_data = data_frame[data_frame['is_fraud'] == 1]

    # Group by the specified periods and count fraud occurrences
    heatmap_data_fraud = fraud_data.groupby([period_1, period_2], observed=False)['is_fraud'].size().unstack()

    plt.figure(figsize=(17, 12))
    sns.heatmap(heatmap_data_fraud, cmap=custom_cmap, linewidths=0.5, annot=True, fmt=".0f")
    plt.title(f"Fraud Transaction Heatmap")

    if period_1 == 'day_of_week':
        label_1 = 'Day of Week'
    elif period_2 == 'day_of_week':
        label_2 = 'Day of Week'

    if period_1 != 'day_of_week':
        label_1 = period_1.title()
    if period_2 != 'day_of_week':
        label_2 = period_2.title()

    plt.xlabel(f"{label_2}")
    plt.ylabel(f"{label_1}")

    # Saving the heatmap
    plt.savefig(os.path.join(OUTPUT_DIR, f"Fraud_Transaction_Occurrences_Heatmap_{period_1}_vs_{period_2}.png"), 
                bbox_inches='tight', 
                facecolor='none', 
                transparent=True)

    # **Heatmap 3: Fraud Rate**
    fraud_rate = heatmap_data_fraud / heatmap_data  # Element-wise division
    fraud_rate.fillna(0, inplace=True)  # Replace NaN with 0 (to handle 0 transactions cases)

    plt.figure(figsize=(17, 12))
    sns.heatmap(fraud_rate, cmap=custom_cmap, linewidths=0.5, annot=True, fmt=".2%")  # Display percentage format
    plt.title(f"Fraud Rate Heatmap")

    if period_1 == 'day_of_week':
        label_1 = 'Day of Week'
    elif period_2 == 'day_of_week':
        label_2 = 'Day of Week'

    if period_1 != 'day_of_week':
        label_1 = period_1.title()
    if period_2 != 'day_of_week':
        label_2 = period_2.title()

    plt.xlabel(f"{label_2}")
    plt.ylabel(f"{label_1}")

    # Saving the heatmap
    plt.savefig(os.path.join(OUTPUT_DIR, f"Fraud_Rate_Heatmap_{period_1}_vs_{period_2}.png"), 
                bbox_inches='tight', 
                facecolor='none', 
                transparent=True)

def plot_fraud_heatmap_statistic(data_frame, period_1, period_2, aggregation='sum'):
    '''
    Creates a heatmap to visualize transactions occurrence by time with a descriptive statisctic parameter
    
    Parameters:
        - data_frame (pd.DataFrame): The dataset containing time-based features and fraud flag.
        - period 1: hour, day, month or day_of_week 
        - period 2: hour, day, month or day_of_week
    
    Returns:
        - None: Displays the heatmap.
    ''' 

    if aggregation == 'sum':
        operation = 'Count'
        heatmap_data = data_frame.groupby([period_1, period_2], observed=False)['amt'].size().unstack()
    elif aggregation == 'mean':
        operation = 'Mean'
        heatmap_data = data_frame.groupby([period_1, period_2], observed=False)['amt'].mean().unstack()
    elif aggregation == 'median':
        operation = 'Median'
        heatmap_data = data_frame.groupby([period_1, period_2], observed=False)['amt'].median().unstack()
    elif aggregation == 'mode':
        operation = 'Mode'
        heatmap_data = fraud_data.groupby([period_1, period_2], observed=False)['amt'].apply(lambda x: x.mode()[0])
    elif aggregation == 'max':
        operation = 'Max'
        heatmap_data = data_frame.groupby([period_1, period_2], observed=False)['amt'].max().unstack()
    elif aggregation == 'min':    
        operation = 'Min'
        heatmap_data = data_frame.groupby([period_1, period_2], observed=False)['amt'].min().unstack()

    plt.figure(figsize=(17, 12))
    sns.heatmap(heatmap_data, cmap="Blues", linewidths=0.5, annot=True, fmt=".1f")
    plt.title(f"Transactions {operation} Heatmap")

    # Saving the heatmap
    plt.savefig(os.path.join(OUTPUT_DIR, f"Transaction_Occurrences_Heatmap_{period_1}_vs_{period_2}_{operation}.png"), 
                bbox_inches='tight', 
                facecolor='none', 
                transparent=True)

    if period_1 == 'day_of_week':
        label_1 = 'Day of Week'
    elif period_2 == 'day_of_week':
        label_2 = 'Day of Week'

    if period_1 != 'day_of_week':
        label_1 = period_1.title()
    if period_2 != 'day_of_week':
        label_2 = period_2.title()

    plt.xlabel(f"{label_2}")
    plt.ylabel(f"{label_1}")

    # Filter for fraud transactions (is_fraud == 1)
    fraud_data = data_frame[data_frame['is_fraud'] == 1]

    # Group by the specified periods and count fraud occurrences

    if aggregation == 'sum':
        operation = 'Count'
        heatmap_data = fraud_data.groupby([period_1, period_2], observed=False)['amt'].size().unstack()
    if aggregation == 'mean':
        operation = 'Mean'
        heatmap_data = fraud_data.groupby([period_1, period_2], observed=False)['amt'].mean().unstack()
    elif aggregation == 'median':
        operation = 'Median'
        heatmap_data = fraud_data.groupby([period_1, period_2], observed=False)['amt'].median().unstack()
    elif aggregation == 'mode':
        operation = 'Mode'
        heatmap_data = fraud_data.groupby([period_1, period_2], observed=False)['amt'].apply(lambda x: x.mode()[0])
    elif aggregation == 'max':
        operation = 'Max'
        heatmap_data = fraud_data.groupby([period_1, period_2], observed=False)['amt'].max().unstack()
    elif aggregation == 'min':    
        operation = 'Min'
        heatmap_data = fraud_data.groupby([period_1, period_2], observed=False)['amt'].min().unstack()

    plt.figure(figsize=(17, 12))
    sns.heatmap(heatmap_data, cmap=custom_cmap, linewidths=0.5, annot=True, fmt=".0f")
    plt.title(f"Fraud Transaction {operation} Heatmap")

    if period_1 == 'day_of_week':
        label_1 = 'Day of Week'
    elif period_2 == 'day_of_week':
        label_2 = 'Day of Week'

    if period_1 != 'day_of_week':
        label_1 = period_1.title()
    if period_2 != 'day_of_week':
        label_2 = period_2.title()

    plt.xlabel(f"{label_2}")
    plt.ylabel(f"{label_1}")


    # Saving the heatmap
    plt.savefig(os.path.join(OUTPUT_DIR, f"Fraud_Transaction_Occurrences_Heatmap_{period_1}_vs_{period_2}_{operation}.png"), 
                bbox_inches='tight', 
                facecolor='none', 
                transparent=True)


def plot_correlation_heatmap(df):
    """
    Generates a heatmap to visualize the correlation coefficients between numerical variables.

    Parameters:
        -df (pd.DataFrame): DataFrame containing the dataset.

    Returns:
        - None: Plot and save the correlation map
    """
    # Select only numerical columns
    num = df.select_dtypes(include="number")
    correlation_matrix = num.corr()

    # Define custom colormap (optional, modify as needed)
    custom_cmap = "coolwarm"

    # Create the heatmap with the specified style
    plt.figure(figsize=(17, 12))
    sns.heatmap(correlation_matrix, cmap=custom_cmap, linewidths=0.5, annot=True, fmt=".2f")

    # Set the title
    plt.title("Correlation Heatmap")

    plt.savefig(os.path.join(OUTPUT_DIR, f"COrrelations_heat_map.png"), 
                bbox_inches='tight', 
                facecolor='none', 
                transparent=True)

def create_fraud_statistics_table(data_frame):
    '''
    Groups by merchant category and fraud status, calculates various statistics for transaction amounts.

    Parameters:
        - data_frame: Pandas DataFrame containing 'category', 'is_fraud', and 'amt' columns.

    Returns:
        - A formatted DataFrame with merchants and fraud status as index, and statistics as columns.
    '''
  
    # Group by category and fraud status, then compute statistics
    stats_df = data_frame.groupby(['category', 'is_fraud'])['amt'].agg([
        ('Mean', 'mean'),
        ('Median', 'median'),
        ('mode', safe_mode),
        ('Min', 'min'),
        ('Max', 'max'),
        ('Std Dev', 'std'),
        ('Count', 'count')
    ]).reset_index()

    # Pivot to make Fraud and Non-Fraud side by side
    stats_pivot = stats_df.pivot(index='category', columns='is_fraud')

    return stats_pivot

def plot_high_risk_transaction_types(data_frame, category):
    '''
    Plots the most frequently values of a categorical column by fraud.
    
    Parameters:
        - data_frame: Pandas DataFrame containing.

    Returns:
        - None: Displays the barplot.
    '''

    # Filter only fraud transactions
    fraud_data = data_frame[data_frame['is_fraud'] == 1]
    
    # Count occurrences 
    fraud_counts = fraud_data[category].value_counts().reset_index()
    fraud_counts.columns = [category, 'fraud_count'] 

    # Bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(y = category, hue=category, x='fraud_count', data=fraud_counts.head(14), palette='Purples_r', legend = False)
    plt.title(f' {category.replace("_", " ").title()} vs. Number of Fraudulent Transactions')
    plt.xlabel('Number of Fraudulent Transactions', fontsize=12)
    plt.ylabel(f'{category.replace("_", " ").title()}', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.5)

    # Saving the plot
    plt.savefig(os.path.join(OUTPUT_DIR, f"Number_of_Fraudulent_Transactions_by_{category}.png"), 
                bbox_inches='tight', 
                facecolor='none', 
                transparent=True)

def plot_high_risk_transaction_types_fre(data_frame, category):
    '''
    Plots the most frequently values of a categorical column by fraud.
    
    Parameters:
        - data_frame: Pandas DataFrame containing.

    Returns:
        - None: Displays the barplot.
    '''

    # Group by category and calculate total and fraud transactions
    category_counts = data_frame.groupby(category)['is_fraud'].agg(['count', 'sum']).reset_index()
    category_counts.columns = [category, 'total_transactions', 'fraud_transactions']
    
    # Compute fraud rate
    category_counts['fraud_rate'] = category_counts['fraud_transactions'] / category_counts['total_transactions']
    
    # Sort by fraud rate and get top 14 categories
    top_categories = category_counts.sort_values(by='fraud_rate', ascending=False).head(20)

    # Bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(y=category, x='fraud_rate', data=top_categories, palette='Purples_r', hue=category, legend=False)
    plt.title(f'{category.replace("_", " ").title()} vs. Fraud Rate')
    plt.xlabel('Fraud Rate', fontsize=12)
    plt.ylabel(f'{category.replace("_", " ").title()}', fontsize=12)
    plt.xlim(0, top_categories['fraud_rate'].max() * 1.1)  # Adjust x-axis range
    plt.grid(axis='x', linestyle='--', alpha=0.5)

    # Save the plot
    plt.savefig(os.path.join(OUTPUT_DIR, f"Fraud_Rate_by_{category}.png"), 
                bbox_inches='tight', 
                facecolor='none', 
                transparent=True)

def plot_donut_histogram_by_net_pos_fraud_only(data_frame, category_column):
    '''
    Plots a donut chart comparing fraud transactions for Net and POS transaction types.
    Only considers fraud transactions.

    Parameters:
        - data_frame: Pandas DataFrame containing the data.
        - category_column: Column name for the transaction categories ("merchant_category").

    Returns:
        - None: Displays the donut chart.
    '''
    
    # Filter data to only include fraud transactions
    fraud_data = data_frame[data_frame['is_fraud'] == 1]

    fraud_data = fraud_data.copy()  # Create a copy of the DataFrame to avoid the warning
    fraud_data['transaction_type'] = np.where(
                                    fraud_data[category_column].str.contains('Net',case=False, na=False), 'Net',  # If 'Net' is found
                                    np.where(fraud_data[category_column].str.contains('POS', case=False, na=False), 'Pos', 'Other')  # If 'POS' is found, else 'Other'
                                    )

    # Filter out any 'Other' categories (those that don't contain 'Net' or 'POS')
    fraud_data = fraud_data[fraud_data['transaction_type'] != 'Other']
    
    # Count fraud transactions for Net and POS
    fraud_counts = fraud_data['transaction_type'].value_counts()
    
    # Plotting donut chart
    fig, ax = plt.subplots(figsize=(8, 8))

    # Creating a donut chart
    wedges, texts, autotexts = ax.pie(fraud_counts, labels=fraud_counts.index, autopct='%1.1f%%', startangle=90, 
                                      colors=['#4BA5F0', '#6EE5D9'], wedgeprops={'width': 0.4})
    
    ax.set_title('Fraud Transactions Net vs. Pos')

    for i, (label, text) in enumerate(zip(texts, autotexts)):
        if i == 0:  # Non-Fraud label
            label.set_position((-0.63, -0.42))  
            text.set_position((-0.73, -0.35))  
            label.set_color('white')
            text.set_color('white') 
            label.set_fontweight('black') 
            label.set_fontsize(14)
        else:  # Fraud label
            label.set_position((0.2, 0.7))   
            text.set_position((0.28, 0.78))
            label.set_color('black')
            text.set_color('black')   
            label.set_fontweight('black') 
            label.set_fontsize(14)

    # Save the plot
    plt.savefig(os.path.join(OUTPUT_DIR, "Fraud_Transactions_by_Net_and_POS.png"), 
                bbox_inches='tight', 
                facecolor='none', 
                transparent=True)

def plot_fraud_vs_nonfraud_means(data_frame, category):
    '''
    Creates a side-by-side bar chart comparing transaction amounts 
    for fraud vs. non-fraud transactions across different categories.

    Parameters:
    - data_frame: Pandas DataFrame containing 'category', 'is_fraud', and 'amt' columns.
    - category: category to compare
    '''


    # Count fraud transactions per category and select the top 14
    top_fraud_categories = (
        data_frame[data_frame['is_fraud'] == 1].groupby(category)['is_fraud']
        .count().nlargest(14).index
    )

    # Filter data to include only the top fraud categories
    filtered_data = data_frame[data_frame[category].isin(top_fraud_categories)]

    # Group by category and fraud status, then calculate the mean transaction amount
    category_means = filtered_data.groupby([category, 'is_fraud'])['amt'].mean().reset_index()

    # Pivot the data for side-by-side comparison
    category_pivot = category_means.pivot(index=category, columns='is_fraud', values='amt').reset_index()
    category_pivot.columns = [category, 'Non-Fraud', 'Fraud']
    category_pivot = category_pivot.sort_values(by='Fraud', ascending=False)
    melted_data = category_pivot.melt(id_vars=category, var_name='Fraud Status', value_name='Mean Amount')

    # Bar chart
    plt.figure(figsize=(12, 6))
    sns.barplot(x=category, y='Mean Amount', hue='Fraud Status', data=melted_data, 
                palette=[non_fraud_color, fraud_color])

    plt.title(f'Mean Transaction Amounts: Non-Fraud vs.Fraud by {category.replace("_"," ").title()}', fontsize=16)
    plt.xlabel(f'{category.replace("_"," ").title()}', fontsize=12)
    plt.ylabel('Mean Transaction Amount', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(fontsize=12, loc= 'upper left', bbox_to_anchor=(1.01, 1))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(currency_formatter))
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Saving the plot
    plt.savefig(os.path.join(OUTPUT_DIR, f"Mean_Transactions_by_{category}.png"), 
                bbox_inches='tight', 
                facecolor='none', 
                transparent=True)

def plot_gender_fraud_and_box_plots(data_frame):
    '''
    Plots:
        1. A bar plot showing the count of fraud and non-fraud transactions by gender (logarithmic scale).
        2. A box plot showing the distribution of transaction amounts for all transactions (logarithmic scale).
    
    Parameters:
    - data_frame: Pandas DataFrame containing 'gender', 'is_fraud', and 'amt' columns.
 
    Return:
        -None: Show and save the plots
    '''

    # Bar Plot: Count of Fraud and Non-Fraud Transactions by Gender (Logarithmic scale)
    plt.figure(figsize=(10, 6))
    sns.countplot(x='gender', hue='is_fraud', data=data_frame, palette=custom_colors)
    plt.title('Count of Fraud vs Non-Fraud Transactions by Gender (Logarithmic Scale)', fontsize=16)
    plt.xlabel('Gender', fontsize=12)
    plt.ylabel('Transaction Count (Log Scale)', fontsize=12)
    plt.legend(labels=['Non-Fraud', 'Fraud'], fontsize=12)
    plt.yscale('log')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(OUTPUT_DIR, "Fraud_vs_NonFraud_by_Gender_Log_Scale.png"), 
                bbox_inches='tight', 
                facecolor='none', 
                transparent=True)

    # Box Plot for All Transactions (Fraud and Non-Fraud Combined) (Logarithmic scale)
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='gender', y='amt', hue='is_fraud', data=data_frame, palette=['#1F4E79', '#8F2C78'])
    plt.title('Transaction Amounts: All Transactions (Fraud & Non-Fraud) by Gender (Logarithmic Scale)', fontsize=16)
    plt.xlabel('Gender', fontsize=12)
    plt.ylabel('Transaction Amount in dollars (Log Scale)', fontsize=12)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=[handles[0], handles[1]], labels=['Non-Fraud','Fraud'], fontsize=12, loc= 'upper left', bbox_to_anchor=(1.01, 1))
    plt.yscale('log')  
    plt.savefig(os.path.join(OUTPUT_DIR, "BoxPlot_All_Transactions_by_Gender_Log_Scale.png"), 
                bbox_inches='tight', 
                facecolor='none', 
                transparent=True)

def plot_fraud_vs_nonfraud_age(data_frame):
    '''
    Creates a side-by-side box plot comparing the age distribution for fraud vs. non-fraud transactions.
    
    Parameters:
        - data_frame: Pandas DataFrame containing 'is_fraud' and 'age' columns.

    Return 
        - None: Plot the box plots
    '''

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='is_fraud', y='age', data=data_frame, palette=custom_colors, hue = 'is_fraud')
    
    plt.title('Age Distribution: Fraud vs Non-Fraud', fontsize=16)
    plt.xlabel('Fraud Status', fontsize=12)
    plt.ylabel('Age', fontsize=12)
    plt.xticks([0, 1], ['Non-Fraud', 'Fraud'])
    plt.legend().set_visible(False)
    plt.savefig(os.path.join(OUTPUT_DIR, "BoxPlot_Fraud_Transactions_by_age.png"), 
                bbox_inches='tight', 
                facecolor='none', 
                transparent=True)

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='gender', y='age', hue='is_fraud', data=data_frame, palette=custom_colors)
    
    plt.title('Age Distribution by Gender and Fraud Status', fontsize=16)
    plt.xlabel('Gender', fontsize=12)
    plt.ylabel('Age', fontsize=12)
    plt.xticks([0, 1], ['Male', 'Female'])
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=[handles[0], handles[1]], labels=['Non-Fraud', 'Fraud'], fontsize=12, loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig(os.path.join(OUTPUT_DIR, "BoxPlot_Fraud_Transactions_by_age_and_gender.png"), 
                bbox_inches='tight', 
                facecolor='none', 
                transparent=True)

def plot_fraud_vs_nonfraud_age_violin(data_frame):
    '''
    Creates a side-by-side violin plot comparing the age distribution for fraud vs. non-fraud transactions.
    
    Parameters:
        - data_frame: Pandas DataFrame containing 'is_fraud' and 'age' columns.
        - OUTPUT_DIR: Directory to save the plots.

    Return 
        - None: Plot the violin plots
    '''

    # Violin plot for age distribution: Fraud vs Non-Fraud
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='is_fraud', y='age', data=data_frame, palette=custom_colors, hue='is_fraud')
    
    plt.title('Age Distribution: Fraud vs Non-Fraud', fontsize=16)
    plt.xlabel('Fraud Status', fontsize=12)
    plt.ylabel('Age', fontsize=12)
    plt.xticks([0, 1], ['Non-Fraud', 'Fraud'])
    plt.legend().set_visible(False)
    
    # Save the plot
    plt.savefig(os.path.join(OUTPUT_DIR, "ViolinPlot_Fraud_Transactions_by_age.png"), 
                bbox_inches='tight', 
                facecolor='none', 
                transparent=True)
    
    # Violin plot for age distribution by Gender and Fraud Status
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='gender', y='age', hue='is_fraud', data=data_frame, palette=custom_colors)
    
    plt.title('Age Distribution by Gender and Fraud Status', fontsize=16)
    plt.xlabel('Gender', fontsize=12)
    plt.ylabel('Age', fontsize=12)
    plt.xticks([0, 1], ['Male', 'Female'])
    
    # Customize the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=[handles[0], handles[1]], labels=['Non-Fraud', 'Fraud'], fontsize=12, loc='upper left', bbox_to_anchor=(1, 1))
    
    # Save the plot
    plt.savefig(os.path.join(OUTPUT_DIR, "ViolinPlot_Fraud_Transactions_by_age_and_gender.png"), 
                bbox_inches='tight', 
                facecolor='none', 
                transparent=True)

def plot_age_grouped_fraud_nonfraud(data_frame):
    '''
    Creates a box plot showing transaction amounts grouped by age ranges, 
    split by fraud and non-fraud transactions.
    
    Parameters:
        - data_frame: Pandas DataFrame containing 'age', 'amt', and 'is_fraud' columns.

    Return 
        - None: Plot the box plots
    '''
    
    # Create custom age groups
    bins = [20, 30, 40, 50, 60, 70, 120]
    labels = ['20-30', '30-40', '40-50', '50-60', '60-70', '70-120']
    data_frame['age_group'] = pd.cut(data_frame['age'], bins=bins, labels=labels, right=False)
    
    # Ensure the 'age_group' column is categorical for proper plotting
    data_frame['age_group'] = data_frame['age_group'].astype("category")
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Box plot: Age group vs amt, split by fraud status
    sns.boxplot(x='age_group', y='amt', hue='is_fraud', data=data_frame, palette=custom_colors, showfliers=False)
    
    # Title and labels
    plt.title('Transaction Amount by Age Group and Fraud Status', fontsize=16)
    plt.xlabel('Age Group', fontsize=12)
    plt.ylabel('Transaction Amount', fontsize=12)
    
    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha="right")
    plt.gca().yaxis.set_major_formatter(FuncFormatter(currency_formatter))
    
    # Adjust the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=[handles[0], handles[1]], labels=['Non-Fraud', 'Fraud'], fontsize=12, loc='upper left', bbox_to_anchor=(1, 1))
    
    # Save the plot
    plt.savefig(os.path.join(OUTPUT_DIR, "BoxPlot_Age_Grouped_Fraud_vs_NonFraud.png"), 
                bbox_inches='tight', 
                facecolor='none', 
                transparent=True)
