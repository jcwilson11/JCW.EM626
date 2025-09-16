'''

In class Excercise 01

We want to analyze a dataset containing the names given to babies in the USA from 2004 to 2014.

This program will read the file US_Baby_Names_right.csv into a pandas data structure and will

Print the data structure info
Delete the column 'Unnamed: 0' and 'Id'
Determine if there are more gender "F" or "M"
Print the top 5 in terms of number of occurrences per name
Print the number of names in the dataset
Print the standard deviation of name occurrence
Print the basic descriptive statistics on the dataset
'''

import pandas as pd

# csv
names_csv = 'US_Baby_Names_right.csv'
# read
names_df = pd.read_csv(names_csv)
names_df.drop(columns=['Unnamed: 0', 'Id'], inplace=True)

# gender "F" or "M"
gender_counts = names_df['Gender'].value_counts()
print("Gender counts:\n", gender_counts)

# top 5 names by occurencnes
top_5_names = names_df.groupby('Name')['Count'].sum().nlargest(5)
print("Top 5 names by occurrences:\n", top_5_names)

# unique names count
unique_names_count = names_df['Name'].nunique()
print("Number of unique names:", unique_names_count)

# std dev
std_dev_occurrences = names_df['Count'].std()
print("Standard deviation of name occurrences:", std_dev_occurrences)

# basidc stats
descriptive_stats = names_df['Count'].describe()
print("Descriptive statistics:\n", descriptive_stats)