'''
Data Exploratory Analysis
'''

# importing libraries
import pandas as pd
import pandas_profiling


# reading the file to be analyzed
df=pd.read_csv('NFL_Census.csv')

# creating and saving the report
print ('\n---Starting the Exploratory Data Analysis\n')
df.profile_report()
profile=df.profile_report(title='Profiling report')
profile.to_file(output_file='Profiling_Output.html')
print ('\n---End of the Analysis\n')
