import numpy as np
import os

##Cleaning data
#------------------------------------------------------------------------------------
        ##check duplicates and ##Checking percentage of duplicates and ##Removing all duplicates

def clean_dupl(df, df_name):

    duplicateRows = df[df.duplicated()]
    print(f'No. duplicated rows - {df_name}: {len(duplicateRows.index)}')
    percentage = int(len(duplicateRows.index) * 100) / len(df.index)
    print(f'Duplicate percentage - {df_name}: {percentage:.2f}%')    
    df.drop_duplicates(keep='last', inplace = True)
    print("\n","Droping duplicates!","\n")
    return df

#------------------------------------------------------------------------------------
        ##Checking for missing data and Handle missing data - no missing data! but if ... then:

def clean_miss(df,df_name):
    sNaN = df.isnull().values.any()
    print(f"Any missing data - {df_name}?: {sNaN}\n")
    if sNaN:
        df.fillna(df.median(), inplace=True)
        print("\n","Droping missing data!","\n")
    return df


#------------------------------------------------------------------------------------
        ##Detect outliers -  Define z_score for checking outliers and Percentage of outliers in whole dataset:

def detect_outliers(df, df_name):
    z_scores = np.abs((df - df.mean(numeric_only=True)) / df.std(numeric_only=True))
    outliers = df[(z_scores > 4).any(axis=1)]
    print(f'No. of outliers - {df_name}: {len(outliers.index)}')
    percout = int((len(outliers.index)*100)/len(df.index))
    print(f' outliers percentage- {df_name}: {percout} %\n')
    return df

#------------------------------------------------------------------------------------
       ##Handling outliers <-- Drop outliers from the dataframes #<<-dziala ale zle <<--- pytanie gdzie umieścic
## df_downloads = df_downloads.drop(outliersD.index,)
## df_uploads = df_uploads.drop(outliersU.index)
## print("\n","Droping outliers!","\n")
## print(f'Rd:{len(df_downloads.index)}');print(f'Ru:{len(df_uploads.index)}')

#------------------------------------------------------------------------------------
        ##final function
def clean_df(df, df_name):

    print(f'{df_name}')
    df = clean_dupl(df, f'{df_name}')
    df = clean_miss(df,f'{df_name}')
    df = detect_outliers(df, f'{df_name}')
    df.to_csv(f'analysis/df_cleaned_{df_name}.csv', index = False)
    print('------------------')
    return df