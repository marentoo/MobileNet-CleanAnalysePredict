import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
# import os
from src.ex2_loading import load_df
from src.ex2_cleaning import clean_df

#-----------------------------------------------------------------------------
#analyze data # create charts, planes...

def analyze(df_down, df_upl):

    df_tuple = (df_down,df_upl)
    for df in df_tuple:
        df_name = [k for k,v in locals().items() if v is df][0]
        # if not os.path.exists('../evaluation'):
        #  os.mkdir('../evaluation')
        
        # Histograms of numeric columns
        df.hist(figsize=(20,15))
        plt.savefig('evaluation/histogram_{}.png'.format(df_name))
        # Boxplots of numeric columns
        _, ax = plt.subplots(figsize=(20,6))
        df.plot(kind='box', ax=ax)
        # Set plot title and axis labels
        plt.title('Box plot of numeric columns')
        plt.xlabel('Columns')
        plt.ylabel('Values')
        plt.savefig('evaluation/boxplots_{}.png'.format(df_name))
        # Correlation matrix
        corr_matrix = df.corr(numeric_only=[False/True])
        # Heatmap of correlation matrix
        plt.matshow(corr_matrix)
        plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
        plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
        plt.colorbar()
        plt.savefig('evaluation/heatmap_{}.png'.format(df_name))

#-----------------------------------------------------------------------------
## let's test it
#load data
# df1 = pd.read_csv('o2_download_nexus5x.csv'); df2 = pd.read_csv('telekom_download_nexus5x.csv'); df3 = pd.read_csv('vodafone_download_nexus5x.csv')
# df4 = pd.read_csv('o2_upload_nexus5x.csv'); df5 = pd.read_csv('telekom_upload_nexus5x.csv'); df6 = pd.read_csv('vodafone_upload_nexus5x.csv')
# df_downloads,df_uploads = load_df(df1,df2,df3,df4,df5,df6)
# print(df_downloads.head(1)); print(df_uploads.head(1))

## Scale - *chose type of scaling
# scaler_norm = MinMaxScaler() #scale between <0,1> (for e.g. algor: KNN or NN)
# scaler_stand = StandardScaler() #scale differently ( for e.g. Logistic Regression, Linear Discriminant Analysis)

#new data frames -scaled- to analyze and put into ML model
# df_scaled_downloads, df_scaled_uploads = clean_df(df_downloads, df_uploads, scaler_norm)
# analyze(df_scaled_downloads,df_scaled_uploads)
# print(df_downloads.head(1)); print(df_uploads.head(1))