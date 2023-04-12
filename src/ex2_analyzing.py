import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

from ex2_loading import load_df
from ex2_cleaning import clean_df

#-----------------------------------------------------------------------------
#load data
df1 = pd.read_csv('o2_download_nexus5x.csv'); df2 = pd.read_csv('telekom_download_nexus5x.csv'); df3 = pd.read_csv('vodafone_download_nexus5x.csv')
df4 = pd.read_csv('o2_upload_nexus5x.csv'); df5 = pd.read_csv('telekom_upload_nexus5x.csv'); df6 = pd.read_csv('vodafone_upload_nexus5x.csv')
df_downloads,df_uploads = load_df(df1,df2,df3,df4,df5,df6)
# print(df_downloads.head(1)); print(df_uploads.head(1))

#-----------------------------------------------------------------------------
#Scale - *chose type of scaling
scaler_norm = MinMaxScaler() #scale between <0,1> (for e.g. algor: KNN or NN)
scaler_stand = StandardScaler() #scale differently ( for e.g. Logistic Regression, Linear Discriminant Analysis)
#new data frames -scaled- to analyze and put into ML model
df_scale_downloads, df_scale_uploads = clean_df(df_downloads, df_uploads, scaler_norm)

#-----------------------------------------------------------------------------
#analyze data # create charts, planes...

def analyze(df1, df2):
    #For download
        # Summary statistics
        # Histograms of numeric columns
    df1.hist(figsize=(20,15))
    plt.savefig('evaluation/histogram_d.png')
        # Boxplots of numeric columns
    fig, ax = plt.subplots(figsize=(20,6))
    df1.plot(kind='box', ax=ax)
    # Set plot title and axis labels
    plt.title('Box plot of numeric columns')
    plt.xlabel('Columns')
    plt.ylabel('Values')
    plt.savefig('evaluation/boxplots_d.png')
        # Correlation matrix
    corr_matrix = df1.corr(numeric_only=[False/True])
        # Heatmap of correlation matrix
    plt.matshow(corr_matrix)
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    plt.colorbar()
    plt.savefig('evaluation/heatmap_d.png')

    #For upload
    # Summary statistics
        # Histograms of numeric columns
    df2.hist(figsize=(20,15))
    plt.savefig('evaluation/histogram_u.png')
        # Boxplots of numeric columns
    fig, ax = plt.subplots(figsize=(20,6))
    df2.plot(kind='box', ax=ax)
    # Set plot title and axis labels
    plt.title('Box plot of numeric columns')
    plt.xlabel('Columns')
    plt.ylabel('Values')
    plt.savefig('evaluation/boxplots_u.png')
        # Correlation matrix
    corr_matrix = df2.corr(numeric_only=[False/True])
        # Heatmap of correlation matrix
    plt.matshow(corr_matrix)
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    plt.colorbar()
    plt.savefig('evaluation/heatmap_u.png')

## let's test it
# analyze(df_downloads,df_uploads)