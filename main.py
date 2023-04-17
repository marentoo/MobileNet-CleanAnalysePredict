import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.ex2_loading import load_df
from src.ex2_cleaning import clean_df
from src.ex2_analyzing import analyze

df1 = pd.read_csv('o2_download_nexus5x.csv'); df2 = pd.read_csv('telekom_download_nexus5x.csv'); df3 = pd.read_csv('vodafone_download_nexus5x.csv')
df4 = pd.read_csv('o2_upload_nexus5x.csv'); df5 = pd.read_csv('telekom_upload_nexus5x.csv'); df6 = pd.read_csv('vodafone_upload_nexus5x.csv')

## Scale - *chose type of scaling
scaler_norm = MinMaxScaler() #scale between <0,1> (for e.g. algor: KNN or NN)
scaler_stand = StandardScaler() #scale differently ( for e.g. Logistic Regression, Linear Discriminant Analysis)

def main():
    df_downloads,df_uploads = load_df(df1,df2,df3,df4,df5,df6)
    df_scaled_downloads, df_scaled_uploads = clean_df(df_downloads, df_uploads, scaler_norm)
    analyze(df_scaled_downloads,df_scaled_uploads)
    # predict(df_scaled_downloads,df_scaled_downloads)
    return df_scaled_downloads, df_scaled_uploads

if __name__ == "__main__":
    main()