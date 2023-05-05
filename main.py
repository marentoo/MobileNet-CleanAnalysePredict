import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.ex2_loading import load_df
from src.ex2_loading import merge_df
from src.ex2_cleaning import clean_df
from src.ex2_analyzing import vis_outl_bxplt
from src.ex2_analyzing import analyze
from src.ex2_scaling import scaling_data

df1 = pd.read_csv('o2_download_nexus5x.csv'); df2 = pd.read_csv('telekom_download_nexus5x.csv'); df3 = pd.read_csv('vodafone_download_nexus5x.csv')
df4 = pd.read_csv('o2_upload_nexus5x.csv'); df5 = pd.read_csv('telekom_upload_nexus5x.csv'); df6 = pd.read_csv('vodafone_upload_nexus5x.csv')

## Scale - *chose type of scaling
scaler_norm = MinMaxScaler() #scale between <0,1> (for e.g. algor: KNN or NN)
scaler_stand = StandardScaler() #scale differently ( for e.g. Logistic Regression, Linear Discriminant Analysis)

def main():
    df_downloads, df_uploads = load_df(df1,df2,df3,df4,df5,df6)
    df_merged = merge_df(df_downloads.copy(), df_uploads.copy())

    df_cleaned_download = clean_df(df_downloads,'downloads')
    df_cleaned_upload = clean_df(df_uploads,'uploads')
    df_cleaned_merged = clean_df(df_merged, 'merged')

    #PreScaling Analysis - visualiza boxplots before cleaning/scaling
    vis_outl_bxplt(df_cleaned_download, ['throughput','tp_cleaned'],'downloads_th_tp')
    vis_outl_bxplt(df_cleaned_download, ['rsrq', 'rsrp', 'rssi'],'downloads__rsrq_rsrp_rssi')

    vis_outl_bxplt(df_cleaned_upload, ['rsrq', 'rsrp', 'rssi'],'uploads_rsrq_rsrp_rssi')
    vis_outl_bxplt(df_cleaned_upload, ['tp_cleaned'],'uploads_tp')

    vis_outl_bxplt(df_cleaned_merged, ['throughput','tp_cleaned'],'merged_th_tp')
    vis_outl_bxplt(df_cleaned_merged, ['rsrq', 'rsrp', 'rssi'],'merged_rsrq_rsrp_rssi')
    
    #PreScaling Analysis - boxplots, histograms...
    analyze(df_cleaned_download,'downloads')
    analyze(df_cleaned_upload,'uploads')
    analyze(df_cleaned_merged,'merged')

    #Scaling data frames by chosen scaling technique
    df_scaled_download = scaling_data(df_cleaned_download, 'downloads', scaler_norm)
    df_scaled_upload = scaling_data(df_cleaned_upload,'uploads', scaler_norm)
    df_scaled_merged = scaling_data(df_cleaned_merged, 'merged', scaler_norm)
    
    #Predictin - building model and predicting
    # predict()
    
    # return df_scaled_downloads, df_scaled_uploads, df_scaled_merged

if __name__ == "__main__":
    main()