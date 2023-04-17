import pandas as pd

##load data for two groups: download and upload
def load_df(df1,df2,df3,df4,df5,df6):
    mobile_provider = ["o2", "telekom", "vodafone"]
    ##let's add column mobile_provider_name taken from each file name to data frames
    df1,df4 = [df.assign(mobiProv_name=mobile_provider[0]) for df in [df1,df4]]
    df2,df5 = [df.assign(mobiProv_name=mobile_provider[1]) for df in [df2,df5]]
    df3,df6 = [df.assign(mobiProv_name=mobile_provider[2]) for df in [df3,df6]]
    
    df_downloads = pd.concat([df1,df2,df3])
    df_uploads = pd.concat([df4,df5,df6])

    return df_downloads,df_uploads

##let's test-load it
# df1 = pd.read_csv('o2_download_nexus5x.csv'); df2 = pd.read_csv('telekom_download_nexus5x.csv'); df3 = pd.read_csv('vodafone_download_nexus5x.csv')
# df4 = pd.read_csv('o2_upload_nexus5x.csv'); df5 = pd.read_csv('telekom_upload_nexus5x.csv'); df6 = pd.read_csv('vodafone_upload_nexus5x.csv')
# df_downloads,df_uploads = load_df(df1,df2,df3,df4,df5,df6)

# print(df_downloads);print(df_downloads.columns);print(df_uploads);print(df_uploads.columns)