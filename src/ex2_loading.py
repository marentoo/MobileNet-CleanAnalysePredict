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


#Merging two dataframes into one
def merge_df(dataFrame_downloads, dataFrame_uploads):
        dataFrame_downloads['tbs'] = dataFrame_downloads['tbs0'] + dataFrame_downloads['tbs1']
        dataFrame_downloads.rename(columns = {'rb0':'rbs', 'mcs0':'mcs'}, inplace = True)
        dataFrame_downloads.drop(['tbs0', 'tbs1', 'mimo','rnti','rb1','scc','caindex','mcs1'], axis=1, inplace = True)
        dataFrame_downloads = dataFrame_downloads.reindex(columns=[
                'chipsettime', 'cellid', 'mcsindex', 'mcs','tbs','throughput', 'rbs', 'tp_cleaned', 'gpstime', 'longitude', 'latitude', 
                'speed', 'rsrq', 'rsrp', 'rssi', 'earfcn', 'cqi', 'mobiProv_name'
                ])
        dataFrame_uploads['throughput'] = dataFrame_uploads['tbs']/1000
        dataFrame_uploads.rename(columns = {'qualitytimestamp':'chipsettime'}, inplace = True)
        dataFrame_uploads.loc[dataFrame_uploads['mcsindex'] == 1,'mcs'] = 'QPSK'
        dataFrame_uploads.loc[dataFrame_uploads['mcsindex'] == 2,'mcs'] = '16QAM'
        dataFrame_uploads.loc[dataFrame_uploads['mcsindex'] == 3,'mcs'] = '64QAM'
        dataFrame_uploads = dataFrame_uploads.reindex(columns=[
                'chipsettime', 'cellid', 'mcsindex', 'mcs', 'tbs', 'throughput', 'rbs', 'tp_cleaned',
                'gpstime', 'longitude', 'latitude', 'speed', 'rsrq', 'rsrp', 'rssi', 'earfcn', 'cqi', 'mobiProv_name'
                ])

        #adding new column with proces information: download or upload data
        proces_downloadOrUpload = ['download', 'upload']
        dataFrame_downloads["proces_DorU"] = proces_downloadOrUpload[0]
        dataFrame_uploads["proces_DorU"] = proces_downloadOrUpload[1]

        dataFrame_merged = pd.concat([dataFrame_downloads, dataFrame_uploads])
        dataFrame_merged.to_csv('analysis/df_merged.csv', index = False)

        return dataFrame_merged