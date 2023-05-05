import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
        ##pre analysis
def vis_outl_bxplt(df, columns, df_name):
    df_num = df.loc[:,columns]
    _, ax = plt.subplots(figsize=(20,6))
    df_num.plot(kind = 'box', ax=ax)
    plt.savefig(f'analysis/boxplot_{df_name}.png')

#-----------------------------------------------------------------------------
        ##analyze data # create charts, planes...

def analyze(df, df_name):
    
    df = df.select_dtypes(include=['int64' , 'float64'])

    #table - describe
    stats = df.describe() #if specific attributes (columns) should be describe. Czy to powinno być tutaj czy przed scalowaniem
    stats.to_csv('analysis/stats_{}.csv'.format(df_name))
    
    # Histograms of numeric columns
    df.hist(figsize=(20,15))
    plt.savefig('analysis/histogram_{}.png'.format(df_name))

    # Boxplots of all numeric columns
    # exclude_cols = ['chipsettime','qualitytimestamp','gpstime']
    # new_cols = df.select_dtypes(include=['int64' , 'float64']).columns.difference(exclude_cols)
    _, ax = plt.subplots(figsize=(20,6))
    # df[new_cols].plot(kind='box', ax=ax)
    df.plot(kind='box', ax=ax)
    plt.title('Box plot of numeric columns')
    plt.xlabel('Columns')
    plt.ylabel('Values')
    plt.savefig('analysis/boxplots_{}.png'.format(df_name))

    # Correlation matrix
    corr_matrix = df.corr(numeric_only=[False/True])
    # Heatmap of correlation matrix
    plt.matshow(corr_matrix)
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    plt.colorbar()
    plt.savefig('analysis/heatmap_{}.png'.format(df_name))
#-----------------------------------------------------------------------------