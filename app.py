import numpy as np
import pandas as pd
#Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
#Text Handling Libraries
import  re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
df2 = pd.read_csv('freelancer_job_postings.csv',index_col='projectId')  
print(df2.shape)
print(df2.head())   #printing all the data present in the data set
print('Null Data Count In Each Column')   
print('-'*30)
print(df2.isnull().sum()) 
print('-'*30)
print('Null Data % In Each Column')  
print('-'*30)
for col in df2.columns:  
    null_count = df2[col].isnull().sum()
    total_count = df2.shape[0]                                     #Null data count in data set
    print("{} : {:.2f}".format(col,null_count/total_count * 100)) #NUll data percentage in each column
df = df2.dropna()  #Dropping the Null data set
print(df.shape)
print(df.dtypes)  #Identifying the data types of the data

counts = df['tags'].value_counts()
count_percentage = df['tags'].value_counts(1)*100
counts_df = pd.DataFrame({'Tags':counts.index,'Counts':counts.values,'Percent':np.round(count_percentage.values,2)})
print(counts_df)
px.bar(data_frame=counts_df,
 x='Tags',
 y='Counts',
 color='Counts',
 color_continuous_scale='blues',
 text_auto=True,
 title=f'Count of job in Each tags')

count2 = CountVectorizer(stop_words='english', lowercase=True)
count_matrix2 = count2.fit_transform(df2['job_title'])
cosine_sim2 = cosine_similarity(count_matrix2, count_matrix2)
cosine_sim_df2 = pd.DataFrame(cosine_sim2)

def content_recommendation_v2(title):
    a = df2.copy().reset_index().drop('projectId', axis=1)
    index = a[a['job_title'] == title].index[0]
    
    # Use cosine_sim_df2 instead of cosine_sim_df
    similar_basis_metric_1 = cosine_sim_df2[cosine_sim_df2[index] > 0][index].reset_index().rename(columns={index: 'sim_1'})
    
    # Continue with cosine_sim_df2
    similar_basis_metric_2 = cosine_sim_df2[cosine_sim_df2[index] > 0][index].reset_index().rename(columns={index: 'sim_2'})
    
    similar_df = similar_basis_metric_1.merge(similar_basis_metric_2, how='left').merge(a[['job_description']].reset_index(), how='left')
    similar_df['sim'] = similar_df[['sim_1', 'sim_2']].fillna(0).mean(axis=1)
    similar_df = similar_df[similar_df['index'] != index].sort_values(by='sim', ascending=False)
    return similar_df[['job_description', 'sim']].head(60)



title = 'Develop a Linux based BT receiving daemon to "grab" BT device values'
print(content_recommendation_v2(title))
print(content_recommendation_v2(title))
