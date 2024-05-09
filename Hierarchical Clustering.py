'''
# # Clustering

# Problem Statement

# Students have to evaluate a lot of factors before taking a decision 
to join a university for their higher education requirements.


# `CRISP-ML(Q)` process model describes six phases:
# 
# 1. Business and Data Understanding
# 2. Data Preparation
# 3. Model Building
# 4. Evaluation
# 5. Deployment
# 6. Monitoring and Maintenance
'''
# Objective(s): Maximize the convenience of the admission process
# Constraints: Minimize the brain drain

'''Success Criteria'''

# Business Success Criteria: Reduce the application process time from anywhere between 20% to 40%
# ML Success Criteria: Achieve Silhouette coefficient of at least 0.6
# Economic Success Criteria: US Higher education department will see an increase in revenues by at least 30%
# 
# **Proposed Plan:**
# Grouping the available universities will allow understanding the characteristics of each group.


'''
# ## Data Collection

# Data: 
#    The university details are obtained from the US Higher Education Body and is publicly available for students to access.
# 
# Data Dictionary:
# - Dataset contains 25 university details
# - 7 features are recorded for each university
# 
# Description:
# - Univ - University Name
# - State - Location (state) of the university
# - SAT - Cutoff SAT score for eligibility
# - Top10 - % of students who ranked in the top 10 in their previous academics
# - Accept - % of students admitted to the universities
# - SFRatio - Student to Faculty ratio
# - Expenses - Overall cost in USD
# - GradRate - % of students who graduate
'''

# Install the required packages if not present already
# pip install sweetviz
# pip install py-AutoClean
# pip install clusteval
# pip install sqlalchemy
# pip install pymysql

# Importing required packages

import pandas as pd
import matplotlib.pyplot as plt
import sweetviz
from AutoClean import AutoClean
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering 
from sklearn import metrics
from clusteval import clusteval
import numpy as np
from sqlalchemy import create_engine

# uni = pd.read_excel(r"D:/DS-ML360/ML code by 360/Hierarchical Clustering_Hands-on/University_Clustering.xlsx")

# Credentials to connect to Database
user = 'root'  # user name
pw = 'anand'  # password
db = 'DSML'  # database name
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# to_sql() - function to push the dataframe onto a SQL table.
# uni.to_sql('univ_tbl', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

sql = 'select * from univ_tbl;'
df = pd.read_sql_query(sql, engine)

# Data types
df.info()
df.head()

# EXPLORATORY DATA ANALYSIS (EDA) / DESCRIPTIVE STATISTICS
# ***Descriptive Statistics and Data Distribution Function***

df.describe()


# Data Preprocessing

# **Cleaning Unwanted columns**
# UnivID is the identity to each university. 
# Analytically it does not have any value (Nominal data). 
# We can safely ignore the ID column by dropping the column.

df.drop(['UnivID'], axis = 1, inplace = True)

df.info()

# ## Automated Libraries

# AutoEDA
# import sweetviz
my_report = sweetviz.analyze([df, "df"])

my_report.show_html('Report.html')


'''
Alternatively, we can use other AutoEDA functions as well.
# D-Tale
########

pip install dtale
import dtale

d = dtale.show(df)
d.open_browser()

'''

# EDA report highlights:
# ------------------------
# Missing Data: Identified Missing Data in columns: SAT, GradRate

# Outliers:  Detected exceptional values in 4 columns: SAT, Top10, Accept, SFRatio
# Boxplot

#Install PyQt5 if you get this warning message - "UserWarning:Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure."
#pip install PyQt5
#import PyQt5

df.plot(kind = 'box', subplots = True, sharey = False, figsize = (15, 8)) 
# increase spacing between subplots
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()

# Encoding: 'State' is categorical data that needs to be encoded into numeric values


# Data Preprocessing
# -----------------------------------------------------------------------------
# Auto Preprocessing and Cleaning
# from AutoClean import AutoClean
clean_pipeline = AutoClean(df.iloc[:, 1:], mode = 'manual', missing_num = 'auto',
                           outliers = 'winz', encode_categ = 'auto')
help(AutoClean)

# Missing values = 'auto': AutoClean first attempts to predict the missing values with Linear Regression
# outliers = 'winz': outliers are handled using winsorization
# encode_categ = 'auto': Label encoding performed (if more than 10 categories are present)

df_clean = clean_pipeline.output
df_clean.head()


# #### Drawback with this approach: If there are more than 10 categories, then Autoclean performs label encoding.

df_clean.drop(['State'], axis = 1, inplace = True)

df_clean.head()

# -----------------------------------------------------------------------------


# ## Normalization/MinMax Scaler - To address the scale differences

# ### Python Pipelines
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import MinMaxScaler

df_clean.info()

cols = list(df_clean.columns)
print(cols)

pipe1 = make_pipeline(MinMaxScaler())

# Train the data preprocessing pipeline on data
df_pipelined = pd.DataFrame(pipe1.fit_transform(df_clean), columns = cols, index = df_clean.index)
df_pipelined.head()

df_pipelined.describe() # scale is normalized to min = 0; max = 1

###### End of Data Preprocessing ######
# -----------------------------------------------------------------------------


######### Model Building #########
# # CLUSTERING MODEL BUILDING

# ### Hierarchical Clustering - Agglomerative Clustering

# from scipy.cluster.hierarchy import linkage, dendrogram
# from sklearn.cluster import AgglomerativeClustering 
# import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline') --- if running in jupyter notebook

plt.figure(1, figsize = (16, 8))
tree_plot = dendrogram(linkage(df_pipelined, method  = "ward"))

plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Euclidean distance')
plt.show()


# Applying AgglomerativeClustering and grouping data into 3 clusters 
# based on the above dendrogram as a reference
hc1 = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'complete')

y_hc1 = hc1.fit_predict(df_pipelined)
y_hc1

# Analyzing the Results obtained
hc1.labels_   # Referring to the cluster labels assigned

cluster_labels = pd.Series(hc1.labels_) 

# Combine the labels obtained with the data
df_clust = pd.concat([cluster_labels, df_clean], axis = 1) 

df_clust.head()

df_clust.columns
df_clust = df_clust.rename(columns = {0: 'cluster'})
df_clust.head()




# # Clusters Evaluation

# **Silhouette coefficient:**  
# Silhouette coefficient is a Metric, which is used for calculating 
# goodness of the clustering technique, and the value ranges between (-1 to +1).
# It tells how similar an object is to its own cluster (cohesion) compared to 
# other clusters (separation).
# A score of 1 denotes the best meaning that the data point is very compact 
# within the cluster to which it belongs and far away from the other clusters.
# Values near 0 denote overlapping clusters.

# from sklearn import metrics
metrics.silhouette_score(df_pipelined, cluster_labels)

'''Alternatively, we can use:'''
# **Calinski Harabasz:**
# Higher value of the CH index means clusters are well separated.
# There is no thumb rule which is an acceptable cut-off value.
metrics.calinski_harabasz_score(df_pipelined, cluster_labels)

# **Davies-Bouldin Index:**
# Unlike the previous two metrics, this score measures the similarity of clusters. 
# The lower the score the better the separation between your clusters. 
# Vales can range from zero and infinity
metrics.davies_bouldin_score(df_pipelined, cluster_labels)



'''Hyperparameter Optimization for Hierarchical Clustering'''
# Experiment to obtain the best clusters by altering the parameters

# ## Cluster Evaluation Library

# pip install clusteval
# Refer to link: https://pypi.org/project/clusteval

# from clusteval import clusteval
# import numpy as np

# Silhouette cluster evaluation. 
ce = clusteval(evaluate = 'silhouette')

df_array = np.array(df_pipelined)

# Fit
ce.fit(df_array)

# Plot
ce.plot()

## Using the report from clusteval library building 2 clusters
# Fit using agglomerativeClustering with metrics: euclidean, and linkage: ward

hc_2clust = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')

y_hc_2clust = hc_2clust.fit_predict(df_pipelined)

# Cluster labels
hc_2clust.labels_

cluster_labels2 = pd.Series(hc_2clust.labels_) 

# Concate the Results with data
df_2clust = pd.concat([cluster_labels2, df_clean], axis = 1)

df_2clust = df_2clust.rename(columns = {0:'cluster'})
df_2clust.head()

# Aggregate using the mean of each cluster
df_2clust.iloc[:, 1:7].groupby(df_2clust.cluster).mean()

# Save the Results to a CSV file
df_3clust = pd.concat([df.Univ, cluster_labels2, df_clean], axis = 1)

df_3clust = df_3clust.rename(columns = {0:'cluster'})
df_3clust.to_csv('University.csv', encoding = 'utf-8')

import os
os.getcwd()
