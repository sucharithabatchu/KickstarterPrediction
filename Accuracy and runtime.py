#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
import datetime
from pyspark.ml import Pipeline

spark = SparkSession.builder.master('local[1]').appName('learn_ml').getOrCreate()


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
df0 = spark.read.csv('BIA 678 dataset.csv', header=True, inferSchema=True, encoding='utf-8')
# df0.toPandas().isna().sum()
# df0.toPandas().isna().values.any()
# False
# StringIndexer
from pyspark.ml.feature import StringIndexer, VectorAssembler
old_columns_names = df0.columns
new_columns_names = [name+'-new' for name in old_columns_names]
for i in range(len(old_columns_names)):
    indexer = StringIndexer(inputCol=old_columns_names[i], outputCol=new_columns_names[i])
    df0 = indexer.fit(df0).transform(df0)
vecAss = VectorAssembler(inputCols=new_columns_names[3:11], outputCol='features')
df0 = vecAss.transform(df0)
# label
df0 = df0.withColumnRenamed(new_columns_names[-1], 'label')

# label, features
dfi = df0.select(['label', 'features'])

# dfi.show(5, truncate=0)

train_data, test_data = dfi.randomSplit([0.9, 0.1], seed=2019)

starttime = datetime.datetime.now()

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression()
lrModel = lr.fit(train_data)
result = lrModel.transform(test_data)

#accuracy
a = result.filter(result.label == result.prediction).count()/result.count()
print('\nLogistic Regression: ', a)
endtime = datetime.datetime.now()
print('\n', endtime - starttime)


starttime = datetime.datetime.now()

from pyspark.ml.classification import DecisionTreeClassifier
dt = DecisionTreeClassifier(maxDepth=5, maxBins=35)
dtModel = dt.fit(train_data)
result = dtModel.transform(test_data)

# accuracy
b = result.filter(result.label == result.prediction).count()/result.count()
print('\nDecisionTree: ', b)
endtime = datetime.datetime.now()
print('\n', endtime - starttime)

starttime = datetime.datetime.now()

from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(numTrees=10, maxDepth=5, maxBins=35)
rfModel = rf.fit(train_data)
result = rfModel.transform(test_data)

# accuracy
c = result.filter(result.label == result.prediction).count()/result.count()
print('\nRandomForest: ', c)
endtime = datetime.datetime.now()
print('\n', endtime - starttime)

starttime = datetime.datetime.now()

from pyspark.ml.classification import GBTClassifier
gbt = GBTClassifier(maxDepth=5, maxBins=35)
gbtModel = gbt.fit(train_data)
result = gbtModel.transform(test_data)

# accuracy
d = result.filter(result.label == result.prediction).count()/result.count()
print('\nGBT: ', d)

endtime = datetime.datetime.now()
print('\n', endtime - starttime)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




