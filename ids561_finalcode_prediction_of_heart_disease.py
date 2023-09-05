# -*- coding: utf-8 -*-


Original file is located at
    https://colab.research.google.com/drive/104u8YmMy_WFBDEo1MWmDJ6Fh-eoHydUg
"""

from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/Colab Notebooks/IDS561/Project

#Spark is written in the Scala programming language and requires the Java Virtual Machine (JVM) to run. Therefore, our first task is to download Java.
!apt-get install openjdk-8-jdk-headless -qq > /dev/null

# Spark installer
!wget -v https://dlcdn.apache.org/spark/spark-3.3.1/spark-3.3.1-bin-hadoop3.tgz

#Spark installer
!tar -xvf spark-3.3.1-bin-hadoop3.tgz

# Install the python library findspark to locate Spark.
!pip install -q findspark
!pip install pyspark
!pip install handyspark

#Set environment variables
#Depending on where they are stored, set Java and Spark to their default locations.
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/drive/MyDrive/Colab Notebooks/IDS561/Project/spark-3.3.1-bin-hadoop3"

#local Spark session creation
import findspark
findspark.init()
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").getOrCreate()

from pyspark.sql import functions as F
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import numpy as np
from pyspark.ml.tuning import CrossValidator
import plotly.graph_objects as go
import matplotlib
import matplotlib.pyplot as plt
df=spark.read.csv('datasets_heart.csv', inferSchema=True, header=True)
df.count()
len(df.columns)

df.dtypes

from pyspark.sql.functions import col,sum
df.select(*(sum(col(c).isNull().cast("int")).alias(c) for c in df.columns)).show()

df.describe().toPandas().transpose()

df2=df.toPandas()
df22=df2.groupby('target').count().reset_index()[['target','age']].rename(columns={'age':'counts'})
colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen']
fig = go.Figure(data=[go.Pie(labels=df22.target,
                             values=df22.counts)])
fig.update_traces(hoverinfo='label+percent', textinfo='value+percent', textfont_size=20, textfont_color='black',
                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))
# fig.show()
fig.update_layout(title='Heart Disease vs. Absence of Heart Disease', title_x=0.5)

from plotly.subplots import make_subplots
fig = make_subplots(rows=4, cols=4, start_cell="top-left",
                   subplot_titles=df2.columns[:-1])
fig.add_trace(go.Histogram(x=df2.age, name='age'),
              row=1, col=1)
fig.add_trace(go.Histogram(x=df2.sex, name='sex'),
              row=1, col=2)
fig.add_trace(go.Histogram(x=df2.cp, name='cp'),
              row=1, col=3)
fig.add_trace(go.Histogram(x=df2.trestbps, name='trestbps'),
              row=1, col=4)
fig.add_trace(go.Histogram(x=df2.chol, name='chol'),
              row=2, col=1)
fig.add_trace(go.Histogram(x=df2.fbs, name='fbs'),
              row=2, col=2)
fig.add_trace(go.Histogram(x=df2.restecg, name='restecg'),
              row=2, col=3)
fig.add_trace(go.Histogram(x=df2.thalach, name='thalach'),
              row=2, col=4)
fig.add_trace(go.Histogram(x=df2.exang, name='exang'),
              row=3, col=1)
fig.add_trace(go.Histogram(x=df2.oldpeak, name='oldpeak'),
              row=3, col=2)
fig.add_trace(go.Histogram(x=df2.slope, name='slope'),
              row=3, col=3)
fig.add_trace(go.Histogram(x=df2.thalach, name='ca'),
              row=3, col=4)
fig.add_trace(go.Histogram(x=df2.thal, name='thal'),
              row=4, col=1)
fig.update_layout(title='Histograms of Variables', title_x=0.5)

df3=df.withColumn('oldpeaklog', F.log(df['oldpeak']+1))
df33=df3.toPandas()
fig = make_subplots(rows=1, cols=2, start_cell="top-left",
                   subplot_titles=['oldpeak','oldpeaklog'])
fig.add_trace(go.Histogram(x=df33.oldpeak, name='oldpeak'),
              row=1, col=1)
fig.add_trace(go.Histogram(x=df33.oldpeaklog, name='oldpeaklog'),
              row=1, col=2)
fig.update_layout(title='Transforming oldpeak', title_x=0.5)

corr = df33.corr()
fig = go.Figure(data=go.Heatmap(z=corr.values,
 x=corr.index.values,
 y=corr.columns.values,
 text=np.round(corr.values,2),
 texttemplate="%{text}"))

fig.update_layout(title=dict(text='Correlation Matrix Heatmap',font=dict(size=20), x=0.5))

#Initialize stages
stages = []
#Target column
label_stringIdx = StringIndexer(inputCol = 'target', outputCol = 'label')
stages += [label_stringIdx]
#Numeric Columns
numericCols = ['age',
 'sex',
 'cp',
 'trestbps',
 'chol',
 'fbs',
 'restecg',
 'thalach',
 'exang',
 'slope',
 'ca',
 'thal',
 'oldpeaklog']
#Create a vector assembler
assemblerInputs = numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features").setHandleInvalid('keep')
stages += [assembler]

from pyspark.ml import Pipeline
pipeline = Pipeline(stages = stages)
pipelineModel = pipeline.fit(df3)
df3 = pipelineModel.transform(df3)
selectedCols = ['label', 'features'] + ['age',
 'sex',
 'cp',
 'trestbps',
 'chol',
 'fbs',
 'restecg',
 'thalach',
 'exang',
 'slope',
 'ca',
 'thal',
 'oldpeaklog','target']
df3 = df3.select(selectedCols)
df3.printSchema()

train, test = df3.randomSplit([0.7, 0.3], seed = 2018)
train.groupby('target').count().show()
test.groupby('target').count().show()

"""**Implementing Models**

**Random Forest**
"""

import matplotlib
import matplotlib.pyplot as plt
from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label', seed=101)
rfModel = rf.fit(train)
predictions_rf=rfModel.transform(test)

#Overall accuracy of Model
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator_rf = MulticlassClassificationEvaluator(predictionCol="prediction")
evaluator_rf.evaluate(predictions_rf)

#Confusion Matrix
predictions_rf.crosstab('label','prediction').show()

#ROC and AUC precision Recall Curves
from handyspark import *
# Creates instance of extended version of BinaryClassificationMetrics
# using a DataFrame and its probability and label columns, as the output
# from the classifier
bcm = BinaryClassificationMetrics(predictions_rf, scoreCol='probability', labelCol='label')
# Get metrics from evaluator
print("Area under ROC Curve: {:.4f}".format(bcm.areaUnderROC))
print("Area under PR Curve: {:.4f}".format(bcm.areaUnderPR))
# Plot both ROC and PR curves
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
bcm.plot_roc_curve(ax=axs[0])
bcm.plot_pr_curve(ax=axs[1])

from pyspark.sql.functions import udf
from pandas import Float64Dtype
from pandas.core.frame import FloatFormatType
from pyspark.sql.types import *

#Testing various thresholds
#confusion matrix based on thresold inputted


split1_udf = F.udf(lambda value: value[0].item(), FloatType())
split2_udf = F.udf(lambda value: value[1].item(), FloatType())
def test_threshold(model, prob):
    output2 = model.select('rawPrediction','target','probability',split1_udf('probability').alias('class_0'), split2_udf('probability').alias('class_1'))
    from pyspark.sql.functions import col, when
    output2=output2.withColumn('prediction', when(col('class_0')> prob, 1).otherwise(0))
    output2.crosstab('prediction','target').show()


test_threshold(predictions_rf,.6)
test_threshold(predictions_rf,.7)

#Feature importances
feat_imps=rfModel.featureImportances
x_values = list(range(len(feat_imps)))
plt.bar(x_values, feat_imps, orientation = 'vertical')
plt.xticks(x_values, ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','slope','ca','thal','oldpeaklog'], rotation=40)
plt.ylabel('Importance')
plt.xlabel('Feature')
plt.title('Feature Importances')

#Tune hyperparameter
paramGrid_rf = ParamGridBuilder() \
    .addGrid(rf.numTrees, [int(x) for x in np.arange(200,221,10)]) \
    .addGrid(rf.maxDepth, [int(x) for x in np.arange(10,11,10)]) \
    .addGrid(rf.featureSubsetStrategy, [x for x in ["sqrt", "log2", "onethird"]]) \
    .addGrid(rf.impurity, [x for x in ['gini','entropy']]) \
    .addGrid(rf.maxBins, [int(x) for x in np.arange(22, 42, 10)]) \
    .build()
evaluator = BinaryClassificationEvaluator()
rf_crossval = CrossValidator(estimator=rf,
                          estimatorParamMaps=paramGrid_rf,
                          evaluator=evaluator,
                          numFolds=3)
rf_cvModel = rf_crossval.fit(train)
predictions_rf_cv = rf_cvModel.transform(test)

#Overall Accuracy of best CV model
evaluator_rf_cv = MulticlassClassificationEvaluator(predictionCol="prediction")
evaluator_rf_cv.evaluate(predictions_rf_cv)

#Feature importance of best CV model
import matplotlib.pyplot as plt
feat_imps=rf_cvModel.bestModel.featureImportances
x_values = list(range(len(feat_imps)))
plt.bar(x_values, feat_imps, orientation = 'vertical')
plt.xticks(x_values, ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','slope','ca','thal','oldpeaklog'], rotation=40)
plt.ylabel('Importance')
plt.xlabel('Feature')
plt.title('Feature Importances')

#Gey hyperparameter values of best Random forests CV model

print('Num Trees: ' + str(rf_cvModel.bestModel.getNumTrees))
print('Max Depth: ' + str(rf_cvModel.bestModel.getMaxDepth()))
print('Feature Subset Strategy: ' + str(rf_cvModel.bestModel.getFeatureSubsetStrategy()))
print('Impurity: ' + str(rf_cvModel.bestModel.getImpurity()))
print('Max Bins: ' + str(rf_cvModel.bestModel.getMaxBins()))

"""**Logistic Regression**"""

from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0,featuresCol = 'features', labelCol = 'label')
lrModel = lr.fit(train)
predictions_lr = lrModel.transform(test)

#Overall Accuracy
evaluator_lr = MulticlassClassificationEvaluator(predictionCol="prediction")
evaluator_lr.evaluate(predictions_lr)

#print coefficients and intercepts for logistic regression
print("Coefficients: " + str(lrModel.coefficients))
print("Intercept: " + str(lrModel.intercept))

#confusion matrix
predictions_lr.crosstab('label','prediction').show()

#ROC and AUC curve
# Creates instance of extended version of BinaryClassificationMetrics
# using a DataFrame and its probability and label columns, as the output
# from the classifier
bcm = BinaryClassificationMetrics(predictions_lr, scoreCol='probability', labelCol='label')
# Get metrics from evaluator
print("Area under ROC Curve: {:.4f}".format(bcm.areaUnderROC))
print("Area under PR Curve: {:.4f}".format(bcm.areaUnderPR))
# Plot both ROC and PR curves
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
bcm.plot_roc_curve(ax=axs[0])
bcm.plot_pr_curve(ax=axs[1])

#Tune hyperparameters
paramGrid_lr = ParamGridBuilder() \
    .addGrid(lr.maxIter, [int(x) for x in np.arange(10,30,10)]) \
    .addGrid(lr.regParam, [int(x) for x in np.arange(.1,.5,.1)]) \
    .addGrid(lr.elasticNetParam, [int(x) for x in np.arange(0,.2,.1)]) \
    .build()
evaluator = BinaryClassificationEvaluator()
lr_crossval = CrossValidator(estimator=lr,
                          estimatorParamMaps=paramGrid_lr,
                          evaluator=evaluator,
                          numFolds=3)
lr_cvModel = lr_crossval.fit(train)
predictions_lr_cv = lr_cvModel.transform(test)

#Overall accuracy of best CV model
evaluator_lr_cv = MulticlassClassificationEvaluator(predictionCol="prediction")
evaluator_lr_cv.evaluate(predictions_lr_cv)

"""**Naive Bayes**"""

# Commented out IPython magic to ensure Python compatibility.
from pyspark.ml.classification import NaiveBayes
nb = NaiveBayes(featuresCol = 'features', labelCol = 'label')
nb_model = nb.fit(train)
predictions_nb=nb_model.transform(test)

#Overall Accuracy of model
evaluator_nb = MulticlassClassificationEvaluator(predictionCol="prediction")
evaluator_nb.evaluate(predictions_nb)

#Confusion matrix
predictions_nb.crosstab('label','prediction').show()

#Confusion matrix
from handyspark import *
from matplotlib import pyplot as plt
# %matplotlib inline
# Creates instance of extended version of BinaryClassificationMetrics
# using a DataFrame and its probability and label columns, as the output
# from the classifier
bcm = BinaryClassificationMetrics(predictions_nb, scoreCol='probability', labelCol='label')
# Get metrics from evaluator
print("Area under ROC Curve: {:.4f}".format(bcm.areaUnderROC))
print("Area under PR Curve: {:.4f}".format(bcm.areaUnderPR))
# Plot both ROC and PR curves
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
bcm.plot_roc_curve(ax=axs[0])
bcm.plot_pr_curve(ax=axs[1])

#Tune Hyperparameter

paramGrid_nb = ParamGridBuilder() \
    .addGrid(nb.smoothing, [int(x) for x in np.arange(1,10,1)]) \
    .build()
evaluator = BinaryClassificationEvaluator()
nb_crossval = CrossValidator(estimator=nb,
                          estimatorParamMaps=paramGrid_nb,
                          evaluator=evaluator,
                          numFolds=3)
nb_cvModel = nb_crossval.fit(train)
predictions_nb_cv = nb_cvModel.transform(test)

#Evaluate best CV model

evaluator_nb_cv = MulticlassClassificationEvaluator(predictionCol="prediction")
evaluator_nb_cv.evaluate(predictions_nb_cv)

"""Conclusion - Interpreting Cp, thalach and Ca"""

# Best fitting model - Random forest
# Feature selection shows - cp, ca and thalach highest
# interpreting three values

#cp
df3.groupby('target','cp').count().show()

#thalach
df34=df33.loc[df33['target']==0,'thalach']
df35=df33.loc[df33['target']==1,'thalach']
fig = go.Figure()
fig.add_trace(go.Histogram(x=df34, name='target:0'))
fig.add_trace(go.Histogram(x=df35,name='target:1'))
# Overlay both histograms
fig.update_layout(barmode='overlay')
# Reduce opacity to see both histograms
fig.update_traces(opacity=.9)
fig.update_xaxes(title='Thalach Level')
fig.update_yaxes(dtick=5, range=[0,30], title='Count')
fig.update_layout(title='Comparing Thalach Levels', title_x=0.5)

#ca

df3.groupby('target','ca').count().show()
