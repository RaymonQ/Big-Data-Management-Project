from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, CountVectorizer, StringIndexer
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType


def base_features_gen_pipeline(input_descript_col="descript", input_category_col="category", output_feature_col="features", output_label_col="label"):
    tokenizer = Tokenizer(inputCol = input_descript_col,outputCol = "words")
    cv = CountVectorizer(inputCol = "words",outputCol = output_feature_col)
    indexer = StringIndexer(inputCol = input_category_col,outputCol = output_label_col)
    pipeline = Pipeline(stages = [tokenizer,cv,indexer])
    return pipeline

def ToInteger(nb_pre,svm_pre):
    return float(int(str(int(nb_pre)) + str(int(svm_pre)),2))

def gen_meta_features(training_df, nb_0, nb_1, nb_2, svm_0, svm_1, svm_2):
    #Cannot always assume the number of group is 5, thus, count the number of group here
    groupNumber = training_df.select("group").distinct().count()
    resultDF = None
    for x in range(groupNumber):
        dfForTrain = training_df[training_df.group.isin([x]) == False]
        dfForGenFeature = training_df[training_df.group.isin([x]) == True]
                
        base_pipeline = Pipeline(stages=[nb_0,nb_1,nb_2,svm_0,svm_1,svm_2])
        
        if resultDF == None:
            resultDF = base_pipeline.fit(dfForTrain).transform(dfForGenFeature)
        else:
            resultDF = resultDF.union(base_pipeline.fit(dfForTrain).transform(dfForGenFeature))
    helper = udf(ToInteger,DoubleType())
    resultDF = resultDF.withColumn("joint_pred_0",helper("nb_pred_0","svm_pred_0")).withColumn("joint_pred_1",helper("nb_pred_1","svm_pred_1")).withColumn("joint_pred_2",helper("nb_pred_2","svm_pred_2"))
    return resultDF

def test_prediction(test_df, base_features_pipeline_model, gen_base_pred_pipeline_model, gen_meta_feature_pipeline_model, meta_classifier):
    rawTestDF = base_features_pipeline_model.transform(test_df)
    rawTestDFTwo = gen_base_pred_pipeline_model.transform(rawTestDF)
    helper = udf(ToInteger,DoubleType())
    rawTestDFThree = rawTestDFTwo.withColumn("joint_pred_0",helper("nb_pred_0","svm_pred_0")).withColumn("joint_pred_1",helper("nb_pred_1","svm_pred_1")).withColumn("joint_pred_2",helper("nb_pred_2","svm_pred_2"))
    featuresDF = gen_meta_feature_pipeline_model.transform(rawTestDFThree)
    outputDF = meta_classifier.transform(featuresDF).select("id","label","final_prediction")
    return outputDF
