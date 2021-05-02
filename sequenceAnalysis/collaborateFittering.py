# pyspark安装有一点问题
from pyspark.sql import HiveContext, SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

'''
    基于spark的算法程序，在本地是不能直接运行的，将.py程序上传至服务器，进而使用spark-submit
    将程序提交给spark进行处理，并得到结果
'''
# 初始化
spark = SparkSession.builder.master("local").appName("WordCount").getOrCreate()
hive_context = HiveContext(spark)
# 切换数据库至tpch
hive_context.sql('use tpch')
# SQL语言
sql = "select p_brand, p_type, p_size, count(distinct ps_suppkey) as supplier_cnt from partsupp, part where p_partkey = ps_partkey and p_brand <> '[BRAND]' and p_type not like '[TYPE]%' and ps_suppkey not in (select s_suppkey from supplier where s_comment like '%Customer%Complaints%') group by p_brand, p_type, p_size"
# 执行SQL语句，得到结果。该结果为DataFrame
df = hive_context.sql(sql)
# 展示结果
df.show()
rows = df.collect()
# 切分训练集和测试集
training, test = df.randomSplit([0.8, 0.2])
# 使用pyspark.ml.recommendation包下的ALS方法实现协同过滤算法，并设置参数
alsExplicit = ALS(maxIter=10, regParam=0.01, userCol="supplier_cnt", itemCol="p_brand", ratingCol="p_size")
# 训练并得到模型
modelExplicit = alsExplicit.fit(training)
# 利用测试集对模型进行检测
predictionsExplicit = modelExplicit.transform(test)
# 结果展示
predictionsExplicit.show()
evaluator = RegressionEvaluator().setMetricName("rmse").setLabelCol("rating").setPredictionCol("prediction")

rmse = evaluator.evaluate(predictionsExplicit)

print("Explicit:Root-mean-square error = " + str(rmse))

print("Explicit:Root-mean-square error = " + str(rmse))
