# Data Preparation in the cloud 

將了解如何利用各種 AWS 雲端服務在雲端設定資料準備。 考慮到資料準備中提取、轉換和載入 (ETL) 操作的重要性，我們將更深入地研究以經濟高效的方式設定和調度 ETL 作業

## Data processing in the cloud 

深度學習 (DL) 專案的成功取決於資料的品質和數量。 因此，資料準備系統必須足夠穩定和可擴展，才能有效處理 TB 和 PB 的資料。

### Introduction to ETL
在整個 ETL 過程中，將從一個或多個來源收集數據，並將其轉換為
根據需要採用不同的形式，並保存在資料記憶體中。 簡而言之，ETL本身涵蓋了整個資料處理管道。

ETL 始終與三種不同類型的資料互動：結構化、非結構化和半結構化

流行的ETL框架包括Apache Hadoop，Presto，Apache Flink，Apache Spark，主要注重Spark

### Data processing system architecture 
建立一個資料處理系統並不是一件簡單的事情，因為它需要定期採購高階機器，正確連接各種資料處理軟體，並確保資料得到正確的處理。
發生故障時不會遺失。 因此，許多公司利用雲端服務，這是透過互聯網按需提供的各種軟體服務。 雖然許多公司提供各種雲端服務，但亞馬遜網路服務（AWS）以其穩定且易於使用的服務脫穎而出

#### Introduction to Apache Spark
作為 Spark 的介紹，我們將涵蓋 Spark 的關鍵概念和一些常見的 Spark 操作。 具體來說，我們將首先介紹彈性分佈式資料集（RDD）和 DataFrame。

#### Resillient distributed datasets and Dataframes

Spark 的獨特優勢來自 RDD，即不可變的分散式資料物件集合。 透過利用 RDD，Spark 可以有效地處理利用並行性的資料。

即使一個或多個處理器發生故障，在 RDD 上運行的 Spark 的內建平行處理功能也有助於資料處理。 當觸發 Spark 作業時，輸入資料的 RDD 表示形式將​​拆分為多個分區並分發到每個節點進行轉換，從而最大化吞吐量。

#### Converting between RDDs and DataFrames
任何 Spark 操作的第一步都是建立 SparkSession 物件。 具體來說，pyspark.sql 中的 SparkSession 模組用於建立 SparkSession 物件。

模組中的 getOrCreate 函數用於建立會話對象，如下所示。 SparkSession 物件是 Spark 應用程式的入口點。 它提供了一種在不同上下文下與Spark應用程式互動的方式，例如Spark上下文，Hive 上下文與 SQL 上下文：
```python
from pyspark.sql import SparkSession

spark_session = SparkSession.builder.appName('covid_analysis').getOrCreate()
```
將 RDD 轉換為 DataFrame 很簡單。 鑑於 RDD 沒有任何模式，您可以建立一個沒有任何schema的DataFrame，如下：
```python
df_ri_freq = rdd_ri_freq.toDF()

```
要將 RDD 轉換為具有模式的 DataFrame，您需要使用 StructType 類，它是 pyspark.sql.types 模組的一部分。 使用 StructType 方法建立模式後，就可以使用 Spark 會話物件的 createDataFrame 方法
將 RDD 轉換為 DataFrame：
```python
from pyspark.sql.types import StructType,StructField.StringType,IntegerType
# rdd for research interest frequency data 
rdd_ri_freq = ...
# convert to df with schema 
schema = StructType(
    [StructField('ri',StringType(),False),
    StructField('frequency',IntergerType(),False),
    ]

)
df = spark.createDataFrame(rdd_ri_freq,schema)
```

#### Loading data 
Spark can load data of different formats that’s stored in various forms of data storage. Loading data
stored in CSV format is a basic operation of Spark.

文件位於本機或雲端中，例如在 S3 儲存桶中，作為 DataFrame。 在下面的程式碼片段中，我們載入儲存在 S3 中的 Google Scholar 資料。

```python
google_scholar_dataset_path =  "s3a://my-bucket/dataset/dataset_csv/dataset-google-scholar/output.csv"
# load google schloar dataset 
df_gs  = spark_session.read.\
    .option("header","True")\
    .csv(google_scholar_dataset_path)
# 也可以讀取json 文件
df = spark_session.read.json(json_file_path)
```
#### Processing data using Spark operations 

Spark 提供了一組將 RDD 轉換為不同結構的 RDD 的操作。 實作 Spark 應用程式是在 RDD 上連結一組 Spark 操作以將資料轉換為目標格式的過程。

在本節中，我們將討論最常用的—即filter、map、flatMap、reduceByKey、take、groupBy和join。

- filter 
  ```python
  df_gs_clean = df_gs.filter("research_interest!='None'")
  
  ```

- map 
  ```python
  rdd_ri = df_gs_clean.rdd.map(lambda x:(x['research_interest']))
  ```


- flatMap
    ```python
    rdd_flattened_ri = rdd_ri.flatMap(lambda x: [(w.lower(), 1) for w in x.split('##')]) 
    ```
- reduceByKey
  ```python
  rdd_ri_freq = rdd_flattened_ri.reduceByKey(add)
  ```
- take 
  ```python
  rdd_ri_freq_5 = rdd_ri_freq.take(5)
  ```


不做無用功

#### Processing data using user-defined functions 
使用者定義函數 (UDF) 是可重複使用的自訂函數，可對 RDD 執行轉換。

UDF 函數可以在多個 DataFrame 上重複使用。 在本節中，我們將提供使用 UDF 處理 Google Scholar 資料集的完整程式碼範例


首先，我們要介紹一下 pyspark.sql.function 模組，它允許您使用 udf 方法定義 UDF，並提供各種按列操作。

pyspark。 sql.function 也包含聚合函數，例如 avg 或 sum 分別用於計算平均值和總計：

```python
import pyspark.sql.functions as F 
```

在 Google Scholar 資料集中，data_science、artificial_intelligence 和 machine_learning 均指同一領域的人工智慧 (AI)。 因此，最好建立一個 UDF 來清理這個欄位。
```python
# list of research_interests that are under same domain 
lst_ai = ['data_science','artifical_intelligence','machine_learning']
@F.udf
def is_ai(research):
    '''return1 if reasearch in AI domain else 0 '''
    try:
        # split the research interest with delimiter 
        lst_research = [w.lower() for w in str(research).split('##')]
        for res in lst_resarch:
            # if present in AI domain 
            if res in lst_ai:
                return1 
            # not present in AI domain 
            return 0 
    except :
        return -1 
df_gs_new = df_gs.withColumn("is_artificial_intelligence",\ is_ai(F.col("research_interest")))
```
#### Exporting data 
在本节中，我们将学习如何将DataFrame保存到S3存储桶中。在RDD的情况下，必须将其转换为要适当保存的DataFrame。
```python
s3_output_path = "s3a:\\my-bucket\output\vaccine_state_avg.csv"
sample_data_frame.coalesce(1),write.mode('overwrite').option('header',True).option("quoteAll",True).csv(s3_output_path)
```
如果您想将DataFrame保存为JSON文件，可以使用write.JSON：
```python
s3_output_path = "s3a:\\my-bucket\output\vaccine_state_avg.csv"
sample_data_frame.write.json(s3_output_path)
```


后面都是aws上面的操作了
