# Data Preparation for Deep Learning Projects
每个机器学习 (ML) 项目的第一步都包括数据收集和数据准备。
作为 ML 的子集，深度学习 (DL) 涉及相同的数据处理过程

## Data collection,data cleaning and data preprocessing 
在本节中，我们将向您介绍数据收集过程中涉及的各种任务。 我们将描述如何从多个来源收集数据并将其转换为数据科学家可以使用的通用形式，无论底层任务是什么。

這個部分可以分為三個部分：數據收集，數據清洗和數據處理
值得一提的是，特定于任务的变换被认为是特征提取，

### Collecting data
- Crawling web pages 
  使用beautilfulsoup來爬取信息。
  ```python
  response = request.get(url)
  html_soup = BeautifulSoup(response.text,'html.parser')
  html_soup.find_all('a',href=True)
  ```

- Collecting JSON data 
  ```JSON
    {
        "first_name":"Ryan",
        "last_name":"Smith",
        "phone":[{  
            "type":"home",
            "number":"111 222-3456"}],
        "pets":["ceasor":"rocky"],
        "job_location":null
    }
  ```

- Popular dataset repositories
  - Kaggle 
  - sklearn,Keras and Tensorflow 

### Cleaning data 
數據清理是拋光原始資料以保持條目一致的過程，也注重保留收集數據中的相關信息

在我們討論單一資料清理作業之前，最好先對 DataFrame 有一些了解，DataFrame 是 pandas 函式庫提供的表狀資料結構
```python
import pandas as pd 
from tabulate import tabulate
in_file = '../xxx/xxx/xxx.csv'
df = pd.read_csv(in_file)
print(tabulate(df.head(5),headers='keys',tablefmt='psql'))
```
- Filling empty fields with default values 
  `df.fillna(value='na',inplace=True)`
- Removing stop words
  ```python
  import pandas as pd 
  import nltk 
  from nltk.stem import PorterStemmer
  from nltk.tokenize import word_tokenize
  import traceback
  from nltk.corpus import stopwords
  nltk.download("punkt")
  nltk.download("stopwords")
  stop_words = set(stopwords.words('english'))

  # read each line in df 
  for index,row in df.iterrows():
    curr_research_interest = str(row['research_interest']).replace("##"," ").replace("_"," ")
  # tokenize text data 
  curr_res_int_tok = tokenize(curr_research_interest)
  # remove stop workds from the words tokens
  curr_filtered_research = [w for w in curr_res_int_tok if not w.lower() in stop_words]
  ```
- Removing text that is not alpha-numeric 
  去除標定符號
  ```python
  def clean_text(in_str):
    clean_txt = re.sub(r"\W+"," ",in_str)
    return clean_txt 

  ```
- Removing newlines
  ```python
  # replace the new line in the given text with empty string 
  text = input_text.replace("\n","")
  ```

### Data preprocessing 
資料預處理的目標是將清理後的資料轉換為適合各種資料分析任務的通用形式。 資料清洗和資料預處理之間沒有明顯的區別。

- Normalization
  有時，欄位的值可能會以不同的方式表示，即使它們含義相同。 就 Google Scholar 數據而言，研究興趣的條目可能使用不同的單詞，即使它們涉及相似的領域
  ```python
  # directionary mapping the values are commonly used for normalization
  dict_norm = {
    "data_science":"artificial_intelligence",
    "machine_learning":"artificial_intelligence"
  }
  if curr in dict_norm:
    return dict_norm[curr]
  else:
    return curr 
  ```
  字段的數值也需要標準化。 對於數值，標準化是將每個值重新調整到特定範圍的過程。 在以下範例中，我們將每個州每週疫苗分發的平均計數調整為 0 到 1 之間。
  ```python
  # 首先，我們計算每個州的平均計數。 然後，我們透過將平均計數除以最大平均計數來計算歸一化平均計數：
  df = df_in.groupby("jurisdiction")["_1st_dose_allocations"].mean().to_frame("mean_vaccine_count").reset_index()
  df['norm_vaccine_count'] = df['mean_vaccine_count']/df['mean_vaccine_count'].max()

  ```
- Case conversion
  在許多情況下，文字資料會轉換為小寫或大寫作為標準化的一種方式。 這帶來了一定程度的一致性.
  ```python
  # word tokenize
  curr_resh_int_tok = word_tokenize(curr_research_interest)
  # remove stop words from the word tokens
  curr_filtered_research = [w for w in curr_res_int_tok if not w.lower() in stop_words]

  # another method
  def convert_lowercase(in_str):
    return str(in_str).lower()
  ```

- Stemming 
  詞幹提取是將單字轉換為其詞根的過程。 詞幹擷取的好處在於，如果單字的基本意義相同，則可以保持單字的一致性。
  ```python
  from nltk.stem import PorterStemmer
  # porter stemmer for stemming word tokens 
  ps = PorterStemmer()
  word = 'information'
  stemmed_word = ps.stem(word) // 'inform'

  ```

## Extracting features from data 
特徵提取（特徵工程）是將資料轉換為特徵的過程，這些特徵以特定的方式表達目標任務的底層資訊。


要求您利用特定於任務的領域知識。 在本節中，我們將介紹
流行的特徵提取技術，包括文字資料的詞袋、術語頻率逆
文件頻率、將彩色影像轉換為灰階影像、序數編碼、one-hot 編碼、
降維和比較兩個字串的模糊匹配。

### Converting text using bag-of-words
**Bag-of-words(BoW)** 詞袋模型，是文件的表示，描述文件中一組單字的出現（詞頻）。

BoW只考慮單字的出現，忽略單字在文件中的順序或單字的結構

```python
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
document_1 = "This is a great place to do holiday shopping"
document_2 = "This is a good place to eat food"
document_3 = "One of the best place to relax is home"
count_vector = CountVectorizer(ngram_range = (1,1),stop_words = 'english')
# transform from the sentences 
count_fit = count_vector.fit_transform([document_1,document_2,document_3])
# create dataframe 
df = pd.DataFrame(count_fit.toarray(),columns =count_vector.get_feature_names_cut())
print(tabulate(df, headers="keys", tablefmt="psql"))
```

### Applying term frequency-inverse document frequency (TF_IDF) transformation
使用詞頻的問題是頻率較高的文件將主導模型或分析。 因此，最好根據單字在所有文件中出現的頻率來重新調整頻率。 這種縮放有助於以文本的數字表示更好地表達上下文的方式來懲罰那些高頻詞（例如 the 和 have）。


在介紹TF-IDF的公式之前，我們必須先定義一些符號。 設 n 為總數
文件且 t 是一個單字（術語）。 `df(t)` 指的是單字 t 的文檔頻率（有多少個文件中包含單字 t），而 `tf(t, d)` 指的是單字 t 在文件 d 的頻率（t 在文件 d 中出現的次數）。 透過這些定義，我們可以定義 idf(t)，即逆文檔
頻率，為 `log [ n / df(t) ] + 1`。
```python
tfidf_vectorizer = TfidfVectorizer(use_idf=True)
# use the tf-idf instance to fit list of research_interest 
tfidf = tfidf_vectorizer.fit_transform(research_interest_list)
df = pd.DataFrame(tfidf[0].T.todense(),index=tfidf_vectorizer.get_feature_names_out(),columns = ['tf-idf'])

df = df.sort_values('tf-idf',ascending=False)
print(df.head())

```

### Creating one-hot encoding (one of k)
One-hot 編碼（one-of-k）是將離散值轉換為二進位值序列的過程。
```python
from sklearn.preprocessing import LabelEncoder 
labelencoder = LabelEncoder()
encoded_data = labelencoder.fit_transfrom(df_research['is_atificial_intelligent'])
```

### Creating ordinal encoding 
序數編碼是將分類值轉換為數值的過程。 
```python
from sklearn.preprocessing import LabelEncoder 
labelencoder = LabelEncoder()
encoded_data = labelencoder.fit_transforms(df_research['research_interest'])

```


### Converting a colored image into a grayscale image 
```python
image = cv2.imread(file)
gray_image  = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
```

### Performing dimensionality reduction
在許多情況下，功能超越了任務的需要； 並非所有功能都有有用的信息。


在這種情況下，您可以使用降維技術，例如主成分分析(PCA)、奇異值分解(SVD)、線性判別分析(LDA)、t-SNE、UMAP 和 ISOMAP 等等。 另一種選擇是使用深度學習。

如果我們更正式地描述 PCA 過程，我們可以說該過程有兩個步驟：
1. 建構一個協方差矩陣，表示每對特徵的相關性。
2. 產生一組新的特徵，透過計算協方差矩陣的特徵值來捕捉不同量的資訊。

```python
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler 
file = '.csv'
df_features = pd.read_csv(file)
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

```