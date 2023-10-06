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
- 