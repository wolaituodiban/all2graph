# spark取数教程

在工业应用中，常常需要将大量的数据处理成json格式，这是个大数据问题，并不是特别容易<br>
本教程提供一个spark sql的模版供参考

假设有一个样本表sample，包含如下字段<br>
uid, timestamp

有一个数据表data，包含如下字段<br>
uid, timestamp, col1, col2

那么sql模版如下


```python
sql = """select
  a.uid,
  a.timestamp,
  to_json(
    collect_list(
      struct(
        b.timestamp, 
        b.col1,
        b.col2
      )
    )
  ) data
from
  sample a
  left join 
  data b
  on a.uid = b.uid
  and b.timestamp <= a.timestamp
group by
  a.uid,
  a.timestamp"""
```

假设数据已经被存储在一个叫做data.csv的文件里了<br>
使用pandas读取的时候可能需要配置一些参数


```python
import pandas as pd
import csv

df = pd.read_csv('data.csv', escapechar='\\', quoting=csv.QUOTE_MINIMAL)
```
