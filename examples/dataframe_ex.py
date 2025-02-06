import pandas as pd

data={
    "name":["a","b","c"],
    "age":[1,2,3],
    "gender":["male","female","other"]
}

df1=pd.DataFrame(data)

df2=pd.DataFrame(data).iloc


print(df1[0])

print(df2)