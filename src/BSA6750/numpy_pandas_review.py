import pandas as pd
import numpy as np

# df

grades_dict = {
    "Wally": [55,55,33],
    "Sally": [66,78,54],
    "Jimbo": [99,99,100]
}

grades_df = pd.DataFrame(grades_dict, index = ['Sem 1', 'Sem 2', 'Sem 3'])
print(grades_df)

# series are ordered, mutable, homogenous, indexable 

# pandas inherits dtype, shape, size, ndim 

'''print(grades_df.count())
print(grades_df.mean())
print(grades_df.min())
print(grades_df.max())'''

print(grades_df.describe())

# select columns via index name
columns = grades_df[["Wally", "Sally", "Jimbo"]]
print(columns)

# Boolean indexing, so condition values taken as an arugment for the index 
print(grades_df[grades_df >= 90])

# slicing rows using loc or iloc
print(grades_df.loc["Sem 1":"Sem 3"]) # loc is inclusive of the last label

print(grades_df.iloc[0:2]) # iloc (index locate) is exclusive of the stop values 

# at

print(grades_df.at["Sem 1", "Sally"])

# iat for index numbers

print(grades_df.iat[1,1]) # row 2, column 2 because index starts at 0 ofc

grades_df.info()


#df["Meters"] = df["Height"].apply(lambda height: height*0.0254) -- transform to meters using lambda and apply


data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Los Angeles', 'Chicago']
}

# Create the DataFrame
df = pd.DataFrame(data)

df.describe()