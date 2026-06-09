import pandas as pd

#Create a Data Frame (like an Excel Table)
data = {
    'Name' : ['Sohaib' , 'Maham' , 'Fahad'],
    'Age' : [21 , 19 , 22],
    'Score' : [48 , 48 , 47]
}
df = pd.DataFrame(data)
print(df)
print(df.head())
print(df.describe())
print(df['Age'])
print(df[df['Score'] > 47])

#Basic Operations
print(df['Score'].mean())
print(df['Age'].max())
df['Grade'] = ['A+' , "A+" , 'A']

print(df)