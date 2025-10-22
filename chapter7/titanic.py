import pandas as pd

pd.set_option('display.max_columns', None)

df = pd.read_csv(r'./chapter7/titanic/train.csv')

df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

df = df.dropna(subset=['Age'])

df = pd.get_dummies(df, columns=['Sex', 'Embarked'], dtype=int)

print(df.head(10))
