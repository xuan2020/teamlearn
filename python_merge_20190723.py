# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 17:33:02 2019

@author: ZhoJi001
"""

 import numpy as np
 import pandas as pd

s=3
s+4
s = pd.Series([1,2,3,4,5])

s = pd.Series([1, 3, 5, np.nan, 6, 8])
s
dates = pd.date_range('20130101', periods=6)
dates
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
df

  df2 = pd.DataFrame({
'A': 1.,
'B': pd.Timestamp('20130102'),
'C': pd.Series(1, index=list(range(4)), dtype='float32'),
'D': np.array([3] * 4, dtype='int32'),
'E': pd.Categorical(["test", "train", "test", "train"]),
'F': 'foo'})
  df2
  df2.dtypes
  df.head(2)
  df.index
  df.describe()
  df.T
  df.sort_values(by='B')
  df['A']
  df[2:3]
  df.loc[dates[0:2]]
  df.loc[:, ['A', 'B']]
  df.loc['20130102':'20130104', ['A', 'B']]
  df.loc['20130102', ['A', 'B']]
  df
  df.loc[dates[0]]
  df.loc[:, ['A', 'B']]
  df.loc['20130102':'20130104', ['A', 'B']]
  
  
 df.iloc[3]
 df.loc[3:5, 1:3]
 df.iloc[[1, 2, 4], [0, 3]]
 df.iloc[1:3, :]
 df.iloc[:, 1:3]
 df[df.A > 0]
    df[df > 0]
    df2 = df.copy()
    df2['E'] = ['one', 'one', 'two', 'three', 'four', 'three']
    df2[df2['E'].isin(['test', 'four'])]
    
s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('20130103', periods=6))
 df['F'] = s1
df[dates[0], 'A'] = 0
 df.iat[0, 1] = 0
 
 df.mean()
 df.mean().max()
 df.apply(np.cumsum)
 df.apply(lambda x: x.max() - x.min())
 df = pd.DataFrame(np.random.randn(10, 4))
 pieces = [df[:3], df[3:7], df[7:]]
 pd.concat(pieces)
   
left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'bar'], 'rval': [4, 5]})
pd.merge(left, right, on='key')

df = pd.DataFrame(np.random.randn(8, 4), columns=['A', 'B', 'C', 'D'])
s = df.iloc[3]
df.append(s, ignore_index=True)

df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
                 'B': ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
                  'C': np.random.randn(8),
               'D': np.random.randn(8)})
 df.groupby('A').sum() 
 df.groupby(['A', 'B']).sum()
 
 
 tuples = list(zip(*[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                    ['one', 'two', 'one', 'two',  'one', 'two', 'one', 'two']]))

index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])
 df2=df[:4]
df2.stack()

df = pd.DataFrame({'A': ['one', 'one', 'two', 'three'] * 3,
                   'B': ['A', 'B', 'C'] * 4,
                    'C': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
                     'D': np.random.randn(12),
                     'E': np.random.randn(12)})
 pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'])
 
 
 ts = pd.Series(np.random.randn(1000),
              index=pd.date_range('1/1/2000', periods=1000))

 ts=ts.cumsum()
 ts.plot()
