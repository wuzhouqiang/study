import pandas as pd
from pandas import Series,DataFrame
import numpy as np

# obj = Series(['d', 'sd'], index=[1, 2])
# print(obj[1])
#
# data = {'a': 100, 'b': 12, 'c': 4, 'g': None}
# ss = Series(data)
# ss = ss.dropna()
# # print(ss)
# print(ss.isnull)

data2 = {
    '60': ['狗子', '嘎子'],
    '70': ['爱国', '建国'],
    '80': ['李磊', '电风扇']
}
#
# frame_data = DataFrame(data2)
# # print(frame_data)
# print(frame_data['60'])

# dates = pd.date_range('20190301', periods=4)
# print(dates)
#
# df = pd.DataFrame(np.random.randn(4, 3), dates, columns=list('ABC'))
# print(df)
# print(df['20190301':'20190303'])

# print(df.loc['20190301':'20190303', ['A', 'B']])
# print(df.at[dates[0], 'A'])
# print(df.tail(3))
# print(df.head(3))

# d1 = Series([3, 2, 45, 65], index=['a', 'b', 'c', 'd'])
# d2 = Series([-1, 2, 4, 5, 53], index=['a', 'b', 'c', 'd', 'e'])
#
# print(d1 + d2)
# print(d1.add(d2, fill_value=0))

# f1 = DataFrame(np.arange(12).reshape(4, 3), columns=list("abc"), index=[1, 2, 3, 4])
# s1 = f1.loc[1]
# # print(f1)
# # print(s1)
# print(f1 - s1)  # 广播相减


# obj = Series(range(4), index=['d', 'e', 'a', 'b'])
# print(obj)
# print(obj.sort_index())
# print(obj.sort_values())

# frame = DataFrame(np.arange(8).reshape(2, 4), index=['two', 'one'], columns=['c', 'd', 'a', 'b'] )
# print(frame)
# print(frame.sum(axis=1))

csv = pd.read_csv('../data/car_news.csv', index_col=0, nrows=7)
print(csv.head())
csv.to_excel('../data/car_new.xlsx')


