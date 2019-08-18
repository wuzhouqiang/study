import time

# add = lambda x, y: x+y
#
# print(add(3, 5))
#
# condition = True  # 三元
# print(3 if condition else 0)
#
# # map
# list1 = [1, 2, 4, 5]
# list2 = [4, 5, 6, 56]
# r = map(lambda x: x*x, list1)
# print(list(r))
#
# r2 = map(lambda x, y: x*y + 4, list1, list2)
# print(list(r2))

# fillter

# def is_not_nons(s):
#     return s and len(s.strip()) > 0
#
#
# list3 = ['', ' ', 'dfas', None]
# print(list(filter(is_not_nons, list3)))

# # reduce
# from functools import reduce
#
# f = lambda x, y: x + y
#
# reduce1 = reduce(f, [2, 3, 4], 19)
# print(reduce1)

# 列表推导式
# list4 = [3, 3, 5, 6]
# list5 = [i*i for i in list4 if i < 5]
# print(list5)
#
# 集合
# list6 = {3, 3, 5, 6}
# list7 = {i*i for i in list4 if i < 5}
# print(list7)


# s = {
#      "zhangsan": 34,
#      "lisi": 45,
#      "王五": 4
#      }
#
# s_key = [key + 'dd' for key, value in s.items()]
# print(s_key)
#
# s2 = {value: key for key, value in s.items()}
# print(s2)

# 闭包返回值是函数的函数


# def runtime():
#     def now_time():
#         print(time.time())
#
#     return now_time
#
# f = runtime()
# print(f())

# 装饰器
# 获取函数的运行时间

# def runtime2(func):
#
#     def get_time(*args, **kwargs):
#         func(*args)
#         print(time.time())
#         print(args)
#         print(kwargs)
#
#     return get_time
#
# @runtime2
# def student_run(*args, **kwargs):
#     print("学生跑")


# student_run(3, i=3, j=5)
# student_run(23, 2)

# def strOnly(func):
#     def check(*args, **kwargs):
#         for i in args:
#             if not isinstance(i, str):
#                 raise TypeError('Argument {} must be {}'.format(i, str))
#         for key, values in kwargs.items():
#             print(key, values)
#         func()
#     return check
#
# @strOnly
# def student(*args, **kwargs):
#     print("student")
#
#
# student("3", "312", i=3, j=3)


