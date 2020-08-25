# def a_new_decorator(a_func):
#     def wrapTheFunction():
#         print("I am doing some boring work before executing a_func()")
#
#         a_func()
#
#         print("I am doing some boring work after executing a_func()")
#
#     return wrapTheFunction
#
#
# # def a_function_requiring_decoration():
# #     print("I am the function which needs some decoration to remove my foul smell")
# #
# # a_function_requiring_decoration = a_new_decorator(a_function_requiring_decoration)
# # # now a_function_requiring_decoration is wrapped by wrapTheFunction()
# #
# # a_function_requiring_decoration()
#
#
#
# @a_new_decorator
# def a_function_requiring_decoration():
#     """Hey you! Decorate me!"""
#     print("I am the function which needs some decoration to "
#           "remove my foul smell")
#
# a_function_requiring_decoration()

from functools import wraps
#
#
# def a_new_decorator(a_func):
#     @wraps(a_func)
#     def wrapTheFunction():
#         print("I am doing some boring work before executing a_func()")
#         a_func()
#         print("I am doing some boring work after executing a_func()")
#
#     return wrapTheFunction
#
#
# @a_new_decorator
# def a_function_requiring_decoration():
#     """Hey yo! Decorate me!"""
#     print("I am the function which needs some decoration to "
#           "remove my foul smell")
#
# print(a_function_requiring_decoration.__name__)
# # Output: a_function_requiring_decoration
#
#
# def decorator_name(f):
#     @wraps(f)
#     def decorated(*args, **kwargs):
#         if not can_run:
#             return "Function will not run"
#         return f(*args, **kwargs)
#
#     return decorated
#
#
# @decorator_name
# def func():
#     return ("Function is running")
#
#
# can_run = True
# print(func())
# # Output: Function is running
#
# can_run = False
# print(func())
# # Output: Function will not run

# def requires_auth(f):
#     @wraps(f)
#     def decorated(*args, **kwargs):
#         auth = request.authorization
#         if not auth or not check_auth(auth.username, auth.password):
#             authenticate()
#         return f(*args, **kwargs)
#     return decorated


# def logit(func):
#     @wraps(func)
#     def with_logging(*args, **kwargs):
#         print(func.__name__ + " was called")
#         print(func(*args, **kwargs))
#         return func(*args, **kwargs)
#
#     return with_logging
#
#
# @logit
# def addition_func(x):
#     """Do some math."""
#     return x + x
#
#
# result = addition_func(4)

def logit(logfile='out.log'):
    def logging_decorator(func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            log_string = func.__name__ + " was called"
            print(log_string)
            # 打开logfile，并写入内容
            with open(logfile, 'a') as opened_file:
                # 现在将日志打到指定的logfile
                opened_file.write(log_string + '\n')
            return func(*args, **kwargs)

        return wrapped_function

    return logging_decorator


@logit()
def myfunc1():
    pass

myfunc1()
# Output: myfunc1 was called
# 现在一个叫做 out.log 的文件出现了，里面的内容就是上面的字符串

@logit(logfile='func2.log')
def myfunc2():
    pass


def a(a):
    print(a)
    def b():
        print('b')
    return b


a('1')()

myfunc2()
