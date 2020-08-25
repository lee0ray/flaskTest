class D(object):
    def test(self):
        print('test in D')

class B(D):
    def test(self):
        print('test in B')
        # super().test() # 与下面的写法等价
        super(B, self).test()  # 返回self对应类的MRO中，类B的下一个类的代理

class C(D):
    def test(self):
        print('test in C')
        # super().test() # 与下面的写法等价
        # super(C, self).test()  # 返回self对应类的MRO中，类C的下一个类的代理





class A(B, C):
    pass


