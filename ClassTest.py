class Test:

    def __init__(self):
        self.a = 1

    from ClassTest2 import print_a


test = Test()
test.print_a()