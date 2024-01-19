import contextlib


@contextlib.contextmanager
def test(s):
    print("{} part".format(s))
    yield
    print("{} part".format(s))


@contextlib.contextmanager
def null_context():
    yield


# 使用可选的上下文管理器
def optional_context_manager():
    return null_context()


if __name__ == '__main__':
    '''
    with test("first"):
        print("*"*10)
    with test("second"):
        print("-"*10)
    '''

    # 使用空的上下文管理器
    with null_context():
        print("Hello, world!")

    with optional_context_manager():
        print("Hello, world!")