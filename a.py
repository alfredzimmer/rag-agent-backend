def fib():
    a, b = 1, 1
    a = 1
    b = 1

    # 1
    a, b = b, a + b  # ????

    # 2 incorrect!!
    a = b
    b = a + b

    # 1
    temp = a + b
    a = b
    b = temp
