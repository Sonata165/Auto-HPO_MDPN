global a
a = 2
def f():
    global a
    a += 1
    print(a)
    return (1, 2, 3, 4)

tup = (1,2,3)
a,b,c = tup
print(c)