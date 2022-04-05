import multiprocessing

output=[]
data = range(0,200000000)

def f(x):
    return x**2

def handler():
    p = multiprocessing.Pool(10)
    r=p.map(f, data)
    return r

a = handler()
if __name__ == '__main__':
    output.append(handler())