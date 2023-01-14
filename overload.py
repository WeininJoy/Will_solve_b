from multipledispatch import dispatch
 
# passing one parameter
 
 
@dispatch(int, int)
def product(first, second):
    result = first*second
    print(result)
 
# passing two parameters
 
 
@dispatch(int, int, int)
def product(first, second, third):
    result = first * second * third
    print(result)

 
# you can also pass data type of any value as per requirement
 
@dispatch(float, float, float)
def product(first, second, third):
    result = first * second * third
    print(result)
 
 
# calling product method with 2 arguments
print(product(2, 3))  # this will give output of 6
 
# calling product method with 3 arguments but all int
print(product(2, 3, 2))  # this will give output of 12
 
# calling product method with 3 arguments but all float
print(product(2.2, 3.4, 2.3))  # this will give output of 17.985999999999997