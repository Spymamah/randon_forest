def factorialNum(num):
    factorial = 1
    if num == 0:
        return factorial
    else:
        for i in range(1, num+1):
            factorial = factorial*i
        return factorial
num3 = factorialNum(5)
print("the factorial of 5 is, ", num3)
    

    