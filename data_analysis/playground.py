import numpy as np


def gear(iteration, n):
   if(iteration == len(n) - 1):
       return 0
   else:
       return n[iteration + 1] - n[iteration] - gear(iteration+1, n)

def solution(n):
    x = gear(0, n)
    numerator, denominator = float((2*x)).as_integer_ratio()
    return numerator, denominator

if __name__ == '__main__':
    print(solution(np.array([4, 20, 50, 70, 90, 180, 190])))