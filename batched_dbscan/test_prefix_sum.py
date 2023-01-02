import numpy as np




def prefix_sum(arr):
        """
        Calculates the prefix sum of pT.
        Warning, requires array to be of size thats log base of 2.
        """
        size_log2 = int(np.log2(arr.shape[0]))

        # up-sweep
        for d in range(0, size_log2, 1):
            step_size = 2**d
            double_step_size = step_size * 2

            for i in range(0, arr.shape[0], double_step_size):
                arr[i + double_step_size - 1] += arr[i + step_size - 1]
        print(arr)
        # down-sweep
        arr[arr.shape[0] - 1] = 0
        d = size_log2 - 1

        while d >= 0:
            step_size = 2**d
            double_step_size = step_size * 2
            for i in range(0, arr.shape[0], double_step_size):
                tmp = arr[i + step_size - 1]
                arr[i + step_size - 1] = arr[i + double_step_size - 1]
                arr[i + double_step_size - 1] += tmp
            d -= 1

        return arr
    
def prefix_sum_cpp(arr):
        """
        Calculates the prefix sum of pT.
        Warning, requires array to be of size thats log base of 2.
        """
        size_log2 = int(np.log2(arr.shape[0]))

        # up-sweep
        for d in range(0, size_log2, 1):
            step_size = 2**d
            double_step_size = step_size * 2

            for i in range(0, arr.shape[0], double_step_size):
                arr[i + double_step_size - 1] += arr[i + step_size - 1]
        print(arr)
        # down-sweep
        arr[arr.shape[0] - 1] = 0
        
        for d in range(size_log2 - 1, -1, -1):
            step_size = 2**d
            double_step_size = step_size * 2
            for i in range(0, arr.shape[0], double_step_size):
                tmp = arr[i+ step_size -1]
                arr[i + step_size - 1] = arr[i + double_step_size - 1]
                arr[i + double_step_size - 1] += tmp

        return arr
    
# pt = np.array([2.35, 2.12, 2.09, 2.71, 4.1, 2.15, 2.45, 3.19])
pt = np.array([2.35, 2.12, 2.09, 2.71, 1.50])

arr = pt.copy()
arrcpp = pt.copy()

rs = prefix_sum(arr)
rs_2 = prefix_sum_cpp(arrcpp)
# print(pt)
print(rs)
print(rs_2)
# print(np.cumsum(pt))