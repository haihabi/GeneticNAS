def vector_bits2int(arr):
    n = arr.shape[0]  # number of columns
    a = arr[0] << n - 1

    for j in range(1, n):
        # "overlay" with the shifted bits of the next column
        a |= arr[j] << n - 1 - j
    return a
