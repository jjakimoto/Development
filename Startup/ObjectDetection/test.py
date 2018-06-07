def search(array):
    left = 0
    right = len(array)
    while left <= right:
        mid = (left + right) // 2
        mid_val = array[mid]
        left_val = array[left]
        right_val = array[right]
        if mid_val > right_val:
            right = mid
        else:
            left = mid
    return array[mid]


def search(array):
    left = 0
    right = len(array) - 1
    while left <= right:
        mid = (left + right) // 2
        if array[mid] < array[mid + 1] and array[mid] < array[mid-1]:
            return array[mid]
        elif array[mid] < array[mid + 1]
            right = mid
        else:
            left = mid
    return array[mid]


def sort(array):
    for i in range(len(array)):
        for j in range(len(array)):
            if array[j] > array[j + 1]:
                tmp = array[j]
                array[j] = array[j + 1]
                array[j + 1] = tmp
    return array
