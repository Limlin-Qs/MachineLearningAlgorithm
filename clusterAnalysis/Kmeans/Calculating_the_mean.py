# 提供给k-means0使用

def calculating(c_list):
    a = len(c_list)
    if a != 0:
        b = 2
        sum_sum = 0
        mean = []
        i = 0
        while i < b:
            for c in c_list:
                sum_sum += c[i]
            mean_one = sum_sum / a
            mean.append(mean_one)
            i += 1
        return mean
    else:
        return 0