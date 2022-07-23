import sys


def calculate_centroids():
    current_centroid = None
    sum_x = 0
    sum_y = 0
    count = 0
    for line in sys.stdin:
        centroid_index, x, y = line.split('\t')
        x = float(x)
        y = float(y)
        if current_centroid == centroid_index:
            count += 1
            sum_x += x
            sum_y += y
        else:
            if count != 0:
                print(str(sum_x / count) + ", " + str(sum_y / count))
            current_centroid = centroid_index
            sum_x = x
            sum_y = y
            count = 1

            
if __name__ == "__main__":
    calculate_centroids()
