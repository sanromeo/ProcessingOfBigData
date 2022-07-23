import sys
from math import sqrt
from typing import List, Tuple


def get_centroids(path: str) -> List[Tuple[float, float]]:
    with open(path, 'r') as fp:
        centroids = [
            tuple(map(float, line.strip().split(', ')))
            for line in fp.read().split('\n')
            if line
        ]
    return centroids


def create_clusters(centroids: List[Tuple[float, float]]):
    for line in sys.stdin:
        x, y = tuple(map(float, line.strip().split(',')))
        min_dist = 10e10
        index = -1
        for x_center, y_center in centroids:
            cur_dist = sqrt(pow(x - x_center, 2) + pow(y - y_center, 2))
            if cur_dist <= min_dist:
                min_dist = cur_dist
                index = centroids.index((x_center, y_center))
        print(f'{index}\t{x}\t{y}')

        
if __name__ == "__main__":
    centroids = get_centroids('centroids.txt')
    create_clusters(centroids)
