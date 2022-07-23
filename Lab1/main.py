import matplotlib.pyplot as plt
from scipy.spatial import distance


def main():
    X = [[[], []], [[], []], [[], []]]
    with open('centroids.txt') as fp:
        centroids = [
            tuple(map(float, line.strip().split(', ')))
            for line in fp.read().split('\n')
            if line
        ]
    print(centroids)
    with open('dataset.txt') as fp:
        line = fp.readline()
        while line:
            if line:
                x = tuple(map(float, line.strip().split(',')))
                dist = 10e100
                selected_m = -1
                for m in centroids:
                    test_distance = distance.euclidean(x, m)
                    if test_distance < dist:
                        dist = test_distance
                        selected_m = centroids.index(m)
                X[selected_m][0].append(x[0])
                X[selected_m][1].append(x[1])
            else:
                break
            line = fp.readline()
    plt.plot(X[0][0], X[0][1], 'ro')
    plt.plot(X[1][0], X[1][1], 'go')
    plt.plot(X[2][0], X[2][1], 'bo')
    plt.plot([x for x, _ in centroids], [y for _, y in centroids], 'yo')
    plt.axis([-22, 20, -50, 40])
    plt.show()


if __name__ == '__main__':
    main()
