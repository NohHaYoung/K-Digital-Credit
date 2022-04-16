import sklearn.decomposition
import sklearn.cluster
import matplotlib.pyplot as plt
import numpy as np
import elice_utils

def main():
    X, attributes = input_data()
    X = normalize(X)
    pca, pca_array = run_PCA(X, 2)
    labels = kmeans(pca_array, 5, [0, 1, 2, 3, 4]) # k=3, 시작중심점 3개[1, 2, 3]
    #print(labels)
    visualize_2d_wine(pca_array, labels)

def input_data():
    X = []
    attributes = []
    
    with open('data/wine.csv') as fp:
        for line in fp:
            X.append([float(x) for x in line.strip().split(',')])
    
    with open('data/attributes.txt') as fp:
        attributes = [x.strip() for x in fp.readlines()]

    return np.array(X), attributes

def run_PCA(X, num_components):
    pca = sklearn.decomposition.PCA(n_components=num_components)
    pca.fit(X)
    pca_array = pca.transform(X)

    return pca, pca_array

def kmeans(X, num_clusters, initial_centroid_indices):
    '''
    X : 178 * 2(차원 축소된 input)
    num_clusters : 클러스터의 갯수
    initial_centroid_indices : 시작할 중심점들의 인덱스
    '''
    
    import time
    N = len(X)
    centroids = X[initial_centroid_indices]
    labels = np.zeros(N) #labels = 각각의 데이터 포인트에 대한 클러스터 초기값 0으로 세팅
    
    while True:
        '''
        Step 1. 각 데이터 포인트 i 에 대해 가장 가까운
        중심점을 찾고, 그 중심점에 해당하는 클러스터를 할당하여
        labels[i]에 넣습니다.
        가까운 중심점을 찾을 때는, 유클리드 거리(= norm @2차원)를 사용합니다.
        미리 정의된 distance 함수를 사용합니다.
        '''
        is_changed = False #centroid 값이 변경되었는지 확인하는 변수
        for i in range(N):
            distances = []
            for k in range(num_clusters):
                #X[i] 와 centroids[k]의 거리 계산(distance 함수 사용)
                k_dist = distance(X[i], centroids[k])
                distances.append(k_dist)
            #print(distances)
            if labels[i] != np.argmin(distances):
                is_changed = True
                labels[i] = np.argmin(distances) #클러스터 할당
        #print(labels)
        
        '''
        Step 2. 할당된 클러스터를 기반으로 새로운 중심점을 계산합니다.
        중심점은 클러스터 내 데이터 포인트들의 위치의 *산술 평균*
        으로 합니다.
        '''
        for k in range(num_clusters):
            x = X[labels == k][ : , 0] #k번째 클러스터에 할당된 데이터의 x 좌표의 모음
            y = X[labels == k][ : , 1] #k번째 클러스터에 할당된 데이터의 y 좌표의 모음
            
            # 산술 평균
            x = np.mean(x)
            y = np.mean(y)
        
            # centroid 변경
            centroids[k] = [x, y]
        '''
        Step 3. 만약 클러스터의 할당이 바뀌지 않았다면 알고리즘을 끝냅니다.
        아니라면 다시 반복합니다.
        '''
        if not is_changed: #바뀌지 않으면 알고리즘 종료
            break
            
    return labels

def distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))
    
def normalize(X):
    for dim in range(len(X[0])):
        X[:, dim] -= np.min(X[:, dim])
        X[:, dim] /= np.max(X[:, dim])
    return X

'''
이전에 더해, 각각의 데이터 포인트에 색을 입히는 과정도 진행합니다.
'''

def visualize_2d_wine(X, labels):
    plt.figure(figsize=(10, 6))
    plt.scatter(X[ : ,0], X[ : , 1], c = labels) # c = labels : 색을 자동으로 입힘
    plt.savefig("image.svg", format="svg")
    elice_utils.send_image("image.svg")

if __name__ == '__main__':
    main()