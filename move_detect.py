import cv2
from sklearn.cluster import DBSCAN
import numpy as np
import os
path = './data/image/poppyes-Scene-003/'
# path = './data/image/poppyes-Scene-004/'
folder_name = path.split('/')[-2]
file_names = os.listdir(path)

# 클러스터링 선언
dbscan = DBSCAN(eps=140, min_samples=7)

for num in range(0, len(file_names)-1):
    print(f'{file_names[num]} 파일 진행중 --------------------- {num} /// {len(file_names)-1}')
    file_name1= path + file_names[num]
    file_name2= path + file_names[num+1]
    img_1 = cv2.imread(file_name1)
    img_2 = cv2.imread(file_name2)

    img_1_gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    img_2_gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(img_1_gray, img_2_gray)

    # _, threshold_diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    _, threshold_diff = cv2.threshold(diff, 30, 255, cv2.THRESH_OTSU)
    print(_)
    contours, _ = cv2.findContours(threshold_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    points = []
    w_h = []

    result_frame = img_1.copy()
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Filter out small contours
            (x, y, w, h) = cv2.boundingRect(contour)

            points.append([x,y])
            w_h.append([w,h])

    # DBSCAN 알고리즘을 사용하여 클러스터링
    labels = dbscan.fit_predict(points)

    # 각 클러스터에 대한 색상 정의
    colors = np.random.randint(0, 255, size=(len(set(labels)), 3), dtype=np.uint8)

    # 최대 클러스터의 바운딩 박스 계산
    max_n = 0
    unique_labels = set(labels)

    for label in unique_labels:
        if label == -1:
            continue
        cluster_points = []
        for i in range(0, len(points)):
            if labels[i] == label:
                cluster_points.append(points[i])
        n = len(cluster_points)
        if n > max_n:
            max_n = n

            x_list = []
            y_list = []
            w_list = []
            h_list = []

            for num in range(0, len(cluster_points)):
                x,y = cluster_points[num]
                w, h = w_h[num]

                x_list.append(x)
                y_list.append(y)
                w_list.append(x+w)
                h_list.append(y+h)

                # 클러스터리된 포인트위 좌상단 포인트
                cv2.circle(result_frame, (x, y), 5, (0, 0, 0), -1)

                # 클러스터링된 포인트의 움직임 박스
                cv2.rectangle(result_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            x_min = np.min(x_list)
            y_min = np.min(y_list)
            x_max = np.max(w_list)
            y_max = np.max(h_list)

            cv2.rectangle(result_frame, (x_min, y_min), (x_max, y_max), (126, 75, 189), 2)

        # 결과 이미지 출력
        cv2.imshow('DBSCAN Clustering', result_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
