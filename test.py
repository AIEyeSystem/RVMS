import cv2
import numpy as np

def thin_image(src, max_iterations=-1):
    assert src.dtype == np.uint8
    dst = src.copy()
    width, height = src.shape[::-1]
    count = 0
    
    while True:
        count += 1
        if max_iterations != -1 and count > max_iterations:
            break
        
        m_flag = []
        
        for i in range(height):
            for j in range(width):
                p1 = src[i, j]
                if p1 != 1:
                    continue
                p4 = src[i, j + 1] if j < width - 1 else 0
                p8 = src[i, j - 1] if j > 0 else 0
                p2 = src[i - 1, j] if i > 0 else 0
                p3 = src[i - 1, j + 1] if i > 0 and j < width - 1 else 0
                p9 = src[i - 1, j - 1] if i > 0 and j > 0 else 0
                p6 = src[i + 1, j] if i < height - 1 else 0
                p5 = src[i + 1, j + 1] if i < height - 1 and j < width - 1 else 0
                p7 = src[i + 1, j - 1] if i < height - 1 and j > 0 else 0
                
                if 2 <= p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9 <= 6:
                    ap = 0
                    if p2 == 0 and p3 == 1:
                        ap += 1
                    if p3 == 0 and p4 == 1:
                        ap += 1
                    if p4 == 0 and p5 == 1:
                        ap += 1
                    if p5 == 0 and p6 == 1:
                        ap += 1
                    if p6 == 0 and p7 == 1:
                        ap += 1
                    if p7 == 0 and p8 == 1:
                        ap += 1
                    if p8 == 0 and p9 == 1:
                        ap += 1
                    if p9 == 0 and p2 == 1:
                        ap += 1
                    
                    if ap == 1 and p2 * p4 * p6 == 0 and p4 * p6 * p8 == 0:
                        m_flag.append((i, j))
        
        for i, j in m_flag:
            dst[i, j] = 0
        
        if not m_flag:
            break
    
    return dst

def filter_over(thin_src):
    assert thin_src.dtype == np.uint8
    width, height = thin_src.shape[::-1]
    
    for i in range(height):
        for j in range(width):
            p1 = thin_src[i, j]
            if p1 != 1:
                continue
            p4 = thin_src[i, j + 1] if j < width - 1 else 0
            p8 = thin_src[i, j - 1] if j > 0 else 0
            p2 = thin_src[i - 1, j] if i > 0 else 0
            p3 = thin_src[i - 1, j + 1] if i > 0 and j < width - 1 else 0
            p9 = thin_src[i - 1, j - 1] if i > 0 and j > 0 else 0
            p6 = thin_src[i + 1, j] if i < height - 1 else 0
            p5 = thin_src[i + 1, j + 1] if i < height - 1 and j < width - 1 else 0
            p7 = thin_src[i + 1, j - 1] if i < height - 1 and j > 0 else 0
            
            if p2 + p3 + p8 + p9 >= 1:
                thin_src[i, j] = 0
    return thin_src

def get_points(thin_src, radius=4, threshold_max=6, threshold_min=4):
    assert thin_src.dtype == np.uint8
    width, height = thin_src.shape[::-1]
    tmp = thin_src.copy()
    points = []
    
    for i in range(height):
        for j in range(width):
            if tmp[i, j] == 0:
                continue
            count = 0
            
            for k in range(i - radius, i + radius + 1):
                for l in range(j - radius, j + radius + 1):
                    if 0 <= k < height and 0 <= l < width and tmp[k, l] == 1:
                        count += 1
            
            if count > threshold_max or count < threshold_min:
                points.append((j, i))
    
    return points

def get_points_2(thin_src, radius=6, threshold_max=7, threshold_min=5):
    assert thin_src.dtype == np.uint8
    width, height = thin_src.shape[::-1]
    tmp = thin_src.copy()
    points = []
    
    for i in range(height):
        for j in range(width):
            if tmp[i, j] == 0:
                continue
            count = 0
            
            for k in range(i - radius, i + radius + 1):
                for l in range(j - radius, j + radius + 1):
                    if 0 <= k < height and 0 <= l < width and tmp[k, l] == 1:
                        count += 1
            
            if count > threshold_max:
                points.append((j, i, 'Intersection'))
            elif count < threshold_min and count > 3:
                points.append((j, i, 'End'))
    
    return points
if __name__ == "__main__":
    src = cv2.imread("0.png", cv2.IMREAD_GRAYSCALE)
    
    if src is None:
        print("Failed to read the file!")
    else:
        _, src = cv2.threshold(src, 128, 1, cv2.THRESH_BINARY)
        dst = thin_image(src)
        dst = filter_over(dst)
        points = get_points_2(dst, 6, 9, 6)
        # dst = np.repeat(dst,[0,0,3])
        dst = cv2.cvtColor(dst,cv2.COLOR_GRAY2BGR)
        print(dst.shape)
        dst = dst * 255
        src = src * 255
        

        # for point in points:
        #     cv2.circle(dst, point, 2, 255, 1)
        for point in points:
            x, y, point_type = point
            if point_type == 'Intersection':
                cv2.circle(dst, (x, y), 2, [0,0,255], 1)  # Draw Intersection Points
            elif point_type == 'End':
                cv2.circle(dst, (x, y), 2, [0,255,0], 1)  # Draw End Points

        cv2.imwrite("dst.jpg", dst)

        cv2.namedWindow("src", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("dst", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("src", src)
        cv2.imshow("dst", dst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # cv2.imwrite("dst.jpg", dst)
        # cv2.imshow("src1", src)
        # cv2.imshow("dst1", dst)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
