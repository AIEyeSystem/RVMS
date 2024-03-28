import skimage
import cv2
import numpy as np


def sk_skeletonize(image,method='lee'):
    # image = cv2.imread("../MorphoSnake/img/2.png", cv2.IMREAD_GRAYSCALE)
    # 骨架提取
    skeleton = skimage.morphology.skeletonize(image,method='lee')
    return skeleton

def kernel_skeleton(image):
    # import skimage.io as io
    # image = io.imread(segmentation_path)
    # print(image.shape,np.max(image))
    image = np.where(image>128,255,0).astype(np.uint8)
    print(image.dtype,np.max(image))

    A = 200
    L = 50

    # Centerline extraction using Zeun-Shang's thinning algorithm
    # Using opencv-contrib-python which provides very fast and efficient thinning algorithm
    # The package can be installed using pip
    thinned = cv2.ximgproc.thinning(image)
    # Filling broken lines via morphological closing using a linear kernel
    kernel = np.ones((1, 10), np.uint8)
    d_im = cv2.dilate(thinned, kernel)
    e_im = cv2.erode(d_im, kernel) 
    num_rows, num_cols = thinned.shape
    for i in range (1, 360//15):
        rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 15*i, 1)
        img_rotation = cv2.warpAffine(thinned, rotation_matrix, (num_cols, num_rows))
        temp_d_im = cv2.dilate(img_rotation, kernel)
        temp_e_im = cv2.erode(temp_d_im, kernel) 
        rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), -15*i, 1)
        im = cv2.warpAffine(temp_e_im, rotation_matrix, (num_cols, num_rows))
        e_im = np.maximum(im, e_im)
    # Skeletonizing again to remove unwanted noise
    thinned1 = cv2.ximgproc.thinning(e_im)
    # thinned1 = thinned1*(mask/255)
    # Removing bifurcation points by using specially designed kernels
    # Can be optimized further! (not the best implementation)
    thinned1 = np.uint8(thinned1)



    thh = thinned1.copy()
    hi = thinned1.copy()
    thi = thinned1.copy()
    hi = cv2.cvtColor(hi, cv2.COLOR_GRAY2BGR)
    thi = cv2.cvtColor(thi, cv2.COLOR_GRAY2BGR)
    thh = thh/255
    kernel1 = np.array([[1,0,1],[0,1,0],[0,1,0]])
    kernel2 = np.array([[0,1,0],[1,1,1],[0,0,0]])
    kernel3 = np.array([[0,1,0],[0,1,1],[1,0,0]])
    kernel4 = np.array([[1,0,1],[0,1,0],[0,0,1]])
    kernel5 = np.array([[1,0,1],[0,1,0],[1,0,1]])
    kernels = [kernel1, kernel2, kernel3, kernel4, kernel5]
    for k in kernels:
        k1 = k
        k2 = cv2.rotate(k1, cv2.ROTATE_90_CLOCKWISE)
        k3 = cv2.rotate(k2, cv2.ROTATE_90_CLOCKWISE)
        k4 = cv2.rotate(k3, cv2.ROTATE_90_CLOCKWISE)
        ks = [k1, k2, k3, k4]
        for kernel in ks:
            th = cv2.filter2D(thh, -1, kernel)
            for i in range(th.shape[0]):
                for j in range(th.shape[1]):
                    if(th[i,j]==4.0):
                        cv2.circle(hi, (j, i), 2, (0, 255, 0), 2)
                        cv2.circle(thi, (j, i), 2, (0, 0, 0), 2)
    thi = cv2.cvtColor(thi, cv2.COLOR_BGR2GRAY)
    cl = thi.copy()
    contours, hierarchy = cv2.findContours(thi, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        if (c.size<L):
            cv2.drawContours(cl, [c], 0, 0, -1)


    # # Centerline superimposed on green channel
    # colors = [(100, 0, 150), (102, 0, 255), (0, 128, 255), (255, 255, 0), (10, 200, 10)]
    # colbgr = [(193, 182, 255), (255, 0, 102), (255, 128, 0), (0, 255, 255), (10, 200, 10)]
    # im = g.copy()
    # im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    # thc = cl
    # thh = thc.copy()
    # thh = cv2.cvtColor(thh, cv2.COLOR_GRAY2BGR)
    # contours, heirarchy = cv2.findContours(thc, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # for c in contours:
        
    #         color = np.random.randint(len(colors))
    #         cv2.drawContours(im, c, -1, colbgr[color], 2, cv2.LINE_AA)

    # cv2.imwrite('00.png',thinned1)
    # cv2.imshow('closed contour', thinned1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    ## d = mv*1.5

    return thinned1

def thin_skeleton(src, max_iterations=-1):
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
    '''
    remove repeat point at cross over ??
    '''
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
    # cv2.imwrite('000.png',thin_src)
    return thin_src