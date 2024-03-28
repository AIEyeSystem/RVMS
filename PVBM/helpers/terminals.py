import itertools
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def generate_nonadjacent_combination(input_list,take_n):
    """ 
    It generates combinations of m taken n at a time where there is no adjacent n.
    INPUT:
        input_list = (iterable) List of elements you want to extract the combination 
        take_n =     (integer) Number of elements that you are going to take at a time in
                     each combination
    OUTPUT:
        all_comb =   (np.array) with all the combinations
    """
    all_comb = []
    for comb in itertools.combinations(input_list, take_n):
        comb = np.array(comb)
        d = np.diff(comb)
        fd = np.diff(np.flip(comb))
        if len(d[d==1]) == 0 and comb[-1] - comb[0] != 7:
            all_comb.append(comb)        
            print(comb)
    return all_comb


def populate_intersection_kernel(combinations):
    """
    Maps the numbers from 0-7 into the 8 pixels surrounding the center pixel in
    a 9 x 9 matrix clockwisely i.e. up_pixel = 0, right_pixel = 2, etc. And 
    generates a kernel that represents a line intersection, where the center 
    pixel is occupied and 3 or 4 pixels of the border are ocuppied too.
    INPUT:
        combinations = (np.array) matrix where every row is a vector of combinations
    OUTPUT:
        kernels =      (List) list of 9 x 9 kernels/masks. each element is a mask.
    """
    n = len(combinations[0])
    template = np.array((
            [-1, -1, -1],
            [-1, 1, -1],
            [-1, -1, -1]), dtype="int")
    match = [(0,1),(0,2),(1,2),(2,2),(2,1),(2,0),(1,0),(0,0)]
    kernels = []
    for n in combinations:
        tmp = np.copy(template)
        for m in n:
            tmp[match[m][0],match[m][1]] = 1
        kernels.append(tmp)
    return kernels


def give_intersection_kernels():
    """
    Generates all the intersection kernels in a 9x9 matrix.
    INPUT:
        None
    OUTPUT:
        kernels =      (List) list of 9 x 9 kernels/masks. each element is a mask.
    """
    input_list = np.arange(8)
    taken_n = [4,3]
    kernels = []
    for taken in taken_n:
        comb = generate_nonadjacent_combination(input_list,taken)
        tmp_ker = populate_intersection_kernel(comb)
        kernels.extend(tmp_ker)
    return kernels


# Find the curve intersections
def find_intersection_points(input_image, show=0):
    """
    Applies morphologyEx with parameter HitsMiss to look for all the curve 
    intersection kernels generated with give_intersection_kernels() function.
    INPUT:
        input_image =  (np.array dtype=np.uint8) binarized m x n image matrix
    OUTPUT:
        output_image = (np.array dtype=np.uint8) image where the nonzero pixels 
                       are the line intersection.
    """
    kernel = np.array(give_intersection_kernels())
    output_image = np.zeros(input_image.shape)
    for i in np.arange(len(kernel)):
        out = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel[i,:,:])
        output_image = output_image + out
    if show == 1:
        show_image = np.reshape(np.repeat(input_image, 3, axis=1),(input_image.shape[0],input_image.shape[1],3))*255
        show_image[:,:,1] = show_image[:,:,1] -  output_image *255
        show_image[:,:,2] = show_image[:,:,2] -  output_image *255
        plt.imshow(show_image)
    return output_image

def get_neighbours(x,y,image):
    """Return 8-neighbours of image point P1(x,y), in a clockwise order"""
    img = image
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
    return [ img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1], img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1] ]   
def find_intersection_points_2(skeleton):
    """ Given a skeletonised image, it will give the coordinates of the intersections of the skeleton.
    
    Keyword arguments:
    skeleton -- the skeletonised image to detect the intersections of
    
    Returns: 
    List of 2-tuples (x,y) containing the intersection coordinates
    """
    # A biiiiiig list of valid intersections             2 3 4
    # These are in the format shown to the right         1 C 5
    #                                                    8 7 6 
    validIntersection = [[0,1,0,1,0,0,1,0],[0,0,1,0,1,0,0,1],[1,0,0,1,0,1,0,0],
                         [0,1,0,0,1,0,1,0],[0,0,1,0,0,1,0,1],[1,0,0,1,0,0,1,0],
                         [0,1,0,0,1,0,0,1],[1,0,1,0,0,1,0,0],[0,1,0,0,0,1,0,1],
                         [0,1,0,1,0,0,0,1],[0,1,0,1,0,1,0,0],[0,0,0,1,0,1,0,1],
                         [1,0,1,0,0,0,1,0],[1,0,1,0,1,0,0,0],[0,0,1,0,1,0,1,0],
                         [1,0,0,0,1,0,1,0],[1,0,0,1,1,1,0,0],[0,0,1,0,0,1,1,1],
                         [1,1,0,0,1,0,0,1],[0,1,1,1,0,0,1,0],[1,0,1,1,0,0,1,0],
                         [1,0,1,0,0,1,1,0],[1,0,1,1,0,1,1,0],[0,1,1,0,1,0,1,1],
                         [1,1,0,1,1,0,1,0],[1,1,0,0,1,0,1,0],[0,1,1,0,1,0,1,0],
                         [0,0,1,0,1,0,1,1],[1,0,0,1,1,0,1,0],[1,0,1,0,1,1,0,1],
                         [1,0,1,0,1,1,0,0],[1,0,1,0,1,0,0,1],[0,1,0,0,1,0,1,1],
                         [0,1,1,0,1,0,0,1],[1,1,0,1,0,0,1,0],[0,1,0,1,1,0,1,0],
                         [0,0,1,0,1,1,0,1],[1,0,1,0,0,1,0,1],[1,0,0,1,0,1,1,0],
                         [1,0,1,1,0,1,0,0]]
    image = skeleton.copy()
    if np.max(image) > 1:
        image = np.where(image==0,0,1).astype(np.uint8)
    intersections = list()
    for x in range(1,len(image)-1):
        for y in range(1,len(image[x])-1):
            # If we have a white pixel
            if image[x][y] == 1:
                neighbours = get_neighbours(x,y,image)
                valid = True
                if neighbours in validIntersection:
                    intersections.append((y,x,'Intersection'))
    # Filter intersections to make sure we don't count them twice or ones that are very close together
    for point1 in intersections:
        for point2 in intersections:
            if (((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2) < 10**2) and (point1 != point2):
                intersections.remove(point2)
    # Remove duplicates
    intersections = list(set(intersections))
    return intersections


#  finding corners
def find_end_points(image, show=0):
    """
    """
    input_image = image.copy()
    if np.max(input_image) > 1:
        input_image = np.where(input_image==0,0,1).astype(np.uint8)
    kernel_0 = np.array((
            [-1, -1, -1],
            [-1, 1, -1],
            [-1, 1, -1]), dtype="int8")

    kernel_1 = np.array((
        [-1, -1, -1],
        [-1, 1, -1],
        [1,-1, -1]), dtype="int8")

    kernel_2 = np.array((
        [-1, -1, -1],
        [1, 1, -1],
        [-1,-1, -1]), dtype="int8")

    kernel_3 = np.array((
        [1, -1, -1],
        [-1, 1, -1],
        [-1,-1, -1]), dtype="int8")

    kernel_4 = np.array((
        [-1, 1, -1],
        [-1, 1, -1],
        [-1,-1, -1]), dtype="int8")

    kernel_5 = np.array((
        [-1, -1, 1],
        [-1, 1, -1],
        [-1,-1, -1]), dtype="int8")

    kernel_6 = np.array((
        [-1, -1, -1],
        [-1, 1, 1],
        [-1,-1, -1]), dtype="int8")

    kernel_7 = np.array((
        [-1, -1, -1],
        [-1, 1, -1],
        [-1,-1, 1]), dtype="int8")

    kernel = np.array((kernel_0,kernel_1,kernel_2,kernel_3,kernel_4,kernel_5,kernel_6, kernel_7))
    output_image = np.zeros(input_image.shape)
    for i in np.arange(8):
        # print(input_image.shape,input_image.dtype,'ff')
        out = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel[i,:,:])
        # print(out.shape)
        output_image = output_image + out

    if show == 1:
        show_image = np.reshape(np.repeat(input_image, 3, axis=1),(input_image.shape[0],input_image.shape[1],3))*255
        show_image[:,:,1] = show_image[:,:,1] -  output_image *255
        show_image[:,:,2] = show_image[:,:,2] -  output_image *255
        plt.imshow(show_image)    



    
    eol_img = output_image.astype(np.uint8)
    end_points_p  = np.argwhere(eol_img == 1).tolist()
    end_points_p = [(p[1],p[0],'End') for p in end_points_p]
    
    return end_points_p#, np.where(output_image == 1)






def get_terminals_circle_kernel(thin_src, radius=6, threshold_max=7, threshold_min=5):
    assert thin_src.dtype == np.uint8
    width, height = thin_src.shape[::-1]
    tmp = thin_src.copy()
    tmp = np.where(tmp == 0,0,1)
    end_points = []
    intersection_points = []
    
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
                end_points.append((j, i, 'Intersection'))
            elif count < threshold_min and count > 3:
                intersection_points.append((j, i, 'End'))
    
    return end_points,intersection_points

def get_terminals_hitmiss(skel,only_points=False):
    img = skel.astype(np.uint8)
    end_points = find_end_points(img)
    intersection_points = find_intersection_points_2(img)
    if only_points:
        end_points = [(p[1],p[0]) for p in end_points] 
        intersection_points = [(p[1],p[0]) for p in intersection_points]   
    return end_points,intersection_points

def preprocess_vascular(vascular):
    image = image.copy()
    kernel = np.ones((5,5), np.uint8)  # note this is a horizontal kernel
    image = cv2.dilate(image, kernel, iterations=2)
    image=cv2.erode(image,kernel,iterations=1)
    return image
 
def show_terminals(skel,e_points,i_points):
    skel_rgb = cv2.cvtColor(skel,cv2.COLOR_GRAY2RGB)
    for point in e_points:
        if len(point) == 3:
            x, y, point_type = point
            if point_type == 'Intersection':
                cv2.circle(skel_rgb, (x, y), 5, [255,0,0], 1)  # Draw Intersection Points
                # pass
            elif point_type == 'End':
                cv2.circle(skel_rgb, (x, y), 5, [0,255,0], 1)  # Draw End Points
        else:
            x,y = point
            cv2.circle(skel_rgb, (x, y), 5, [0,255,0], 1)
            
    for point in i_points:
        if len(point) == 3:
            x, y, point_type = point
            if point_type == 'Intersection':
                cv2.circle(skel_rgb, (x, y), 5, [255,0,0], 1)  # Draw Intersection Points
                # pass
            elif point_type == 'End':
                cv2.circle(skel_rgb, (x, y), 5, [0,255,0], 1)  # Draw End Points
        else:
            x,y = point
            cv2.circle(skel_rgb, (x, y), 5, [255,0,0], 1)    
    return skel_rgb               
if __name__ == "__main__":
    from skeleton import sk_skeletonize,filter_over
    from plantcv import plantcv as pcv
    # src = cv2.imread("../../MorphoSnake/img/2.png", cv2.IMREAD_GRAYSCALE)
    src = cv2.imread("../Results/M2/artery_vein/raw/1.png")
    cv2.imwrite('src.png',src)
    src_v = cv2.add(src[:,:,0],src[:,:,1])
    cv2.imwrite('v.png',src_v)
    src_a = cv2.add(src[:,:,2],src[:,:,1])
    cv2.imwrite('a.png',src_a)
    

    kernel = np.ones((5,5), np.uint8)  # note this is a horizontal kernel
    src_v = cv2.dilate(src_v, kernel, iterations=2)
    src_v=cv2.erode(src_v,kernel,iterations=1)
    cv2.imwrite('open_v.png',src_v)

    
    src_a = cv2.dilate(src_a, kernel, iterations=2)
    src_a=cv2.erode(src_a,kernel,iterations=1)
    cv2.imwrite('open_a.png',src_a) 
      
      
      
    print(np.max(src_v))
    
    if src is None:
        print("Failed to read the file!")
    else:
        # _, src = cv2.threshold(src, 128, 1, cv2.THRESH_BINARY)
        # dst = thin_image(src)
        dst = sk_skeletonize(src_v)
        # dst = src
        dst = filter_over(dst)### (0-255)
        pruned_dst,segmented_img, segment_objects = pcv.morphology.prune(skel_img=dst, size=15)
        cv2.imwrite('pruned_dst.png',pruned_dst)
        # dst = dst.astype(np.uint8)
        # h,w = dst.shape
        # skel_rgb = np.zeros((*dst.shape,3))
        # print(skel_rgb.shape)
        # skel_rgb[:,:,0] = dst.copy()
        skel_rgb = cv2.cvtColor(pruned_dst,cv2.COLOR_GRAY2RGB)
    
        # print(type(dst),dst.dtype,np.max(dst),np.min(dst))

        cv2.imwrite('dst.png',pruned_dst)
        # points = get_points_2(dst, 6, 9, 6)
        # points = getSkeletonIntersection(dst)

    # 0- Find end of lines
    pruned_dst = np.where(pruned_dst==0,0,1)
    input_image = pruned_dst.astype(np.uint8) # must be blaack and white thin network image
    e_points,i_points = get_terminals_hitmiss(input_image)

 
    # print(end_points_p)
    

    for point in e_points:
        x, y, point_type = point
        if point_type == 'Intersection':
            cv2.circle(skel_rgb, (x, y), 5, [255,0,0], 1)  # Draw Intersection Points
            # pass
        elif point_type == 'End':
            cv2.circle(skel_rgb, (x, y), 5, [0,255,0], 1)  # Draw End Points
    for point in i_points:
        x, y, point_type = point
        if point_type == 'Intersection':
            cv2.circle(skel_rgb, (x, y), 5, [255,0,0], 1)  # Draw Intersection Points
            # pass
        elif point_type == 'End':
            cv2.circle(skel_rgb, (x, y), 5, [0,255,0], 1)  # Draw End Points

    cv2.imwrite('p.png',skel_rgb)
    plt.imshow((skel_rgb).astype(np.uint8))
