import cv2
import numpy as np
import os

def two_main_branch(skel_no_optic):
    # Load the binary vascular mask image (single-channel)
    vascular_mask = skel_no_optic.copy()###cv2.imread('img/Vein.png', cv2.IMREAD_GRAYSCALE)

    # Find connected components using OpenCV's connectedComponentsWithStats function
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(vascular_mask, connectivity=8)

    # Create a blank canvas for visualization
    canvas = np.zeros_like(vascular_mask)

    # Sort the connected components by area in descending order
    sorted_indices = np.argsort(-stats[:, cv2.CC_STAT_AREA])
    num_trees_to_keep = min(2, num_labels - 1)  # Limit to the number of components
    trees_to_keep = sorted_indices[1:num_trees_to_keep+1]

    branchs = []

    # Iterate through the connected components and show them one by one
    for label in trees_to_keep:
        # Create a mask for the current connected component
        component_mask = (labels == label).astype(np.uint8)

        # Set the canvas to the current component mask
        canvas = component_mask * 255
        branchs.append(canvas)
        # # Show the current connected component
        # cv2.imshow(f'Connected Component {label}', canvas)

        # # Wait for a key press to move to the next component
        # cv2.waitKey(0)
        # cv2.destroyWindow(f'Connected Component {label}')
        # # cv2.imwrite(os.path.join('img',str(label)+'.png'),canvas)

    # Close all windows when done
    # cv2.destroyAllWindows()

    # Display the total number of connected trees
    print(f'Total connected trees: {num_labels - 1}')
    return branchs

def remove_optic_cup(skel,optic_mask):
    center_points,OD = optic_center(optic_mask)
    OD_mask = np.zeros_like(optic_mask)
    # Fundus_Area_Mask = np.where(Fundus_img==0,0,1)


    # 画一个环

    cv2.circle(OD_mask, center_points, int(OD*0.2), 1, thickness=-1)  # wai圈
    # cv2.circle(OD_mask, optic_center_point, int(OD), (0,0,0), thickness=-1)  # nei圈

    skel_no_optic = np.where(OD_mask==0,skel,0)
    
    # vascular_mask = skel.copy()
    # optic_binary_inv = np.where(optic_mask==0,1,0)
    # skel_no_optic = optic_binary_inv*vascular_mask
    # skel_no_optic = skel_no_optic.astype(np.uint8)
    
    return skel_no_optic
def remove_optic(skel,optic_mask):
    vascular_mask = skel.copy()
    optic_binary_inv = np.where(optic_mask==0,1,0)
    skel_no_optic = optic_binary_inv*vascular_mask
    skel_no_optic = skel_no_optic.astype(np.uint8)
    
    return skel_no_optic

def optic_center(optic_mask):

    # 读取二值图像，其中圆形的值为0，其他位置的值为1
    binary_image = np.where(optic_mask==0,0,255).astype(np.uint8)

    # 查找图像中的轮廓
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 如果找到轮廓
    if contours:
        # 计算每个轮廓的边界矩形，并找到最大的边界矩形
        max_area = -1
        largest_rect = None

        for contour in contours:
            rect = cv2.boundingRect(contour)
            x, y, w, h = rect
            area = w * h

            if area > max_area:
                max_area = area
                largest_rect = rect

        if largest_rect is not None:
            # 计算最大边界矩形的中心坐标并绘制矩形
            x, y, w, h = largest_rect
            center_x = x + w // 2
            center_y = y + h // 2
            OD = np.max((w,h))
            return (center_x,center_y),OD
            # 在图像上标记矩形的中心
            # cv2.circle(binary_image, (center_x, center_y), 2, (0, 0, 255), 3)  # 用红色标记矩形中心
            # cv2.rectangle(binary_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 用绿色绘制边界矩形

            # # 显示带有矩形中心标记的图像
            # cv2.imshow('Largest Rectangle', binary_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # # 打印最大矩形的中心坐标
            # print(f'Largest Rectangle Center Coordinates: ({center_x}, {center_y})')
        else:
            print('No rectangles found in the image.')
            return None
    else:
        print('No contours found in the image.')
        return None
def mask_circle(image, center, radius):
    # Create a mask with the same dimensions as the image, initialized to zero
    mask = np.zeros_like(image)
    
    # Draw a filled circle on the mask with the given center and radius
    cv2.circle(mask, center, radius, (1,1,1), -1)
    
    # Apply the mask to the image
    masked_image = cv2.bitwise_and(image, mask)
    
    return masked_image

if __name__ == '__main__':
    
    optic_mask_pth = '../Results/M2/optic_disc_cup/raw/1.png'
    skel_pth = 'pruned_dst.png'
    
    optic_mask = cv2.imread(optic_mask_pth,0)
    skel_img = cv2.imread(skel_pth,0)
    
    vascular_no_optic = remove_optic(skel_img,optic_mask)

    two_main_branch(vascular_no_optic)
    cv2.imshow(f'optic', vascular_no_optic)

    # Wait for a key press to move to the next component
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite(os.path.join('img',str(label)+'.png'),canvas)    
    


    # 读取二值图像，其中圆形的值为0，其他位置的值为1
    optic_center(optic_mask)


