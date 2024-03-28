import cv2
import os 
import glob
from PIL import Image,ImageFilter #Import Pillow library to load the images
import numpy as np #Import numpy
import matplotlib.pyplot as plt #Import matplotlib library for visualization
from skimage.morphology import skeletonize,square,dilation #Import functions to compute morphological operations
import csv

import networkx as nx
from PVBM.helpers.main_branch import remove_optic,two_main_branch,remove_optic_cup
from PVBM.helpers.skeleton import sk_skeletonize,filter_over
from PVBM.helpers.terminals import get_terminals_hitmiss,show_terminals
from PVBM.helpers.vascular2graph import *
from plantcv import plantcv as pcv

# segmentation_path = "images/"+image1 #replace image1 by image2 if you want to use the second example
def measure_single_img(img_name):
    Fundus_img_pth = os.path.join('./Results/M0/images',img_name)
    AV_segmentation_path = image1 = os.path.join('./Results/M2/artery_vein/raw/',img_name)
    optic_mask_pth = os.path.join('./Results/M2/optic_disc_cup/raw/',img_name)
    AV_mask = cv2.imread(AV_segmentation_path)
    Fundus_img = cv2.imread(Fundus_img_pth)

    mask_V = cv2.add(AV_mask[:,:,0],AV_mask[:,:,1])
    mask_A = cv2.add(AV_mask[:,:,2],AV_mask[:,:,1])

    src_mask_V = mask_V.copy()
    src_mask_A = mask_A.copy()
    # AV_segmentation_path = image1 = '/home/zack/Desktop/workspace/vessel/RETA/RAVIR/train/training_masks/IR_Case_022.png'
    # AV_mask = cv2.imread(AV_segmentation_path,0)

    # mask_V = np.where(AV_mask==255,255,0).astype(np.uint8)
    # mask_A = np.where(AV_mask==128,255,0).astype(np.uint8)
    
    ########### fill the break line by dilate and erode 
    ##
    kernel = np.ones((5,5), np.uint8)  # note this is a horizontal kernel

    mask_V = cv2.dilate(mask_V, kernel, iterations=2)
    mask_V=cv2.erode(mask_V,kernel,iterations=1)
    # cv2.imwrite('open_v.png',mask_V)

    mask_A = cv2.dilate(mask_A, kernel, iterations=2)
    mask_A=cv2.erode(mask_A,kernel,iterations=1)
    # cv2.imwrite('open_v.png',mask_V)
    
    ## Skeleton and remove small branches
    # skel_A = filter_over(sk_skeletonize(mask_A))
    ### scikit skel can keep each point has tow neighbor at most.!!!
    skel_A = np.where(morphology.skeletonize(mask_A)==0,0,255).astype(np.uint8)
    skel_V = np.where(morphology.skeletonize(mask_V)==0,0,255).astype(np.uint8)

    skel_V,segmented_img, segment_objects = pcv.morphology.prune(skel_img=skel_V, size=15)
    skel_A,segmented_img, segment_objects = pcv.morphology.prune(skel_img=skel_A, size=15)
    #### remove optic in skeleton

    optic_mask = cv2.imread(optic_mask_pth,0)

    skel_A = remove_optic(skel_A,optic_mask)
    skel_V = remove_optic(skel_V,optic_mask)
    # skel_A = remove_optic_cup(skel_A,optic_mask)
    # skel_V = remove_optic_cup(skel_V,optic_mask)

    skel_A = two_main_branch(skel_A)
    skel_V = two_main_branch(skel_V)

    #### terminals points (end points, intersction points)
    skel_V_0 = skel_V[0]
    skel_V_1 = skel_V[1]
    skel_A_0 = skel_A[0]
    skel_A_1 = skel_A[1]

    skel_V = skel_V[0]+skel_V[1]
    skel_A = skel_A[0]+skel_A[1]

    b_skel_V = np.where(skel_V==0,0,1).astype(np.uint8)
    b_skel_A = np.where(skel_A==0,0,1).astype(np.uint8)

    b_skel_V_0 = np.where(skel_V_0==0,0,1).astype(np.uint8)
    b_skel_V_1 = np.where(skel_V_1==0,0,1).astype(np.uint8)
    b_skel_A_0 = np.where(skel_A_0==0,0,1).astype(np.uint8)
    b_skel_A_1 = np.where(skel_A_1==0,0,1).astype(np.uint8)


    # b_skel_V = b_skel_V.astype(np.uint8) # must be blaack and white thin network image
    # b_skel_A = b_skel_A.astype(np.uint8)

    e_points_V,i_points_V = get_terminals_hitmiss(b_skel_V)
    e_points_A,i_points_A = get_terminals_hitmiss(b_skel_A)

    e_points_V_0,i_points_V_0 = get_terminals_hitmiss(b_skel_V_0)
    e_points_V_1,i_points_V_1 = get_terminals_hitmiss(b_skel_V_1)
    e_points_A_0,i_points_A_0 = get_terminals_hitmiss(b_skel_A_0)
    e_points_A_1,i_points_A_1 = get_terminals_hitmiss(b_skel_A_1)
    
    # %xmode
    # %debug
    
    px_mm = 43.0
    root = None #(417,196)
    select = None

    from PVBM.helpers.main_branch import optic_center
    global optic_center_point,OD
    optic_center_ = optic_center(optic_mask)
    # optic_center_point,OD = optic_center(optic_mask)
    if optic_center_ is None:
        return None
    else:
        optic_center_point,OD = optic_center_
    # img = mh.imread(filename)


    # t_skel = skel_V_0

    skel_V_list = [skel_V_0,skel_V_1]
    e_points_V_list = [e_points_V_0,e_points_V_1]
    i_points_V_list = [i_points_V_0,i_points_V_1]
    img = src_mask_V.copy()
    G_V = []
    roots_V=[]
    for t_skel,t_e_points_V,t_i_points_V in zip(skel_V_list,e_points_V_list,i_points_V_list):
        contours, hierarchy = cv2.findContours(t_skel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        t_e_points = [(p[0],p[1]) for p in t_e_points_V]
        t_i_points = [(p[0],p[1]) for p in t_i_points_V]
        

        root = None
        if root == None:
            root = find_root_based_optic(t_e_points+t_i_points,optic_center_point)
        else:
            if root not in t_e_points+t_i_points:
                newroot = findClosest(t_e_points+t_i_points, root)
                print('Moving the root to a node in the tree. New root is', str(newroot), 'old root was', str(root), 'distance', dist(newroot, root))     
                root = newroot
        roots_V.append(root)       
        t_skel = np.ma.masked_where(t_skel==0, t_skel)

        superposed = False
        # contour = findContour(img)

        if(len(contours)>2):
            superposed = True
            print("image with loops detected, useless to use an older graph, constructing the graph")
            G = buildGraph(img, t_skel,root,t_e_points,t_i_points)
            if(len(contours)>2):
                img,t_skel,skel2,cutPoints = loopSolution(G,t_skel,img,contours,root)
                G = buildGraph(img, t_skel,root,t_e_points,t_i_points)
                # contour = findContour(img)
                markCutPoints(G,cutPoints,20)
        else:
            # # read in the graph from a previous run, if it exists
            # try:
            #     pkl_path = path_basename+'_graph.pkl'
            #     if os.path.exists(pkl_path):
            #         with open() as f:
            #             G = pickle.load(f)
            #         print('Loaded graph from ' + path_basename + '_graph.pkl')
            #         print('Graphs created by an older version of this software wont load, please delete ' + path_basename + '_graph.pkl file and retry in case of error.')
            #     else:
            #         G = buildGraph(img, t_skel,root,t_e_points,t_i_points)
            # except:
            #     # could not read the graph. Constructing it now
            #     G = buildGraph(img, t_skel,root,t_e_points,t_i_points)
            G = buildGraph(img, t_skel,root,t_e_points,t_i_points)

        width = t_skel.shape[1]
        height = t_skel.shape[0]

        # plt.axis((0,width,height,0))
        # plt.imshow(~img, cmap=leaf_colors, interpolation="nearest")
        # plt.imshow(t_skel, cmap=skel_colors,  interpolation="nearest")


        # handles to plot elements
        nodes = None
        edges = None
        node_labels = None
        edge_labels = None

        rad=[]
        # from IPython.core.debugger import set_trace

        # plot_graph(G,nodes,edges,node_labels,edge_labels,rad)
        # # set_trace()
        # a,b = report(G,root,px_mm,'basename')
        G_V.append(G)
        # print(a,b)

    skel_A_list = [skel_A_0,skel_A_1]
    e_points_A_list = [e_points_A_0,e_points_A_1]
    i_points_A_list = [i_points_A_0,i_points_A_1]
    img = src_mask_A.copy()
    G_A = []
    roots_A=[]
    for t_skel,t_e_points_A,t_i_points_A in zip(skel_A_list,e_points_A_list,i_points_A_list):
        contours, hierarchy = cv2.findContours(t_skel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        t_e_points = [(p[0],p[1]) for p in t_e_points_A]
        t_i_points = [(p[0],p[1]) for p in t_i_points_A]
        

        root = None
        if root == None:
            root = find_root_based_optic(t_e_points+t_i_points,optic_center_point)
        else:
            if root not in t_e_points+t_i_points:
                newroot = findClosest(t_e_points+t_i_points, root)
                print('Moving the root to a node in the tree. New root is', str(newroot), 'old root was', str(root), 'distance', dist(newroot, root))     
                root = newroot
        roots_A.append(root)       
        t_skel = np.ma.masked_where(t_skel==0, t_skel)

        superposed = False
        contour = findContour(img)

        if(len(contours)>2):
            superposed = True
            print("image with loops detected, useless to use an older graph, constructing the graph")
            G = buildGraph(img, t_skel,root,t_e_points,t_i_points)
            if(len(contours)>2):
                img,t_skel,skel2,cutPoints = loopSolution(G,t_skel,img,contours,root)
                # def loopSolution(G,skel,img):
                # print("Image with Loops detected, attempting automatic unLooping")
                # alpha = np.pi - findAngleMoy(G)
                # img,cutPoints = finalStrat(img,contours,alpha,d_alpha,root,d_pixels)
                
                G = buildGraph(img, t_skel,root,t_e_points,t_i_points)
                contour = findContour(img)
                markCutPoints(G,cutPoints,20)
        else:
            # read in the graph from a previous run, if it exists
            try:
                pkl_path = path_basename+'_graph.pkl'
                if os.path.exists(pkl_path):
                    with open() as f:
                        G = pickle.load(f)
                    print('Loaded graph from ' + path_basename + '_graph.pkl')
                    print('Graphs created by an older version of this software wont load, please delete ' + path_basename + '_graph.pkl file and retry in case of error.')
                else:
                    G = buildGraph(img, t_skel,root,t_e_points,t_i_points)
            except:
                # could not read the graph. Constructing it now
                G = buildGraph(img, t_skel,root,t_e_points,t_i_points)

        width = t_skel.shape[1]
        height = t_skel.shape[0]

        # plt.axis((0,width,height,0))
        # plt.imshow(~img, cmap=leaf_colors, interpolation="nearest")
        # plt.imshow(t_skel, cmap=skel_colors,  interpolation="nearest")


        # handles to plot elements
        nodes = None
        edges = None
        node_labels = None
        edge_labels = None

        rad=[]
        # from IPython.core.debugger import set_trace

        # plot_graph(G,nodes,edges,node_labels,edge_labels,rad)
        # # set_trace()
        # a,b = report(G,root,px_mm,'basename')
        G_A.append(G)
        # print(a,b)
        
    save_data(G_A,roots_A,img_name,'Artery.csv')
    save_data(G_V,roots_V,img_name,'Vein.csv')


def save_data(G_ts,roots,img_name,filename):
    # G_t = G.copy()
    last_trunk_nodes_Vein = []

    trunk_branch_node=[]
    trunk_branch_angle=[]
    trunk_branch_diameter=[]
    edge_branch_node=[]
    edge_branch_angle=[]
    edge_branch_diameter=[]

    for G_t,root in zip(G_ts,roots):
        nodes = nx.dfs_preorder_nodes(G_t, root)
        # edges = nx.dfs_edges(G_t,root)
        edges = G_t.edges()

        print('root',root)
        last_trunk_node = []
        truck_branch = []
        leaf_branch = []
        # for n in G:
        #     print('n',n)
        #     print(G[n])
        #     for n2 in G[n]:
        #         print('n2',n2)
        order_of_root = G_t.nodes[root]['Strahler']
        for p in nodes:
            # print(p)
            if G_t.nodes[p]['Strahler']==order_of_root:
                last_trunk_node.append(p)
        last_trunk_nodes_Vein.append(last_trunk_node[-1])
        # for e in edges: 
        #     if G_t.edges[e[0],e[1]]['Strahler']==1:
        #             # print(G_t.edges[e[0],e[1]].keys())
        #             ## 'theta', 'alpha', 'alpha_e'
        #             print((G_t.edges[e[0],e[1]]['alpha'],G_t.edges[e[0],e[1]]['alpha_e']))#*(180.0/np.pi))
        #             print((G_t.edges[e[0],e[1]]['alpha']+G_t.edges[e[0],e[1]]['alpha_e'])*(180.0/np.pi))#*(180.0/np.pi))
        #             # print(G_t.edges[e[0],e[1]]['theta'])

        nodes = nx.dfs_preorder_nodes(G_t, root) ## each time you use, need dfs
        order_of_root = G_t.nodes[root]['Strahler']
        print("The Strahler Order of Root: ",order_of_root)
        print('nodes:',nodes)
        for n in nodes: 
            if G_t.nodes[n]['Strahler'] == order_of_root:
                n_parent = G_t.nodes[n]['parent']
                num_branch = 0
                branch_angle = {}
                branch_diameter = {}
                branch_order = []
                for n_2 in G_t[n]: # neighbor
                    if n_2 == n_parent and n_2 is not None:
                        pp = G_t.nodes[n_2]['parent']
                        if pp is None:
                            continue
                        branch_diameter[(*n_2,'parent')] = G_t.edges[n_2,pp]['W_mean']
                    # elif n_2 != n_parent and n_2 is not None and G_t.nodes[n_2]['Strahler']==(order_of_root-1): ##BiA
                    #     num_branch+=1
                    #     branch_angle[n_2] = (180.0-G_t.edges[n,n_2]['alpha_e']*(180.0/np.pi))
                    #     branch_diameter[n_2]=G_t.edges[n,n_2]['W_mean']
                    elif n_2 != n_parent and n_2 is not None and G_t.nodes[n_2]['Strahler']==(order_of_root-1) or G_t.nodes[n_2]['Strahler']==(order_of_root): ## BA
                        num_branch+=1
                        branch_order.append(G_t.nodes[n_2]['Strahler'])
                        branch_angle[n_2] = (180.0-G_t.edges[n,n_2]['alpha_e']*(180.0/np.pi))
                        branch_diameter[n_2]=G_t.edges[n,n_2]['W_mean']  
                if num_branch == 2 and n != root and sum(branch_order)==2*order_of_root-1:
                    trunk_branch_node.append(n)
                    trunk_branch_angle.append(branch_angle)
                    trunk_branch_diameter.append(branch_diameter)                    
                # if num_branch == 2 and n != root: ### BiA
                #     trunk_branch_node.append(n)
                #     trunk_branch_angle.append(branch_angle)
                #     trunk_branch_diameter.append(branch_diameter)
                #     # print(G_t.nodes[n_2]['parent'])
                #     # if G_t.nodes[n_2]['Strahler'] == order_of_root:   
                #     # if G_t.edges[n,n_2]['Strahler'] 
            elif G_t.nodes[n]['Strahler'] == 2: ## should be 2 not order_of_root-1 ## order_of_root==4
                n_parent = G_t.nodes[n]['parent']
                num_branch = 0
                branch_angle = {}
                branch_diameter={}
                for n_2 in G_t[n]: # neighbor
                    if n_2 == n_parent and n_2 is not None:
                        pp = G_t.nodes[n_2]['parent']
                        if pp is None:
                            continue
                        branch_diameter[(*n_2,'parent')] = G_t.edges[n_2,pp]['W_mean']
                    elif n_2 != n_parent and n_2 is not None and G_t.nodes[n_2]['Strahler']==1:
                        num_branch+=1
                        branch_angle[n_2] = (180.0-G_t.edges[n,n_2]['alpha_e']*(180.0/np.pi))
                        branch_diameter[n_2]=G_t.edges[n,n_2]['W_mean']
                        ### W_mean
                        
                if num_branch == 2 and n != root:
                    edge_branch_node.append(n)
                    edge_branch_angle.append(branch_angle)
                    edge_branch_diameter.append(branch_diameter)
                    # print(G_t.nodes[n_2]['parent'])
                    # if G_t.nodes[n_2]['Strahler'] == order_of_root:   
                    # if G_t.edges[n,n_2]['Strahler']         
        # print('number of trunk and edge branch node: ',len(trunk_branch_node),len(edge_branch_node))  
        # print('all branch theta angle:',trunk_branch_angle) 
        # print('all edge theta angle:',edge_branch_angle)

        trunk_branch_angle_different_list=[]
        trunk_branching_coeff_list = []
        for ta,td in zip(trunk_branch_angle,trunk_branch_diameter):
            angle_list = list(ta.values())
            angle_different = np.fabs(angle_list[0]-angle_list[1]) 
            trunk_branch_angle_different_list.append(angle_different)
            
            d = 0.0
            d0=0
            for nkey in td.keys():
                if 'parent' in nkey:
                    d0 = td[nkey]*td[nkey]
                else:
                    d = d+td[nkey]*td[nkey]
            print('d,d0:',d,d0)
            if d0 == 0:
                trunk_branching_coeff_list.append(1.0)
            else:
                trunk_branching_coeff_list.append(d/d0)
            # print(angle_different)
        if len(trunk_branch_angle_different_list)==0:
            mean_trunk_branch_angle_different = -1.0
        else:
            mean_trunk_branch_angle_different = sum(trunk_branch_angle_different_list)/len(trunk_branch_angle_different_list)
        if len(trunk_branching_coeff_list) == 0:
            mean_trunk_branching_coeff =-1.0
        else:
            mean_trunk_branching_coeff = sum(trunk_branching_coeff_list)/len(trunk_branching_coeff_list)
        print('Number of trunk branch,Average angle different of trunk branch: ',len(trunk_branch_angle_different_list),mean_trunk_branch_angle_different)
        print('Average trunk branch coefficient: ',len(trunk_branching_coeff_list),mean_trunk_branching_coeff)

        edge_branch_angle_different_list=[]
        edge_branching_coeff_list = []
        for ta,td in zip(edge_branch_angle,edge_branch_diameter):
            # for bkey in ta.keys():
                # x = [bkey[0],bf[0]]
                # y = [bkey[1],bf[1]]
            angle_list = list(ta.values())
            angle_different = np.fabs(angle_list[0]-angle_list[1]) 
            edge_branch_angle_different_list.append(angle_different)
            
            d = 0.0
            d0=0
            for nkey in td.keys():
                if 'parent' in nkey:
                    d0 = td[nkey]*td[nkey]
                else:
                    d = d+td[nkey]*td[nkey]
            if d0==0:
                edge_branching_coeff_list.append(1.0)
            else:
                edge_branching_coeff_list.append(d/d0)
            # print(angle_different)
        if len(edge_branch_angle_different_list) == 0:
            mean_edge_branch_angle_different = -1.0
        else:
            mean_edge_branch_angle_different = sum(edge_branch_angle_different_list)/len(edge_branch_angle_different_list)
        if len(edge_branching_coeff_list) == 0:
            mean_edge_branching_coeff = -1.0
        else:
            mean_edge_branching_coeff = sum(edge_branching_coeff_list)/len(edge_branching_coeff_list)
        print('Number fo edge branch,Average angle different of edge branch: ',len(edge_branch_angle_different_list),mean_edge_branch_angle_different)
        print('Average edge branch coefficient: ',len(edge_branching_coeff_list),mean_edge_branching_coeff)
    
    if len(last_trunk_nodes_Vein) == 2:
        main_vessel_angle = angle(optic_center_point,last_trunk_nodes_Vein[0],last_trunk_nodes_Vein[1])*(180.0/np.pi)
    else:
        main_vessel_angle = -1.0
    print("Main angle of vessel tree:",main_vessel_angle)          
    vessel_param = [img_name,main_vessel_angle,mean_trunk_branch_angle_different,mean_trunk_branching_coeff,mean_edge_branch_angle_different,mean_edge_branching_coeff]
    writefile(filename,vessel_param)    
          
def writefile(filename,data):
    # 使用 Python open 方法打开文件进行写入，
    csv_header = ['ID','MainAngle','MeanTrunkBAngle','MeanTrunkBCoef','MeanEdgeBAngle','MeanEdgeBcoef']
    filexists =  os.path.exists(filename)
    with open(filename, mode='a') as f:
        
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting =csv.QUOTE_MINIMAL)
        # 用 writerow 方法写入表头行
        if not filexists:
            writer.writerow(csv_header)
        # 用 writerow 方法写入数据行
        # for data in datalist:
        
        data = [data[0]]+[round(d,3) for d in data[1:]]
        writer.writerow(data)



if __name__ == '__main__':
    img_root = './Results/M1/Good_quality'
    img_name_list = glob.glob(os.path.join(img_root,'*.png'))
    img_name_list = [os.path.split(img_name)[-1] for img_name in img_name_list]
    img_name_list = sorted(img_name_list)
    print(img_name_list)
    
    resume_measure = True
    img_names_in_csv=[]
    if not resume_measure:
        import pandas as pd
        if os.path.exists('Artery.csv'):
            os.remove('Artery.csv')
        if os.path.exists('Vein.csv'):
            os.remove('Vein.csv') 
        img_names_in_csv = pd.read_csv('Artery.csv')['ID'].tolist()

        # # 提取第一列（假设第一列的列名为 'ID'）
        # id_list = df['ID'].tolist()
    for img_id,img_name in enumerate(img_name_list):
        print('the img number is: ', img_id,' img name is : ',img_name)
        try:
            if resume_measure:
                if img_name not in img_names_in_csv:
                    measure_single_img(img_name)
                
            else:
                
                # if(img_name == '0003441522_20200820_080825_Color_L_001.png'):
                measure_single_img(img_name)
                # pass
        except:
            continue


