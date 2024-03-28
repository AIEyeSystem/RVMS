import numpy as np
from scipy.signal import convolve2d

from queue import PriorityQueue
from PVBM.helpers.terminals import get_terminals_hitmiss

# def iterative(i_or, j_or, skeleton, all_important_index, visited, connected):
#   pq = PriorityQueue()
#   pq.put((0, i_or, j_or, i_or, j_or,0))
#   priotities = [0,1,2,3,4,5,6,7]
#   distances = [1, 1, 1, 1, 2**0.5, 2**0.5, 2**0.5, 2**0.5]
#   while not pq.empty():
#     _, i_or, j_or, i, j,d = pq.get()
#     directions = [(i-1,j),(i+1,j),(i,j-1),(i,j+1),(i-1,j-1),(i-1,j+1),(i+1,j-1),(i+1,j+1)]
#     for direction, distance,priority in zip(directions,distances,priotities):
#       x,y = direction
#       if x >= 0 and x < skeleton.shape[0] and y >= 0 and y < skeleton.shape[1] and visited[direction] == 0:
#         if direction not in all_important_index:
#           visited[direction] = 1
#         if skeleton[x][y] == 1:
#           point = direction
#           if direction in all_important_index:
#             connected[(i_or,j_or)] = connected.get((i_or,j_or),[]) + [(direction,d + distance)]
            
#           else : 
#             pq.put((priority, i_or, j_or, x, y,d + distance ))

   
# def connected_pixels(skeleton, all_important_index):
#     connected = {}
#     visited = np.zeros_like(skeleton)
#     for i, j in all_important_index:
#         if skeleton[i][j] == 1 and not visited[i][j]:
#             # print(skeleton[i][j])
#             iterative(i, j, skeleton, all_important_index,visited,connected)
#             #recursive(i, j, skeleton, i, j, visited, all_important_index, connected)
#     return connected

def iterative(i_or, j_or, skeleton, visited, connected):
    pq = PriorityQueue()
    pq.put((0, i_or, j_or))
    priorities = [0, 1, 2, 3, 4, 5, 6, 7]
    distances = [1, 1, 1, 1, 2**0.5, 2**0.5, 2**0.5, 2**0.5]
    
    while not pq.empty():
        _, i, j = pq.get()
        visited[i][j] = True
        directions = [(i-1, j), (i+1, j), (i, j-1), (i, j+1), (i-1, j-1), (i-1, j+1), (i+1, j-1), (i+1, j+1)]
        
        for direction, distance, priority in zip(directions, distances, priorities):
            x, y = direction
            if 0 <= x < skeleton.shape[0] and 0 <= y < skeleton.shape[1] and not visited[x][y]:
                visited[x][y] = True
                if skeleton[x][y] == 1:
                    point = direction
                    connected[(i_or, j_or)].add((point, distance))
                    # print(connected)
                    pq.put((priority, x, y))

def connected_pixels(skeleton, important_points):
    connected = {}
    visited = np.zeros_like(skeleton, dtype=bool)
    
    for i, j in important_points:
        if skeleton[i][j] == 1 and not visited[i][j]:
            connected[(i, j)] = set()  # Initialize a set to store connected points
            iterative(i, j, skeleton, visited, connected)
    
    return connected



from scipy.signal import convolve2d
filter_ = np.ones((3,3))
filter_[1,1] = 10
filter_


def point2img(points,img):
    for p in points:
        img[p[0],p[1]] = 1
    return img

def compute_tortuosity(skeleton):
    # tmp = convolve2d(skeleton, filter_, mode="same")
    # endpoints = tmp == 11
    # intersection = tmp >= 13
    # particular = endpoints + intersection
    # origin_points = [(i, j) for i in range(particular.shape[0]) for j in range(particular.shape[1]) if particular[i, j]]
    
    # print(origin_points)
    # [(8, 818), (9, 829), (13, 821), (13, 822), (13, 823), (14, 822), (63, 1011), (150, 1157), (191, 1094), (191, 1095), (192, 1094), (206, 259), (260, 169), (263, 669), (291, 275), (292, 274), (292, 275), (341, 1332), (345, 1011), (346, 1010), (346, 1011), (346, 1012), (459, 840), (533, 27), (616, 783), (617, 782), (617, 783), (617, 784), (618, 782), (679, 646), (679, 647), (679, 648), (680, 647), (689, 1200), (689, 1201), (690, 1201), (690, 1202), (691, 1201), (711, 652), (712, 651), (712, 652), (712, 653), (762, 1227), (782, 725), (783, 724), (783, 725), (784, 725), (837, 726), (844, 734), (845, 734), (845, 735), (846, 734), (861, 704), (861, 705), (861, 706), (862, 705), (890, 1419), (952, 976), (952, 977), (953, 977), (984, 50), (1113, 1325), (1307, 306), (1428, 854)]
    
    # [(8, 818), (9, 829), (13, 822), (63, 1011), (150, 1157), (191, 1094), (206, 259), (260, 169), (263, 669), (292, 275), (341, 1332), (346, 1011), (459, 840), (533, 27), (617, 783), (679, 647), (689, 1201), (711, 652), (762, 1227), (783, 725), (837, 726), (845, 734), (861, 705), (890, 1419), (952, 977), (984, 50), (1113, 1325), (1307, 306), (1428, 854)]
    end_points,inter_points = get_terminals_hitmiss(skeleton,only_points=1)
    eimg = np.zeros_like(skeleton)
    iimg = np.zeros_like(skeleton)
    for p in end_points:
        eimg[p[0],p[1]] = 1
    for p in inter_points:
        iimg[p[0],p[1]] = 1 
    
    particular = eimg+iimg
    # pimg = point2img(origin_points,np.zeros_like(skeleton))
    origin_points = [(i, j) for i in range(skeleton.shape[0]) for j in range(skeleton.shape[1]) if particular[i, j]]
    # print(origin_points)
    # origin_points = sorted(origin_points, key=lambda x: (x[0],x[1]), reverse=False)
    # for p in origin_points:
    #     print(skeleton[p[0],p[1]],' : ',skeleton[p[1],p[0]])
    
    connection_dico = connected_pixels(skeleton, origin_points)
    # print(connection_dico)
    tor = []
    chord = []
    arc = []
    for key, value in connection_dico.items():
        x, y = key
        for p, d in value:
            if d > 10:
                arc.append(d)
                x2, y2 = p
                real_d = ((x - x2) ** 2 + (y - y2) ** 2) ** 0.5
                tor.append(d/real_d)
                chord.append(real_d)
    return np.median(tor), np.sum(chord), arc, chord, connection_dico


    
