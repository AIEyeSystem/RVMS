import torch
import os
import shutil
import cv2
import pandas as pd
from models.model import Net,predict_fovea_location,show_single_box

def fovea_centre(img_result_path,loc_result_path,fundus_path,device):
    if os.path.exists(img_result_path+'.ipynb_checkpoints'):
        shutil.rmtree(img_result_path+'.ipynb_checkpoints')   
    if os.path.exists(loc_result_path+'.ipynb_checkpoints'):
        shutil.rmtree(loc_result_path+'.ipynb_checkpoints') 
    
    if not os.path.exists(img_result_path):
        os.makedirs(img_result_path)
    if not os.path.exists(loc_result_path):
        os.makedirs(loc_result_path)

    ########### update something for resolution scale important!!!
    
    fundus_imgs_list = sorted(os.listdir(fundus_path))

      
    
    fovea_locx_list = []
    fovea_locy_list = []
    fovea_name_list = []
    for img_name in fundus_imgs_list:
        img_path = fundus_path+img_name
        # img = cv2.imread(img_path)
        cx,cy = predict_fovea_location(model,img_path,img_result_path,device)
        fovea_locx_list.append(cx)
        fovea_locy_list.append(cy)
        fovea_name_list.append(img_name)
        
    Pd_macular_centre = pd.DataFrame({'Name':fovea_name_list, 'cx':fovea_locx_list, 'cy':fovea_locy_list})
    Pd_macular_centre.to_csv(loc_result_path + 'Macular_loc.csv', index = None, encoding='utf8')  
    return cx,cy
if __name__ == '__main__':
    params_model={
        "input_shape": (3,512,512),
        "initial_filters": 16, 
        "num_outputs": 2,
            }
    # # create model
    # model = Net(params_model)
    # model.eval()
    
    # # move model to cuda/gpu device

    # path2weights="./weights/weights_smoothl1.pt"
    # model.load_state_dict(torch.load(path2weights))
    
    ### Yolo model
    from ultralytics import YOLO
    import cv2 
    # Load a model
    model = YOLO('weights/best.pt')  # load a pretrained model (recommended for training)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model=model.to(device)  
    
    # ## infer
    # img = cv2.imread('8_left.jpg')
    from PIL import Image
    import torchvision.transforms.functional as TF
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms.functional as tv_F
    img = Image.open('8_left.jpg') #PIL Image
    img_resized = img.resize((256,256))
    img_tensor = TF.to_tensor(img_resized)

    with torch.no_grad():
        results = model.predict(source=img_resized)

        # print(results[0])
        for box in results[0].boxes:
            if box.cls == 2:
                cx,cy = box.xywhn[0][0],box.xywhn[0][1]
                break
    show_single_box(img_tensor,(cx,cy),'./r.png')
