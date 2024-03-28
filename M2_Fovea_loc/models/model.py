import matplotlib.pylab as plt
import torch
import torchvision.transforms.functional as TF
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tv_F
from PIL import Image,ImageDraw
import numpy as np
import cv2
import os

def rescale_label(a,b):
    div = [ai*bi for ai,bi in zip(a,b)]
    return div

def show_single_box(img,label,fovea_result_path,w_h=(50,50)): 

    scale_cx,scale_cy = label
    label=rescale_label(label,img.shape[1:])
    img=tv_F.to_pil_image(img) 
    w,h=w_h 
    cx,cy=label
    draw = ImageDraw.Draw(img)
    draw.rectangle(((cx-w/2, cy-h/2), (cx+w/2, cy+h/2)),outline="yellow",width=2)

    cv2.imwrite(fovea_result_path,np.asarray(img)[:,:,::-1])
    # print(cx,cy)
    
    # plt.imshow(np.asarray(img))
    # plt.show()
    return scale_cx.detach().cpu().numpy(),scale_cy.detach().cpu().numpy()

    
def predict_fovea_location(model,img_path,img_result_path,device):
    from PIL import Image
    img_name = os.path.split(img_path)[-1]
    img = Image.open(img_path) #PIL Image
    img_resized = img.resize((256,256))
    img_tensor = TF.to_tensor(img_resized)
    with torch.no_grad():
        # label_pred=model(img_tensor.unsqueeze(0).to(device))[0].cpu()
        #print(label_pred)

        ### YOLO
        
        results = model.predict(source=img_resized)

        # # print(results[0])
        for box in results[0].boxes:
            if box.cls == 2:
                label_pred = box.xywhn[0][0],box.xywhn[0][1]
                print(label_pred)
                break
    # plt.figure(figsize=(5,4),facecolor='lightgrey')
    # plt.title('Fovea Detector-'+img_path)
    # plt.axis('off')
    fovea_result_path = os.path.join(img_result_path,img_name)
    return show_single_box(img_tensor,label_pred,fovea_result_path)
    

class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()

        C_in,H_in,W_in=params["input_shape"]
        init_f=params["initial_filters"] 
        num_outputs=params["num_outputs"] 

        self.conv1 = nn.Conv2d(C_in, init_f, kernel_size=3,stride=2,padding=1)
        self.conv2 = nn.Conv2d(init_f+C_in, 2*init_f, kernel_size=3,stride=1,padding=1)
        self.conv3 = nn.Conv2d(3*init_f+C_in, 4*init_f, kernel_size=3,padding=1)
        self.conv4 = nn.Conv2d(7*init_f+C_in, 8*init_f, kernel_size=3,padding=1)
        self.conv5 = nn.Conv2d(15*init_f+C_in, 16*init_f, kernel_size=3,padding=1)
        self.fc1 = nn.Linear(16*init_f, num_outputs)

        
    def forward(self, x):
        identity=F.avg_pool2d(x,4,4)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = torch.cat((x, identity), dim=1)

        identity=F.avg_pool2d(x,2,2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = torch.cat((x, identity), dim=1)

        identity=F.avg_pool2d(x,2,2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = torch.cat((x, identity), dim=1)

        identity=F.avg_pool2d(x,2,2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        x = torch.cat((x, identity), dim=1)

        x = F.relu(self.conv5(x))

        x=F.adaptive_avg_pool2d(x,1)
        x = x.reshape(x.size(0), -1)

        x = self.fc1(x)
        return x
    
if __name__ == '__main__':
    params_model={
        "input_shape": (3,256,256),
        "initial_filters": 16, 
        "num_outputs": 2,
            }
    # create model
    model = Net(params_model)
    model.eval()
    
    # move model to cuda/gpu device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model=model.to(device)  
    
    path2weights="../weights/weights_smoothl1.pt"
    model.load_state_dict(torch.load(path2weights))
    
    predict_fovea_location(model,'8_left.jpg','./',device='cuda')