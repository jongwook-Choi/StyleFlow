import os
import torch
import torch.nn as nn
from model.StyleGRU import StyleGRU




class TripletNet(nn.Module):
    def __init__(self, model):
        super(TripletNet, self).__init__()
        self.model = model
    
    def forward(self, c1, c2, c3):
        x1, E1 = self.model(c1)
        x2, E2 = self.model(c2)
        x3, E3 = self.model(c3)
        return x1, x2, x3, E1, E2, E3

def get_model(args, device):
    if args.dtype == 'total':
        input_size = 18*512
    elif args.dtype == 'coarse':
        input_size = 3*512
    elif args.dtype == 'middle':
        input_size = 4*512
    elif args.dtype == 'fine':
        input_size = 11*512
    elif args.dtype == 'middle+':
        input_size = 15*512
    elif args.dtype == 'coarse+':
        input_size = 7*512
    else:
        raise Exception('invalid dtype!')
    
    add_weights = args.add_weigths
    weights_name = args.weights_name
        
    g = StyleGRU(feature_size=input_size)
    
    model = TripletNet(g)
    model = nn.DataParallel(model, device_ids=args.gpu_devices)
    
    model = model.to(device)
    
    if args.ckp:
        ckp_path = os.path.join(add_weights, weights_name)
        if os.path.isfile(ckp_path):
            g.load_state_dict(torch.load(ckp_path))
        else:
            print(f"=> No checkpoint found at '{args.ckp}'")
            
    return model
