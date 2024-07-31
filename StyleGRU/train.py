import argparse

import os
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F

from model.net import get_model
from dataloader.base_dset import FF_Dataset_con
from dataloader.triplet_clip_loader import get_loader



def save_checkpoint(state, file_name):
    torch.save(state, file_name)


def calculate_accuracy(predict, target):
    return torch.sum(predict == target).float().mean().item()


def train(data, model, criterion_list, optimizer, epoch, mode='cls_tri'):
    print("******** Training ********")
    print(f"mode: {mode}")
    print(f"con version: {args.con}")
    total_loss = 0.0
    acc_sum, samples_sum = 0.0, 0
    model.train()
    for batch_idx, clip_triplet in enumerate(data):
        anchor_clip, pos_clip, neg_clip, labels = clip_triplet
        anchor_clip, pos_clip, neg_clip = anchor_clip.to(torch.float32), pos_clip.to(torch.float32), neg_clip.to(torch.float32)
        anchor_clip, pos_clip, neg_clip, = anchor_clip.to(device), pos_clip.to(device), neg_clip.to(device)
        x1, x2, x3, E1, E2, E3 = model(anchor_clip, pos_clip, neg_clip)
        
        outputs = torch.stack([x1, x2, x3], dim=0).squeeze(2)
        target_clf = torch.stack(labels, dim=0).to(device)
        
        
        loss = 0.0
        ##############################
        ### 1. classification loss ###
        ##############################
        loss_clf = criterion_list[0](outputs.view(-1), target_clf.view(-1).float())
        loss += loss_clf
        total_loss += loss_clf
        
        #######################
        ### 2. triplet loss ###
        #######################
        dist_E1_E2 = F.pairwise_distance(E1, E2, 2) # ||T(a) - T(p)||
        dist_E1_E3 = F.pairwise_distance(E1, E3, 2) # ||T(a) - T(n)||
        
        target_tri = torch.FloatTensor(dist_E1_E2.size()).fill_(-1)
        target_tri = target_tri.to(device)
        loss_tri = criterion_list[1](dist_E1_E2, dist_E1_E3, target_tri)
        loss += loss_tri
        total_loss += loss_tri
      
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Generating Log
        preds = (outputs > 0.500).to(torch.float32)
        acc_sum += calculate_accuracy(preds, target_clf)
        samples_sum += torch.numel(preds)
        log_step = args.train_log_step
        
        if (batch_idx % log_step == 0) and (batch_idx != 0):
            print('Train Epoch: {} [{}/{}]  Loss: {:.4f}  Acc: {:.4f}'.format(epoch,
                                                                              batch_idx,
                                                                              len(data),
                                                                              loss / log_step,
                                                                              acc_sum / samples_sum))
    print()
            
    print("****************")


def validation(data, model, criterion_list):
    print("******** Validation ********")
    with torch.no_grad():
        model.eval()
        
        total_loss = 0
        acc_sum, samples_sum = 0.0, 0
        for batch_idx, clip_triplet in enumerate(data):
            anchor_clip, pos_clip, neg_clip, labels = clip_triplet
            anchor_clip, pos_clip, neg_clip = anchor_clip.to(torch.float32), pos_clip.to(torch.float32), neg_clip.to(torch.float32)
            anchor_clip, pos_clip, neg_clip, = anchor_clip.to(device), pos_clip.to(device), neg_clip.to(device)
            x1, x2, x3, E1, E2, E3 = model(anchor_clip, pos_clip, neg_clip)
            
            
            outputs = torch.stack([x1, x2, x3], dim=0).squeeze(2)
            target_clf = torch.stack(labels, dim=0).to(device)
            
            preds = (outputs > 0.500).to(torch.float32)
            
            acc_sum += calculate_accuracy(preds, target_clf)
            samples_sum += torch.numel(preds)
        
        accuracy = acc_sum / samples_sum
        print(f"Evaluation - ACC: {accuracy:.4}")
        print("****************")
        return accuracy
        

def main():
    torch.manual_seed(1)
    if args.cuda:
        torch.cuda.manual_seed(1)
    cudnn.benchmark = True
    
    exp_dir = os.path.join(args.result_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Bulid Model
    model = get_model(args, device)
    if model is None: 
        return
    
    params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            params += [{'params': [value]}]

    criterion_clf = torch.nn.BCEWithLogitsLoss()
    criterion_tri = torch.nn.MarginRankingLoss(margin=args.margin)
    criterion_ap = torch.nn.MSELoss()
    
    criterion_list = [criterion_clf, criterion_tri, criterion_ap]
    
    optimizer = optim.Adam(params, lr=args.lr)
    
    # Train Test Loop
    with torch.autograd.set_detect_anomaly(True):
        best_val, best_test = None, None
        best_val_epoch, best_test_epoch = None, None
        for epoch in range(1, args.epochs + 1):
   
            train_data_loader = get_loader(FF_Dataset_con(split='train', dtype=args.dtype), args)
            val_data_loader = get_loader(FF_Dataset_con(split='val', dtype=args.dtype), args)

            
            train(train_data_loader, model, criterion_list, optimizer, epoch, mode=args.mode)
            
            ### Check the parameter update ###
            
            val_acc = validation(val_data_loader, model, criterion_list)
            
            if epoch == 1:
                best_val = val_acc,
                best_val_epoch = epoch
       
            
            model_to_save = {
                "epoch": epoch + 1,
                'state_dict': model.state_dict(),
            }
       
            file_name = os.path.join(exp_dir, "checkpoint_" + str(epoch))
            save_checkpoint(model_to_save, file_name)
            
            if best_val < val_acc:
                best_val = val_acc
                best_val_epoch = epoch
                print(f"best val accuracy: {best_val:.4f}, saved!!!")
            else:
                print(f"best val accuracy: {best_val:.4f}, epoch: {best_val_epoch}")
                


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Siamese Example')
    parser.add_argument('--result_dir', default='data', type=str,
                        help='Directory to store results')
    parser.add_argument('--exp_name', default='exp0', type=str,
                        help='name of experiment')
    parser.add_argument('--cuda', action='store_true', default=True,            
                        help='enables CUDA training')
    parser.add_argument("--gpu_devices", type=int, nargs='+', default=[0],      
                        help="List of GPU Devices to train on")
    parser.add_argument('--epochs', type=int, default=100, metavar='N',          
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--ckp_freq', type=int, default=1, metavar='N',
                        help='Checkpoint Frequency (default: 1)')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--margin', type=float, default=1.0, metavar='M',
                        help='margin for triplet loss (default: 1.0)')
    parser.add_argument('--ckp', default=None, type=str,
                        help='path to load checkpoint')
    parser.add_argument('--num_train_samples', type=int, default=50000, metavar='M',
                        help='number of training samples (default: 50000)')
    parser.add_argument('--num_test_samples', type=int, default=10000, metavar='M',
                        help='number of test samples (default: 10000)')
    parser.add_argument('--train_log_step', type=int, default=100, metavar='M',
                        help='Number of iterations after which to log the loss')
    
    parser.add_argument('--add_weigths', type=str, default='./weights')
    parser.add_argument('--weights_name', type=str, default='g_temp.pth')
    parser.add_argument('--dtype', type=str, default='total',
                        choices=['total', 'coarse', 'middle', 'fine', 'middle+', 'coarse+'],
                        help='used latent level for training')
    parser.add_argument('--mode', type=str, default='cls_tri', 
                        help='training mode (cls_tri: classification loss + triplet loss)')
    parser.add_argument('--con', type=str, default='sup', 
                        help='contrastive learning mode (sup: supervised contrastive)')

    
    global args, device
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    if args.cuda:
        device = 'cuda'
        if args.gpu_devices is None:
            args.gpu_devices = [0]
    else:
        device = 'cpu'
    main()