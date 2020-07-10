import os.path as osp
import argparse
import numpy as np 

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa

from torch.nn import Parameter

from sklearn.utils import class_weight


from sklearn.metrics import confusion_matrix


#dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
#data = dataset[0]

data_path = "../../data/processed/" + "data2.dataset"
data = torch.load(data_path)

print(data.y[data.train_mask])
print(data.y.unique())

print(data.y.shape)


print(data.y[data.train_mask].shape)


print(data.y.unique())
print(data.y)




parser = argparse.ArgumentParser()

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')\

#early stopping criteria
parser.add_argument('--early_stopping', type=int, default = 50,
                    help = 'Stopping criteria for validation')



parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
args = parser.parse_args()



print(f'number of nodes: {data.x.shape}')


if args.use_gdc:
    gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                normalization_out='col',
                diffusion_kwargs=dict(method='ppr', alpha=0.05),
                sparsification_kwargs=dict(method='topk', k=128,
                                           dim=0), exact=True)
    data = gdc(data)



#cached = True is for transductive learning
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = GCNConv(data.x.shape[1], 16, cached=True, \
        #      normalize=not args.use_gdc)
        # self.conv2 = GCNConv(16, 32, cached=True,\
        #      normalize=not args.use_gdc)        
        # self.conv3 = GCNConv(32, 64, cached=True, \
        #      normalize=not args.use_gdc)        
        # self.conv4 = GCNConv(64, 6, cached=True, \
        #      normalize=not args.use_gdc)       

        
        self.conv1 = ChebConv(data.x.shape[1], 16, K=3)
        self.conv2 = ChebConv(16, 32, K=3)
        self.conv3 = ChebConv(32, 64, K=3)
        self.conv4 = ChebConv(64, 6, K=3)
    
    

        self.reg_params = self.conv1.parameters()
        
        self.non_reg_params = self.conv4.parameters()

    def forward(self):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        
        x = F.relu(self.conv1(x, edge_index))
       
        x = F.relu(self.conv2(x, edge_index))
      
        x = F.relu(self.conv3(x, edge_index))
        
        #x = F.relu(self.conv4(x, edge_index))
        
        x = self.conv4(x, edge_index)

        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)


optimizer = torch.optim.Adam([
    dict(params=model.reg_params, weight_decay=5e-4),
    dict(params=model.non_reg_params, weight_decay=0)
], lr=0.01)



def train():
    model.train()
    optimizer.zero_grad()
    # print(model()[data.val_mask])
    # print(data.y[data.train_mask])

    r"""class weights for imbalanced data"""
    class_weights = class_weight.compute_class_weight('balanced',
                                                 data.y.unique().numpy() ,
                                                 data.y.numpy() )


    #FIX WEIGHTS
    weights = torch.FloatTensor(class_weights)
    
    #print(weights)
    #weights = torch.tensor([3.3419, 9.3718, 8.9813, 8.9526, 0.1905,8.1577])
    
    loss = F.nll_loss(model()[data.train_mask], data.y[data.train_mask]-1, weight=weights)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test():
    model.eval()
    
    #F.nll_loss(model()[data.val_mask], data.y[data.val_mask])
    logits, accs = model(), []
    for mask_name, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]


        acc = pred.eq(data.y[mask]-1).sum().item() / mask.sum().item()  


        print(mask_name)
        print(pred)
        print(f'unique predictions: {pred.unique()}')
        print(data.y[mask]-1)
        print('correct predicted :  ', pred.eq(data.y[mask]-1).sum().item())
        print('total items: ' , mask.sum().item())

        print(f'{mask_name} accuracy is {acc}')


        # print(f'{mask}', pred.eq(data.y[mask]).sum().item() )
        # print( mask.sum().item()   )

        r"printing predicted classes and number of elements for preds"
        print('pred')
        unique_elements, counts_elements = np.unique(pred, return_counts=True)
        print(np.asarray((unique_elements, counts_elements)))


        r"printing predicted classes and number of elements for actual"
        print('actual')
        unique_elements, counts_elements = np.unique((data.y[mask]-1), return_counts=True)
        print(np.asarray((unique_elements, counts_elements)))

        nb_classes = 6


        conf_mat=confusion_matrix((data.y[mask]-1).numpy(), pred.numpy())
      

    # Per-class accuracy
        class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)
        print(class_accuracy)

       # print(acc)
        accs.append(acc)

    return accs


if __name__ == '__main__':
#stopping criteria 
    counter = 0


    for epoch in range(1, args.epochs):
        

        loss = train()
        train_acc, val_acc, test_acc = test()

        with torch.no_grad():
            #print(model()[data.val_mask])
            
            loss_val = F.nll_loss(model()[data.val_mask], data.y[data.val_mask]-1)
            #print(model()[data.val_mask])
        
    
        
        log = 'Epoch: {:03d}, train_loss:{:.4f}, val_loss:{:.4f}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch,loss,loss_val, train_acc, val_acc, test_acc))


        #for first epoch
        if epoch == 1:
            largest_val_loss = loss_val

        #early stopping if the loss val does not improve/decrease for a number of epochs
        if loss_val >= largest_val_loss:
            counter += 1 
            best_val_loss = loss_val
            if counter >= args.early_stopping:
                print(f'EarlyStopping counter: validation loss did not increase for {args.early_stopping}!!')
                break