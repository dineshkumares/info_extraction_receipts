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


data_path = "../../data/processed/" + "data2.dataset"

data = torch.load(data_path)

print(f'training nodes: {data.y[data.train_mask].shape}')
print(f'validation nodes: {data.y[data.val_mask].shape}')
print(f'testing nodes: {data.y[data.test_mask].shape}')

# Training settings
parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default ='ChebConv',
                    help = 'GCN or ChebConv model')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--verbose', type=int, default=0,
                    help='print confusion matrix')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--early_stopping', type=int, default = 50,
                    help = 'Stopping criteria for validation')
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
args = parser.parse_args()


print(f'number of nodes: {data.x.shape}')

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        if args.model == 'GCN':
            #cached = True is for transductive learning
            self.conv1 = GCNConv(data.x.shape[1], 16, cached=True)
            self.conv2 = GCNConv(16, 32, cached=True)  
            self.conv3 = GCNConv(32, 64, cached=True) 
            self.conv4 = GCNConv(64, 6, cached=True)   
        elif args.model == 'ChebConv': 
            self.conv1 = ChebConv(data.x.shape[1], 16, K=3)
            self.conv2 = ChebConv(16, 32, K=3)
            self.conv3 = ChebConv(32, 64, K=3)
            self.conv4 = ChebConv(64, 6, K=3)
            

    def forward(self):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
    
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = self.conv4(x, edge_index)

        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

def train():
    model.train()
    optimizer.zero_grad()
    # print(model()[data.val_mask])
    # print(data.y[data.train_mask])

    r"""class weights for imbalanced data"""
    class_weights = class_weight.compute_class_weight('balanced',
                                                 data.y.unique().numpy() ,
                                                 data.y.numpy() )
  
    weights = torch.FloatTensor(class_weights)
    #weights = torch.tensor([8.1577,3.3419, 9.3718, 8.9813, 8.9526, 0.1905])
    
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
        if args.verbose == 1:
            # r"printing predicted classes and number of elements for preds"
            # print('pred')
            # unique_elements, counts_elements = np.unique(pred, return_counts=True)
            # print(np.asarray((unique_elements, counts_elements)))

            # r"printing predicted classes and number of elements for actual"
            # print('actual')
            # unique_elements, counts_elements = np.unique((data.y[mask]-1), return_counts=True)
            # print(np.asarray((unique_elements, counts_elements)))
        #confusion matrix
            if mask_name == 'test_mask':
                conf_mat=confusion_matrix((data.y[mask]-1).numpy(), pred.numpy())
                print(f'confusion_matrix: \n   {conf_mat}')
                class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)
                print(class_accuracy)
        accs.append(acc)
    return accs
  
if __name__ == '__main__':
#stopping criteria 
    counter = 0
    for epoch in range(1, args.epochs+1):
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
