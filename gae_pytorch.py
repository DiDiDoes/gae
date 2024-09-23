import torch
from torch import nn
from torch import optim
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.transforms import RandomLinkSplit

# hyperparameters
dim = 16
lr = 0.01
n_epoch = 200

# prepare data
# choose name from "Cora", "CiteSeer" and "PubMed"
dataset = Planetoid(root="/scratch/chengdicao/Planetoid", name="Cora")
transform = RandomLinkSplit(is_undirected=True, split_labels=True)
train_data, val_data, test_data = transform(dataset[0])

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2*out_channels, cached=True)
        self.relu = nn.ReLU()
        self.conv2 = GCNConv(2*out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        return x

# components
encoder = Encoder(dataset.num_node_features, dim)
model = GAE(encoder)
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(n_epoch):
    # train
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)
    loss = model.recon_loss(z, train_data.pos_edge_label_index, train_data.neg_edge_label_index)
    loss.backward()
    optimizer.step()

    # valid
    model.eval()
    with torch.no_grad():
        z = model.encode(val_data.x, val_data.edge_index)
        auc, ap = model.test(z, val_data.pos_edge_label_index, val_data.neg_edge_label_index)

    # print
    print(f'[{epoch+1:03d}], train_loss: {loss.item():.4f}, valid_auc: {auc.item():.4f}, valid_ap: {ap.item():.4f}')

# test
with torch.no_grad():
    z = model.encode(test_data.x, test_data.edge_index)
    auc, ap = model.test(z, test_data.pos_edge_label_index, test_data.neg_edge_label_index)
print(f'[test], auc: {auc.item():.4f}, ap: {ap.item():.4f}')