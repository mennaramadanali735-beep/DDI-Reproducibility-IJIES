import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


from utils import *
import torch.nn as nn
import numpy as np
from torch_geometric.nn import global_mean_pool, global_add_pool
from method import GNN, Encoder_MultipleLayers, SchNet
from torch_geometric.utils import degree
from torch_geometric.nn import radius_graph
from torch_scatter import scatter_add
from method.model_helper import Encoder_1d, Embeddings
from torch_geometric.nn import GCNConv,RGCNConv,RGATConv
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from loadtrain import loadtrain
import torch.nn.functional as F
import csv
import argparse
from torch.optim import Adam
from train_model import train_model
from process_dataset import PCQM4Mv2Dataset,Drug3Dataset
import torch_geometric
import copy
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
SAVE = 'save/model_finalDeng.pth'

class transformer_1d(nn.Sequential):
    def __init__(self):
        super(transformer_1d, self).__init__()
        input_dim_drug = 2587
        transformer_emb_size_drug = 128
        transformer_dropout_rate = 0.1
        transformer_n_layer_drug = 8
        transformer_intermediate_size_drug = 512
        transformer_num_attention_heads_drug = 8
        transformer_attention_probs_dropout = 0.1
        transformer_hidden_dropout_rate = 0.1

        self.emb = Embeddings(input_dim_drug,
                              transformer_emb_size_drug,
                              50,
                              transformer_dropout_rate)

        self.encoder = Encoder_1d(transformer_n_layer_drug,
                                  transformer_emb_size_drug,
                                  transformer_intermediate_size_drug,
                                  transformer_num_attention_heads_drug,
                                  transformer_attention_probs_dropout,
                                  transformer_hidden_dropout_rate)

    def forward(self, emb, mask):
        e = emb.long().to(device)
        e_mask = mask.long().to(device)
        ex_e_mask = e_mask.unsqueeze(1).unsqueeze(2)
        ex_e_mask = (1.0 - ex_e_mask) * -10000.0

        emb = self.emb(e)
        encoded_layers, _ = self.encoder(emb.float(), ex_e_mask.float())
        return encoded_layers



class projection_head(nn.Module):
    def __init__(self):
        super(projection_head, self).__init__()
        self.line1 = nn.Linear(128, 128)
        self.line2 = nn.Linear(128, 64)

        self.relu = nn.ReLU()

    def forward(self, emb):
        out1 = self.line1(emb)
        out = self.relu(out1)
        out = self.line2(out)

        return out, out1


def floyd_warshall(adjacency_matrix):
    (nrows, ncols) = adjacency_matrix.shape
    assert nrows == ncols
    n = nrows

    adj_mat_copy = adjacency_matrix.astype(float, order='C', casting='safe', copy=True)
    assert adj_mat_copy.flags['C_CONTIGUOUS']
    M = adj_mat_copy

    for i in range(n):
        for j in range(n):
            if i == j:
                M[i][j] = 0
            elif M[i][j] == 0:
                M[i][j] = 510

    # floyed algo
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if M[i][j] > M[i][k] + M[k][j]:
                    M[i][j] = M[i][k] + M[k][j]
                    # path[i][j] = k

    return M


def get_spatil_pos(batch_node, edge_index_s, batch=None):
    if type(batch_node) != list:
        batch_node = batch_node.tolist()
    N = 0
    row, col = [], []
    adj = torch.zeros([1])
    spe = []
    N_last = 0
    for x in range(batch_node[len(batch_node) - 1] + 1):
        N = batch_node.count(x)
        edge_index = batch[x].edge_index
        N_last = N_last + N

        adj = torch.zeros([N, N], dtype=torch.bool)
        row, col = edge_index
        adj[row, col] = True
        shortest_path_result = floyd_warshall(adj.numpy())
        spatial_pos = torch.from_numpy((shortest_path_result)).long()
        if N < 50:
            spatial_pos = torch.nn.functional.pad(spatial_pos, pad=(0, 50 - N, 0, 50 - N), value=0)
        else:
            spatial_pos = spatial_pos[:50, :50]
        spe.append(spatial_pos)
    spe = torch.stack(spe).to(device)
    return spe

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(32, 32, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)

        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk=None):
        if msk is None:
            return torch.mean(seq, 0)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 0) / torch.sum(msk)

class CL_model(nn.Module):
    def __init__(self, device, feature, hidden1, hidden2,dropout):
        super(CL_model, self).__init__()

        self.max = 0

        self.device = device
        self.class_prompts = None

        # Multiple GPUs need to remove ‘modules’
        model_dict = torch.load(SAVE, map_location=device)
        new_state_dict = {}
        for k, v in model_dict['cl_model'].items():
            new_state_dict[k[7:]] = v

        self.cl_model = CL_model_2d().to(device)

        self.cl_model.load_state_dict(new_state_dict)


        self.encoder_o1 = RGCNConv(feature, hidden1, num_relations=65).to(device)
        self.encoder_o2 = RGCNConv(hidden1, hidden2, num_relations=65).to(device)

        self.attt = torch.zeros(2)
        self.attt[0] = 0.5
        self.attt[1] = 0.5

        self.attt = nn.Parameter(self.attt)


        self.dropout = dropout
        self.sigm = nn.Sigmoid()
        self.read = AvgReadout().to(device)
        self.num_classes = 65
        self.mlp = nn.ModuleList([nn.Linear(704, 256),
                                  nn.ELU(),
                                  nn.Dropout(p=0.1),
                                  nn.Linear(256, 128),
                                  nn.ELU(),
                                  nn.Dropout(p=0.1),
                                  nn.Linear(128, 65)
                                  ])

    def MLP(self, vectors, layer):
            for i in range(layer):
                vectors = self.mlp[i](vectors)

            return vectors




    def forward(self,out, edge_index, label_list, idx):

        x_o = out
        adj = edge_index
        e_type = label_list
        e_type = torch.tensor(e_type, dtype=torch.int64).to(device)

        x1_o = F.relu(self.encoder_o1(x_o, adj, e_type))
        x1_o = F.dropout(x1_o, self.dropout, training=self.training)
        x1_os = x1_o
        x2_o = self.encoder_o2(x1_os, adj, e_type)

        a = [int(i) for i in list(idx[0])]
        b = [int(i) for i in list(idx[1])]

        aa = torch.tensor(a, dtype=torch.long)
        bb = torch.tensor(b, dtype=torch.long)
        final = torch.cat((self.attt[0] * x1_o, self.attt[1] * x2_o), dim=1)
        entity1 = final[aa]
        entity2 = final[bb]
        # skip connection
        entity1_res = out[aa].to(device)
        entity2_res = out[bb].to(device)
        entity1 = torch.cat((entity1, entity1_res), dim=1)
        entity2 = torch.cat((entity2, entity2_res), dim=1)
        concatenate = torch.cat((entity1, entity2), dim=1)
        feature = self.MLP(concatenate, 7)
        log1 = feature

        return log1,final


class Transformer_E(nn.Sequential):
    def __init__(self):
        super(Transformer_E, self).__init__()
        transformer_emb_size_drug = 128
        transformer_n_layer_drug = 4
        transformer_intermediate_size_drug = 256
        transformer_num_attention_heads_drug = 4
        transformer_attention_probs_dropout = 0.1
        transformer_hidden_dropout_rate = 0.1

        self.encoder = Encoder_MultipleLayers(transformer_n_layer_drug,
                                              transformer_emb_size_drug,
                                              transformer_intermediate_size_drug,
                                              transformer_num_attention_heads_drug,
                                              transformer_attention_probs_dropout,
                                              transformer_hidden_dropout_rate)

    def forward(self, h_node, mask, if_2d, in_degree_2d=None, out_degree_2d=None, dist_3d=None, dist_m=None, spe=None):
        encoded_layers, _, all = self.encoder(h_node.float(), mask.float(), if_2d, in_degree_2d=in_degree_2d,
                                              out_degree_2d=out_degree_2d, dist_3d=dist_3d, dist_m=dist_m, spe=spe)
        return encoded_layers, all


class CL_model_2d(nn.Module):
    def __init__(self):
        super(CL_model_2d, self).__init__()
        self.model_2d = GNN()
        self.model_3d = SchNet()
        self.model_T_2d = Transformer_E()
        self.model_T_3d = Transformer_E()
        self.model_1d = transformer_1d()
        self.projection_2d_low = projection_head()
        self.projection_2d_high = projection_head()
        self.projection_3d_low = projection_head()
        self.projection_3d_high = projection_head()
        self.projection_1d_low = projection_head()
        self.projection_1d_high = projection_head()

    def forward(self, batch):
        # 1d
        smile_emb = torch.from_numpy(np.asarray(batch.smiles))
        smile_mask = torch.from_numpy(np.asarray(batch.mask))
        emb_1d = self.model_1d(smile_emb, smile_mask)
        emb_1d_low = emb_1d[3]
        emb_1d_high = emb_1d[7]

        # suit for transfomer encoder
        out_2d, out_2d_res, _ = self.model_2d(batch.x, batch.edge_index, batch.edge_attr)
        emb_2d_low = global_mean_pool(out_2d, batch.batch)
        out_2d = out_2d + out_2d_res
        batch_node = batch.batch.tolist()
        h_node_two = []
        mask = []
        mask_out = []
        in_degree_2d_final = []
        out_degree_2d_final = []
        in_degree_2d = degree(batch.edge_index[0], num_nodes=len(batch.x)).int()
        out_degree_2d = degree(batch.edge_index[1], num_nodes=len(batch.x)).int()
        spe = get_spatil_pos(batch.batch, batch.edge_index, batch=batch)

        flag = 0
        # some mole is too long
        for x in range(batch_node[len(batch_node) - 1] + 1):
            if batch_node.count(x) < 50:
                mask.append([] + batch_node.count(x) * [0] + (50 - batch_node.count(x)) * [-10000])
                mask_out.append([] + batch_node.count(x) * [1] + (50 - batch_node.count(x)) * [0])
            else:
                mask.append([] + 50 * [0])
                mask_out.append([] + 50 * [1])

            oral_node_2d = out_2d[flag:flag + batch_node.count(x)]
            in_degree_2d_oral = in_degree_2d[flag:flag + batch_node.count(x)]
            out_degree_2d_oral = out_degree_2d[flag:flag + batch_node.count(x)]
            flag += batch_node.count(x)
            if batch_node.count(x) < 50:
                temp_2d = torch.full([(50 - oral_node_2d.size()[0]), 128], 0).to(device)
                temp_in_degree_2d = torch.full([(50 - oral_node_2d.size()[0])], 0).to(device)
                temp_out_degree_2d = torch.full([(50 - oral_node_2d.size()[0])], 0).to(device)
                in_degree_2d_oral = torch.cat((in_degree_2d_oral, temp_in_degree_2d), 0)
                out_degree_2d_oral = torch.cat((out_degree_2d_oral, temp_out_degree_2d), 0)
                final_node_2d = torch.cat((oral_node_2d, temp_2d), 0)
            else:
                final_node_2d = oral_node_2d[:][:][:50]
                in_degree_2d_oral = in_degree_2d_oral[:][:][:50]
                out_degree_2d_oral = out_degree_2d_oral[:][:][:50]

            h_node_two.append(final_node_2d)
            in_degree_2d_final.append(in_degree_2d_oral)
            out_degree_2d_final.append(out_degree_2d_oral)

        h_node_two = torch.stack(h_node_two).to(device)
        mask_2d = torch.tensor(mask, dtype=torch.float)
        mask_2d = mask_2d.to(device).unsqueeze(1).unsqueeze(2)
        mask_2d_out = torch.tensor(mask_out, dtype=torch.float).to(device)
        mask_1d_out = smile_mask.to(device)
        in_degree_2d_final = torch.stack(in_degree_2d_final).to(device)
        out_degree_2d_final = torch.stack(out_degree_2d_final).to(device)

        # 2d transformer emb
        emb_2d_high, _ = self.model_T_2d(h_node_two, mask_2d, True, in_degree_2d_final, out_degree_2d_final, spe=spe)
        emb_2d_high_out2 = torch.div(torch.sum(torch.mul(emb_2d_high, mask_2d_out.unsqueeze(2)), dim=1),
                                     torch.sum(mask_2d_out, dim=1).unsqueeze(1))
        # 1d transormer emb
        emb_1d_low = torch.div(torch.sum(torch.mul(emb_1d_low, mask_1d_out.unsqueeze(2)), dim=1),
                               torch.sum(mask_1d_out, dim=1).unsqueeze(1))
        emb_1d_high = torch.div(torch.sum(torch.mul(emb_1d_high, mask_1d_out.unsqueeze(2)), dim=1),
                                torch.sum(mask_1d_out, dim=1).unsqueeze(1))

        emb_2d_high = emb_2d_high_out2

        return emb_2d_low, emb_2d_high, emb_1d_low, emb_1d_high




class Data_class(Dataset):

    def __init__(self, triple):
        self.entity1 = triple[:, 0]
        self.entity2 = triple[:, 1]
        self.relationtype=triple[:, 2]
        #self.label = triple[:, 3]

    def __len__(self):
        return len(self.relationtype)

    def __getitem__(self, index):
        return  (self.entity1[index], self.entity2[index], self.relationtype[index])

def set_random_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



if __name__ == '__main__':
    set_random_seed(1, deterministic=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="Deng")
    parser.add_argument('--lr', type=float, default=1e-3,  help='Initial learning rate. Default is 5e-4.')
    parser.add_argument('--weight_decay', default=5e-4,  help='Weight decay (L2 loss on parameters) Default is 5e-4.')
    parser.add_argument('--zhongzi', default=0, help='Number of zhongzi.')
    parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
    parser.add_argument('--out_file', required=False, default='result.txt', help='Path to data result file. e.g., result.txt')

    parser.add_argument('--drugbank_tr', type=str, default='ddi_training1.csv', required=False, help='..')
    parser.add_argument('--drugbank_val', type=str, default='ddi_validation1.csv', required=False, help='..')
    parser.add_argument('--drugbank_te', type=str, default='ddi_test1.csv', required=False, help='..')

    parser.add_argument("--num_tasks", type=int, default=65)

    args = parser.parse_args()

    print(args.batch_size)
    print(args.lr)

    df = pd.read_csv("dataset/Deng/drug_listxiao.csv")

    drug_list = df['drug_id'].tolist()


    dataset = PCQM4Mv2Dataset()
    train_loader_nol = torch_geometric.loader.DataLoader(dataset, batch_size=572, shuffle=False)

    train_data, val_data, test_data = loadtrain(args)

    params = {'batch_size': args.batch_size, 'shuffle': False, 'num_workers': 0, 'drop_last': False}
    training_set = Data_class(train_data)
    train_loader = DataLoader(training_set, **params)

    validation_set = Data_class(val_data)
    val_loader = DataLoader(validation_set, **params)

    test_set = Data_class(test_data)
    test_loader = DataLoader(test_set, **params)

    positive1 = copy.deepcopy(train_data)

    edge_index_o = []
    label_list = []
    for i in range(positive1.shape[0]):
        # for h, t, r ,label in positive1:
        a = []
        a.append(int(positive1[i][0]))
        a.append(int(positive1[i][1]))
        edge_index_o.append(a)
        label_list.append(int(positive1[i][2]))
        a = []
        a.append(int(positive1[i][1]))
        a.append(int(positive1[i][0]))
        edge_index_o.append(a)
        label_list.append(int(positive1[i][2]))
    edge_index = torch.tensor(edge_index_o, dtype=torch.long).to(device)
    edge_index = edge_index.t().contiguous()

    model= CL_model(device, 256, 64, 32, 0.5).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_model(model, device, optimizer, edge_index, label_list, train_loader_nol, train_loader, val_loader, test_loader, args)






