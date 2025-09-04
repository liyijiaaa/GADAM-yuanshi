import torch
import torch.nn as nn
import dgl
from dgl.data import register_data_args
import argparse, time
from model import *
from utils import *
from sklearn.metrics import roc_auc_score, recall_score, average_precision_score
from pytorch_memlab import LineProfiler, profile
# 自适应采样添加
from numpy.linalg import inv
from torch.nn.functional import normalize


def train_local(net, graph, feats, opt, args, init=True):
    memo = {}
    labels = graph.ndata['label']
    num_nodes = graph.num_nodes()

    device = args.gpu
    if device >= 0:
        torch.cuda.set_device(device)
        net = net.to(device)
        labels = labels.cuda()
        feats = feats.cuda()

    def init_xavier(m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)

    if init:
        net.apply(init_xavier)

    print('train on:', 'cpu' if device < 0 else 'gpu {}'.format(device))

    cnt_wait = 0
    best = 999
    dur = []

    for epoch in range(args.local_epochs):
        net.train()
        if epoch >= 3:
            t0 = time.time()

        opt.zero_grad()
        loss, l1, l2 = net(feats)

        loss.backward()
        opt.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        if loss.item() < best:
            best = loss.item()
            torch.save(net.state_dict(), 'best_local_model.pkl')

        print("Epoch {} | Time(s) {:.4f} | Loss {:.4f} | l1 {:.4f} | l2 {:.4f}"
              .format(epoch + 1, np.mean(dur), loss.item(), l1.item(), l2.item()))

    memo['graph'] = graph
    net.load_state_dict(torch.load('best_local_model.pkl'))
    h, mean_h = net.encoder(feats)
    h, mean_h = h.detach(), mean_h.detach()
    memo['h'] = h
    memo['mean_h'] = mean_h

    torch.save(memo, 'memo.pth')


def load_info_from_local(local_net, device):
    if device >= 0:
        torch.cuda.set_device(device)
        local_net = local_net.to(device)

    memo = torch.load('memo.pth')
    local_net.load_state_dict(torch.load('best_local_model.pkl'))
    graph = memo['graph']
    pos = graph.ndata['pos']
    scores = -pos.detach()
    ano_topk = 0.05  # k_ano
    nor_topk = 0.3  # k_nor
    num_nodes = graph.num_nodes()

    num_ano = int(num_nodes * ano_topk)
    _, ano_idx = torch.topk(scores, num_ano)

    num_nor = int(num_nodes * nor_topk)
    _, nor_idx = torch.topk(-scores, num_nor)

    feats = graph.ndata['feat']

    h, _ = local_net.encoder(feats)

    center = h[nor_idx].mean(dim=0).detach()

    if device >= 0:
        memo = {k: v.to(device) for k, v in memo.items()}
        nor_idx = nor_idx.cuda()
        ano_idx = ano_idx.cuda()
        center = center.cuda()

    return memo, nor_idx, ano_idx, center


def train_global(global_net, opt, graph, args):
    epochs = args.global_epochs

    labels = graph.ndata['label'].cpu().numpy()
    num_nodes = graph.num_nodes()
    device = args.gpu
    feats = graph.ndata['feat']
    pos = graph.ndata['pos']

    if device >= 0:
        torch.cuda.set_device(device)
        global_net = global_net.to(device)
        # labels = labels.cuda()
        feats = feats.cuda()

    def init_xavier(m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)

    init = True
    if init:
        global_net.apply(init_xavier)

    print('train on:', 'cpu' if device < 0 else 'gpu {}'.format(device))

    cnt_wait = 0
    best = 999
    dur = []

    pred_labels = np.zeros_like(labels)
    for epoch in range(epochs):
        global_net.train()
        if epoch >= 3:
            t0 = time.time()

        opt.zero_grad()
        loss, scores = global_net(feats, epoch)
        loss.backward()
        opt.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        if loss.item() < best:
            best = loss.item()
            torch.save(global_net.state_dict(), 'best_global_model.pkl')

        mix_score = -(scores + pos)
        mix_score = mix_score.detach().cpu().numpy()

        mix_auc = roc_auc_score(labels, mix_score)

        sorted_idx = np.argsort(mix_score)
        k = int(sum(labels))
        topk_idx = sorted_idx[-k:]
        pred_labels[topk_idx] = 1

        recall_k = recall_score(np.ones(k), labels[topk_idx])
        ap = average_precision_score(labels, mix_score)

        # print("Epoch {} | Time(s) {:.4f} | Loss {:.4f} | auc {:.4f} | mix_auc {:.4f}"
        #       .format(epoch+1, np.mean(dur), loss.item(), auc, mix_auc))
        print("Epoch {} | Time(s) {:.4f} | Loss {:.4f} | mix_auc {:.4f} | recall@k {:.4f} | ap {:.4f}"
              .format(epoch + 1, np.mean(dur), loss.item(), mix_auc, recall_k, ap))

    return mix_auc, recall_k, ap


# def train_global(global_net, opt, graph, args):
#     epochs = args.global_epochs
#
#     labels = graph.ndata['label'].cpu().numpy()
#     num_nodes = graph.num_nodes()
#     device = args.gpu
#     feats = graph.ndata['feat']
#     pos = graph.ndata['pos']
#
#     if device >= 0:
#         torch.cuda.set_device(device)
#         global_net = global_net.to(device)
#         # labels = labels.cuda()
#         feats = feats.cuda()
#
#     def init_xavier(m):
#         if type(m) == nn.Linear:
#             nn.init.xavier_normal_(m.weight)
#
#     init = True
#     if init:
#         global_net.apply(init_xavier)
#
#     print('train on:', 'cpu' if device < 0 else 'gpu {}'.format(device))
#
#     cnt_wait = 0
#     best = 999
#     dur = []
#
#     # 自适应邻居采用修改开始点——初始化采样概率
#     # 移除自环
#     graph = dgl.remove_self_loop(graph)
#     # # # 添加自环
#     graph = dgl.add_self_loop(graph)
#     # 邻接矩阵处理
#     adj_sp = graph.adj_external(scipy_fmt='coo') # 正确用法
#
#     # 4种采样方式
#     sampling_ways = 4
#
#     normalized_adj = adj_normalize(adj_sp)  # 归一化邻接矩阵
#     column_normalized_adj = column_normalize(adj_sp)  # 列归一化
#     ppr_c = 0.15
#
#     # 幂次邻接矩阵（1-hop, 2-hop）
#     power_adj_list = [normalized_adj]
#     for m in range(2):
#         power_adj_list.append(power_adj_list[0] * power_adj_list[m])
#     #随机游走修改
#     ppr_adj = ppr_c * inv((sp.eye(adj_sp.shape[0]) - (1 - ppr_c) * column_normalized_adj).toarray())  # PPR
#     hop1_adj = power_adj_list[0].toarray()
#     hop2_adj = power_adj_list[1].toarray()
#     x = normalize(feats, dim=1).cpu()
#     knn_adj = np.array(torch.matmul(x, x.transpose(1, 0)))
#
#     # 四种采样方式
#     sampling_weight = np.ones(4)
#     # 最小采样概率
#     p_min = 0.05
#
#     p = (1 - 4 * p_min) * sampling_weight / sum(sampling_weight) + p_min
#
#     warm_up_epoch = 3
#     #奖励函数的计算次数
#     update_internal = 5
#     update_day = -1
#     torch.autograd.set_detect_anomaly(True)
#
#     pred_labels = np.zeros_like(labels)
#
#     for epoch in range(epochs):
#         global_net.train()
#         if epoch >= 3:
#             t0 = time.time()
#
#         opt.zero_grad()
#         #自适应邻居采样修改——自适应采样
#         sampled_result = adaptive_sampler(num_nodes, ppr_adj, hop1_adj, hop2_adj, knn_adj, p=p, total_sample_size=15)
#
#         ada_neighbor_nodes = torch.stack(sampled_result).to(device).detach()
#
#         # 模型前向传播
#         loss, scores = global_net(feats, epoch, ada_neighbor_nodes)
#
#         mix_score = -(scores + pos)
#         if epoch >= warm_up_epoch and (epoch - update_day) >= update_internal:
#             # 计算奖励（采样效果评估）
#             r = get_reward(device, p, ppr_adj, hop1_adj, hop2_adj, knn_adj, num_nodes,
#                            ada_neighbor_nodes, cost_mat=mix_score)
#
#             # 基于奖励更新采样权重_两个0.01是可变参数
#             updated_param = np.exp((p_min / 2.0) * (r + 0.01 / p) * 100 * np.sqrt(
#                 np.log(15 / 0.01) / (sampling_ways * update_internal)))
#             sampling_weight = sampling_weight * updated_param
#             p = (1 - 4 * p_min) * sampling_weight / sum(sampling_weight) + p_min
#             update_day = epoch
#         loss.backward()
#         opt.step()
#
#         if epoch >= 3:
#             dur.append(time.time() - t0)
#
#         if loss.item() < best:
#             best = loss.item()
#             torch.save(global_net.state_dict(), 'best_global_model.pkl')
#
#         #mix_score = -(scores + pos)
#         mix_score = mix_score.detach().cpu().numpy()
#
#         mix_auc = roc_auc_score(labels, mix_score)
#
#         sorted_idx = np.argsort(mix_score)
#         k = int(sum(labels))
#         topk_idx = sorted_idx[-k:]
#         pred_labels[topk_idx] = 1
#
#         recall_k = recall_score(np.ones(k), labels[topk_idx])
#         ap = average_precision_score(labels, mix_score)
#
#         # print("Epoch {} | Time(s) {:.4f} | Loss {:.4f} | auc {:.4f} | mix_auc {:.4f}"
#         #       .format(epoch+1, np.mean(dur), loss.item(), auc, mix_auc))
#         print("Epoch {} | Time(s) {:.4f} | Loss {:.4f} | mix_auc {:.4f} | recall@k {:.4f} | ap {:.4f}"
#               .format(epoch + 1, np.mean(dur), loss.item(), mix_auc, recall_k, ap))
#
#     return mix_auc, recall_k, ap

def main(args):
    seed_everything(args.seed)

    graph = my_load_data(args.data)
    # graph = graph.add_self_loop() test encoder=GCN
    feats = graph.ndata['feat']

    if args.gpu >= 0:
        graph = graph.to(args.gpu)

    in_feats = feats.shape[1]

    local_net = LocalModel(graph,
                           in_feats,
                           args.out_dim,
                           nn.PReLU(), )

    local_opt = torch.optim.Adam(local_net.parameters(),
                                 lr=args.local_lr,
                                 weight_decay=args.weight_decay)
    t1 = time.time()
    train_local(local_net, graph, feats, local_opt, args)

    # load information from LIM module
    memo, nor_idx, ano_idx, center = load_info_from_local(local_net, args.gpu)
    t2 = time.time()
    graph = memo['graph']
    global_net = GlobalModel(graph,
                             in_feats,
                             args.out_dim,
                             nn.PReLU(),
                             nor_idx,
                             ano_idx,
                             center)
    opt = torch.optim.Adam(global_net.parameters(),
                           lr=args.global_lr,
                           weight_decay=args.weight_decay)
    t3 = time.time()

    mix_auc, recall_k, ap = train_global(global_net, opt, graph, args)
    t4 = time.time()

    t_all = t2 + t4 - t1 - t3
    print('mean_t:{:.4f}'.format(t_all / (args.local_epochs + args.global_epochs)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model')
    register_data_args(parser)
    parser.add_argument("--data", type=str, default="Cora",
                        help="dataset")
    parser.add_argument("--seed", type=int, default=717,
                        help="random seed")
    parser.add_argument("--dropout", type=float, default=0.,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=1,
                        help="gpu")
    parser.add_argument("--local-lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--global-lr", type=float, default=5e-4,
                        help="learning rate")
    parser.add_argument("--local-epochs", type=int, default=100,
                        help="number of training local model epochs")
    parser.add_argument("--global-epochs", type=int, default=50,
                        help="number of training global model epochs")
    parser.add_argument("--out-dim", type=int, default=64,
                        help="number of hidden gcn units")
    parser.add_argument("--train-ratio", type=float, default=0.05,
                        help="train ratio")
    parser.add_argument("--weight-decay", type=float, default=0.,
                        help="Weight for L2 loss")
    parser.add_argument("--patience", type=int, default=20,
                        help="early stop patience condition")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.set_defaults(self_loop=True)

    args = parser.parse_args()
    print(args)
    main(args)
    # multi_run(args)


