import argparse
import os.path
import datetime
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import math
import numpy as np
from scipy.spatial.distance import cosine
import scipy as scipy
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
import metrics, dataloader, utils, val,eval,mini_batch_test
import ast

def split_bacth_items(items,popular,split=0.5):
    G1,G2=[],[]
    items_sorted=list(np.array(items)[np.argsort(np.array(popular)[items])])
    num=int(len(items_sorted)*split)
    G1.extend(items_sorted[0:num])
    G2.extend(items_sorted[num:])
    return np.array(G1),np.array(G2)

def gini_coefficient(popularity):
    # Sort the popularity values
    popularity_sorted = np.sort(popularity)
    
    # Calculate the cumulative sum
    cum_popularity = np.cumsum(popularity_sorted)
    cum_sum = np.sum(popularity_sorted)
    
    # Calculate the Gini coefficient
    gini_numerator = np.sum(cum_popularity) / cum_sum
    gini_denominator = len(popularity) + 1
    gini_coeff = 1 - 2 * (gini_numerator / gini_denominator)
    
    return gini_coeff

def alignment_user(x, y):
    x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
    return (x - y).norm(p=2, dim=1).pow(2).mean()

def main_args():

    # model description
    args = argparse.ArgumentParser(description="ours_lightGCN")

    # dataset
    args.add_argument('--dataset_name', default='gowalla', type=str)


    # args.add_argument('--dataset_path', default='/home/cmm/ICML22/dataset', type=str)
    # args.add_argument('--result_path', default='/home/d1/cmm/ICML22/IID_result', type=str)

    ##OOD
    args.add_argument('--dataset_path', default='./Data', type=str)
    args.add_argument('--result_path', default='OOD_result', type=str)

    args.add_argument('--bpr_num_neg', default=1, type=int)
    args.add_argument('--split_list', default='[0.4]', type=str)
    # LightGCN model
    args.add_argument('--model', default='PAAC_adp_lyn', type=str)
    args.add_argument('--decay', default=0.0001, type=float)
    args.add_argument('--lr', default=0.001, type=float)
    args.add_argument('--batch_size', default=2048, type=int)
    args.add_argument('--layers_list', default='[6]', type=str)
    args.add_argument('--eps', default=0.2, type=float)
    args.add_argument('--cl_rate_list', default='[0.01,0.1,0.5,1,5,10,20]', type=str)
    args.add_argument('--temperature_list', default='[0.2]', type=str)
    args.add_argument('--seed', default=12345, type=int)
    args.add_argument('--align_reg_list', default='[400]', type=str)
    args.add_argument('--weight_list', default='[2]', type=str)
    args.add_argument('--gini_list', default='[1]', type=str)

    #args.add_argument('--align_reg_list', default='[100]', type=str)

    # train
    args.add_argument('--device', default=0, type=int)
    args.add_argument('--EarlyStop', default=20, type=int)
    args.add_argument('--emb_size', default=64, type=int)
    args.add_argument('--num_epoch', default=1000, type=int)
    

    args.add_argument(
        '--topks', default='[20]', type=str)

    return args.parse_args()


class ours_lightGCN(torch.nn.Module):
    def __init__(self, config, data):
        super(ours_lightGCN, self).__init__()

        # model
        self.emb_size = config.emb_size
        self.decay = config.decay
        self.layers = config.layers
        self.device = torch.device(config.device)
        self.eps = config.eps
        self.cl_rate = config.cl_rate
        self.temperature = config.temperature
        self.pop_train=data.pop_train_count
        self.split=config.split
        self.gamma=config.gini

        # data
        self.num_users = data.num_users
        self.num_items = data.num_items
        self.adj = data.norm_adj.to(self.device)

        # embedding
        user_emb_weight = torch.nn.init.normal_(torch.empty(
            self.num_users, self.emb_size), mean=0, std=0.1)
        item_emb_weight = torch.nn.init.normal_(torch.empty(
            self.num_items, self.emb_size), mean=0, std=0.1)
        self.user_embeddings = torch.nn.Embedding(
            self.num_users, self.emb_size, _weight=user_emb_weight)
        self.item_embeddings = torch.nn.Embedding(
            self.num_items, self.emb_size, _weight=item_emb_weight)

    # 
    def forward(self, perturbed=False):
        ego_embeddings = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0)

        #all_emb = [ego_embeddings]

        all_emb = []

        for _ in range(self.layers):
            ego_embeddings = torch.sparse.mm(self.adj, ego_embeddings)
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).to(self.device)
                ego_embeddings = ego_embeddings + torch.sign(ego_embeddings) * F.normalize(random_noise,
                                                                                           dim=1) * self.eps
            all_emb = all_emb + [ego_embeddings]
        all_emb = torch.stack(all_emb, dim=1)
        all_emb = torch.mean(all_emb, dim=1)
        user_emb, item_emb = torch.split(
            all_emb, [self.num_users, self.num_items])
        return user_emb, item_emb
    
    def lyn_forward(self):
        ego_embeddings = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0)
        
        item_emb=[]
        item_emb.append(self.item_embeddings.weight)

        for _ in range(self.layers):

            ego_embeddings = torch.sparse.mm(self.adj, ego_embeddings)
            user_emb, item_embeding = torch.split(
            ego_embeddings, [self.num_users, self.num_items])
            item_emb.append(item_embeding)
        return item_emb

    def bpr_loss(self, user_emb, pos_emb, neg_emb):
        pos_score = torch.mul(user_emb, pos_emb).sum(dim=1)
        neg_score = torch.mul(user_emb, neg_emb).sum(dim=1)
        bpr_loss = - \
            torch.log(10e-8 + torch.sigmoid(pos_score - neg_score)).mean()

        l2_loss = torch.norm(user_emb, p=2) + torch.norm(pos_emb, p=2)
        l2_loss = self.decay * l2_loss

        return bpr_loss, l2_loss
    
    def cl_loss(self, u_idx, i_idx, j_idx):

        gamma=1-gini_coefficient(np.array(self.pop_train)[i_idx])
        # gamma=self.gamma
        # batch里采样
        u_idx = torch.tensor(u_idx)
        bacth_pop,batch_unpop=split_bacth_items(i_idx,self.pop_train,split=self.split)
        #i_idx = torch.tensor(i_idx)
        batch_users = torch.unique(u_idx).type(torch.long).to(self.device)
        #batch_items = torch.unique(i_idx).type(torch.long).to(self.device)
        bacth_pop=torch.tensor(bacth_pop)
        bacth_pop = torch.unique(bacth_pop).type(torch.long).to(self.device)
        batch_unpop=torch.tensor(batch_unpop)
        batch_unpop = torch.unique(batch_unpop).type(torch.long).to(self.device)
        user_view_1, item_view_1 = self.forward(perturbed=True)
        user_view_2, item_view_2 = self.forward(perturbed=True)
        user_cl_loss = metrics.InfoNCE(
            user_view_1[batch_users], user_view_2[batch_users], self.temperature) * self.cl_rate
        item_cl_pop=gamma*metrics.InfoNCE_i(item_view_1[bacth_pop], item_view_2[bacth_pop],item_view_2[batch_unpop], self.temperature,gamma)
        item_cl_unpop=(1-gamma)*metrics.InfoNCE_i(item_view_1[batch_unpop], item_view_2[batch_unpop],item_view_2[bacth_pop], self.temperature,gamma)
        item_cl_loss=(item_cl_pop+item_cl_unpop)* self.cl_rate
        cl_loss = user_cl_loss + item_cl_loss
        return cl_loss, user_cl_loss, item_cl_loss

    def batch_loss(self, u_idx, i_idx, j_idx):

        user_embedding, item_embedding = self.forward(perturbed=False)
        user_emb = user_embedding[u_idx]
        pos_emb = item_embedding[i_idx]
        neg_emb = item_embedding[j_idx]
        bpr_loss, l2_loss = self.bpr_loss(user_emb, pos_emb, neg_emb)
        cl_loss, user_cl_loss, item_cl_loss = self.cl_loss(u_idx, i_idx, j_idx)
        batch_loss = bpr_loss+l2_loss+cl_loss
        return batch_loss, bpr_loss, l2_loss, cl_loss, user_cl_loss, item_cl_loss


def test(model):
    user_embedding, item_embedding = model.forward()
    return user_embedding.detach().cpu().numpy(), item_embedding.detach().cpu().numpy()


def train(config, data, model, optimizer, early_stopping, logger,train_step=1):
    model.train()
    for epoch in range(config.num_epoch):
        start = datetime.datetime.now()
        train_res = {
            'bpr_loss': 0.0,
            'emb_loss': 0.0,
            'cl_loss':0.0,
            'batch_loss': 0.0,
            'align_loss':0.0,
        }
        # train
        with tqdm(total=math.ceil(len(data.training_data) / config.batch_size), desc=f'Epoch {epoch}',
                  unit='batch')as pbar:
            for n, batch in enumerate(dataloader.next_batch_pairwise(data, config.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                batch_loss, bpr_loss, l2_loss, cl_loss, user_cl_loss, item_cl_loss= model.batch_loss(
                    user_idx, pos_idx, neg_idx)
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                train_res['bpr_loss']+= bpr_loss.item()
                train_res['emb_loss'] += l2_loss.item()
                train_res['batch_loss'] += batch_loss.item()
                train_res['cl_loss']+= cl_loss.item()

                pbar.set_postfix({'loss (batch)': batch_loss.item()})
                pbar.update(1)
        train_res['bpr_loss'] = train_res['bpr_loss'] / math.ceil(len(data.training_data) / config.batch_size)
        train_res['emb_loss'] = train_res['emb_loss'] / math.ceil(len(data.training_data) / config.batch_size)
        train_res['batch_loss'] = train_res['batch_loss'] / math.ceil(len(data.training_data) / config.batch_size)
        train_res['cl_loss'] = train_res['cl_loss'] / math.ceil(len(data.training_data) / config.batch_size)
        

        # user_emb, item_emb = model.forward()
        item_emb=model.lyn_forward()
        G1,G2=dataloader.user_items_2_group_pop(data)
        align_loss=0
        for layer in range(config.layers+1):
            weight=1 / (config.weight**layer)
            item_emb_layer=item_emb[layer]
            align_loss_ly=alignment_user(item_emb_layer[G1],item_emb_layer[G2])*weight
            align_loss=align_loss+align_loss_ly
        align_loss=align_loss*config.align_reg/(config.layers+1)
        optimizer.zero_grad()
        align_loss.backward()
        optimizer.step()
        train_res['align_loss'] += align_loss.item()
            


        train_res['align_loss'] = train_res['align_loss'] / train_step

        training_logs = 'epoch: %d, ' % epoch
        for name, value in train_res.items():
            training_logs += name + ':' + '%.6f' % value + ' '
        logger.info(training_logs)
        trin_time = datetime.datetime.now()

        # val
        model.eval()
        user_embedding, item_embedding = test(model)
        val_hr,val_recall, val_ndcg = mini_batch_test.test_acc_batch(
            data.val_U2I, data.train_U2I, user_embedding, item_embedding)
        logger.info('val_hr@100:{:.6f}   val_recall@100:{:.6f}   val_ndcg@100:{:.6f}   train_time:{}s   test_tiem:{}s'.format(
            val_hr, val_recall, val_ndcg, (trin_time-start).seconds, (datetime.datetime.now() - trin_time).seconds))

        # early_stopping
        early_stopping(val_hr, model, epoch)
        if early_stopping.early_stop:
            logger.info("Early stopping")
            break
    



def main(config):
    # parser
    #config = main_args()

    # # train continue
    # if config.if_continue:
    #     sys.exit()

    # Logger
    ISOTIMEFORMAT = '%m%d-%H%M%S'
    timestamp = str(datetime.datetime.now().strftime(ISOTIMEFORMAT))
    file_name = '_'.join((str(config.gini),str(config.weight),str(config.layers),str(config.cl_rate),str(config.align_reg),timestamp))

    result_path = '/'.join((config.result_path,
                           config.model, config.dataset_name, file_name))
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    logger_file_name = os.path.join(result_path, 'train_logger')
    logger = utils.get_logger(logger_file_name)
    for name, value in vars(config).items():
        logger.info('%20s =======> %-20s' % (name, value))

    # Seed
    if config.seed:
        utils.setup_seed(config.seed)

    # Load Data
    logger.info('------Load Data-----')
    data = dataloader.Data(config, logger)
    data.norm_adj = dataloader.LaplaceGraph(
        data.num_users, data.num_items, data.train_U2I).generate()
        
   # data.user_pop,data.unpopular_users,data.popular_users=data.split_user_pop()


    # Load Model
    logger.info('------Load Model-----')
    model = ours_lightGCN(config, data)
    model.to(model.device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # EarlyStopping
    early_stopping = utils.EarlyStopping(logger,
        config.EarlyStop, verbose=True, path=result_path)

    # train model
    train(config, data, model, optimizer, early_stopping, logger)

    # #test model

    model_dict = model.load_state_dict(torch.load(result_path + "/best_val_epoch.pt"))
    user_embedding,item_embedding=test(model)

    val_hr,val_recall, val_ndcg = mini_batch_test.test_acc_batch(
        data.val_U2I, data.train_U2I, user_embedding, item_embedding)
    logger.info('=======Best   performance=====\nval_hr@20:{:.6f}   val_recall@20:{:.6f}   val_ndcg@20:{:.6f} '.format(
        val_hr, val_recall, val_ndcg))
    test_OOD_hr, test_OOD_recall, test_OOD_ndcg = mini_batch_test.test_acc_batch(
        data.test_U2I, data.train_U2I, user_embedding, item_embedding)
    logger.info('=======Best   performance=====\ntest_OOD_hr@20:{:.6f}   test_OOD_recall@20:{:.6f}   test_OOD_ndcg@20:{:.6f} '.format(
        test_OOD_hr, test_OOD_recall, test_OOD_ndcg))
    test_IID_hr, test_IID_recall, test_IID_ndcg = mini_batch_test.test_acc_batch(
        data.test_iid_U2I, data.train_U2I, user_embedding, item_embedding)
    logger.info('=======Best   performance=====\ntest_IID_hr@20:{:.6f}   test_IID_recall@20:{:.6f}   test_IID_ndcg@20:{:.6f} '.format(
        test_IID_hr, test_IID_recall, test_IID_ndcg))
    return val_hr, val_recall, val_ndcg,test_OOD_hr, test_OOD_recall, test_OOD_ndcg ,test_IID_hr, test_IID_recall, test_IID_ndcg,result_path

if __name__ == '__main__':
    config=main_args()
    result_path = '/'.join((config.result_path,
                                config.model, config.dataset_name))
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    f=open('/'.join((config.result_path,config.model, config.dataset_name))+'/best_performace.txt','a+')
    for cl_rate in ast.literal_eval(config.cl_rate_list):
        for layers in ast.literal_eval(config.layers_list):
            for align_reg in ast.literal_eval(config.align_reg_list):
                for temperature in ast.literal_eval(config.temperature_list):
                    for split in ast.literal_eval(config.split_list):
                        for weight in ast.literal_eval(config.weight_list):
                            for gini in ast.literal_eval(config.gini_list):
                                config.gini=gini
                                config.weight=weight
                                config.split=split
                                config.temperature=temperature
                                config.cl_rate=cl_rate
                                config.layers=layers
                                config.align_reg=align_reg
                                
                                val_hr, val_recall, val_ndcg,test_OOD_hr, test_OOD_recall, test_OOD_ndcg ,test_IID_hr, test_IID_recall, test_IID_ndcg,result_path= main(config)
                                f.write('\n')
                                f.write('\n gini:{}=======layers:{}===align_reg:{}====weight:{}===cl-rate:{}====split:{}\n  best_hr@20:{}=====best_recall@20:{}====best_ndcg@20:{}\n test_OOD_hr@20:{:.6f}   test_OOD_recall@20:{:.6f}   test_OOD_ndcg@20:{:.6f}\n test_IID_hr@20:{:.6f}   test_IID_recall@20:{:.6f}   test_IID_ndcg@20:{:.6f} \n  Resulst_path:{}\n '
                                        .format(gini,config.layers,config.align_reg,config.weight,config.cl_rate,config.split,val_hr,val_recall,val_ndcg,test_OOD_hr, test_OOD_recall, test_OOD_ndcg ,test_IID_hr, test_IID_recall, test_IID_ndcg,result_path))
                                f.write('\n')
        f.close()
