import pandas as pd
from collections import defaultdict
import pandas as pd
import numpy as np
import torch
import os
#os.chdir('/home/cmm/beifen/importance_detach/')
import random 
import time

class Data(object):
    def __init__(self, config, logger):
        self.dataset_name=config.dataset_name
        self.dataset_path = '/'.join((config.dataset_path,
                                     config.dataset_name))
        self.num_neg = config.bpr_num_neg
        self.num_users, self.num_items, self.train_U2I, self.training_data, self.test_U2I, self.pop_train_count, _ ,self.test_I2U,self.train_I2U,self.val_U2I,self.test_iid_U2I,self.val_sum,self.test_iid_sum= self.load_data()
        logger.info('num_users:{:d}   num_items:{:d}   density:{:.6f}%'.format(
            self.num_users, self.num_items, _/self.num_items/self.num_users*100))
        #self.weight=self.get_weight()

    def load_data(self):
        training_data = []
        num_items, num_users = 0, 0
        train_num, test_num = 0, 0
        train_interation = []
        train_U2I, test_U2I,test_I2U,train_I2U= defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
        train_file = pd.read_table(
            self.dataset_path + '/train.txt', header=None)
        test_file = pd.read_table(self.dataset_path + '/test.txt', header=None)
        

        
        # 读取train
        for l in range(len(train_file)):
            line = train_file.iloc[l, 0]
            text = line.split(" ")
            text = list(map(lambda x: int(x), text))
            uid = int(text[0])
            num_users = max(num_users, uid)
            items = text[1:]
            if len(items) == 0:
                continue
            train_interation.extend(items)
            num_items = max(num_items, max(items))
            train_num = train_num + len(items)
            train_U2I[uid].extend(items)
            for item in items:
                training_data.append([uid, item])
                train_I2U[item].append(uid)

        # 读取test
        for l in range(len(test_file)):
            line = test_file.iloc[l, 0]
            text = line.split(" ")
            text = list(map(lambda x: int(x), text))
            uid = int(text[0])
            num_users = max(num_users, uid)
            items = text[1:]
            test_num = test_num + len(items)
            num_items = max(num_items, max(items))
            test_U2I[uid].extend(items)
            for item in items:
                test_I2U[item].append(uid)
        if self.dataset_name=='amazon-book.new' or self.dataset_name=='tencent.new':
            val_U2I,test_iid_U2I=defaultdict(list),defaultdict(list)
            val_sum,test_iid_sum=0,0
            val_file=pd.read_table(self.dataset_path + '/valid.txt', header=None)
            test_iid_file=pd.read_table(self.dataset_path + '/test_id.txt', header=None)
            for l in range(len(val_file)):
                line = val_file.iloc[l, 0]
                text = line.split(" ")
                text = list(map(lambda x: int(x), text))
                uid = int(text[0])
                num_users = max(num_users, uid)
                items = text[1:]
                val_sum = val_sum + len(items)
                num_items = max(num_items, max(items))
                val_U2I[uid].extend(items)
            for l in range(len(test_iid_file)):
                line = test_iid_file.iloc[l, 0]
                text = line.split(" ")
                text = list(map(lambda x: int(x), text))
                uid = int(text[0])
                num_users = max(num_users, uid)
                items = text[1:]
                test_iid_sum = test_iid_sum + len(items)
                num_items = max(num_items, max(items))
                test_iid_U2I[uid].extend(items)
        elif self.dataset_name=='meituan' or self.dataset_name=='douban.new' or self.dataset_name=='coat' or self.dataset_name=='yahoo.new':
            val_U2I=defaultdict(list)
            val_sum=0
            val_file=pd.read_table(self.dataset_path + '/valid.txt', header=None)
            for l in range(len(val_file)):
                line = val_file.iloc[l, 0]
                text = line.split(" ")
                text = list(map(lambda x: int(x), text))
                uid = int(text[0])
                num_users = max(num_users, uid)
                items = text[1:]
                val_sum = val_sum + len(items)
                num_items = max(num_items, max(items))
                val_U2I[uid].extend(items)
            test_iid_U2I=test_U2I
            test_iid_sum=0
        else:
            val_U2I=test_U2I
            test_iid_U2I=test_U2I
            val_sum,test_iid_sum=0,0
        # 统计信息
        num_users = num_users + 1
        num_items = num_items + 1

        # 训练数据[user,item]
        training_data = [
            val for val in training_data for i in range(self.num_neg)]

        # 获得item的分布
        ps = pd.Series(train_interation)
        vc = ps.value_counts(sort=False)
        vc.sort_index(inplace=True)
        pop_train = []

        if num_items == len(np.unique(np.array(train_interation))):
            for item in range(num_items):
                pop_train.append(vc[item])
        else:
            for item in range(num_items):
                if item not in list(vc.index):
                    pop_train.append(0)
                else:
                    pop_train.append(vc[item])

        return num_users, num_items, train_U2I, training_data, test_U2I, pop_train, train_num+test_num+val_sum+test_iid_sum,test_I2U,train_I2U,val_U2I,test_iid_U2I,val_sum,test_iid_sum
    
    def user_items_2_group(self):
        G1=[]
        G2=[]
        random.seed(time.time())
        for u in self.train_U2I.keys():
                items=self.train_U2I[u]
                if len(items)%2!=0:
                    items=np.delete(items,random.sample(range(len(items)),1)[0])
                np.random.shuffle(items)
                num=int(len(items)/2)
                G1.extend(items[0:num])
                G2.extend(items[num:])
        return np.array(G1),np.array(G2)
    
    def bc_loss_data(self):
        pop_user = {key: len(value) for key, value in self.train_U2I.items()}
        pop_item = {key: len(value) for key, value in self.train_I2U.items()}
        sorted_pop_user = list(set(list(pop_user.values())))
        sorted_pop_item = list(set(list(pop_item.values())))
        sorted_pop_user.sort()
        sorted_pop_item.sort()
        self.n_user_pop = len(sorted_pop_user)
        self.n_item_pop = len(sorted_pop_item)
        user_idx = {}
        item_idx = {}
        for i, item in enumerate(sorted_pop_user):
            user_idx[item] = i
        for i, item in enumerate(sorted_pop_item):
            item_idx[item] = i
        self.user_pop_idx = np.zeros(self.num_users, dtype=int)
        self.item_pop_idx = np.zeros(self.num_items, dtype=int)
        for key, value in pop_user.items():
            self.user_pop_idx[key] = user_idx[value]
        for key, value in pop_item.items():
            self.item_pop_idx[key] = item_idx[value]
        # user_pop_max = max(self.user_pop_idx)
        # item_pop_max = max(self.item_pop_idx)

        # self.user_pop_max = user_pop_max
        # self.item_pop_max = item_pop_max  
    


    def split_data_2(self):
        if self.dataset_name=='ml-1m':
            split=[50]
        elif self.dataset_name=='Yelp2018':
            split=[10]
        elif self.dataset_name=='douban-book':
            split=[22]
        elif self.dataset_name=='gowalla':
            split=[22]
        elif self.dataset_name=='amazon-book':
            split=[33]
        elif self.dataset_name=='ml-20m':
            split=[100]
        elif self.dataset_name=='addressa':
            split=[30]

        unpopular_item=[]
        popular_item=[]
        for item,pop in enumerate(self.pop_train_count):
            if pop <=split[0]:
                unpopular_item.append(item)
            else:
                popular_item.append(item)
        return unpopular_item,popular_item
    
    def get_weight(self):

        pop = self.pop_train_count
        pop = np.clip(pop, 1, max(pop))
        pop = pop / np.linalg.norm(pop, ord=np.inf)
        pop = 1 / pop
        pop = np.clip(pop, 1, np.median(pop))
        pop = pop / np.linalg.norm(pop, ord=np.inf)

        return pop

    
    def split_test(self):
        unpopular_test=np.load(self.dataset_path+'/unpopular_test.npy',allow_pickle=True).tolist()
        norm_test=np.load(self.dataset_path+'/norm_test.npy',allow_pickle=True).tolist()
        popular_test=np.load(self.dataset_path+'/popular_test.npy',allow_pickle=True).tolist()
        return unpopular_test,norm_test,popular_test
    
    def split_test_2(self):
        unpopular_test=np.load(self.dataset_path+'/unpopular_test_8.npy',allow_pickle=True).tolist()
        popular_test=np.load(self.dataset_path+'/popular_test_2.npy',allow_pickle=True).tolist()
        return unpopular_test,popular_test

    def pri_list(self):
        test_U2I=self.test_U2I
        pri_dict= defaultdict(set)
        for user in test_U2I.keys():
            for item in test_U2I[user]:
                pri_dict[item].add(user)
        
        return pri_dict

    def split_user_pop(self):
        user_pop=[]
        for user in range(self.num_users):
            user_pop.append(len(self.train_U2I[user]))
        if self.dataset_name=='ml-1m':
            split=[45]
        elif self.dataset_name=='douban-book':
            split=[80]
        elif self.dataset_name=='amazon-book':
            split=[50]
        elif self.dataset_name=='yelp2018':
            split=[100]
        elif self.dataset_name=='ml-20m':
            split=[30]
        elif self.dataset_name=='gowalla':
            split=[20]
        elif self.dataset_name=='adressa':
            split=[10]
        
        unpopular_users,popular_users=[],[]

        for user,pop in enumerate(user_pop):
            if pop <=split[0]:
                unpopular_users.append(user)
            else:
                popular_users.append(user)
        return user_pop,unpopular_users,popular_users


    def split_user_pop_5(self):
        user_pop=[]
        for user in range(self.num_users):
            user_pop.append(len(self.train_U2I[user]))
        
        if self.dataset_name=='amazon-book' or self.dataset_name=='ml-1m':
            split_list=[25,45,85,180]
        elif self.dataset_name=='douban-book':
            split_list=[33,64,105,184]
        pop_cnt=np.array(user_pop)
        tmp = np.arange(len(pop_cnt))

        unpopular_user_1 = list(tmp[pop_cnt < split_list[0]])

        unpopular_user_2=list(tmp[pop_cnt < split_list[1]])
        unpopular_user_2=list(set(unpopular_user_2) - set(unpopular_user_1))

        unpopular_user_3=list(tmp[pop_cnt < split_list[2]])
        unpopular_user_3=list(set(unpopular_user_3) - set(unpopular_user_1)-set(unpopular_user_2))

        unpopular_user_4=list(tmp[pop_cnt < split_list[3]])                                                                                                        
        unpopular_user_4=list(set(unpopular_user_4) - set(unpopular_user_1)-set(unpopular_user_2)-set(unpopular_user_3))

        unpopular_user_5=list(set(tmp)-set(unpopular_user_4) - set(unpopular_user_1)-set(unpopular_user_2)-set(unpopular_user_3))


        return user_pop,unpopular_user_1,unpopular_user_2,unpopular_user_3,unpopular_user_4,unpopular_user_5

    def split_item_pop_5(self):
        user_pop=self.pop_train_count
        
        
        if self.dataset_name=='amazon-book':
            split_list=[16,30,52,105]
        elif self.dataset_name=='douban-book':
            split_list=[13,36,93,265]
        elif self.dataset_name=='ml-1m':
            split_list=[5,10,20,30]
        pop_cnt=np.array(user_pop)
        tmp = np.arange(len(pop_cnt))

        unpopular_user_1 = list(tmp[pop_cnt < split_list[0]])

        unpopular_user_2=list(tmp[pop_cnt < split_list[1]])
        unpopular_user_2=list(set(unpopular_user_2) - set(unpopular_user_1))

        unpopular_user_3=list(tmp[pop_cnt < split_list[2]])
        unpopular_user_3=list(set(unpopular_user_3) - set(unpopular_user_1)-set(unpopular_user_2))

        unpopular_user_4=list(tmp[pop_cnt < split_list[3]])                                                                                                        
        unpopular_user_4=list(set(unpopular_user_4) - set(unpopular_user_1)-set(unpopular_user_2)-set(unpopular_user_3))

        unpopular_user_5=list(set(tmp)-set(unpopular_user_4) - set(unpopular_user_1)-set(unpopular_user_2)-set(unpopular_user_3))


        return unpopular_user_1,unpopular_user_2,unpopular_user_3,unpopular_user_4,unpopular_user_5

        
    
    
    
    

class Graph(object):

    def __init__(self, num_users, num_items, train_U2I,gama):
        self.num_users = num_users
        self.num_items = num_items
        self.train_U2I = train_U2I
        self.gama=gama

    def to_edge(self):
        # 得到图的对应边的权重和索引二元组
        train_U, train_I = [], []

        for u, items in self.train_U2I.items():
            train_U.extend([u] * len(items))
            train_I.extend(items)

        train_U = np.array(train_U)
        train_I = np.array(train_I)

        row = np.concatenate([train_U, train_I + self.num_users])
        col = np.concatenate([train_I + self.num_users, train_U])

        edge_weight = np.ones_like(row).tolist()
        # 列表里面包括两个列表，分别对应的是u-i和i-u，train——U2I所有有交互记录的edge都为1
        edge_index = np.stack([row, col]).tolist()

        return train_U, train_I, edge_index, edge_weight


class LaplaceGraph(Graph):

    def __init__(self, num_users, num_items, train_U2I,gama=0.5):
        Graph.__init__(self, num_users, num_items, train_U2I,gama)

    def generate(self):
        graph_u, graph_i, edge_index, edge_weight = self.to_edge()
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
        edge_index, edge_weight = self.add_self_loop(edge_index, edge_weight)
        original_adj = self.mat(edge_index, edge_weight)
        edge_index, edge_weight = self.norm(edge_index, edge_weight)
        norm_adj = self.mat(edge_index, edge_weight)
        return  norm_adj

    # 增加自己对自己的一个影响；【0，0】【1，1】这种
    def add_self_loop(self, edge_index, edge_weight):
        """ add self-loop """
        # 【0：num_user+num_item-1]的一个列表
        loop_index = torch.arange(0, self.num_nodes, dtype=torch.long)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        loop_weight = torch.ones(self.num_nodes, dtype=torch.float32)
        edge_index = torch.cat([edge_index, loop_index], dim=-1)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=-1)

        return edge_index, edge_weight

    # 归一化上面的数据
    def norm(self, edge_index, edge_weight):
        """ D^{-1/2} * A * D^{-1/2}"""

        row, col = edge_index[0], edge_index[1]
        # 计算图中的度
        deg = torch.zeros(self.num_nodes, dtype=torch.float32)
        deg = deg.scatter_add(0, col, edge_weight)
        deg_inv_sqrt = deg.pow(-1*self.gama)
        # masked_fill_(mask, value) 用value填充tensor中与mask中值为1位置相对应的元素。mask的形状必须与要填充的tensor形状一致
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)

        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        return edge_index, edge_weight

    @property
    def num_nodes(self):
        return self.num_users + self.num_items

    def mat(self, edge_index, edge_weight):
        return torch.sparse.FloatTensor(edge_index, edge_weight, torch.Size([self.num_nodes, self.num_nodes]))

from random import shuffle,choice


def next_batch_pairwise(data,batch_size):
    training_data = data.training_data
    shuffle(training_data)
    batch_id = 0
    data_size = len(training_data)
    while batch_id < data_size:
        if batch_id + batch_size <= data_size:
            users = [training_data[idx][0] for idx in range(batch_id, batch_size + batch_id)]
            items = [training_data[idx][1] for idx in range(batch_id, batch_size + batch_id)]
            batch_id += batch_size
        else:
            users = [training_data[idx][0] for idx in range(batch_id, data_size)]
            items = [training_data[idx][1] for idx in range(batch_id, data_size)]
            batch_id = data_size
        u_idx, i_idx, j_idx = [], [], []
        item_list = list(range(data.num_items))
        for i, user in enumerate(users):
            i_idx.append(items[i])
            u_idx.append(user)
            neg_item = choice(item_list)
            while neg_item in data.train_U2I[user]:
                neg_item = choice(item_list)
            j_idx.append(neg_item)
        yield u_idx, i_idx, j_idx


def next_batch_pairwise_OS_item(data,batch_size):
    
    training_data = data.training_data
    shuffle(training_data)
    batch_id = 0
    data_size = len(training_data)
    while batch_id < data_size:
        if batch_id + batch_size <= data_size:
            users = [training_data[idx][0] for idx in range(batch_id, batch_size + batch_id)]
            items = [training_data[idx][1] for idx in range(batch_id, batch_size + batch_id)]
            batch_id += batch_size
        else:
            users = [training_data[idx][0] for idx in range(batch_id, data_size)]
            items = [training_data[idx][1] for idx in range(batch_id, data_size)]
            batch_id = data_size
        u_idx, i_idx, j_idx = [], [], []
        item_list = list(range(data.num_items))
        for i, user in enumerate(users):
            i_idx.append(items[i])
            u_idx.append(user)
            neg_item = choice(item_list)
            while neg_item in data.train_U2I[user]:
                neg_item = choice(item_list)
            j_idx.append(neg_item)
        yield u_idx, i_idx, j_idx


def next_batch_user_item(data,batch_size):
    users = list(data.train_U2I.keys())
    shuffle(users)
    batch_id = 0
    data_size = len(users)
    while batch_id < data_size:
        if batch_id + batch_size <= data_size:
            batch_users = users[batch_id:batch_id+batch_size]
            batch_id += batch_size
        else:
            batch_users = users[batch_id:data_size]
            batch_id = data_size
        u_idx, i_idx, j_idx = [], [], []
        item_list = list(range(data.num_items))
        for user in batch_users:
            items = data.train_U2I[user]
            for item in items:
                u_idx.append(user)
                i_idx.append(item)
                neg_item = choice(item_list)
                while neg_item in items:
                    neg_item = choice(item_list)
                j_idx.append(neg_item)
            
        
        yield u_idx, i_idx, j_idx
    


def split_test(data):
    test_U2I = data.test_U2I
    pop_cnt = np.array(data.pop_train_count)
    tmp = np.arange(len(pop_cnt))
    if data.dataset_name == 'amazon-book':
        split_list = [9, 33]
    elif data.dataset_name == 'yelp2018':
        split_list = [10, 40]
    elif data.dataset_name=='gowalla':
        split_list=[10,22]
    elif data.dataset_name=='ml-1m':
        split_list=[3,50]
    elif data.dataset_name=='douban-book':
        split_list=[5,22]

    unpopular_items = list(tmp[pop_cnt < split_list[0]])
    print("unpopular group item pop: {}-{}  item numbers: {}  The total of all items is: {:.6f}".format(0,
                                                                                                        split_list[0],
                                                                                                        len(
                                                                                                            unpopular_items),
                                                                                                        len(
                                                                                                            unpopular_items) * 100 / len(
                                                                                                            pop_cnt)))
    norm_items = list(tmp[pop_cnt < split_list[1]])
    norm_items = list(set(norm_items) - set(unpopular_items))
    print("norm group item pop: {}-{}  item numbers: {}  The total of all items is: {:.6f}".format(split_list[0],
                                                                                                   split_list[1],
                                                                                                   len(norm_items),
                                                                                                   len(
                                                                                                       norm_items) * 100 / len(
                                                                                                       pop_cnt)))
    popular_items = list(set(tmp) - set(unpopular_items) - set(norm_items))
    print("popular group item pop: {}-{}  item numbers: {}  The total of all items is: {:.6f}".format(split_list[1],
                                                                                                      max(pop_cnt),
                                                                                                      len(
                                                                                                          popular_items),
                                                                                                      len(
                                                                                                          popular_items) * 100 / len(
                                                                                                          pop_cnt)))
    unpopular_test, norm_test, popular_test = defaultdict(list), defaultdict(list), defaultdict(list)
    for user in test_U2I.keys():
        for item in test_U2I[user]:
            if item in unpopular_items:
                unpopular_test[user].append(item)
            elif item in norm_items:
                norm_test[user].append(item)
            elif item in popular_items:
                popular_test[user].append(item)
    return unpopular_test, norm_test, popular_test


def split_test_8_2(data):
    test_U2I = data.test_U2I
    pop_cnt = np.array(data.pop_train_count)
    tmp = np.arange(len(pop_cnt))
    if data.dataset_name == 'amazon-book':
        split_list = [33]
    elif data.dataset_name == 'yelp2018':
        split_list = [40]
    elif data.dataset_name=='gowalla':
        split_list=[22]
    elif data.dataset_name=='ml-1m':
        split_list=[50]
    elif data.dataset_name=='douban-book':
        split_list=[22]
    elif data.dataset_name=='ml-20m':
        split_list=[60]


    unpopular_items = list(tmp[pop_cnt < split_list[0]])
    print("unpopular group item pop: {}-{}  item numbers: {}  The total of all items is: {:.6f}".format(0,
                                                                                                        split_list[0],
                                                                                                        len(
                                                                                                            unpopular_items),
                                                                                                        len(
                                                                                                            unpopular_items) * 100 / len(
                                                                                                            pop_cnt)))
   
    popular_items = list(set(tmp) - set(unpopular_items))
    print("popular group item pop: {}-{}  item numbers: {}  The total of all items is: {:.6f}".format(split_list[0],
                                                                                                      max(pop_cnt),
                                                                                                      len(
                                                                                                          popular_items),
                                                                                                      len(
                                                                                                          popular_items) * 100 / len(
                                                                                                          pop_cnt)))
    unpopular_test, popular_test = defaultdict(list), defaultdict(list)
    for user in test_U2I.keys():
        for item in test_U2I[user]:
            if item in unpopular_items:
                unpopular_test[user].append(item)
            else:
                popular_test[user].append(item)
    return unpopular_test, popular_test


def split_user_2(data):
    user_pop=[]
    for user in range(data.num_users):
        user_pop.append(len(data.train_U2I[user]))
    pop_cnt = np.array(user_pop)
    tmp = np.arange(len(pop_cnt))
    if data.dataset_name == 'amazon-book':
        split_list = [50]
    elif data.dataset_name == 'yelp2018':
        split_list = [40]
    elif data.dataset_name=='gowalla':
        split_list=[22]
    elif data.dataset_name=='ml-1m':
        split_list=[45]
    elif data.dataset_name=='douban-book':
        split_list = [33, 64, 105, 184]
    elif data.dataset_name == 'ml-20m':
        split_list = [30]

    unpopular_items = list(tmp[pop_cnt < split_list[0]])
    print("unpopular group item pop: {}-{}  item numbers: {}  The total of all items is: {:.6f}".format(0,
                                                                                                        split_list[0],
                                                                                                        len(
                                                                                                            unpopular_items),
                                                                                                        len(
                                                                                                            unpopular_items) * 100 / len(
                                                                                                            pop_cnt)))
   
    popular_items = list(set(tmp) - set(unpopular_items))
    print("popular group item pop: {}-{}  item numbers: {}  The total of all items is: {:.6f}".format(split_list[0],
                                                                                                      max(pop_cnt),
                                                                                                      len(
                                                                                                          popular_items),
                                                                                                      len(
                                                                                                          popular_items) * 100 / len(
                                                                                                          pop_cnt)))
    unpopular_test, popular_test = defaultdict(list), defaultdict(list)

    return unpopular_test,popular_test

def split_user_test_8_2(data):
    #u2i=data.train_U2I
    pop_cnt = np.array(data.user_pop)
    tmp = np.arange(len(pop_cnt))
    if data.dataset_name == 'amazon-book':
        split_list = [25,45,85,180]
    elif data.dataset_name == 'yelp2018':
        split_list = [40]
    elif data.dataset_name=='gowalla':
        split_list=[22]
    elif data.dataset_name=='ml-1m':
        split_list=[45]
    elif data.dataset_name=='douban-book':
        split_list=[33,64,105,184]
    sum_all=np.sum(pop_cnt)
    unpopular_user_1 = list(tmp[pop_cnt < split_list[0]])
    print("unpopular group 1 user pop: {}-{}  user numbers: {}  The total of all items is: {:.6f}  The total of all users is: {:.6f}".format(0,
                                                                                                        split_list[0],
                                                                                                        len(
                                                                                                            unpopular_user_1),
                                                                                                        np.sum(pop_cnt[unpopular_user_1]) * 100 / sum_all,len(
                                                                                                          unpopular_user_1) * 100 / len(
                                                                                                          pop_cnt)))
                                                                                           
    unpopular_user_2=list(tmp[pop_cnt < split_list[1]])

    unpopular_user_2=list(set(unpopular_user_2) - set(unpopular_user_1))
    print("unpopular group 2 user pop: {}-{}  user numbers: {}  The total of all items is: {:.6f}  The total of all users is: {:.6f}".format(split_list[0],split_list[1],
                                                                                                        len(
                                                                                                            unpopular_user_2),
                                                                                                        np.sum(pop_cnt[unpopular_user_2]) * 100 / sum_all,len(
                                                                                                          unpopular_user_2) * 100 / len(
                                                                                                          pop_cnt)))      
                                                                                                             
    unpopular_user_3=list(tmp[pop_cnt < split_list[2]])

    unpopular_user_3=list(set(unpopular_user_3) - set(unpopular_user_1)-set(unpopular_user_2))
    print("unpopular group 3 user pop: {}-{}  user numbers: {}  The total of all items is: {:.6f}  The total of all users is: {:.6f}".format(split_list[1],split_list[2],
                                                                                                        len(
                                                                                                            unpopular_user_3),
                                                                                                        np.sum(pop_cnt[unpopular_user_3]) * 100 / sum_all,len(
                                                                                                          unpopular_user_3) * 100 / len(
                                                                                                          pop_cnt)))    
    unpopular_user_4=list(tmp[pop_cnt < split_list[3]])                                                                                                        
    unpopular_user_4=list(set(unpopular_user_4) - set(unpopular_user_1)-set(unpopular_user_2)-set(unpopular_user_3))
    print("unpopular group 4 user pop: {}-{}  user numbers: {}  The total of all items is: {:.6f}  The total of all users is: {:.6f}".format(split_list[2],split_list[3],
                                                                                                        len(
                                                                                                            unpopular_user_4),
                                                                                                        np.sum(pop_cnt[unpopular_user_4]) * 100 / sum_all,len(
                                                                                                          unpopular_user_4) * 100 / len(
                                                                                                          pop_cnt)))                                                                                                       
                                                                                                
    unpopular_user_5=list(set(tmp)-set(unpopular_user_4) - set(unpopular_user_1)-set(unpopular_user_2)-set(unpopular_user_3))
    print("unpopular group 5 user pop: {}-{}  user numbers: {}  The total of all items is: {:.6f}  The total of all users is: {:.6f}".format(split_list[3],max(pop_cnt),
                                                                                                        len(
                                                                                                            unpopular_user_5),
                                                                                                        np.sum(pop_cnt[unpopular_user_5]) * 100 / sum_all,len(
                                                                                                          unpopular_user_5) * 100 / len(
                                                                                                          pop_cnt)))   

                                                                                                        
    return 0

def split_item_test_5(data):
    #u2i=data.train_U2I
    #test_U2I = data.test_U2I
    pop_cnt = np.array(data.pop_train_count)
    tmp = np.arange(len(pop_cnt))
    if data.dataset_name == 'amazon-book':
        split_list = [16,30,52,105]
    elif data.dataset_name=='douban-book':
        split_list=[13,36,93,265]
    sum_all=np.sum(pop_cnt)
    unpopular_user_1 = list(tmp[pop_cnt < split_list[0]])
    print("unpopular group 1 item pop: {}-{}  item numbers: {}  The total of all item is: {:.6f}  The total of all item is: {:.6f}".format(0,
                                                                                                        split_list[0],
                                                                                                        len(
                                                                                                            unpopular_user_1),
                                                                                                        np.sum(pop_cnt[unpopular_user_1]) * 100 / sum_all,len(
                                                                                                          unpopular_user_1) * 100 / len(
                                                                                                          pop_cnt)))
                                                                                           
    unpopular_user_2=list(tmp[pop_cnt < split_list[1]])

    unpopular_user_2=list(set(unpopular_user_2) - set(unpopular_user_1))
    print("unpopular group 2 item pop: {}-{}  useitemr numbers: {}  The total of all item is: {:.6f}  The total of all item is: {:.6f}".format(split_list[0],split_list[1],
                                                                                                        len(
                                                                                                            unpopular_user_2),
                                                                                                        np.sum(pop_cnt[unpopular_user_2]) * 100 / sum_all,len(
                                                                                                          unpopular_user_2) * 100 / len(
                                                                                                          pop_cnt)))      
                                                                                                             
    unpopular_user_3=list(tmp[pop_cnt < split_list[2]])

    unpopular_user_3=list(set(unpopular_user_3) - set(unpopular_user_1)-set(unpopular_user_2))
    print("unpopular group 3 item pop: {}-{}  item numbers: {}  The total of all item is: {:.6f}  The total of all item is: {:.6f}".format(split_list[1],split_list[2],
                                                                                                        len(
                                                                                                            unpopular_user_3),
                                                                                                        np.sum(pop_cnt[unpopular_user_3]) * 100 / sum_all,len(
                                                                                                          unpopular_user_3) * 100 / len(
                                                                                                          pop_cnt)))    
    unpopular_user_4=list(tmp[pop_cnt < split_list[3]])                                                                                                        
    unpopular_user_4=list(set(unpopular_user_4) - set(unpopular_user_1)-set(unpopular_user_2)-set(unpopular_user_3))
    print("unpopular group 4 item pop: {}-{}  item numbers: {}  The total of all item is: {:.6f}  The total of all item is: {:.6f}".format(split_list[2],split_list[3],
                                                                                                        len(
                                                                                                            unpopular_user_4),
                                                                                                        np.sum(pop_cnt[unpopular_user_4]) * 100 / sum_all,len(
                                                                                                          unpopular_user_4) * 100 / len(
                                                                                                          pop_cnt)))                                                                                                       
                                                                                                
    unpopular_user_5=list(set(tmp)-set(unpopular_user_4) - set(unpopular_user_1)-set(unpopular_user_2)-set(unpopular_user_3))
    print("unpopular group 5 item pop: {}-{}  item numbers: {}  The total of all item is: {:.6f}  The total of all item  is: {:.6f}".format(split_list[3],max(pop_cnt),
                                                                                                        len(
                                                                                                            unpopular_user_5),
                                                                                                        np.sum(pop_cnt[unpopular_user_5]) * 100 / sum_all,len(
                                                                                                          unpopular_user_5) * 100 / len(
                                                                                                          pop_cnt)))   

                                                                                                        
    return 0




def batch_user_item_2_group(batch_user,data):
    G1=[]
    G2=[]
    random.seed(time.time())
    for u in batch_user:
            items=data.train_U2I[u]
            if len(items)%2!=0:
                items=np.delete(items,random.sample(range(len(items)),1)[0])
            np.random.shuffle(items)
            num=int(len(items)/2)
            G1.extend(items[0:num])
            G2.extend(items[num:])
    return np.array(G1),np.array(G2)

def batch_item_usetr_2_group(batch_item,data):
    G1=[]
    G2=[]
    random.seed(time.time())
    batch_item=list(np.unique(batch_item))
    for i in batch_item:
            item=i+data.num_users
            items=np.array(data.norm_adj[item].coalesce().indices()[0])
            items=np.delete(items,-1)
            if len(items)%2!=0:
                items=np.delete(items,random.sample(range(len(items)),1)[0])
            np.random.shuffle(items)
            num=int(len(items)/2)
            G1.extend(items[0:num])
            G2.extend(items[num:])
    return np.array(G1),np.array(G2)


def item_user_2_group(data):
    G1=[]
    G2=[]
    random.seed(time.time())
    for i in range(data.num_items):
            item=i+data.num_users
            items=np.array(data.norm_adj[item].coalesce().indices()[0])
            items=np.delete(items,-1)
            if len(items)%2!=0:
                items=np.delete(items,random.sample(range(len(items)),1)[0])
            np.random.shuffle(items)
            num=int(len(items)/2)
            G1.extend(items[0:num])
            G2.extend(items[num:])
    return np.array(G1),np.array(G2)

def user_items_2_group_sup(scores,train_U2I):
    G1,G2=[],[]
    for u in train_U2I.keys():
        items=train_U2I[u]
        items_sorted=list(np.array(items)[np.argsort(scores[u][items])])
        if len(items)%2!=0:
            items_sorted=np.delete(items_sorted,random.sample(range(len(items_sorted)),1)[0])
        num=int(len(items_sorted)/2)
        G1.extend(items_sorted[0:num])
        G2.extend(items_sorted[num:])
    return np.array(G1),np.array(G2)

def item_user_2_group_sup(scores,data):
    G1,G2=[],[]
    for i in range(data.num_items):
        item=i+data.num_users
        items=np.array(data.norm_adj[item].coalesce().indices()[0])
        items=np.delete(items,-1)
        items_sorted=items[np.argsort(scores[:,i][items])]
        if len(items)%2!=0:
            items_sorted=np.delete(items_sorted,random.sample(range(len(items_sorted)),1)[0])
        num=int(len(items_sorted)/2)
        G1.extend(items_sorted[0:num])
        G2.extend(items_sorted[num:])
    return np.array(G1),np.array(G2)


def user_items_2_group_pop(data):
    G1,G2=[],[]
    for u in data.train_U2I.keys():
        items=data.train_U2I[u]
        items_sorted=list(np.array(items)[np.argsort(np.array(data.pop_train_count)[items])])
        if len(items)%2!=0:
            items_sorted=np.delete(items_sorted,random.sample(range(len(items_sorted)),1)[0])
        num=int(len(items_sorted)/2)
        G1.extend(items_sorted[0:num])
        G2.extend(items_sorted[num:])
    return np.array(G1),np.array(G2)



def split_item_2(data):
    split=[10]
    pop_lable=[]
    pop_num,unpop_num=0,0
    for item,pop in enumerate(data.pop_train_count):
        if pop<=split[0]:
            pop_lable.append(0)
            unpop_num=unpop_num+1
        else:
            pop_lable.append(1)
            pop_num=pop_num+1
    print(pop_num,unpop_num)
    return pop_lable
