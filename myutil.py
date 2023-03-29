import torch
import pickle
import numpy as np
import sklearn.metrics as sk_m
from torch.utils.data import Dataset
import time

def encode_EC(EC_NUMBER, conversion_dictionaries):
    ECa_to_hot, ECb_to_hot, ECc_to_hot = conversion_dictionaries
    A, B, C, _ = [int(i) for i in EC_NUMBER.split('.')]
    veca, vecb, vecc = [field[value] for field, value in zip([ECa_to_hot, ECb_to_hot, ECc_to_hot], [A,B,C])]
    return np.concatenate([veca, vecb, vecc])

def decode_EC(EC_hot_vec, conversion_dictionaries, lentuple):
    len_ECa, len_ECb, len_ECc = lentuple
    hot_to_ECa, hot_to_ECb, hot_to_ECc = conversion_dictionaries
    veca, vecb, vecc = [EC_hot_vec[:len_ECa], EC_hot_vec[len_ECa:len_ECa+len_ECb], EC_hot_vec[len_ECa+len_ECb:]]
    integers = [field[tuple(value)] for field, value in zip([hot_to_ECa, hot_to_ECb, hot_to_ECc],[veca, vecb, vecc])] 
    return ".".join([str(i) for i in integers])+".-"

def encode_KOs(kos, ko_to_pos, num_ko):
    enc = np.zeros(num_ko)
    for ko in kos:
        enc[ko_to_pos[ko]] = 1
    return enc

def decode_KOs(vector,pos_to_ko):
    return [pos_to_ko[index[0]] for index in np.where(vector==1)]

# def legacy_load_interaction_data():
#     with open('./data/interaction.pkl', 'rb') as fi:
#         data = pickle.load(fi)
#     tr_p, va_p, te_p, va_pn, te_pn, n_all_exclusive, num_compound, num_enzyme, compound_i2n, \
#     enzyme_i2n, fp_label, ec_label = data['tr_p'], data['va_p'], data['te_p'], data['va_pn'], data['te_pn'], data['n_all_exclusive'],\
#                                      data['num_compound'], data['num_enzyme'], data['compound_i2n'], data['enzyme_i2n'], data['fp_label'], data['ec_label']
#     return tr_p, va_p, te_p, va_pn, te_pn, n_all_exclusive, num_compound, num_enzyme, compound_i2n, enzyme_i2n, fp_label, ec_label

def first_load_interaction_data():
    with open('./data/interaction.pkl', 'rb') as fi:
        data = pickle.load(fi)

        tr_p               = data['tr_p']
        va_p               = data['va_p']                  # TODO: where need these?
        te_p               = data['te_p']                  # TODO: where need this?
        n_all_exclusive    = data['n_all_exclusive']
        va_pn              = data['va_pn']
        te_pn              = data['te_pn']
        num_compound       = data['num_compound']
        num_enzyme         = data['num_enzyme']
        fp_label           = data['fp_label']
        ec_label           = data['ec_label']
        len_EC_fields      = data['len_EC_fields']
        EC_to_hot_dicts    = data['EC_to_hot_dicts']
        hot_to_EC_dicts    = data['hot_to_EC_dicts']
        pos_to_ko_dict     = data['pos_to_ko_dict']
        ko_to_pos_dict     = data['ko_to_pos_dict']
        num_ko             = data['num_ko']
        rpairs_pos         = data['rpairs_pos']
        enzyme_ko_hot      = data['enzyme_ko_hot']
    
    return tr_p, va_p, te_p, n_all_exclusive, va_pn, te_pn, num_compound, num_enzyme,\
        fp_label, ec_label, len_EC_fields, EC_to_hot_dicts, hot_to_EC_dicts, \
        pos_to_ko_dict,ko_to_pos_dict,num_ko,rpairs_pos,enzyme_ko_hot

def load_interaction_data():
    with open('./data/interaction.pkl', 'rb') as fi:
        data = pickle.load(fi)

        pairs              = data['pairs']
        num_compound       = data['num_compound']
        num_enzyme         = data['num_enzyme']
        fp_label           = data['fp_label']
        ec_label           = data['ec_label']
        len_EC_fields      = data['len_EC_fields']
        EC_to_hot_dicts    = data['EC_to_hot_dicts']
        hot_to_EC_dicts    = data['hot_to_EC_dicts']
        pos_to_ko_dict     = data['pos_to_ko_dict']
        ko_to_pos_dict     = data['ko_to_pos_dict']
        num_ko             = data['num_ko']
        rpairs_pos         = data['rpairs_pos']         # TODO: I don't think needed any more
        CC_dict            = data['CC_dict']
        enzyme_ko_hot      = data['enzyme_ko_hot']
    
    return pairs, num_compound, num_enzyme,\
        fp_label, ec_label, len_EC_fields, EC_to_hot_dicts, hot_to_EC_dicts, \
        pos_to_ko_dict,ko_to_pos_dict,num_ko,rpairs_pos,CC_dict,enzyme_ko_hot

# def legacy_load_mt_data():
#     with open('./data/auxiliary.pkl', 'rb') as fi:
#         data = pickle.load(fi)
#     rpairs_pos, cpd_module, cpd_pathway, enzyme_ko, enzyme_ko_hot, enzyme_module, enzyme_pathway = data['rpairs_pos'], data['cpd_module'], data['cpd_pathway'],\
#                                           data['enzyme_ko'], data['enzyme_ko_hot'], data['enzyme_module'], data['enzyme_pathway']
#     return rpairs_pos, cpd_module, cpd_pathway, enzyme_ko, enzyme_ko_hot, enzyme_module, enzyme_pathway

# def load_mt_data():
#     with open('./data/auxiliary.pkl', 'rb') as fi:
#         data = pickle.load(fi)
#     rpairs_pos, enzyme_ko, enzyme_ko_hot = data['rpairs_pos'], data['enzyme_ko'], data['enzyme_ko_hot']
#     return rpairs_pos, enzyme_ko, enzyme_ko_hot

# TODO: IMPLEMENT THIS AS A FLAG INSTEAD OF TWO DIFFERENT CLASSES (if train or testset I mean)
class TrainDataset(Dataset):
    # apart from interactions, also samples CC (anchor, pos, neg) triplets
    def __init__(self, data, CC_dict, num_compound, device):
        self.data = data            # compound enzyme {0,1}
        self.CC = CC_dict           # anchor : {positive1, positive2} (have cc relation)
        self.num_cpd = num_compound # total number of compounds in dataset
        self.device = device

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        es_interaction = self.data[index].unsqueeze(0)
        # TODO: check syntax now that just single point
        compound_id = es_interaction[:,0].to(self.device)
        enzyme_id   = es_interaction[:,1].to(self.device)
        # get all (anchor, positive) cc relations for this compound
        # NOTE: the original had 25x the positive samples and 25x sampled negative ones
        positives = torch.Tensor(list(self.CC[int(compound_id)]))
        anchors = torch.Tensor(np.repeat(int(compound_id), len(positives)))
        negatives = torch.from_numpy(np.random.choice(np.arange(self.num_cpd), len(positives)))
        triplets = torch.stack([anchors, positives, negatives], dim=1).long().to(self.device)
        return es_interaction, compound_id, enzyme_id, triplets

class TrainDatasetPairTable(Dataset):
    # apart from interactions, also samples CC (anchor, pos, neg) triplets
    # old implementation without hash table
    def __init__(self, data, rpairs, num_compound, device):
        self.data = data            # compound enzyme {0,1}
        self.rpairs = rpairs        # anchor positive (have cc relation)
        self.num_cpd = num_compound # total number of compounds in dataset
        self.device = device

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        es_interaction = self.data[index].unsqueeze(0)
        # TODO: check syntax now that just single point
        compound_id = es_interaction[:,0].to(self.device)
        enzyme_id   = es_interaction[:,1].to(self.device)
        # get all (anchor, positive) cc relations for this compound
        # TODO: re-implement with hash table
        mask = (self.rpairs[:,0].unsqueeze(1) == compound_id).any(dim=1)
        indices = torch.nonzero(mask)[:, 0]
        relevant_rpairs = self.rpairs[indices]
        # sample a (negative) counterpart compound for each -> (anchor pos neg)
        # NOTE: the original had 25x the positive samples and 25x sampled negative ones
        negs = np.random.choice(np.arange(self.num_cpd), relevant_rpairs.shape[0])
        negs = torch.LongTensor(negs).unsqueeze(-1)
        triplets = torch.cat((relevant_rpairs, negs), dim=-1).to(self.device)
        return es_interaction, compound_id, enzyme_id, triplets
    
class TestDataset(Dataset):
    # does not return CC triplets
    def __init__(self, data, device):
        self.data = data            # compound enzyme {0,1}
        self.device = device

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        es_interaction = self.data[index].unsqueeze(0)
        # TODO: check syntax now that just single point
        compound_id = es_interaction[:,0].to(self.device)
        enzyme_id   = es_interaction[:,1].to(self.device)
        return es_interaction, compound_id, enzyme_id


# old custom dataset with active negative sampling of es-interactions
class CustomDatasetSample(Dataset):
    def __init__(self, data, neg_data, rpairs, num_compound, neg_rate, device):
        self.data = data            # compound enzyme 1
        self.neg_data = neg_data    # compound enzyme 0
        self.rpairs = rpairs        # anchor positive (have cc relation)
        self.neg_rate = neg_rate    # ratio neg / pos enzyme-substr interactions
        self.num_cpd = num_compound # total number of compounds in dataset
        self.device = device

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        # get one positive enzyme-substrate interaction
        pos_interaction = self.data[index].unsqueeze(0)
        # sample several (assumed) negative ones
        neg_index = np.random.choice(np.arange(self.neg_data.shape[0]), pos_interaction.shape[0]*self.neg_rate)
        neg_interactions = self.neg_data[neg_index]
        # create data object
        obj = torch.cat([pos_interaction, neg_interactions], dim=0).to(self.device)
        obj_compound_ids = obj[:, 0].to(self.device)
        obj_enzyme_ids = obj[:, 1].to(self.device)
        # also get all (anchor, positive) cc relations for the compounds involved
        mask = (self.rpairs[:,0].unsqueeze(1) == obj_compound_ids).any(dim=1)
        indices = torch.nonzero(mask)[:, 0]
        relevant_rpairs = self.rpairs[indices]
        # and sample a (negative) counterpart compound for each -> (anchor pos neg)
        # NOTE: the original had 25x the positive samples and 25x sampled negative ones
        negs = np.random.choice(np.arange(self.num_cpd), relevant_rpairs.shape[0])
        negs = torch.LongTensor(negs).unsqueeze(-1)
        triplets = torch.cat((relevant_rpairs, negs), dim=-1).to(self.device)
        return obj, obj_compound_ids, obj_enzyme_ids, triplets
        
def custom_collate_triplets(batch):
    obj, obj_compound_ids, obj_enzyme_ids, triplets = [], [], [], []
    for (obj_, obj_compound_ids_, obj_enzyme_ids_, triplets_) in batch:
        obj.append(obj_)
        obj_compound_ids.append(obj_compound_ids_)
        obj_enzyme_ids.append(obj_enzyme_ids_)
        triplets.append(triplets_)
    obj = torch.cat(obj, dim=0)
    obj_compound_ids = torch.cat(obj_compound_ids, dim=0)
    obj_enzyme_ids = torch.cat(obj_enzyme_ids, dim=0)
    triplets = torch.cat(triplets, dim=0)
    return obj, obj_compound_ids, obj_enzyme_ids, triplets 

def custom_collate_no_triplets(batch):
    obj, obj_compound_ids, obj_enzyme_ids, = [], [], []
    for (obj_, obj_compound_ids_, obj_enzyme_ids_) in batch:
        obj.append(obj_)
        obj_compound_ids.append(obj_compound_ids_)
        obj_enzyme_ids.append(obj_enzyme_ids_)
    obj = torch.cat(obj, dim=0)
    obj_compound_ids = torch.cat(obj_compound_ids, dim=0)
    obj_enzyme_ids = torch.cat(obj_enzyme_ids, dim=0)
    return obj, obj_compound_ids, obj_enzyme_ids 

def report_metric(num_compound, num_enzyme, true_interaction, pred_interaction, te_pn):
    metric = {}

    # compute map
    def map(n, dim, k=None):
        rst = []
        for i in range(n):
            indices = te_pn[:, dim] == i
            if indices.sum() == 0: continue
            x = true_interaction[indices]
            y = pred_interaction[indices]
            if k is not None:
                y_sorted_indices = np.argsort(-y)[:k]
                x = x[y_sorted_indices]
                y = y[y_sorted_indices]
                if x.sum() == 0:
                    rst.append(0)
                    continue
            rst.append(sk_m.average_precision_score(y_true=x, y_score=y))
        rst = (np.mean(rst), np.std(rst) / np.sqrt(len(rst)))
        return rst
    metric['compound_map'] = map(num_compound, 0, k=None)
    metric['enzyme_map'] = map(num_enzyme, 1, k=None)

    metric['compound_map_3'] = map(num_compound, 0, k=3)
    metric['enzyme_map_3'] = map(num_enzyme, 1, k=3)

    # compute r precision and precision@k(1, 3)
    def precision(n, k=None, dim=None):
        def h(x, y):
            m_true = int(x.sum()) if k is None else k
            if m_true == 0: return -1

            xy = np.vstack([x, y]).T
            xy_sorted_indices = np.argsort(-xy[:, 1])
            xy = xy[xy_sorted_indices]

            z = xy[:m_true, 0].sum() / m_true

            return z

        rst = []
        if dim is None:
            x = true_interaction
            y = pred_interaction
            z = h(x, y)
            if z != -1:
                rst.append(z)
        else:
            for i in range(n):
                indices = te_pn[:, dim] == i
                if indices.sum() == 0: continue
                x = true_interaction[indices]
                y = pred_interaction[indices]
                z = h(x, y)
                if z != -1:
                    rst.append(z)
        if k is None and dim is None:
            rst = np.mean(rst)
        else:
            rst = (np.mean(rst), np.std(rst) / np.sqrt(len(rst)))
        return rst

    metric['compound_rprecision'] = precision(num_compound, k=None, dim=0)
    metric['enzyme_rprecision'] = precision(num_enzyme, k=None, dim=1)
    metric['rprecision'] = precision(num_enzyme, k=None, dim=None)

    metric['compound_precision_1'] = precision(num_compound, k=1, dim=0)
    metric['enzyme_precision_1'] = precision(num_enzyme, k=1, dim=1)

    return metric
