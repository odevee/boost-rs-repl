import copy
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score
import sklearn.metrics as sk_m
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from myutil import load_interaction_data, report_metric, TrainDataset, TestDataset, custom_collate_triplets, custom_collate_no_triplets

from sklearn.model_selection import StratifiedShuffleSplit

import time
from tqdm import tqdm

torch.set_num_threads(1) # TODO: was 12

class MLPModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, sigmoid_last_layer=False):
        super(MLPModel, self).__init__()

        # construct layers
        layers = [torch.nn.Linear(input_dim, hidden_dim),
                  torch.nn.ReLU(),
                  torch.nn.Dropout(dropout),
                  torch.nn.Linear(hidden_dim, output_dim)]
        if sigmoid_last_layer:
            layers.append(torch.nn.Sigmoid())

        # construct model
        self.predictor = torch.nn.Sequential(*layers)

    def forward(self, X):
        X = self.predictor(X)
        return X

class Recommender(torch.nn.Module):
    def __init__(self, num_compound, num_enzyme, hidden_dim, dropout=0.5, device='cpu'):
        super(Recommender, self).__init__()

        # embedding layer for compound and enzyme
        self.MF_Embedding_Compound = torch.nn.Embedding(num_compound, hidden_dim).to(device)
        self.MF_Embedding_Enzyme = torch.nn.Embedding(num_enzyme, hidden_dim).to(device)

        self.MLP_Embedding_Compound = torch.nn.Embedding(num_compound, hidden_dim).to(device)
        self.MLP_Embedding_Enzyme = torch.nn.Embedding(num_enzyme, hidden_dim).to(device)
        self.dropout = torch.nn.Dropout(p=dropout)

        # main-task: compound-enzyme interaction prediction net. * 2 since concatenation
        self.ce_predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, 1),
            torch.nn.Sigmoid()
        )
        # multi-task: fingerprint
        self.fp_predictor = MLPModel(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=167, dropout=dropout, sigmoid_last_layer=True)
        # multi-task: ec
        self.ec_predictor = torch.nn.ModuleList()
        for ec_dim in [7, 68, 231]: # TODO: fix these numbers, I thiink?; aslo why sigmoid=False?
            self.ec_predictor.append(MLPModel(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=ec_dim, dropout=dropout, sigmoid_last_layer=False))
        # multi-task: ko
        # TODO: don't harcode this output dimension
        self.ko_predictor = MLPModel(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=8476, dropout=dropout, sigmoid_last_layer=True)
        # MLP co-embedding of enzyme and substrate from input embedding
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU()
        )

        # save parameters
        self.num_compound = num_compound
        self.num_enzyme = num_enzyme

    def forward(self, compound_ids, enzyme_ids):
        mf_embedding_compound = self.MF_Embedding_Compound(compound_ids)
        mf_embedding_enzyme = self.MF_Embedding_Enzyme(enzyme_ids)

        mlp_embedding_compound = self.MLP_Embedding_Compound(compound_ids)
        mlp_embedding_enzyme = self.MLP_Embedding_Enzyme(enzyme_ids)

        

        mf_vector = mf_embedding_enzyme * mf_embedding_compound

        mlp_vector = torch.cat([mlp_embedding_enzyme, mlp_embedding_compound], dim=-1)

        #print('MLP-MLP CONCAT', mlp_vector[:5,:]) # TODO: remove these

        mlp_vector = self.fc1(mlp_vector)

        #print('MLP AFTER FC1 (COEMBEDDING)', mlp_vector[:5,:])

        predict_vector = torch.cat([mf_vector, mlp_vector], dim=-1)

        #print('AFTER MF/MLP CONCAT', predict_vector[:5,:])

        predict_vector = self.dropout(predict_vector)

        #print('AFTER DROPOUT', predict_vector[:5,:])

        predict_vector = self.ce_predictor(predict_vector)

        #print('AFTER CE-PREDICT (FINAL)', predict_vector[:5,:])

        return predict_vector

    def predict_fp(self, compound_ids, mf=True):
        if mf:
            emb_compound_sub = self.MF_Embedding_Compound(compound_ids)
        else:
            emb_compound_sub = self.MLP_Embedding_Compound(compound_ids)

        pred = self.fp_predictor(emb_compound_sub)
        return pred

    def predict_ec(self, ec_ids, ec_i, mf=True):
        if mf:
            emb_enzyme_sub = self.MF_Embedding_Enzyme(ec_ids)
        else:
            emb_enzyme_sub = self.MLP_Embedding_Enzyme(ec_ids)

        pred = self.ec_predictor[ec_i](emb_enzyme_sub)
        return pred

    def predict_ko(self, ec_ids, mf=True):
        if mf:
            emb_enzyme_sub = self.MF_Embedding_Enzyme(ec_ids)
        else:
            emb_enzyme_sub = self.MLP_Embedding_Enzyme(ec_ids)

        pred = self.ko_predictor(emb_enzyme_sub)
        return pred

    def triplet_loss(self, triplets, fp, margin, MF=True):
        if fp:
            embedding_anchor = self.MF_Embedding_Compound(triplets[:, 0]) if MF else self.MLP_Embedding_Compound(triplets[:, 0])
            embedding_positive = self.MF_Embedding_Compound(triplets[:, 1]) if MF else self.MLP_Embedding_Compound(triplets[:, 1])
            embedding_negative = self.MF_Embedding_Compound(triplets[:, 2]) if MF else self.MLP_Embedding_Compound(triplets[:, 2])
        else:
            embedding_anchor = self.MF_Embedding_Enzyme(triplets[:, 0]) if MF else self.MF_Embedding_Enzyme(triplets[:, 0])
            embedding_positive = self.MF_Embedding_Enzyme(triplets[:, 1]) if MF else self.MF_Embedding_Enzyme(triplets[:, 1])
            embedding_negative = self.MF_Embedding_Enzyme(triplets[:, 2]) if MF else self.MF_Embedding_Enzyme(triplets[:, 2])

        cosine_positive = F.cosine_similarity(embedding_anchor, embedding_positive, dim=-1)
        cosine_negative = F.cosine_similarity(embedding_anchor, embedding_negative, dim=-1)

        loss = torch.clamp_min(-cosine_positive + cosine_negative + margin, min=0.0)

        loss = loss.mean()

        return loss

def weighted_binary_cross_entropy(output, target, weights=None):
    output = torch.clamp(output, 1e-6, 1.0 - 1e-6) # TODO: ?

    if weights is not None:
        assert len(weights) == 2

        loss = weights[1] * (target * torch.log(output)) + \
               weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))

def contrastive_loss(triplets, fp):
    loss_pair_mf = model.triplet_loss(triplets, fp=fp, margin=args.margin, MF=True)
    loss_pair_mlp = model.triplet_loss(triplets, fp=fp, margin=args.margin, MF=False)

    loss = loss_pair_mf + loss_pair_mlp

    return loss

def main_task_loss(target, output, pi=0.03, old=False):
    if old:
        return weighted_binary_cross_entropy(output, target)
    else:
        eps = 1e-6 # to avoid log(0)=-inf in second term if output==1
        loss = target * torch.log(pi + (1-pi)*output) + (1-target) * torch.log(1 - pi + (output-eps)*(pi-1))
        return torch.neg(torch.mean(loss))

def multi_task_loss(pred_interactions, true_interactions, cpds, enzs, triplets, epoch):
    # main task 
    # TODO: enable new loss
    interaction_loss = main_task_loss(pred_interactions, true_interactions, old=False)
   
    # auxilliary tasks
    # fingerprints (for compounds in this batch)
    loss_fp_mf  = weighted_binary_cross_entropy(model.predict_fp(cpds, mf=True), fp_label[cpds])
    loss_fp_mlp = weighted_binary_cross_entropy(model.predict_fp(cpds, mf=False), fp_label[cpds])
    loss_fp = loss_fp_mf + loss_fp_mlp
    # EC-numbers (for enzymes in this batch)
    loss_ec_mf, loss_ec_mlp = 0.0, 0.0
    ec_loss_w = [1./3., 1./3., 1./3.]
            # TODO: don't hardcode these dimensionalities (they are passed in interaction data already -> just use them)
    for j, ec_dim in enumerate([(0, 7), (7, 7+25), (7+25, 7+25+23)]):
        ec_label_j = ec_label[enzs, ec_dim[0]:ec_dim[1]]
        _, ec_label_j = ec_label_j.max(dim=1) # indexes of the 1s in the 1-hot encodings
        loss_ec_mf += ec_loss_w[j] * torch.nn.CrossEntropyLoss()(model.predict_ec(enzs, j, mf=True), ec_label_j)
        loss_ec_mlp += ec_loss_w[j] * torch.nn.CrossEntropyLoss()(model.predict_ec(enzs, j, mf=False), ec_label_j)
    loss_ec = loss_ec_mf + loss_ec_mlp
    # cc-relations (for compounds in this batch)
    loss_rpair = contrastive_loss(triplets, fp=True)
    # ko (for enzymes in this batch)
            # BUG: they changed the KO loss from the paper, I think? -> not contrastive!
            # TODO: think about what would be an appropriate loss, also considering it's a multi-label problem
            # loss_enzyme_ko = contrastive_loss(enzyme_ko, num_enzyme, fp=False)  # never tested
    loss_ko_mf = weighted_binary_cross_entropy(model.predict_ko(enzs, mf=True), enzyme_ko_hot[enzs], weights=[1.0, 1.0])
    loss_ko_mlp = weighted_binary_cross_entropy(model.predict_ko(enzs, mf=False), enzyme_ko_hot[enzs], weights=[1.0, 1.0])
    loss_enzyme_ko = loss_ko_mf + loss_ko_mlp
    
    # total loss
    T = 2000
    w_m = 1.0 if epoch > T else epoch / float(T)
    w_a = 0.0 if epoch > T else (1 - epoch / float(T))
    loss = w_m * interaction_loss + w_a * (loss_fp + loss_ec + loss_rpair + loss_enzyme_ko)

    return loss

def train():
    val_maps = []
    best_valid_map = 0.0
    best_model_state = None

    for epoch in (range(args.epochs)):
        model.train()

        # TODO: implement and test batches-by-hand on euler (no num_workers) -> that is fastest for eval on laptop

        epoch_loss = 0
        for batch_idx, batch_data in tqdm(enumerate(train_loader)):
            # get data
            interactions, cpds, enzs, triplets = batch_data
            true_interactions = interactions[:, -1].reshape([-1, 1]).float()
            # forward and loss
            pred_interactions = model(cpds, enzs)
            loss = multi_task_loss(pred_interactions, true_interactions, cpds, enzs, triplets, epoch)
            epoch_loss += loss.item()
            # backprop
            opt.zero_grad()
            loss.backward()
            opt.step()
        
            # TODO: reinstate this block
            # # print eval metrics on valset every so often
            # if batch_idx % args.eval_freq == 0:
            #     print('LAST TRAIN BATCH LOSS: {}'.format(loss.item()))
            #     _, val_map = evaluate(model, val_data, iteration=epoch)
            #     if val_map > best_valid_map:
            #         best_valid_map = val_map
            #         best_model_state = copy.deepcopy(model.state_dict())
            #     # early stop on map
            #     val_maps.append(val_map)
            #     if len(val_maps) == args.early_stop_window // args.eval_freq:
            #         if val_maps[0] > np.max(val_maps[1:]):
            #             print('STOPPING EARLY !!!') # TODO: understand condition exactly
            #             break
            #         val_maps.pop(0)

        print('AVERAGE EPOCH LOSS: {}'.format(epoch_loss / len(train_loader)))

    # test at end of training
    model.load_state_dict(best_model_state)
    evaluate(model, test_data, report_metric_bool=True, iteration=-1, num_compound=num_compound, num_enzyme=num_enzyme)

def evaluate(model, data, report_metric_bool=False, **kwargs):
    model.eval()
    with torch.no_grad():
        # batches by hand implementation -> FASTEST on laptop
        batch_size = 20480              # TODO: pass to evaluate as argument
        true_interactions = data[:,2]
        pred_interactions = []
        for bi in tqdm(range(int(np.ceil(data.shape[0] / batch_size)))):
            indices_s = bi * batch_size
            indices_e = min(data.shape[0], (bi + 1) * batch_size)
            compound_indices = data[indices_s:indices_e, 0]
            ec_indices = data[indices_s:indices_e, 1]
            pred_interaction_ = model(compound_indices, ec_indices)
            pred_interactions.append(pred_interaction_)
        pred_interactions = torch.cat(pred_interactions, dim=0)

        # # dataloader implementation -> last two lines give syntax error -> test this on euler with more 4 workers or so
        # pred_interactions = []
        # true_interactions = []
        # for batch_idx, batch_data in tqdm(enumerate(dataloader)):
        #     interactions, cpds, enzs = batch_data
        #     true_interactions.append(interactions)
        #     pred_interactions.append(model(cpds, enzs))
        # pred_interactions = torch.cat(pred_interactions, dim=0)
        # true_interactions = torch.cat(pred_interactions, dim=0)

         # convert ground truth and prediction to numpy
        all_zero_pred = torch.zeros_like(true_interactions).float().cpu().detach().numpy().reshape(-1)
        all_one_pred  = torch.ones_like(true_interactions).float().cpu().detach().numpy().reshape(-1)
        true_interactions = true_interactions.cpu().detach().numpy().reshape(-1)
        pred_interactions = pred_interactions.cpu().detach().numpy().reshape(-1)
       
        # report metrics for evaluation
        # TODO: more metrics, plotting ...
        te_auc = roc_auc_score(y_true=true_interactions, y_score=pred_interactions)
        te_map = sk_m.average_precision_score(y_true=true_interactions, y_score=pred_interactions)

        zero_pred_auc = roc_auc_score(y_true=true_interactions, y_score=all_zero_pred)
        zero_pred_map = sk_m.average_precision_score(y_true=true_interactions, y_score=all_zero_pred)
        one_pred_auc = roc_auc_score(y_true=true_interactions, y_score=all_one_pred)
        one_pred_map = sk_m.average_precision_score(y_true=true_interactions, y_score=all_one_pred)

        print('Iteration %d: auc %.3f, map %.3f' % (kwargs['iteration'], te_auc, te_map))
        print('Compared to All-Zero-Prediction: auc %.3f, map %.3f' % (zero_pred_auc, zero_pred_map))
        print('Compared to All-One-Prediction: auc %.3f, map %.3f' % (one_pred_auc, one_pred_map))

        # TODO: fix the commented out section
        # if report_metric_bool:
        #     test_rst = report_metric(kwargs['num_compound'], kwargs['num_enzyme'], true_interactions, pred_interactions, pn_.cpu().detach().numpy())
        #     test_rst['auc'] = te_auc
        #     test_rst['map'] = te_map

        #     for key in ['map', 'rprecision', 'auc', 'enzyme_map', 'enzyme_rprecision', 'enzyme_map_3', 'enzyme_precision_1',
        #                                             'compound_map', 'compound_rprecision', 'compound_map_3', 'compound_precision_1']:
        #         if isinstance(test_rst[key], tuple):
        #             print('%.3f' % (test_rst[key][0]), end=' ')
        #         else:
        #             print('%.3f' % (test_rst[key]), end=' ')
        #     print()

    return te_auc, te_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MLT with NMF.")
    parser.add_argument('--gpu', type=int, default=0)
    # training parameters
    parser.add_argument('--epochs', type=int, default=350) # changed from 3500
    parser.add_argument('--lr', type=float, default=5e-3) # TODO: consider grid search
    parser.add_argument('--l2_reg', type=float, default=1e-6) # TODO: consider grid search
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--neg_rate', type=int, default=25)
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--eval_freq', type=int, default=50) # changed from 50
    parser.add_argument('--early_stop_window', type=int, default=200)
    # model structure
    parser.add_argument('--hidden_dim', type=int, default=256)
    args = parser.parse_args()
    print(args)

    device = 'cuda:' + str(args.gpu) if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'

    # load data
    pairs, num_compound, num_enzyme,\
        fp_label, ec_label, len_EC_fields, EC_to_hot_dicts, hot_to_EC_dicts, \
        pos_to_ko_dict,ko_to_pos_dict,num_ko,rpairs_pos,CC_dict,enzyme_ko_hot = load_interaction_data()
    # single train-val-test split 7:2:1 (preserves class imbalance)
    # TODO: think about creating multiple train/test splits
    # TODO: think about implications of class imbalance -> force balance? asymmetric loss?
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.33, random_state=0)
    for i, (train_index, valtest_index) in enumerate(sss1.split(pairs[:,:2], pairs[:,2])):
        for j, (val_index, test_index) in enumerate(sss2.split(pairs[valtest_index,:2], pairs[valtest_index,2])):
            #train_data = pairs[train_index, :].to(device)
            train_data = TrainDataset(pairs[train_index,:], CC_dict, num_compound, device) 
            val_data  = pairs[val_index, :].to(device)
            test_data = pairs[test_index,:].to(device)

            #val_data   = TestDataset(pairs[val_index,:] , device) 
            #test_data  = TestDataset(pairs[test_index,:], device) 

            # TODO: consider influence of batch size on gradient accuracy given class imbalance
            train_loader = DataLoader(train_data, batch_size=64, collate_fn=custom_collate_triplets, shuffle=True, pin_memory=True, num_workers=1) # TODO num_worker was 4
            # val_loader   = DataLoader(val_data,   batch_size=20480,  collate_fn=custom_collate_no_triplets, shuffle=False, pin_memory=True, num_workers=4)
            # test_loader  = DataLoader(test_data,  batch_size=len(test_data), collate_fn=custom_collate_no_triplets, shuffle=False, pin_memory=True, num_workers=4)
        
            # tr_p, va_p, te_p, va_pn, te_pn, n_all_exclhmusive, num_compound, num_enzyme, compound_i2n, enzyme_i2n, fp_label, ec_label = load_interaction_data()

            # tr_p = tr_p.to(device)
            # va_p = va_p.to(device)
            # te_p = te_p.to(device)
            # va_pn = va_pn.to(device)
            # te_pn = te_pn.to(device)
            # n_all_exclusive = n_all_exclusive.to(device)
            fp_label = fp_label.to(device)
            ec_label = ec_label.to(device)
            # rpairs_pos = rpairs_pos.to(device)
                        # # enzyme_ko = enzyme_ko.to(device)  # needed for contrastive loss later
            enzyme_ko_hot = enzyme_ko_hot.to(device)

            # construct model
            model = Recommender(num_compound=num_compound, num_enzyme=num_enzyme,
                                hidden_dim=args.hidden_dim, dropout=args.dropout, device=device).to(device)

            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)

            train()
