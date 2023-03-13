import pickle
import numpy as np
import sklearn.metrics as sk_m

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

def load_interaction_data():
    with open('./data/interaction.pkl', 'rb') as fi:
        data = pickle.load(fi)
    tr_p, va_p, te_p, n_all_exclusive, va_pn, te_pn, \
    num_compound,\
    num_enzyme, fp_label, ec_label,len_EC_fields,\
    EC_to_hot_dicts,hot_to_EC_dicts,pos_to_ko_dict,\
    ko_to_pos_dict,num_ko,rpairs_pos,enzyme_ko_hot = \
                data['tr_p'], data['va_p'], data['te_p'], data['n_all_exclusive'],\
                data['va_pn'], data['te_pn'],\
                data['num_compound'], data['num_enzyme'], data['fp_label'], data['ec_label'],\
                data['len_EC_fields'],data['EC_to_hot_dicts'],data['hot_to_EC_dicts'],\
                data['pos_to_ko_dict'],data['ko_to_pos_dict'], data['num_ko'], data['rpairs_pos'],\
                data['enzyme_ko_hot']
    
    return tr_p, va_p, te_p, n_all_exclusive, va_pn, te_pn, num_compound, num_enzyme,\
        fp_label, ec_label, len_EC_fields, EC_to_hot_dicts, hot_to_EC_dicts, \
        pos_to_ko_dict,ko_to_pos_dict,num_ko,rpairs_pos,enzyme_ko_hot

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
