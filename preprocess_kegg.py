import pandas as pd
import numpy as np
import random
from rdkit.Chem import MACCSkeys, MolFromMolFile
from myutil import encode_KOs, encode_EC

KEGG        = '/Users/odysseasvavourakis/Documents/2022-2023/Studium/5. Semester/Thesis Work/datasets.nosync/kegg/'
MOL_FILES   = KEGG+'ligand/compound/mol'
ES_FILE     = KEGG+'ligand/compound/links/compound_enzyme.list'
EC_KO_FILE  = KEGG+'ligand/enzyme/links/enzyme_ko.list'
CC_FILE     = KEGG+'ligand/rclass/rclass_cpair.lst'

cp_blacklist = {'C00001'}   # ignore these compounds: water

print('GETTING E/S INTERACTIONS')
esi = pd.read_csv(ES_FILE, sep='\t', header=None)
esi[0]=esi[0].str.split(':').apply(lambda list : list[1]) # drop cpd: prefix
esi[1]=esi[1].str.split(':').apply(lambda list : list[1]) # drop ec: prefix
esi = esi.set_axis(['compound', 'EC'], axis=1)

print('GETTING SUBSTRATE FINGERPRINTS')
substrates = set(esi['compound'])
mol_FPs = {}
for S in substrates:
    path = MOL_FILES+'/'+S+'.mol'
    try:
        rep = MolFromMolFile(MOL_FILES+'/'+S+'.mol')
        mol_FPs[S] = MACCSkeys.GenMACCSKeys(rep)
    except: # generic functions, photons, macromolecular, faulty files
        cp_blacklist.add(S)

print('GETTING COMPOUND-COMPOUND REACTION RELATIONS')
cc = pd.read_csv(CC_FILE, sep='\t', header=None).drop(axis=1,columns=[0,2])
cc = cc[1].str.split('_').apply(lambda list: pd.Series([list[0],list[1]]))
# also consider reverse pairs
cc_rev = pd.concat([cc[1],cc[0]], axis=1).set_axis([0,1], axis=1)
cc = pd.concat([cc,cc_rev], axis=0, ignore_index=True) 
# create compound : compound interaction dictionary
CC = cc.groupby(by=cc[0]).agg(set).to_dict()[1]

print('GETTING ENZYME KOs')
ec_ko = pd.read_csv(EC_KO_FILE, sep='\t', header=None)
ec_ko[0]=ec_ko[0].str.split(':').apply(lambda list : list[1]) # drop ec: prefix
ec_ko[1]=ec_ko[1].str.split(':').apply(lambda list : list[1]) # drop ko: prefix
ko_set  = set(ec_ko[1])
num_ko  = len(ko_set)
ec_ko = ec_ko.groupby(by=ec_ko[0]).agg(set)

print('FILTERING DOWN ENZYMES & COMPOUNDS')
# consider enzymes with BOTH an interaction AND an assigned KO; drop others
enzymes = set(esi['EC']) & set(ec_ko.index)
ec_ko = ec_ko.drop(set(ec_ko.index)-enzymes, axis=0)
esi = esi.drop(esi[esi['EC'].apply(lambda ec: not ec in enzymes)].index)
# consider substrates with ALL of (fingerprints, CC relations, ESI); drop others
substrates = (substrates - cp_blacklist) & set(CC.keys()) & set(esi['compound'])
for key in set(mol_FPs.keys())-substrates:                                
    mol_FPs.pop(key)
for key in set(CC.keys())-substrates: 
    CC.pop(key)
esi = esi[esi['compound'].apply(lambda cmp: cmp in substrates)] 
# last step has removed more enzymes
enzymes = set(esi['EC']) & set(ec_ko.index)
ec_ko = ec_ko.drop(set(ec_ko.index)-enzymes, axis=0)
esi = esi.drop(esi[esi['EC'].apply(lambda ec: not ec in enzymes)].index)
assert(set(esi['compound']) == substrates)
assert(set(mol_FPs.keys()) == substrates)
assert(set(CC.keys()) == substrates)
assert(enzymes == set(esi['EC']))
assert(enzymes == set(ec_ko.index))

# analyse EC numbers, set up encode/decoder dictionaries
As, Bs, Cs = set(),set(),set()
for E in enzymes:
    A, B, C, _ = [int(i) for i in E.split('.')]
    As.add(A); Bs.add(B); Cs.add(C)
len_ECa, len_ECb, len_ECc = [len(x) for x in [As, Bs, Cs]]
ECa_to_hot, ECb_to_hot, ECc_to_hot = \
        [ { EC_field : one_hot for EC_field, one_hot in zip(list(field), list(np.identity(len(field), dtype=int))) } for field in [As, Bs, Cs]]
hot_to_ECa, hot_to_ECb, hot_to_ECc = \
        [ { tuple(v): k for k, v in dic.items()} for dic in [ECa_to_hot, ECb_to_hot, ECc_to_hot] ]

# create dictionary EC : {compounds}
enzymes, substrates, num_compound, num_enzyme = sorted(enzymes), sorted(substrates), len(substrates), len(enzymes)
E_S = esi.groupby(by=esi['EC']).agg(set).to_dict()['compound']

print('CREATING INTERACTION MATRIX')
matrix = np.zeros((num_compound,num_enzyme), dtype=bool)

Eid_to_EC = {} # maps numerical enzyme id to EC
EC_to_Eid = {} # maps ECs back to numerical enzyme ids

Sid_to_cmpd = {} # maps numerial compound id to KEGG C_id
cmpd_to_Sid = {} # maps KEGG C_id to numerical compound id

for E in E_S:
    for S in E_S[E]:
        assert(S in substrates)
        E_index, S_index = enzymes.index(E), substrates.index(S)

        Eid_to_EC[E_index], EC_to_Eid[E] = E, E_index

        Sid_to_cmpd[S_index], cmpd_to_Sid[S] = S, int(S_index)
        
        matrix[S_index,E_index] = 1

print('BUILDING INTERACTION TABLE')
interaction_pairs = np.array(np.where(matrix)).T
all_ones = np.ones(len(interaction_pairs), dtype=int).reshape(-1,1)
interaction_pairs = np.append(interaction_pairs, all_ones , axis=1)
interaction_pairs = pd.DataFrame(interaction_pairs)
interaction_pairs.columns = ['compound', 'enzyme', 'interaction']

print('BUILDING NON-INTERACTION TABLE')
non_interaction_pairs = np.array(np.where(np.invert(matrix))).T
all_zeros = np.zeros(len(non_interaction_pairs), dtype=int).reshape(-1,1)
non_interaction_pairs = np.append(non_interaction_pairs, all_zeros , axis=1)
non_interaction_pairs = pd.DataFrame(non_interaction_pairs)                  
non_interaction_pairs.columns = ['compound', 'enzyme', 'interaction']

print('SPLITTING INTERACTIONS INTO TRAIN/VALIDATION/TEST')
interaction_pairs = interaction_pairs.sample(frac=1).reset_index(drop=True) # shuffle rows
end_of_train_fraction = int(np.ceil(interaction_pairs.shape[0] * 0.7))
end_of_val_fraction = end_of_train_fraction + int(np.ceil(interaction_pairs.shape[0] * 0.2))
tr_interaction_pairs = interaction_pairs.iloc[ : end_of_train_fraction]
va_interaction_pairs = interaction_pairs.iloc[ end_of_train_fraction : end_of_val_fraction]
te_interaction_pairs = interaction_pairs.iloc[ end_of_val_fraction : ]

print('BUILDING SUBSTRATE FINGERPRINT TABLE')
fps = pd.DataFrame.from_dict(mol_FPs, orient='index').reset_index().set_axis(['compound', 'fingerprint'], axis=1)
fps['fingerprint'] = fps['fingerprint'].apply(lambda fp: np.array(fp))
fps['compound'] = fps['compound'].apply(lambda cp: cmpd_to_Sid[cp])
fps = fps.sort_values('compound').set_index('compound')

print('ENCODING EC TABLE')
EC_table = pd.DataFrame.from_dict(Eid_to_EC, orient='index')
EC_table=EC_table[0].apply(lambda EC_num: encode_EC(EC_num, (ECa_to_hot, ECb_to_hot, ECc_to_hot)))

print('BUILDING CC PAIR TABLE')
pairlist=[]
for anchor, paired_compounds in CC.items():
    for P in paired_compounds:
        if anchor in substrates and P in substrates:
            pairlist.append([cmpd_to_Sid[anchor],cmpd_to_Sid[P]])
cc_pairs = np.array(pairlist)

print('ENCODING KOs')
# build pos : KO dictionary
pos_to_ko, ko_to_pos = {}, {}
for i, ko in enumerate(sorted(ko_set)):
    pos_to_ko[i] = ko
    ko_to_pos[ko] = i
# encode each enzyme's KO vector (multi-hot)
ec_ko = ec_ko.reset_index()
ec_ko[0] = ec_ko[0].apply(lambda ec : EC_to_Eid[ec])
ec_ko[1] = ec_ko[1].apply(lambda kos : encode_KOs(kos, ko_to_pos, num_ko))
ec_ko.set_index(0)

# TODO: convert arrays to torch
# TODO: re-write the data-read in functions in myutil

interaction_data = { 'tr_p' : tr_interaction_pairs,
                     'va_p' : va_interaction_pairs,
                     'te_p' : te_interaction_pairs,
                     'n_all_exclusive' : non_interaction_pairs,
                     'num_compound' : num_compound,
                     'num_enzyme' : num_enzyme,
                     'fp_label' : fps,
                     'ec_label' : EC_table,
                     'len_EC_fields' : (len_ECa, len_ECb, len_ECc),
                     'EC_to_hot_dicts' : [ECa_to_hot, ECb_to_hot, ECc_to_hot],
                     'hot_to_EC_dicts' : [hot_to_ECa, hot_to_ECb, hot_to_ECc],
                     'pos_to_ko_dict' : pos_to_ko,
                     'ko_to_pos_dict' : ko_to_pos,
                     'num_ko' : num_ko,
                     'rpairs_pos' : cc_pairs,
                     'enzyme_ko_hot' : ec_ko
}

# TODO: pickle this data







