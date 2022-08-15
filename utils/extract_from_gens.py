import argparse, json
from tqdm import tqdm
from datasets import load_dataset
from utils import ExtractConceptOrder
from clf_utils import ContentTypeData, MultiLabelClassifier
from coref_resolve import CorefResolver

def get_ct_list_from_arr(arr):
    ct_list = []
    if arr[0] == 1:
        ct_list.append('B')
    if arr[1] == 1:
        ct_list.append('W')
    if arr[2] == 1:
        ct_list.append('A')
    return ct_list

def main(SYS_NAME='ent', season='2014'):
    print(f'Constructing...')

    ctd = ContentTypeData()
    clf = MultiLabelClassifier()
    coref_obj = CorefResolver()
    eco_obj = ExtractConceptOrder()

    dataset = load_dataset('GEM/sportsett_basketball')
    gen_out = open(f'gens/{SYS_NAME}.txt', 'r').readlines()
    gen_out = [line.strip() for line in gen_out]

    print(f'Constructed!!!')

    gen_concept_order = []
    for idx, gen in tqdm(enumerate(gen_out)):
        if idx % 100 == 0:
            print(f'\n\nthis is {idx}\n\n')
        entry = dataset[f'test'][idx]

        all_sents = coref_obj.process_one_summary(gen)
        ner_abs_sents = ctd.abstract_sents(all_sents)
        ct_y = clf.predict_multilabel_classif(ner_abs_sents)
        all_sents_with_ct = [{"coref_sent": sent, "content_types": get_ct_list_from_arr(ct_y[idx])} for idx, sent in enumerate(all_sents)]
        gen_concept_order.append(eco_obj.extract_concept_order(entry, all_sents_with_ct))

    json.dump(gen_concept_order, open(f'sportsett/output/{SYS_NAME}/concepts.json', 'w'), indent='\t')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sys', '-sys', type=str, default='ent', choices=['ent', 'hir', 'mp', 'cbr', 'temp'])
    parser.add_argument('-season', '--season', type=str, default='2014', \
                            choices=['2014', '2015', '2016', '2017', '2018'])
    args = parser.parse_args()
    main(SYS_NAME=args.sys, season=args.season)
    print('Done!')
