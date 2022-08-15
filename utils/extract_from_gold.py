import json
import argparse
from tqdm import tqdm
from datasets import load_dataset
from utils import ExtractConceptOrder

parser = argparse.ArgumentParser()
parser.add_argument('-season', '--season', type=str, default='2014', \
                        choices=['2014', '2015', '2016', '2017', '2018', 'all', 'bens', 'juans'])
args = parser.parse_args()
season = args.season
print(season)
print(f'Constructing...')
if season in ['2014', '2015', '2016', 'bens']:
    part = 'train'
if season in ['2017', 'juans']:
    part = 'validation'
if season in ['2018', 'all']:
    part = 'test'
data_split = json.load(open(f'data/seasonal_splits.json'))
train_ids = data_split[f'{season}']['train']
test_ids = data_split[f'{season}']['test']
validation_ids = data_split[f'{season}']['validation']

dataset = load_dataset('GEM/sportsett_basketball')
sents_data = json.load(open(f'data/{part}_data_ct.json'))
eco_obj = ExtractConceptOrder()

concepts = []
for idx, entry in tqdm(enumerate(dataset[f'{part}'])):
    if entry['gem_id'] not in test_ids:
        continue
    entry_sents = list(filter(lambda x: x['gem_id'] == entry['gem_id'], sents_data))
    unique_summary_idx = set([x['summary_idx'] for x in entry_sents])
    for us_id in unique_summary_idx:
        if us_id != 0:
            continue
        summary_sents = list(filter(lambda x: x['summary_idx'] == us_id, entry_sents))
        sentences = [x['coref_sent'] for x in summary_sents]
        concept_order_w_ents = eco_obj.extract_concept_order(entry, summary_sents)
        concepts.append(concept_order_w_ents)

print(f"concepts len: {len(concepts)}")
json.dump(concepts, open(f'sportsett/concepts/{season}/gold.json', 'w'), indent='\t')

