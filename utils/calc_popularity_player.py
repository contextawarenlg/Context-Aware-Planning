import json, argparse
from tqdm import tqdm
from datasets import load_dataset
from utils import ExtractEntities

argparser = argparse.ArgumentParser()
argparser.add_argument('-season', '--season', type=str, default='all', \
                        choices=['2014', '2015', '2016', '2017', '2018', 'all', 'bens', 'juans'])
args = argparser.parse_args()
season = args.season
print(args, season)

dataset = load_dataset('GEM/sportsett_basketball')
ee_obj = ExtractEntities()

if season in ['2014', '2015', '2016', 'bens', 'all']:
    part = 'train'
if season in ['2017', 'juans']:
    part = 'validation'
if season in ['2018']:
    part = 'test'

js = json.load(open(f'data/{part}_data_ct.json', 'r'))
data_split = json.load(open(f'data/seasonal_splits.json'))
train_ids = data_split[f'{season}']['train']
test_ids = data_split[f'{season}']['test']
validation_ids = data_split[f'{season}']['validation']

popularity_dict = {}
for idx, entry in tqdm(enumerate(dataset[f'{part}'])):
    if entry['gem_id'] not in train_ids:
        continue
    hbs = entry['teams']['home']['box_score']
    vbs = entry['teams']['vis']['box_score']
    hplayers_name = [player['name'] for player in hbs]
    vplayers_name = [player['name'] for player in vbs]
    all_players = hplayers_name + vplayers_name
    all_ents, teams, players = ee_obj.get_all_ents(entry)

    entry_sents = list(filter(lambda x: x['gem_id'] == entry['gem_id'], js))
    unique_summary_idx = set([x['summary_idx'] for x in entry_sents])
    for us_id in unique_summary_idx:
        summary_sents = list(filter(lambda x: x['summary_idx'] == us_id, entry_sents))
        players_mentioned = []
        for sent in summary_sents:
            player_ents_unresolved = ee_obj.extract_entities(players, sent['coref_sent'])
            player_ents = ee_obj.get_full_player_ents(player_ents_unresolved, entry)
            players_mentioned.extend(player_ents)

        for player in all_players:
            if player not in popularity_dict:
                popularity_dict[player] = {'games_played': 0, 'summary_mentions': 0}
            popularity_dict[player]['games_played'] += 1
            if player in players_mentioned:
                popularity_dict[player]['summary_mentions'] += 1

pop_score = {}
for key, val in tqdm(popularity_dict.items()):
    pop_score[key] = f"{(val['summary_mentions'] / val['games_played'])*100:.2f}"
# print(pop_score)
json.dump(pop_score, open(f'popularity/{season}/pop_score.json', 'w'), indent='\t')
