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

all_teams = json.load(open(f'data/all_teams.json', 'r'))
js = json.load(open(f'data/{part}_data_ct.json', 'r'))
data_split = json.load(open(f'data/seasonal_splits.json'))
train_ids = data_split[f'{season}']['train']
test_ids = data_split[f'{season}']['test']
validation_ids = data_split[f'{season}']['validation']

teams_mentions = {team: 0 for team, _ in all_teams.items()}
for idx, item in enumerate(tqdm(dataset[f'{part}'])):
    if item['gem_id'] not in train_ids:
        continue
    gemid = item['gem_id']
    all_ents, teams, players = ee_obj.get_all_ents(item)
    hteam = f"{item['teams']['home']['place']} {item['teams']['home']['name']}"
    vteam = f"{item['teams']['vis']['place']} {item['teams']['vis']['name']}"
    item_summs = list(filter(lambda x: x['gem_id'] == gemid, js))
    summ_uids = set([sent['summary_idx'] for sent in item_summs])
    hteam_mention, vteam_mention = 0, 0
    for summ_uid in summ_uids:
        summary_sents = list(filter(lambda x: x['summary_idx'] == summ_uid, item_summs))
        for sent in summary_sents:
            teams_ents_in_sent1 = ee_obj.extract_entities(teams, sent['coref_sent'])
            teams_ents_in_sent = ee_obj.get_full_team_ents(teams_ents_in_sent1, item)
            if hteam in teams_ents_in_sent:
                hteam_mention += 1
            if vteam in teams_ents_in_sent:
                vteam_mention += 1
    teams_mentions[hteam] += hteam_mention/len(item_summs)
    teams_mentions[vteam] += vteam_mention/len(item_summs)

pop_score = {}
for team, mention in teams_mentions.items():
    # pop_score[team] = round((mention/len(train_ids))*100, 2)
    pop_score[team] = round((mention/82)*100, 2)

json.dump(pop_score, open(f'popularity/{season}/pop_score-team.json', 'w'), indent='\t')
