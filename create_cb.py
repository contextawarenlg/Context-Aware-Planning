import argparse
import json, numpy as np
from datasets import load_dataset
from tqdm import tqdm
from utils.utils import ExtractConceptOrder, ExtractEntities, GetEntityRepresentation
from utils.imp_players_utility import get_imp_players_train, get_game_repr_imp_players
from sentence_transformers import SentenceTransformer

argparser = argparse.ArgumentParser()
argparser.add_argument('-side', '--side', type=str, default='both', \
                        help='problem or solution side of case-base', \
                        choices=['prob', 'sol', 'both'])
argparser.add_argument('-season', '--season', type=str, default='2014', \
                        choices=['2014', '2015', '2016', '2017', '2018', 'bens', 'all', 'juans'])
argparser.add_argument('-players', '--players', type=str, default='imp', choices=['all', 'imp'])
argparser.add_argument('-ftrs', '--ftrs', type=str, default='num', choices=['text', 'num', 'set'])
argparser.add_argument('-pop', '--pop', action='store_true', help='use popularity for players')
argparser.add_argument('-tpop', '--tpop', action='store_true', help='use popularity for teams')
argparser.add_argument('-stands', '--stands', action='store_true', help='use teams standings')
argparser.add_argument('-week', '--week', action='store_true', help='use week/date of season')

args = argparser.parse_args()
cb_side = args.side
pop = args.pop
tpop = args.tpop
players = args.players
season = args.season
ftrs = args.ftrs
stands = args.stands
week = args.week

bstr = f"*"*150
print(f"\nCreating CB\n{bstr}\nPopularity: {pop}\tTeam Popularity: {tpop}\tWeek: {week}\tStands: {stands}\tSeason: {season}\tFeature Type: {ftrs}\tPlayers: {players}\n{bstr}\n")

if season in ['2014', '2015', '2016', 'bens', 'all']:
    part = 'train'
if season in ['2017', 'juans']:
    part = 'validation'
if season in ['2018']:
    part = 'test'

data_split = json.load(open(f'data/seasonal_splits.json'))
train_ids = data_split[f'{season}']['train']
test_ids = data_split[f'{season}']['test']
validation_ids = data_split[f'{season}']['validation']

dataset = load_dataset('GEM/sportsett_basketball')
sents_data = json.load(open(f'data/{part}_data_ct.json'))
eco_obj = ExtractConceptOrder()
ee_obj = ExtractEntities()
ger_obj = GetEntityRepresentation(popularity=pop, team_popularity=tpop, season=season, standing=stands, season_week=week)
embedding_model = SentenceTransformer('bert-base-nli-mean-tokens')


print(f"\nPlayers: {players}\tFeature Type: {ftrs}\tPopularity: {pop}\tTeam Popularity: {tpop}")
prob, sol = [], []
for idx, entry in tqdm(enumerate(dataset[f'{part}'])):
    if entry['gem_id'] not in train_ids:
        continue
    entry_sents = list(filter(lambda x: x['gem_id'] == entry['gem_id'], sents_data))
    unique_summary_idx = set([x['summary_idx'] for x in entry_sents])
    for us_id in unique_summary_idx:
        if us_id != 0 and season != 'all':
            continue
        summary_sents = list(filter(lambda x: x['summary_idx'] == us_id, entry_sents))
        sentences = [x['coref_sent'] for x in summary_sents]
        if cb_side == 'both' or cb_side == 'sol':
            concept_order_w_ents = eco_obj.extract_concept_order(entry, summary_sents)
            concept_order = [x.split('|')[1] for x in concept_order_w_ents]
            sol.append(concept_order)
        if cb_side == 'both' or cb_side == 'prob':
            if players == "imp":
                imp_players = get_imp_players_train(entry, sentences, ee_obj)
            elif players == "all":
                hplayers = [player['name'] for player in entry['teams']['home']['box_score']]
                vplayers = [player['name'] for player in entry['teams']['vis']['box_score']]
                imp_players = hplayers + vplayers
            game_repr = get_game_repr_imp_players(entry, ger_obj, imp_players, embedding_model, ftrs_type=ftrs, players=players)
            prob.append(game_repr)

if cb_side == 'both' or cb_side == 'prob':
    prob1 = np.array(prob)
    file_name = f"{players}_players_{ftrs}_ftrs{'_pop' if pop else ''}{'_tpop' if tpop else ''}{'_stands' if stands else ''}{'_week' if week else ''}_cb_prob.npz"
    print(f"Prob shape: {prob1.shape}\t{file_name}")
    np.savez(f'cbs/{season}/{file_name}', prob1)
if cb_side == 'both' or cb_side == 'sol':
    print(f"Sol shape: {len(sol)}")
    json.dump(sol, open(f'cbs/{season}/cb_sol.json', 'w'), indent='\t')
