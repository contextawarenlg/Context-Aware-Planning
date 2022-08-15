import time
import json
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from utils.entity_ranking import RankEntities
from utils.utils import GetEntityRepresentation
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from utils.imp_players_utility import get_imp_players_test, get_game_repr_imp_players

def get_ents_with_concepts(item, concept_order, re_obj):
    ts = re_obj.get_ranked_teams(item)
    ps = re_obj.get_ranked_players(item)
    t_combs = re_obj.get_ranked_teams_comb(item)
    pt_combs = re_obj.get_ranked_player_team_comb(item)
    pp_combs = re_obj.get_ranked_players_comb(item)
    ts_idx, ps_idx, t_combs_idx, pt_combs_idx, pp_combs_idx = 0, 0, 0, 0, 0
    concepts = concept_order
    delim = '|'
    new_concepts = []
    for c in concepts:
        concept_type = c.split('-')[0]
        if concept_type == 'T':
            try:
                new_concepts.append(f"{ts[ts_idx]}{delim}{c}")
                ts_idx += 1
            except:
                new_concepts.append(f"{ts[0]}{delim}{c}")
        elif concept_type == 'P':
            new_concepts.append(f"{ps[ps_idx]}{delim}{c}")
            ps_idx += 1
        elif concept_type == 'T&T':
            try:
                new_concepts.append(f"{t_combs[t_combs_idx]}{delim}{c}")
                t_combs_idx += 1
            except:
                new_concepts.append(f"{t_combs[0]}{delim}{c}")
        elif concept_type == 'P&T':
            new_concepts.append(f"{pt_combs[pt_combs_idx]}{delim}{c}")
            pt_combs_idx += 1
        elif concept_type == 'P&P':
            new_concepts.append(f"{pp_combs[pp_combs_idx]}{delim}{c}")
            pp_combs_idx += 1
    return new_concepts

def get_sol_idx(dists, cb_sol, top_k=15):
    dists_1d = dists.ravel()
    dists_sorted = np.argsort(dists_1d)
    sol_idx = dists_sorted[0]
    max_len = 0
    for _, d in enumerate(dists_sorted[:top_k]):
        sol = cb_sol[d]
        if len(sol) > max_len:
            max_len = len(sol)
            sol_idx = d
    return cb_sol[int(sol_idx)]

def get_sol_by_median(dists, cb_sol, top_k=15):
    dists_1d = dists.ravel()
    dists_sorted = np.argsort(dists_1d)
    sols_top_k_idx = dists_sorted[:top_k]
    sols_top_k = [cb_sol[sol_idx] for sol_idx in sols_top_k_idx]
    sols_len = [len(i) for i in sols_top_k]
    median_sol_len_idx = np.argsort(sols_len)[int(len(sols_len)//2)]
    return sols_top_k[median_sol_len_idx]

def get_solution(dists, cb_sol, reuse_type='long', ret_set_size=15):
    if reuse_type == 'long':
        entry_sol = get_sol_idx(dists, cb_sol)
    elif reuse_type == 'median':
        entry_sol =  get_sol_by_median(dists, cb_sol)
    elif reuse_type == 'first':
        dists_1d = dists.ravel()
        dists_sorted = np.argsort(dists_1d)
        sol_idx = dists_sorted[0]
        entry_sol = cb_sol[sol_idx]
    return entry_sol

def main():

    argparser = argparse.ArgumentParser()
    argparser.add_argument('-season', '--season', type=str, default='all', \
                            choices=['2014', '2015', '2016', '2017', '2018', 'bens', 'all', 'juans'])
    argparser.add_argument('-topk', '--topk', type=str, default='15', \
                            choices=['5', '10', '15', '20', '25', '30'])
    argparser.add_argument('-players', '--players', type=str, default='imp', choices=['all', 'imp'])
    argparser.add_argument('-ftrs', '--ftrs', type=str, default='num', choices=['text', 'num', 'set'])
    argparser.add_argument('-reuse', '--reuse', type=str, default='median', choices=['median', 'first', 'long'])
    argparser.add_argument('-sim', '--sim', type=str, default='euclidean', choices=['euclidean', 'cosine'])
    argparser.add_argument('-stands', '--stands', action='store_true', help='use teams standings')
    argparser.add_argument('-week', '--week', action='store_true', help='use week/date of season')
    argparser.add_argument('--pop', '-pop', action='store_true', help='Use popularity for players')
    argparser.add_argument('--tpop', '-tpop', action='store_true', help='Use top popularity for teams')

    args = argparser.parse_args()
    POP = args.pop
    TPOP = args.tpop
    SEASON = args.season
    TOPK = int(args.topk)
    PLAYERS = args.players
    FTRS = args.ftrs
    REUSE = args.reuse
    SIM = args.sim
    STANDS = args.stands
    WEEK = args.week

    bstr = f"*"*150
    print(f"\n\nGenerating from CB")
    print(f"{bstr}")
    print(f"Player Popularity: {POP}\tTeam Popularity: {TPOP}\tStands: {STANDS}\tWeek: {WEEK}")
    print(f"{bstr}")
    print(f"Players: {PLAYERS}\tSeason: {SEASON}\tFeatures: {FTRS}\tSim: {SIM}\tReuse: {REUSE}")
    print(f"{bstr}")

    if SEASON in ['2014', '2015', '2016', 'bens']:
        part = 'train'
    if SEASON in ['2017', 'juans']:
        part = 'validation'
    if SEASON in ['2018', 'all']:
        part = 'test'
    data_split = json.load(open(f'data/seasonal_splits.json'))
    train_ids = data_split[f'{SEASON}']['train']
    test_ids = data_split[f'{SEASON}']['test']
    validation_ids = data_split[f'{SEASON}']['validation']

    ftr_file_name = f"base{'-pop' if POP else ''}{'-tpop' if TPOP else ''}{'-stands' if STANDS else ''}{'-week' if WEEK else ''}"
    print(f"ftr_file_name: {ftr_file_name}\n{bstr}")
    ftr_weights_js = json.load(open(f"ftr_weights/{ftr_file_name}.json"))

    ftr_weights = np.array(list(ftr_weights_js.values()))
    print(f"ftr_weights: {ftr_weights_js.keys()}\n{bstr}")

    dataset = load_dataset('GEM/sportsett_basketball')
    embedding_model = SentenceTransformer('bert-base-nli-mean-tokens')
    imp_player_clf = pickle.load(open(f'player_clf/model/model_{SEASON}.pkl', 'rb'))
    ger_obj = GetEntityRepresentation(popularity=POP, team_popularity=TPOP, season=SEASON, standing=STANDS, season_week=WEEK)
    ger_obj1 = GetEntityRepresentation(popularity=False, team_popularity=False, season=SEASON) # because, the imp player clf doesnt use popularity
    re_obj = RankEntities()

    cb_sol = json.load(open(f'cbs/{SEASON}/cb_sol.json'))
    prob_file = f"{PLAYERS}_players_{FTRS}_ftrs{'_pop' if POP else ''}{'_tpop' if TPOP else ''}{'_stands' if STANDS else ''}{'_week' if WEEK else ''}_cb_prob"
    cb_prob = np.load(f'cbs/{SEASON}/{prob_file}.npz')['arr_0']
    print(f"Prob Set: {cb_prob.shape}\tSol Set: {len(cb_sol)}\tFeature Weights: {ftr_weights.shape}\n{prob_file}")
    test_set_sol_pred = []

    time1 = time.time()

    for _, entry in tqdm(enumerate(dataset[f'{part}'])):
        if entry['gem_id'] not in test_ids:
            continue

        if PLAYERS == 'imp':
            if POP or TPOP:
                imp_players = get_imp_players_test(entry, ger_obj1, imp_player_clf)
            else:
                imp_players = get_imp_players_test(entry, ger_obj, imp_player_clf)
        elif PLAYERS == 'all':
            hplayers = [player['name'] for player in entry['teams']['home']['box_score']]
            vplayers = [player['name'] for player in entry['teams']['vis']['box_score']]
            imp_players = hplayers + vplayers

        target_problem_rep = get_game_repr_imp_players(entry, ger_obj, imp_players, embedding_model, ftrs_type=FTRS, players=PLAYERS)

        target_problem_rep = target_problem_rep * ftr_weights
        cb_prob = cb_prob * ftr_weights

        if SIM == 'cosine':
            dists = cosine_distances(cb_prob, target_problem_rep.reshape(1, -1))
        elif SIM == 'euclidean':
            dists = euclidean_distances(cb_prob, target_problem_rep.reshape(1, -1))

        entry_sol = get_solution(dists, cb_sol, reuse_type=REUSE, ret_set_size=TOPK)

        enrty_sol_ents_with_concepts = get_ents_with_concepts(entry, entry_sol, re_obj)
        test_set_sol_pred.append(enrty_sol_ents_with_concepts)

    time2 = time.time()
    print(f"\nTime taken: {time2-time1}\n")
    # sol_file = f"{PLAYERS}_players-{FTRS}_ftrs-{SIM}_sim-{REUSE}_reuse{'-pop' if POP else ''}{'-tpop' if TPOP else ''}{'-stands' if STANDS else ''}{'-week' if WEEK else ''}{'-weighted' if WEIGHTED else ''}"
    sol_file = f"{PLAYERS}_players-{FTRS}_ftrs-{SIM}_sim-{REUSE}_reuse{'-pop' if POP else ''}{'-tpop' if TPOP else ''}{'-stands' if STANDS else ''}{'-week' if WEEK else ''}"
    print(f"Saving at: {SEASON}/{sol_file}\n\n")
    json.dump(test_set_sol_pred, open(f'sportsett/concepts/{SEASON}/{sol_file}.json', 'w'), indent='\t')

if __name__ == '__main__':
    main()
