import json, pandas as pd, argparse
from tqdm import tqdm

argparser = argparse.ArgumentParser()
argparser.add_argument('-season', '--season', type=str, default='all', \
                        choices=['2014', '2015', '2016', '2017', '2018', 'bens', 'all', 'juans'])
args = argparser.parse_args()
season = args.season
print(season)

lens = json.load(open(f'sportsett/res/{season}/concept_lengths.json'))
concepts = pd.read_csv(f'sportsett/res/{season}/concepts.csv', index_col=0)
entities = pd.read_csv(f'sportsett/res/{season}/entities.csv', index_col=0)
concepts['systems'] = concepts.index
entities['systems'] = entities.index

all_systems = [row['systems'] for _, row in concepts.iterrows()]
new_systems = [row['systems'] for _, row in concepts.iterrows() if 'players' in row['systems']]
other_systems = [row['systems'] for _, row in concepts.iterrows() if 'players' not in row['systems']]
print(len(new_systems))

dictionc = {
    'players': [], 'features': [], 'similarity': [], 'reuse': [], 
    'pop': [], 'tpop': [], 'week': [], 'stands': [], 'weighted': [], 'topk': [], 
    'f1': [], 'f2': [], 'prec': [], 'rec': [], 'dld': [], 'length': []
    }
dictione = {
    'players': [], 'features': [], 'similarity': [], 'reuse': [], 
    'pop': [], 'tpop': [], 'week': [], 'stands': [], 'weighted': [], 'topk': [], 
    'f1': [], 'f2': [], 'prec': [], 'rec': [], 'dld': [], 'length': []
    }

# taken_combinations = []
for sys in tqdm(new_systems):

    # pop = True if 'pop' in sys else False
    # tpop = True if 'tpop' in sys else False

    info = [i.split('_')[0] for i in sys.split('-')]
    players = info[0]
    ftrs = info[1]
    sim = info[2]
    reuse = info[3]

    pop = True if 'pop' in info else False
    tpop = True if 'tpop' in info else False
    week = True if 'week' in info else False
    stands = True if 'stands' in info else False
    weighted = True if 'weighted' in info else False

    # if len(info) == 5:
    #     pop = True if info[-1][0] == 'pop' else False
    #     tpop = True if info[-1][0] == 'tpop' else False
    #     stands = True if info[-1][0] == 'stands' else False
    #     week = True if info[-1][0] == 'week' else False
    # elif len(info) == 6:
    #     pop = True if info[-2][0] == 'pop' else False
    #     tpop = True if info[-1][0] == 'tpop' else False
    # else:
    #     pop = False
    #     tpop = False

    # print(info, len(info), pop, tpop)

    cscore = concepts.loc[concepts['systems'] == sys]
    escore = entities.loc[entities['systems'] == sys]
    
    dictionc['players'].append(players)
    dictionc['features'].append(ftrs)
    dictionc['similarity'].append(sim)
    dictionc['reuse'].append(reuse)
    dictionc['pop'].append(pop)
    dictionc['tpop'].append(tpop)
    dictionc['week'].append(week)
    dictionc['stands'].append(stands)
    dictionc['weighted'].append(weighted)
    dictionc['f2'].append(cscore['f2'].values[0])
    dictionc['prec'].append(cscore['prec'].values[0])
    dictionc['rec'].append(cscore['rec'].values[0])
    dictionc['f1'].append(cscore['f1'].values[0])
    dictionc['dld'].append(cscore['dld'].values[0])
    dictionc['length'].append(float(f"{lens[sys]:.2f}"))
    
    dictione['players'].append(players)
    dictione['features'].append(ftrs)
    dictione['similarity'].append(sim)
    dictione['reuse'].append(reuse)
    dictione['pop'].append(pop)
    dictione['tpop'].append(tpop)
    dictione['week'].append(week)
    dictione['stands'].append(stands)
    dictione['weighted'].append(weighted)
    dictione['f2'].append(escore['f2'].values[0])
    dictione['prec'].append(escore['prec'].values[0])
    dictione['rec'].append(escore['rec'].values[0])
    dictione['f1'].append(escore['f1'].values[0])
    dictione['dld'].append(escore['dld'].values[0])
    dictione['length'].append(float(f"{lens[sys]:.2f}"))

column_names = ['players', 'similarity', 'reuse', 'features', 'pop', 'tpop', 'week', 'stands', \
                'f1', 'f2', 'prec', 'rec', 'dld', 'length']
dfe = pd.DataFrame(dictione, columns=column_names)
dfe.to_csv(f'sportsett/res/{season}/eval_entities.csv', index=0)

dfc = pd.DataFrame(dictionc, columns=column_names)
# print(dfc.shape)
dfc.to_csv(f'sportsett/res/{season}/eval_concepts.csv', index=0)

if season == "all":
    print("Benchmarks and Baselines")
    dictionc = {'system': [], 'f1': [], 'f2': [], 'prec': [], 'rec': [], 'dld': [], 'length': []}
    dictione = {'system': [], 'f1': [], 'f2': [], 'prec': [], 'rec': [], 'dld': [], 'length': []}

    for sys in tqdm(other_systems):
        cscore = concepts.loc[concepts['systems'] == sys]
        escore = entities.loc[entities['systems'] == sys]

        dictionc['system'].append(sys)
        dictionc['f2'].append(cscore['f2'].values[0])
        dictionc['prec'].append(cscore['prec'].values[0])
        dictionc['rec'].append(cscore['rec'].values[0])
        dictionc['f1'].append(cscore['f1'].values[0])
        dictionc['dld'].append(cscore['dld'].values[0])
        dictionc['length'].append(float(f"{lens[sys]:.2f}"))
        
        dictione['system'].append(sys)
        dictione['f2'].append(escore['f2'].values[0])
        dictione['prec'].append(escore['prec'].values[0])
        dictione['rec'].append(escore['rec'].values[0])
        dictione['f1'].append(escore['f1'].values[0])
        dictione['dld'].append(escore['dld'].values[0])
        dictione['length'].append(float(f"{lens[sys]:.2f}"))

    dfc = pd.DataFrame(dictionc)
    dfc.to_csv(f'sportsett/res/{season}/eval_concepts_other.csv', index=0)
    dfe = pd.DataFrame(dictione)
    dfe.to_csv(f'sportsett/res/{season}/eval_entities_other.csv', index=0)
