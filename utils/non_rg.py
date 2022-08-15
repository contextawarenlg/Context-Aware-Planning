"""
# usage:
    python non_rg_metrics.py gold_tuple_fi pred_tuple_fi
"""
import json, argparse
import pandas as pd
from tqdm import tqdm
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance

class NonRGMetrics:

    full_names = ['Atlanta Hawks', 'Boston Celtics', 'Brooklyn Nets',
                'Charlotte Hornets', 'Chicago Bulls', 'Cleveland Cavaliers',
                'Detroit Pistons', 'Indiana Pacers', 'Miami Heat',
                'Milwaukee Bucks', 'New York Knicks', 'Orlando Magic',
                'Philadelphia 76ers', 'Toronto Raptors', 'Washington Wizards',
                'Dallas Mavericks', 'Denver Nuggets', 'Golden State Warriors',
                'Houston Rockets', 'Los Angeles Clippers', 'Los Angeles Lakers',
                'Memphis Grizzlies', 'Minnesota Timberwolves',
                'New Orleans Pelicans', 'Oklahoma City Thunder', 'Phoenix Suns',
                'Portland Trail Blazers', 'Sacramento Kings', 'San Antonio Spurs',
                'Utah Jazz']

    cities, teams = set(), set()
    ec = {}  # equivalence classes
    for team in full_names:
        pieces = team.split()
        if len(pieces) == 2:
            ec[team] = [pieces[0], pieces[1]]
            cities.add(pieces[0])
            teams.add(pieces[1])
        elif pieces[0] == "Portland":  # only 2-word team
            ec[team] = [pieces[0], " ".join(pieces[1:])]
            cities.add(pieces[0])
            teams.add(" ".join(pieces[1:]))
        else:  # must be a 2-word City
            ec[team] = [" ".join(pieces[:2]), pieces[2]]
            cities.add(" ".join(pieces[:2]))
            teams.add(pieces[2])


    def same_ent(self, e1, e2):
        if e1 in self.cities or e1 in self.teams:
            return e1 == e2 or any((e1 in fullname and e2 in fullname for fullname in self.full_names))
        else:
            return e1 in e2 or e2 in e1


    def trip_match(self, t1, t2, ent_or_concept='concepts'):
        if ent_or_concept == 'concepts':
            return t1[1] == t2[1]
        elif ent_or_concept == 'entities':
            return self.same_ent(t1[0], t2[0])


    def dedup_triples(self, triplist):
        """
        this will be inefficient but who cares
        """
        dups = set()
        for i in range(1, len(triplist)):
            for j in range(i):
                if self.trip_match(triplist[i], triplist[j]):
                    dups.add(i)
                    break
        return [thing for i, thing in enumerate(triplist) if i not in dups]


    def get_triples(self, fi):
        all_triples = []
        curr = []
        with open(fi) as f:
            for line in f:
                if line.isspace():
                    all_triples.append(self.dedup_triples(curr))
                    curr = []
                else:
                    pieces = line.strip().split('|')
                    curr.append(tuple(pieces))
        if len(curr) > 0:
            all_triples.append(self.dedup_triples(curr))
        return all_triples


    def get_triples_new(self, fi, ent_or_concept='concepts'):
        delim = "|"
        js = json.load(open(f'{fi}'))
        data = []
        for item in js:
            temp = []
            for val in item:
                if ent_or_concept == 'concepts':
                    temp.append(tuple(['', val.split(delim)[1]])) # only using the concepts
                elif ent_or_concept == 'entities':
                    ents = val.split(delim)[0].split(' & ')
                    for ent in ents:
                        temp.append(tuple([ent, '']))
            data.append(temp)
        return data


    def calc_precrec(self, goldfi, predfi, ent_or_concept='concepts'):

        gold_triples = self.get_triples_new(goldfi, ent_or_concept=ent_or_concept)
        pred_triples = self.get_triples_new(predfi, ent_or_concept=ent_or_concept)

        total_tp, total_predicted, total_gold = 0, 0, 0
        assert len(gold_triples) == len(pred_triples)

        for i, triplist in enumerate(pred_triples):
            # tp = sum(
            #         (1 for j in range(len(triplist))
            #             if any(self.trip_match(triplist[j], gold_triples[i][k], ent_or_concept=ent_or_concept) 
            #                     for k in range(len(gold_triples[i]))
            #             )
            #         )
            #     )

            total_predicted += len(triplist)
            total_gold += len(gold_triples[i])

            tp = 0
            gold_triplist = gold_triples[i]
            for j, itemp in enumerate(triplist):
                if len(gold_triplist) > 0:
                    for k, itemg in enumerate(gold_triplist):
                        if self.trip_match(itemp, itemg, ent_or_concept=ent_or_concept):
                            tp += 1
                            del gold_triplist[k]
                            break

            total_tp += tp
        avg_prec = float(total_tp) / total_predicted
        avg_rec = float(total_tp) / total_gold
        avg_jac = float(total_tp) / (total_predicted + (total_gold - total_tp))
        avg_f1 = 2 * float(avg_prec * avg_rec) / (avg_prec + avg_rec)
        avg_dice = float(2 * total_tp) / (total_predicted + total_gold)
        avg_f2 = 5 * float(avg_prec * avg_rec) / ((4 * avg_prec) + avg_rec)
        return avg_prec, avg_rec, avg_jac, avg_f1, avg_dice, avg_f2


    def norm_dld(self, l1, l2, ent_or_concept='concepts'):
        ascii_start = 0
        # make a string for l1
        # all triples are unique...
        s1 = ''.join((chr(ascii_start + i) for i in range(len(l1))))
        s2 = ''
        next_char = ascii_start + len(s1)
        for j in range(len(l2)):
            found = None
            # next_char = chr(ascii_start+len(s1)+j)
            for k in range(len(l1)):
                if self.trip_match(l2[j], l1[k], ent_or_concept=ent_or_concept):
                    found = s1[k]
                    # next_char = s1[k]
                    break
            if found is None:
                s2 += chr(next_char)
                next_char += 1
                assert next_char <= 128
            else:
                s2 += found
        # return 1- , since this thing gives 0 to perfect matches etc
        return 1.0 - normalized_damerau_levenshtein_distance(s1, s2)


    def calc_dld(self, goldfi, predfi, ent_or_concept='concepts'):
        gold_triples = self.get_triples_new(goldfi, ent_or_concept=ent_or_concept)
        pred_triples = self.get_triples_new(predfi, ent_or_concept=ent_or_concept)
        assert len(gold_triples) == len(pred_triples)
        total_score = 0
        for i, triplist in enumerate(pred_triples):
            total_score += self.norm_dld(triplist, gold_triples[i], ent_or_concept=ent_or_concept)
        avg_score = float(total_score) / len(pred_triples)
        return avg_score


argparser = argparse.ArgumentParser()
argparser.add_argument('-season', '--season', type=str, default='all', \
                        choices=['2014', '2015', '2016', '2017', '2018', 'bens', 'all', 'juans'])
argparser.add_argument('-eoc', '--eoc', type=str, default='concepts', \
                        choices=['entities', 'concepts', 'len'])
args = argparser.parse_args()
args = argparser.parse_args()
season = args.season
ent_or_concept = args.eoc

obj = NonRGMetrics()

benchmarks = ['ent', 'hir', 'mp']
baselines = ['temp', 'cbr']
dists = ['euclidean', 'cosine']
reuses = ['median', 'first']
players = ['imp'] 
ftrs = ['set'] 

systems = baselines + benchmarks

for player in players:
    for dist in dists:
        for reuse in reuses:
            for ftr in ftrs:
                systems.append(f"{player}_players-{ftr}_ftrs-{dist}_sim-{reuse}_reuse")

systems.append(f"imp_players-set_ftrs-euclidean_sim-median_reuse-pop")
systems.append(f"imp_players-set_ftrs-euclidean_sim-median_reuse-tpop")
systems.append(f"imp_players-set_ftrs-euclidean_sim-median_reuse-week")
systems.append(f"imp_players-set_ftrs-euclidean_sim-median_reuse-stands")
systems.append(f"imp_players-set_ftrs-euclidean_sim-median_reuse-pop-tpop")
systems.append(f"imp_players-set_ftrs-euclidean_sim-median_reuse-stands-week")
systems.append(f"imp_players-set_ftrs-euclidean_sim-median_reuse-tpop-stands-week")
systems.append(f"imp_players-set_ftrs-euclidean_sim-median_reuse-pop-tpop-stands-week")

if season != "all":
    systems = [
        "imp_players-set_ftrs-euclidean_sim-median_reuse",
        "imp_players-set_ftrs-euclidean_sim-median_reuse-pop",
        "imp_players-set_ftrs-euclidean_sim-median_reuse-tpop",
        "imp_players-set_ftrs-euclidean_sim-median_reuse-week",
        "imp_players-set_ftrs-euclidean_sim-median_reuse-stands",
        "imp_players-set_ftrs-euclidean_sim-median_reuse-pop-tpop",
        "imp_players-set_ftrs-euclidean_sim-median_reuse-stands-week",
        "imp_players-set_ftrs-euclidean_sim-median_reuse-tpop-stands-week",
        "imp_players-set_ftrs-euclidean_sim-median_reuse-pop-tpop-stands-week"
    ]
print(f"season: {season}, ent_or_concept: {ent_or_concept}, total systems: {len(systems)}")
print(f"systems: {systems}")

if ent_or_concept != 'len':
    res = {}
    print(f'\nThis is {ent_or_concept}\n')
    for sys_name in tqdm(systems):
        sys_res = {}
        predfi = f'sportsett/concepts/{season}/{sys_name}.json'
        goldfi = f'sportsett/concepts/{season}/gold.json'
        try:
            prec, rec, jac, f1, dice, f2 = obj.calc_precrec(goldfi, predfi, ent_or_concept=ent_or_concept)
            dld = obj.calc_dld(goldfi, predfi, ent_or_concept=ent_or_concept)
            sys_res['f2'] = f"{f2*100:.2f}"
            sys_res['prec'] = f"{prec*100:.2f}"
            sys_res['rec'] = f"{rec*100:.2f}"
            sys_res['f1'] = f"{f1*100:.2f}"
            sys_res['dld'] = f"{dld*100:.2f}"
            res[sys_name] = sys_res
        except:
            pass
    df = pd.DataFrame(res)
    df.transpose().to_csv(f'sportsett/res/{season}/{ent_or_concept}.csv')

else:
    print(f'\nThis is {ent_or_concept}\n')
    systems.append('gold')
    lens = {}
    for sys_name in tqdm(systems):
        try:
            if sys_name == 'gold':
                tpls = obj.get_triples_new(f'sportsett/concepts/{season}/gold.json')
            else:
                tpls = obj.get_triples_new(f'sportsett/concepts/{season}/{sys_name}.json')
            avg = 0
            for tpl in tpls:
                avg += len(tpl)
            avg /= len(tpls)
            # print(f"{sys_name}\t{avg}")
            lens[sys_name] = avg
        except:
            pass
    json.dump(lens, open(f'sportsett/res/{season}/concept_lengths.json', 'w'), indent='\t')
