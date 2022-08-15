"""
Extract Concept Ordering from a given summary.
"""
import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

class ExtractEntities:
    alias_dict = {
        'Mavs': 'Mavericks',
        'Cavs': 'Cavaliers',
        'Sixers': '76ers',
    }

    def get_full_team_ents(self, team_ents_in_sent, item):
        teams_mentioned = [] #set()
        vname, vplace = item['teams']['vis']['name'], item['teams']['vis']['place']
        hname, hplace = item['teams']['home']['name'], item['teams']['home']['place']
        hnon, hnop = item['teams']['home']['next_game']['opponent_name'], item['teams']['home']['next_game']['opponent_place']
        vnon, vnop = item['teams']['vis']['next_game']['opponent_name'], item['teams']['vis']['next_game']['opponent_place']
        hng_ents = [hnon, hnop, f"{hnop} {hnon}"]
        vng_ents = [vnon, vnop, f"{vnop} {vnon}"]
        home_team = [hname, hplace, f'{hplace} {hname}']
        vis_team = [vname, vplace, f'{vplace} {vname}']
        home_flag = False
        vis_flag = False
        hng_flag = False
        vng_flag = False
        for ent in team_ents_in_sent:
            if ent in home_team:
                home_flag = True
            if ent in vis_team:
                vis_flag = True
            if ent in hng_ents:
                hng_flag = True
            if ent in vng_ents:
                vng_flag = True
        if home_flag:
            teams_mentioned.append(f'{hplace} {hname}')
        if vis_flag:
            teams_mentioned.append(f'{vplace} {vname}')
        if hng_flag:
            teams_mentioned.append(f"{hnop} {hnon}")
        if vng_flag:
            teams_mentioned.append(f"{vnop} {vnon}")
        teams_mentioned_final = []
        for ent in teams_mentioned:
            if ent not in teams_mentioned_final:
                teams_mentioned_final.append(ent)
        return teams_mentioned_final
        # return teams_mentioned
        # return list(teams_mentioned)

    def get_full_player_ents(self, player_ents_in_sent, item):

        hbs = item['teams']['home']['box_score']
        vbs = item['teams']['vis']['box_score']

        home_full_names = [player['name'] for player in hbs]
        home_first_names = [player['first_name'] for player in hbs]
        home_last_names = [player['last_name'] for player in hbs]

        vis_full_names = [player['name'] for player in vbs]
        vis_first_names = [player['first_name'] for player in vbs]
        vis_last_names = [player['last_name'] for player in vbs]

        home_player_mention_idxs, vis_player_mention_idxs = [], []

        for player_ent in player_ents_in_sent:
            home_player_idx = -1
            vis_player_idx = -1
            if player_ent in home_full_names:
                for idx2, player in enumerate(hbs):
                    if player['name'] == player_ent:
                        home_player_idx = idx2
            elif player_ent in home_first_names:
                for idx2, player in enumerate(hbs):
                    if player['first_name'] == player_ent:
                        home_player_idx = idx2
            elif player_ent in home_last_names:
                for idx2, player in enumerate(hbs):
                    if player['last_name'] == player_ent:
                        home_player_idx = idx2

            if player_ent in vis_full_names:
                for idx2, player in enumerate(vbs):
                    if player['name'] == player_ent:
                        vis_player_idx = idx2
            elif player_ent in vis_first_names:
                for idx2, player in enumerate(vbs):
                    if player['first_name'] == player_ent:
                        vis_player_idx = idx2
            elif player_ent in vis_last_names:
                for idx2, player in enumerate(vbs):
                    if player['last_name'] == player_ent:
                        vis_player_idx = idx2

            if home_player_idx != -1:
                home_player_mention_idxs.append(home_player_idx)
            if vis_player_idx != -1:
                vis_player_mention_idxs.append(vis_player_idx)

        full_player_ents = []
        for i in home_player_mention_idxs:
            full_player_ents.append(hbs[i]['name'])
        for i in vis_player_mention_idxs:
            full_player_ents.append(vbs[i]['name'])

        players_final = []
        for ent in full_player_ents:
            if ent not in players_final:
                players_final.append(ent)
        return players_final
        # return full_player_ents
        # print(player_ents_in_sent, full_player_ents)

    def get_all_ents(self, score_dict):
        players = []#set()
        teams = []#set()

        teams.append(score_dict['teams']['home']['name'])
        teams.append(score_dict['teams']['vis']['name'])
        
        teams.append(score_dict['teams']['home']['place'])
        teams.append(score_dict['teams']['vis']['place'])
        
        teams.append(f"{score_dict['teams']['home']['place']} {score_dict['teams']['home']['name']}")
        teams.append(f"{score_dict['teams']['vis']['place']} {score_dict['teams']['vis']['name']}")

        teams.append(score_dict['teams']['home']['next_game']['opponent_name'])
        teams.append(score_dict['teams']['vis']['next_game']['opponent_name'])

        teams.append(score_dict['teams']['home']['next_game']['opponent_place'])
        teams.append(score_dict['teams']['vis']['next_game']['opponent_place'])

        teams.append(f"{score_dict['teams']['home']['next_game']['opponent_place']} {score_dict['teams']['home']['next_game']['opponent_name']}")
        teams.append(f"{score_dict['teams']['vis']['next_game']['opponent_place']} {score_dict['teams']['vis']['next_game']['opponent_name']}")

        for player in score_dict['teams']['home']['box_score']:
            players.append(player['first_name'])
            players.append(player['last_name'])
            players.append(player['name'])

        for player in score_dict['teams']['vis']['box_score']:
            players.append(player['first_name'])
            players.append(player['last_name'])
            players.append(player['name'])

        teams_final, players_final = [], []
        for ent in teams:
            if ent not in teams_final:
                teams_final.append(ent)
        for ent in players:
            if ent not in players_final:
                players_final.append(ent)
        
        # all_ents = teams | players
        all_ents = teams + players

        return all_ents, teams, players

    def extract_entities(self, all_ents, sent):

        new_toks = []
        for tok in sent.split(' '):
            if tok in self.alias_dict:
                new_toks.append(self.alias_dict[tok])
            else:
                new_toks.append(tok)
        new_sent = ' '.join(new_toks)

        toks = new_sent.split(' ')
        sent_ents = []
        i = 0
        while i < len(toks):
            if toks[i] in all_ents:
                j = 1
                while i+j <= len(toks) and " ".join(toks[i:i+j]) in all_ents:
                    j += 1
                sent_ents.append(" ".join(toks[i:i+j-1]))
                i += j-1
            else:
                i += 1
        sent_ents_final = []
        for ent in sent_ents:
            if ent not in sent_ents_final:
                sent_ents_final.append(ent)
        return sent_ents_final
        # return list(set(sent_ents))


class ExtractConceptOrder:
    clust_keys = [
        'P-B', 'P-W', 'P-A', 'P-B&W', 'P-B&A', 'P-W&A', 'P-B&W&A',
        'T-B', 'T-W', 'T-A', 'T-B&W', 'T-B&A', 'T-W&A', 'T-B&W&A',
        'P&P-B', 'P&P-W', 'P&P-A', 'P&P-B&W', 'P&P-B&A', 'P&P-W&A', 'P&P-B&W&A',
        'T&T-B', 'T&T-W', 'T&T-A', 'T&T-B&W', 'T&T-B&A', 'T&T-W&A', 'T&T-B&W&A',
        'P&T-B', 'P&T-W', 'P&T-A', 'P&T-B&W', 'P&T-B&A', 'P&T-W&A', 'P&T-B&W&A'
    ]

    def __init__(self):
        """
        score_dict: Datasets dict item for a single game
        summary_sentences: List of sentences from the summary
        """
        self.delim = '|'
        self.ents = ExtractEntities()
        self.clust_dict = {key: [] for key in self.clust_keys}
        self.ng_sent_clf = pickle.load(open(f'data/ng_sent_clf.pkl', 'rb'))
        self.embedding_model = SentenceTransformer('bert-base-nli-mean-tokens')

    def extract_concept_order(self, score_dict, summary_sentences):
        concept_order = []
        all_ents, teams, players = self.ents.get_all_ents(score_dict)

        # emb_sents = self.embedding_model.encode([x['coref_sent'] for x in summary_sentences[1:]])
        # ng_sent_lab = self.ng_sent_clf.predict(emb_sents)

        for idx, sent in enumerate(list(summary_sentences)[1:]):
            # if ng_sent_lab[idx] == 0:
            player_ents_unresolved = self.ents.extract_entities(players, sent['coref_sent'])
            team_ents_unresolved = self.ents.extract_entities(teams, sent['coref_sent'])
            # "_unresolved" because some extracted entities may be either first or last name, or in case of teams, either just name or place of the team
            # we want to make sure that we don't have any duplicates in the list of entities and also entities have full name 
            # (both place & name in case of teams and first & last name in case of players)
            player_ents = self.ents.get_full_player_ents(player_ents_unresolved, score_dict)
            team_ents = self.ents.get_full_team_ents(team_ents_unresolved, score_dict)

            if len(player_ents) == 1 and len(team_ents) == 0:
                ent_str = f'{player_ents[0]}'
                if 'B' in sent['content_types'] and 'W' in sent['content_types'] and 'A' in sent['content_types']:
                    concept_order.append(f'{ent_str}{self.delim}P-B&W&A')
                elif 'B' in sent['content_types'] and 'W' in sent['content_types']:
                    concept_order.append(f'{ent_str}{self.delim}P-B&W')
                elif 'B' in sent['content_types'] and 'A' in sent['content_types']:
                    concept_order.append(f'{ent_str}{self.delim}P-B&A')
                elif 'W' in sent['content_types'] and 'A' in sent['content_types']:
                    concept_order.append(f'{ent_str}{self.delim}P-W&A')
                elif 'B' in sent['content_types']:
                    concept_order.append(f'{ent_str}{self.delim}P-B')
                elif 'W' in sent['content_types']:
                    concept_order.append(f'{ent_str}{self.delim}P-W')
                elif 'A' in sent['content_types']:
                    concept_order.append(f'{ent_str}{self.delim}P-A')
            
            if len(team_ents) == 1 and len(player_ents) == 0:
                ent_str = f'{team_ents[0]}'
                if 'B' in sent['content_types'] and 'W' in sent['content_types'] and 'A' in sent['content_types']:
                    concept_order.append(f'{ent_str}{self.delim}T-B&W&A')
                elif 'B' in sent['content_types'] and 'W' in sent['content_types']:
                    concept_order.append(f'{ent_str}{self.delim}T-B&W')
                elif 'B' in sent['content_types'] and 'A' in sent['content_types']:
                    concept_order.append(f'{ent_str}{self.delim}T-B&A')
                elif 'W' in sent['content_types'] and 'A' in sent['content_types']:
                    concept_order.append(f'{ent_str}{self.delim}T-W&A')
                elif 'B' in sent['content_types']:
                    concept_order.append(f'{ent_str}{self.delim}T-B')
                elif 'W' in sent['content_types']:
                    concept_order.append(f'{ent_str}{self.delim}T-W')
                elif 'A' in sent['content_types']:
                    concept_order.append(f'{ent_str}{self.delim}T-A')

            if len(player_ents) > 0 and len(team_ents) > 0:
                ent_str_list = [player_ents[i] for i in range(len(player_ents))]
                for te in team_ents:
                    ent_str_list.append(te)
                ent_str = f'{" & ".join(ent_str_list)}'
                if 'B' in sent['content_types'] and 'W' in sent['content_types'] and 'A' in sent['content_types']:
                    concept_order.append(f'{ent_str}{self.delim}P&T-B&W&A')
                elif 'B' in sent['content_types'] and 'W' in sent['content_types']:
                    concept_order.append(f'{ent_str}{self.delim}P&T-B&W')
                elif 'B' in sent['content_types'] and 'A' in sent['content_types']:
                    concept_order.append(f'{ent_str}{self.delim}P&T-B&A')
                elif 'W' in sent['content_types'] and 'A' in sent['content_types']:
                    concept_order.append(f'{ent_str}{self.delim}P&T-W&A')
                elif 'B' in sent['content_types']:
                    concept_order.append(f'{ent_str}{self.delim}P&T-B')
                elif 'W' in sent['content_types']:
                    concept_order.append(f'{ent_str}{self.delim}P&T-W')
                elif 'A' in sent['content_types']:
                    concept_order.append(f'{ent_str}{self.delim}P&T-A')

            if len(player_ents) > 1 and len(team_ents) == 0:
                ent_str_list = [player_ents[i] for i in range(len(player_ents))]
                ent_str = f'{" & ".join(ent_str_list)}'
                # print(f'\n{ent_str}\n{player_ents}')
                if 'B' in sent['content_types'] and 'W' in sent['content_types'] and 'A' in sent['content_types']:
                    concept_order.append(f'{ent_str}{self.delim}P&P-B&W&A')
                elif 'B' in sent['content_types'] and 'W' in sent['content_types']:
                    concept_order.append(f'{ent_str}{self.delim}P&P-B&W')
                elif 'B' in sent['content_types'] and 'A' in sent['content_types']:
                    concept_order.append(f'{ent_str}{self.delim}P&P-B&A')
                elif 'W' in sent['content_types'] and 'A' in sent['content_types']:
                    concept_order.append(f'{ent_str}{self.delim}P&P-W&A')
                elif 'B' in sent['content_types']:
                    concept_order.append(f'{ent_str}{self.delim}P&P-B')
                elif 'W' in sent['content_types']:
                    concept_order.append(f'{ent_str}{self.delim}P&P-W')
                elif 'A' in sent['content_types']:
                    concept_order.append(f'{ent_str}{self.delim}P&P-A')

            if len(player_ents) == 0 and len(team_ents) > 1:
                ent_str_list = [team_ents[i] for i in range(len(team_ents))]
                ent_str = f'{" & ".join(ent_str_list)}'
                # print(f'\n{ent_str}\n{team_ents}\n')
                if 'B' in sent['content_types'] and 'W' in sent['content_types'] and 'A' in sent['content_types']:
                    concept_order.append(f'{ent_str}{self.delim}T&T-B&W&A')
                elif 'B' in sent['content_types'] and 'W' in sent['content_types']:
                    concept_order.append(f'{ent_str}{self.delim}T&T-B&W')
                elif 'B' in sent['content_types'] and 'A' in sent['content_types']:
                    concept_order.append(f'{ent_str}{self.delim}T&T-B&A')
                elif 'W' in sent['content_types'] and 'A' in sent['content_types']:
                    concept_order.append(f'{ent_str}{self.delim}T&T-W&A')
                elif 'B' in sent['content_types']:
                    concept_order.append(f'{ent_str}{self.delim}T&T-B')
                elif 'W' in sent['content_types']:
                    concept_order.append(f'{ent_str}{self.delim}T&T-W')
                elif 'A' in sent['content_types']:
                    concept_order.append(f'{ent_str}{self.delim}T&T-A')
            
        return concept_order


class GetEntityRepresentation:
    def __init__(self, popularity=False, team_popularity=False, season="all", standing=False, season_week=False):
        """
        Initialize the class.
        Parameters:
            popularity (bool): If True, use popularity.
        """
        self.NUM_PLAYERS = 13
        self.bs_keys = ['PTS', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 
                        'OREB', 'DREB', 'TREB', 'AST', 'STL', 'BLK', 'TOV', 'PF']
        self.ls_keys = ['FG3A', 'FG3M', 'FGA', 'FGM', 'FTA', 'FTM', 'DREB', 'OREB', 
                        'TREB', 'BLK', 'AST', 'STL', 'TOV', 'PF', 'PTS', 'MIN']
        self.season = season
        self.popularity = popularity
        self.team_popularity = team_popularity
        self.standing = standing
        self.season_week = season_week

        if self.popularity:
            self.pop_score = json.load(open(f'popularity/{season}/pop_score.json'))
        if self.team_popularity:
            self.team_pop_score = json.load(open(f'popularity/{season}/pop_score-team.json'))
        
        self.team_standings = json.load(open(f'data/team_standings.json'))

    def sort_players_by_pts(self, entry, type='HOME'):
        """
        Sort players by points and return the indices sorted by points
        bs --> [{'pts': 10}, {'pts': 30}, {'pts': 35}, {'pts': 5}]
        return --> [2, 1, 0, 3]
        """
        all_pts = [int(item['PTS']) for item in entry['teams'][type.lower()]['box_score']]
        all_pts1 = [[item, idx] for idx, item in enumerate(all_pts)]
        all_pts1.sort()
        all_pts1.reverse()
        return [item[1] for item in all_pts1]

    def get_one_player_data(self, player_stats, winner=1):
        """
        basic ones: 15 ['PTS', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'TREB', 'AST', 'STL', 'BLK', 'TOV', 'PF']
        double, winner, starter, min: 4
        pop: 1
        total: 20
        """
        starter = 1 if player_stats['starter'] == True else 0
        double = 0 
        if player_stats['DOUBLE'] == 'double':
            double = 1
        elif player_stats['DOUBLE'] == 'triple':
            double = 2
        elif player_stats['DOUBLE'] == 'quad':
            double = 3
        player_min = int(player_stats['MIN'])
        player_dict = {'PLAYER-starter': starter, 'PLAYER-double': double, 'PLAYER-MIN': player_min, 'PLAYER-winner': winner}
        for key in self.bs_keys:
            player_dict[f"PLAYER-{key}"] = int(player_stats[key])
        if self.popularity:
            player_dict['PLAYER-popularity'] = float(self.pop_score[player_stats['name']]) if player_stats['name'] in self.pop_score else 0.0
        return player_dict

    def get_empty_bs_dict(self, winner=0):
        bs_dict = {'PLAYER-starter': 0, 'PLAYER-double': 0, 'PLAYER-MIN': 0, 'PLAYER-winner': winner}
        for key in self.bs_keys:
            bs_dict[f"PLAYER-{key}"] = 0
        if self.popularity:
            bs_dict['PLAYER-popularity'] = 0
        return bs_dict

    def get_box_score(self, entry, type='HOME'):
        bs = entry['teams'][type.lower()]['box_score']
        sorted_idx = self.sort_players_by_pts(entry, type)
        player_lines = [self.get_one_player_data(bs[idx]) for rank, idx in enumerate(sorted_idx)]
        if len(player_lines) < self.NUM_PLAYERS:
            player_lines += [self.get_empty_bs_dict() for _ in range(self.NUM_PLAYERS - len(player_lines))]
        return player_lines[:self.NUM_PLAYERS]

    def get_team_line(self, entry, type='HOME', winner='HOME'):
        """
        basic ones: 16 ['FG3A', 'FG3M', 'FGA', 'FGM', 'FTA', 'FTM', 'DREB', 'OREB', 'TREB', 'BLK', 'AST', 'STL', 'TOV', 'PF', 'PTS', 'MIN']
        4 quarters: 4
        winner, wins, losses: 3
        popularity: 1
        season_week, season_date, standing: 3
        total: 16 + 4 + 3 + 1 + 3 = 27
        """
        line_score = entry['teams'][type.lower()]['line_score']['game']
        team_name = entry['teams'][type.lower()]['name']
        line_score_dict = {}
        for key in self.ls_keys:
            line_score_dict[f"TEAM-{key}"] = int(line_score[key])
        for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:
            line_score_dict[f'TEAM-PTS_{quarter}'] = int(entry['teams'][type.lower()]['line_score'][quarter]['PTS'])
        team_winner = 1 if winner == type else 0
        line_score_dict['TEAM-IS_WINNER'] = team_winner
        line_score_dict['TEAM-WINS'] = int(entry['teams'][type.lower()]['wins'])
        line_score_dict['TEAM-LOSSES'] = int(entry['teams'][type.lower()]['losses'])
        if self.team_popularity:
            line_score_dict['TEAM-popularity'] = float(self.team_pop_score[team_name]) if team_name in self.team_pop_score else 0.0

        team_standing = int(self.team_standings[entry['gem_id']][type.lower()]['current']['standing'])
        season_date = int(self.team_standings[entry['gem_id']][type.lower()]['current']['season_date'])
        season_week = season_date//7 + 1
        if self.season_week:
            line_score_dict['TEAM-SEASON_WEEK'] = season_week
            line_score_dict['TEAM-SEASON_DATE'] = season_date
        if self.standing:
            line_score_dict['TEAM-STANDING'] = team_standing

        return line_score_dict


class GetGameRepresentation:
    def __init__(self):
        self.gep_obj = GetEntityRepresentation()

    def get_full_game_repr(self, entry):
        all_features_for_orbering_cb = []
        home_team_pts = entry['teams']['home']['line_score']['game']['PTS']
        vis_team_pts = entry['teams']['home']['line_score']['game']['PTS']
        winner = 'HOME' if int(home_team_pts) > int(vis_team_pts) else 'VIS'

        if winner == 'HOME':
            t1_line_score_dict = self.gep_obj.get_team_line(entry, type='HOME', winner='HOME')
            t2_line_score_dict = self.gep_obj.get_team_line(entry, type='VIS', winner='HOME')
            t1_bs = self.gep_obj.get_box_score(entry, type='HOME')
            t2_bs = self.gep_obj.get_box_score(entry, type='VIS')
        elif winner == 'VIS':
            t1_line_score_dict = self.gep_obj.get_team_line(entry, type='VIS', winner='VIS')
            t2_line_score_dict = self.gep_obj.get_team_line(entry, type='HOME', winner='VIS')
            t1_bs = self.gep_obj.get_box_score(entry, type='VIS')
            t2_bs = self.gep_obj.get_box_score(entry, type='HOME')

        t1ps = [t1_line_score_dict | player for player in t1_bs]
        t2ps = [t2_line_score_dict | player for player in t2_bs]

        t1_pps = []
        for idx1, player1 in enumerate(t1_bs):
            for idx2, player2 in enumerate(t1_bs[idx1+1:]):
                temp1 = {f"{k}-P1": v for k, v in player1.items()}
                temp2 = {f"{k}-P2": v for k, v in player2.items()}
                t1_pps.append(temp1 | temp2)
                # t1_pps.append(player1 | player2)
        t2_pps = []
        for idx1, player1 in enumerate(t2_bs):
            for idx2, player2 in enumerate(t2_bs[idx1+1:]):
                temp1 = {f"{k}-P1": v for k, v in player1.items()}
                temp2 = {f"{k}-P2": v for k, v in player2.items()}
                t1_pps.append(temp1 | temp2)
                # t2_pps.append(player1 | player2)

        all_features_for_orbering_cb.extend([t1_line_score_dict, t2_line_score_dict])
        all_features_for_orbering_cb.extend(t1_bs)
        all_features_for_orbering_cb.extend(t2_bs)
        all_features_for_orbering_cb.extend(t1ps)
        all_features_for_orbering_cb.extend(t2ps)
        all_features_for_orbering_cb.extend(t1_pps)
        all_features_for_orbering_cb.extend(t2_pps)

        final_rep = [np.mean(list(feature_dict.values())) for feature_dict in all_features_for_orbering_cb]
        return np.array(final_rep)