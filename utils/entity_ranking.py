import numpy as np
from utils.utils import GetEntityRepresentation

class RankEntities:
    def __init__(self):
        self.players_weights = np.load('ranking_outs/players_weights.npy', allow_pickle=True)
        self.teams_weights = np.load('ranking_outs/teams_weights.npy', allow_pickle=True)
        self.gep_obj = GetEntityRepresentation()
        self.NUM_PLAYERS = 13
        self.players_weights1 = np.ones(self.players_weights.shape)
        self.teams_weights1 = np.ones(self.teams_weights.shape)


    def get_ranked_players(self, item):
        hbs = item['teams']['home']['box_score']
        vbs = item['teams']['vis']['box_score']
        hpts = item['teams']['home']['line_score']['game']['PTS']
        vpts = item['teams']['vis']['line_score']['game']['PTS']
        home_players_rank_val, vis_players_rank_val = [], []

        for player in hbs[:self.NUM_PLAYERS]:
            winner = 1 if hpts > vpts else 0
            player_data = np.array(list(self.gep_obj.get_one_player_data(player, winner=winner).values()))
            home_players_rank_val.append(player_data.dot(self.players_weights1))

        for player in vbs[:self.NUM_PLAYERS]:
            winner = 1 if hpts < vpts else 0
            player_data = np.array(list(self.gep_obj.get_one_player_data(player, winner=winner).values()))
            vis_players_rank_val.append(player_data.dot(self.players_weights1))

        all_players_rank_val = home_players_rank_val + vis_players_rank_val
        all_players_rank_sorted_idx = np.argsort(all_players_rank_val)[::-1]

        final_sorted_players = []
        for rank in all_players_rank_sorted_idx:
            if rank > len(home_players_rank_val)-1:
                # its a vis player
                final_sorted_players.append(item['teams']['vis']['box_score'][rank - len(home_players_rank_val) - 1]['name'])
            else:
                # its a home player
                final_sorted_players.append(item['teams']['home']['box_score'][rank]['name'])

        return final_sorted_players

    def get_ranked_player_team_comb(self, item):
        hbs = item['teams']['home']['box_score']
        vbs = item['teams']['vis']['box_score']
        hls = item['teams']['home']['line_score']['game']
        vls = item['teams']['vis']['line_score']['game']
        hpts = int(hls['PTS'])
        vpts = int(vls['PTS'])
        winner = 'HOME' if hpts > vpts else 'VIS'

        hrep = np.array(list(self.gep_obj.get_team_line(item, type='HOME', winner=winner).values()))
        vrep = np.array(list(self.gep_obj.get_team_line(item, type='VIS', winner=winner).values()))
        hbs_rep = np.array([list(i.values()) for i in self.gep_obj.get_box_score(item, type='HOME')])
        vbs_rep = np.array([list(i.values()) for i in self.gep_obj.get_box_score(item, type='VIS')])

        ent_names = []
        ent_rank_vals = []
        for idx, p in enumerate(hbs[:self.NUM_PLAYERS]):
            ent_rank_vals.append(hrep.dot(self.teams_weights1) + hbs_rep[idx].dot(self.players_weights1))
            ent_names.append(f"{p['name']} & {item['teams']['home']['place']} {item['teams']['home']['name']}")
        for idx, p in enumerate(vbs[:self.NUM_PLAYERS]):
            ent_rank_vals.append(vrep.dot(self.teams_weights1) + vbs_rep[idx].dot(self.players_weights1))
            ent_names.append(f"{p['name']} & {item['teams']['vis']['place']} {item['teams']['vis']['name']}")

        ent_names_ranked = [ent_names[i] for i in np.argsort(ent_rank_vals)[::-1]]
        return ent_names_ranked

    def get_ranked_players_comb(self, item):
        hbs = item['teams']['home']['box_score']
        vbs = item['teams']['vis']['box_score']

        hbs_rep = np.array([list(i.values()) for i in self.gep_obj.get_box_score(item, type='HOME')])
        vbs_rep = np.array([list(i.values()) for i in self.gep_obj.get_box_score(item, type='VIS')])

        ent_names = []
        ent_rank_vals = []
        for idx1, p1 in enumerate(hbs[:self.NUM_PLAYERS]):
            for idx2, p2 in enumerate(hbs[idx1+1:self.NUM_PLAYERS]):
                ent_rank_vals.append(hbs_rep[idx1].dot(self.players_weights1) + hbs_rep[idx2].dot(self.players_weights1))
                ent_names.append(f"{p1['name']} & {p2['name']}")
        for idx1, p1 in enumerate(vbs[:self.NUM_PLAYERS]):
            for idx2, p2 in enumerate(vbs[idx1+1:self.NUM_PLAYERS]):
                ent_rank_vals.append(vbs_rep[idx1].dot(self.players_weights1) + vbs_rep[idx2].dot(self.players_weights1))
                ent_names.append(f"{p1['name']} & {p2['name']}")

        ent_names_ranked = [ent_names[i] for i in np.argsort(ent_rank_vals)[::-1]]
        return ent_names_ranked

    def get_ranked_teams_comb(self, item):
        hls = item['teams']['home']['line_score']['game']
        vls = item['teams']['vis']['line_score']['game']
        hpts, vpts = int(hls['PTS']), int(vls['PTS'])
        hteam = f"{item['teams']['home']['place']} {item['teams']['home']['name']}"
        vteam = f"{item['teams']['vis']['place']} {item['teams']['vis']['name']}"
        winner = 'HOME' if hpts > vpts else 'VIS'
        team_comb_str = f"{hteam} & {vteam}" if winner == 'HOME' else f"{vteam} & {hteam}"
        return [team_comb_str]
    
    def get_ranked_teams(self, item):
        hls = item['teams']['home']['line_score']['game']
        vls = item['teams']['vis']['line_score']['game']
        hpts, vpts = int(hls['PTS']), int(vls['PTS'])
        hteam = f"{item['teams']['home']['place']} {item['teams']['home']['name']}"
        vteam = f"{item['teams']['vis']['place']} {item['teams']['vis']['name']}"
        winner = 'HOME' if hpts > vpts else 'VIS'
        teams_ranked = [hteam, vteam] if winner == 'HOME' else [vteam, hteam]
        return teams_ranked
