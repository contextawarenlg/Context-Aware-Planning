"""
Use alignment for feature weighting
Taken from:
@inproceedings{upadhyay2021case,
  title={A Case-Based Approach to Data-to-Text Generation},
  author={Upadhyay, Ashish and Massie, Stewart and Singh, Ritwik Kumar and Gupta, Garima and Ojha, Muneendra},
  booktitle={International Conference on Case-Based Reasoning},
  pages={232--247},
  year={2021},
  organization={Springer}
}
"""
import json
import random
import argparse
import numpy as np
import pyswarms as ps
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
random.seed(42)

class CaseAlignMeasure:
    def __init__(self, cb_sols: list, cb_probs, num_ftrs, ftr_names, top_k=100):
        self.top_k = top_k
        self.cb_sols = cb_sols
        self.cb_probs = cb_probs
        self.num_ftrs = num_ftrs
        self.ftr_names = ftr_names

    def calc_f1(self, pred_lists, gold_list: list):
        all_f1s = []
        for _, triplist in enumerate(pred_lists):
            tp = 0
            gold_triplist = gold_list.copy()
            for j, itemp in enumerate(triplist):
                if len(gold_triplist) > 0:
                    for k, itemg in enumerate(gold_triplist):
                        if itemp == itemg:
                            tp += 1
                            del gold_triplist[k]
                            break

            prec = tp / len(triplist)
            rec = tp / len(gold_list)
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            all_f1s.append(f1)
        return all_f1s


    def get_align_score(self, problem_list, solution_list):
        """
        input:
            problem/solution lists
        return:
            align score
        """
        prob_rank = np.copy(problem_list)
        sol_rank = np.copy(solution_list)

        prob_rank[self.top_k:] = 1
        for i in range(self.top_k):
            prob_rank[i] = (self.top_k + 1) - i

        sol_rank[:] = 1
        for i in range(self.top_k):
            sol_ind = solution_list.index(problem_list[i])
            sol_rank[sol_ind] = (self.top_k + 1) - i

        ndcg = ndcg_score([prob_rank[:self.top_k]], [sol_rank[:self.top_k]])
        return ndcg


def problem_lists_function(p):
    """ Calculate roll-back the weights and biases

    Inputs
    ------
    p: np.ndarray
        The dimensions should include an unrolled version of the
        weights and biases.

    Returns
    -------

    This should give the list of problem side lists for all examples in the train set
    """

    problem_lists, solutions = [], []
    for idx, case in enumerate(cam_obj.cb_probs):
        target_problem_arr = np.multiply(case, p)
        case_base_arr = np.multiply(np.delete(cam_obj.cb_probs, idx, axis=0), p)
        dists = euclidean_distances(case_base_arr, [target_problem_arr])
        dists_1d = dists.ravel()
        dists_arg = np.argsort(dists_1d)
        problem_lists.append(dists_arg)
        solutions.append(cam_obj.cb_sols[dists_arg[1]])

    return problem_lists, solutions

# Forward propagation
def forward_prop(params):
    """Forward propagation as objective function

    This computes for the forward propagation of the neural network, as
    well as the loss.

    Inputs
    ------
    params: np.ndarray
        The dimensions should include an unrolled version of the
        weights and biases.

    Returns
    -------
    float
        The computed case-alignment given the parameters

    1. get the problem_side lists and their corresponding best solution index for all the cases from previous function 
    2. get the solution_side lists for all the cases
    3. calculate the align score for whole case-base - this should be our loss function
    """

    problem_lists, generated_solutions = problem_lists_function(params)
    solution_side_cb = cam_obj.cb_sols

    solution_lists = []
    for idx, case in enumerate(solution_side_cb):
        target_solution_arr = generated_solutions[idx]
        case_base_solution_list = solution_side_cb.copy()
        del case_base_solution_list[idx]
        dists = cam_obj.calc_f1(case_base_solution_list, [target_solution_arr])
        dists_arg = np.argsort(dists)
        solution_lists.append(dists_arg)

    all_align = []
    for pl, sl in zip(problem_lists, solution_lists):
        all_align.append(cam_obj.get_align_score(pl.tolist(), sl.tolist()))

    loss = 1 - np.mean(np.array(all_align))
    print("loss", loss)

    return loss

def f(x):
    """Higher-level method to do forward_prop in the
    whole swarm.

    Inputs
    ------
    x: numpy.ndarray of shape (n_particles, dimensions)
        The swarm that will perform the search

    Returns
    -------
    numpy.ndarray of shape (n_particles, )
        The computed loss for each particle
    """
    n_particles = x.shape[0]
    j = [forward_prop(x[i]) for i in range(n_particles)]
    return np.array(j)

def train_pso(cam_obj):
    print("Initialize swarm")
    options = {'c1':2, 'c2':2, 'w':1}
    dimensions = cam_obj.num_ftrs
    bounds = (np.array([0]*dimensions), np.array([1]*dimensions))

    print("Call instance of PSO")
    optimizer = ps.single.GlobalBestPSO(n_particles=3, dimensions=dimensions, options=options, bounds=bounds)

    print("Perform optimization")
    cost, pos = optimizer.optimize(f, iters=2)

    print("Saving Features")
    ftrs_weights = {ftr: pos[idx] for idx, ftr in enumerate(cam_obj.ftr_names)}
    # json.dump(ftrs_weights, open(f'ftr_weights.json', 'w'), indent='\t')
    return ftrs_weights

def cb_prob_data_scaling(cb_probs):
    scaler_model = MinMaxScaler(feature_range=(0, 1))
    scaler_model.fit(cb_probs)
    return scaler_model.transform(cb_probs)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Weight features of a Case-Base using Case-Alignment Measure")
    argparser.add_argument('-stands', '--stands', action='store_true', help='use teams standings')
    argparser.add_argument('-week', '--week', action='store_true', help='use week/date of season')
    argparser.add_argument('--pop', '-pop', action='store_true', help='Use popularity for players')
    argparser.add_argument('--tpop', '-tpop', action='store_true', help='Use popularity for teams')

    args = argparser.parse_args()
    print(args)

    POP = args.pop
    TPOP = args.tpop
    STANDS = args.stands
    WEEK = args.week

    bstr = f"*"*150
    print(f"\n\nGenerating from CB\n{bstr}")
    print(f"Popularity: {POP}\tTeam Popularity: {TPOP}\tStandings: {STANDS}\tWeek: {WEEK}")
    print(f"{bstr}")

    ftr_weights_file_name = f"base{'-pop' if POP else ''}{'-tpop' if TPOP else ''}{'-stands' if STANDS else ''}{'-week' if WEEK else ''}"
    cb_prob_file_name = f"imp_players_set_ftrs{'_pop' if POP else ''}{'_tpop' if TPOP else ''}{'_stands' if STANDS else ''}{'_week' if WEEK else ''}_cb_prob"
    print(f"cb_prob_file_name: {cb_prob_file_name}")
    print(f"ftr_weights_file_name: {ftr_weights_file_name}")
    print(f"{bstr}")

    cb_probs = np.load(f'cbs/all/{cb_prob_file_name}.npz')['arr_0']
    cb_sols = json.load(open(f'cbs/all/cb_sol.json'))

    player_ftrs = ['starter', 'double', 'MIN', 'winner', \
                    'PTS', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'TREB', 'AST', 'STL', 'BLK', 'TOV', 'PF']#, \
                    # 'popularity']
    if POP:
        player_ftrs.append('popularity')

    team_ftrs = ['FG3A', 'FG3M', 'FGA', 'FGM', 'FTA', 'FTM', 'DREB', 'OREB', 'TREB', 'BLK', 'AST', 'STL', 'TOV', 'PF', 'PTS', 'MIN', \
                    'PTS_Q1', 'PTS_Q2', 'PTS_Q3', 'PTS_Q4', \
                    'IS_WINNER', 'WINS', 'LOSSES'] #, \
                    # 'popularity', \
                    # 'SEASON_WEEK', 'SEASON_DATE', 'STANDING']
    if TPOP:
        team_ftrs.append('popularity')
    if WEEK:
        team_ftrs.append('SEASON_WEEK')
        team_ftrs.append('SEASON_DATE')
    if STANDS:
        team_ftrs.append('STANDING')

    ftr_names = [f"PLAYER-{item}" for item in player_ftrs] + [f"TEAM-{item}" for item in team_ftrs]
    num_ftrs = len(ftr_names)

    scaled_cb_probs = cb_prob_data_scaling(cb_probs)
    cam_obj = CaseAlignMeasure(cb_sols, scaled_cb_probs, num_ftrs, ftr_names)

    print("Constructed!!\n\n")
    print(ftr_names)
    print(num_ftrs, len(ftr_names), len(player_ftrs), len(team_ftrs))
    print(len(cam_obj.cb_sols))
    print(cam_obj.cb_probs.shape, len(cam_obj.cb_sols))
    ftrs_weights = train_pso(cam_obj)
    json.dump(ftrs_weights, open(f"ftr_weights/{ftr_weights_file_name}.json", 'w'), indent='\t')
    print("Done")
