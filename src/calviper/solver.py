
import numpy as np
import xarray as xr
from scipy.optimize import least_squares, minimize
import matplotlib.pyplot as plt

class ScipySolverLeastSquares:
    '''
    Currently just for testing the full execution path of a gain cal
    This is forcing the return to be antenna based gains, This needs a more general solution
    Probably a matrix of calibrated values?
    '''

    def __init__(self, obs, model=[1+0j,0+0j,0+0j,1+0j], start_guess=[(1+0j), (1+0j)]):
        self.obs = obs
        self.model = model
        self.start_guess = start_guess
        self.pols = [(0,0), (0,1), (1,0), (1,1)]

        # tracking solns
        self.tracked_vals = {}
        self.solved_vals = {}

    def vis_equation(self, params, vis_obs, vis_model):
        ant1 = params[:4].reshape(2,2)
        ant2 = params[4:].reshape(2,2)
        vis_model = np.asarray(vis_model).reshape(2,2)
        vis_obs = vis_obs.reshape(2,2)

        res = np.dot(ant1, vis_model)
        res = np.dot(vis_model, np.conj(ant2))

        return ((vis_obs - res).flatten())
    
    def chi_square(self, ant_set, vis_dict, vis_model, stepsize=0.1):
        """
        This is the derivative of chi squared to minimize
        This is more of a temp solver TODO: see Josh's verison for more effective solution
        """
        tmp = self.solved_vals
        diff = 100
        # ignoring weights for the moment
        weight = 1 #/ (0.5 * 0.5)

        for i in ant_set:
            # init sum for numerator and denominator
            # This is 2d since we calculate a sum for the X and Y polarizations
            num_sum = np.asarray([0+0j, 0+0j])
            denom_sum = np.asarray([0+0j, 0+0j])

            # sum over all j where i != j
            for j in ant_set:
                if i == j: continue;
                # iterate over each polarization for estimated Gx and Gy values
                num_sum, denom_sum = self.calc_grad_components(vis_dict, i, j, num_sum, denom_sum)

            gain = num_sum / denom_sum

            # again make sure we use the right pol
            diff = gain - self.solved_vals[i]#[pols[p][0]]
            tmp[i] = self.solved_vals[i] + (stepsize * diff)

            # if tracking for plots
            self.tracked_vals[i].append(tmp[i].copy())

            self.solved_vals = tmp

        return diff


    def calc_grad_components(self, vis: dict, ant1: int, ant2: int,
                            num_sum, denom_sum):
        '''
        Use all correlated baseline polarizations to enumerate on the numerator and denominator
        for the gradient calculation
        '''

        # Find a solution for Gx Gy using each of the baseline polarizations
        for baseline_pol in range(4):

            # see self.pols for more info but this is the polarization of each ant in the baseline
            ant1_pol = self.pols[baseline_pol][0]
            ant2_pol = self.pols[baseline_pol][1]

            # update the numerator
            if (ant1, ant2) in vis:
                cur_obs = vis[(ant1, ant2)][baseline_pol]
                num_sum[ant1_pol] += cur_obs * self.solved_vals[ant2][ant2_pol] * self.model[baseline_pol]
            # for the case where antenna1 is on the other side
            elif (ant2, ant1) in vis:
                cur_obs = np.conj(vis[(ant2, ant1)][baseline_pol])
                num_sum[ant1_pol] += cur_obs * self.solved_vals[ant2][ant2_pol] * (np.conj(self.model[baseline_pol]))
            
            # update the denominator
            denom_sum[ant1_pol] += (self.solved_vals[ant2][ant2_pol] * np.conj(self.solved_vals[ant2][ant2_pol]) *
                                    self.model[baseline_pol] * np.conj(self.model[baseline_pol]))
        
        return num_sum, denom_sum


    def build_vis_dict(self):
        """
        returns the matrix that gives the observered vis value for each baseline
        ant1 x ant2 with each ant val being a 2x2 of the polarizations
        """
        ants = set(self.obs.matrix.baseline_antenna1_name.values)
        ants = ants.union(set(self.obs.matrix.baseline_antenna2_name.values))
        vis_dict = {}

        # for each baseline and the reverse assign the observed visibility
        for item in self.obs.matrix:
            #print(item)
            _vis = item.values
            _ant1 = str(item.baseline_antenna1_name.values)
            _ant2 = str(item.baseline_antenna2_name.values)
            
            # 
            if (_ant1, _ant2) not in vis_dict.keys():
                vis_dict[(_ant1, _ant2)] = _vis

        return vis_dict
    
    def _to_vector(self, m1, m2):
        vec = np.hstack([m1.flatten(), m2.flatten()])
        return abs(vec)

    def solve(self, tracking=False) -> dict:
        '''
        Enable tracking to store the gain vals and use them for plotting
        This is just so I can see what it's doing for now...
        '''
        self.solved_vals = {}
        vis_dict = self.build_vis_dict()
        diff = 1 + 0j

        ant_set = set(self.obs.matrix.baseline_antenna1_name.values)
        ant_set = ant_set.union(set(self.obs.matrix.baseline_antenna2_name.values))
        
        # For all ants in baselines init solution to starting guess Gx = 1+0j and Gy = 1+0j
        for i in self.obs.matrix:
            self.solved_vals.setdefault(str(i.baseline_antenna1_name.values), np.asarray(self.start_guess))
            self.solved_vals.setdefault(str(i.baseline_antenna2_name.values), np.asarray(self.start_guess))

            # Tracking for plots
            if tracking:
                self.tracked_vals.setdefault(str(i.baseline_antenna1_name.values), [np.asarray(self.start_guess)])
                self.tracked_vals.setdefault(str(i.baseline_antenna2_name.values), [np.asarray(self.start_guess)])

        # probably worse since we are doing a lot of looping but it lets us solve for all pols at once?
        # single iteration or does scipy do multiple iterations on it's own? Not sure
        '''for i in ant_set:
            for j in ant_set:
                if i == j: continue;

                ant1_gain = self.solved_vals[i]
                ant2_gain = self.solved_vals[j]
                ants = self._to_vector(ant1_gain, ant2_gain)

                res = least_squares(self.vis_equation, ants, args=(vis_dict[(i, j)], self.start_guess))
                self.solved_vals[i] = res.x[:4]
                self.solved_vals[j] = res.x[4:]

                if tracking:
                    self.tracked_vals[i].append(res.x[:4])
                    self.tracked_vals[j].append(res.x[4:])
        '''
        for i in range(100):
        #while abs(diff) > 0.1:
            stepsize = .1 #/ (i+1)
            diff = self.chi_square(ant_set, vis_dict, self.model, stepsize)
            #print(diff)
        print(abs(diff))

        #print(self.solved_vals)
        return self.solved_vals
    
    def get_tracked_vals(self):
        return self.tracked_vals
    
    def plot_solution(self, pol=0):
        for k, v in self.tracked_vals.items():
            #print(k, np.asarray(v)[:, pol])
            p = [abs(i) for i in np.asarray(v)[:, :]]
            plt.plot(p)
            break
        
        plt.show()

    def generate_cal_table(self, coords: dict) -> xr.Dataset:
        '''
        Take in a dictionary of the solved antenna gains and convert it to an xarray dataset.
        This will be called by the VisEquation solve to generate the output cal table

        coords: dict of antenna and solutions for each polarization
        returns: xarray dataset

        NOTE: Need some clarification on the desired structure of this...
        Where should this live? Doesn't make too much sense here?
        '''
        xds = xr.Dataset(coords=coords)

        return xds
    
