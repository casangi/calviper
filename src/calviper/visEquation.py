import xarray as xr

import numpy as np
import bisect
from enum import Enum, IntEnum

from toolviper.utils import logger
from calviper.solver import ScipySolverLeastSquares
from abc import ABC, abstractmethod
from typing import TypeVar, Type, Union
import calviper as cv

# Vis equation components
# Ordered VisEquation Enum
class StandardVisEqTypeEnum(IntEnum):
    UNDEF = 0
    K = 10  # Geometry (Fourier Kernel errors)
    B = 20  # Bandpass
    G = 30  # Electronic Gain (per poln)
    J = 40  # General Jones
    D = 50  # Instrumental Polarization Leakage
    X = 60  # (Forward) Cross-hand phase
    C = 70  # Feed configuration (absolute orientation)
    P = 80  # Parallactic angle
    E = 90  # Efficiency (on-axis)    (review order w.r.t. C,P and measurement conv.)
    T = 100  # Tropospheric Gain (unpol)
    F = 110  # Ionospheric Gain (disp. delay and birefringence)
    M = 300  # Multiplicative non-closing
    A = 400  # Additive non-closing

# Calibration parameter data type
class ParTypeEnum(Enum):
    # Must we support combined Float & Complex case?
    UNDEF = 0
    FLOAT = 1
    COMPLEX = 2


# Calibration Jones matrix type
class MatTypeEnum(Enum):
    # Is this needed at all, if all matrices rendered as general?
    # Must we support linearized general?  (ignores 2nd order terms, for traditional poln alg.)
    UNDEF = 0
    SCALAR = 1
    DIAGONAL = 2
    GENERAL = 4


# Visibility polarization basis
class PolBasisEnum(Enum):
    # What about antenna-dependent pol basis?
    UNDEF = 0
    LINEAR = 1
    CIRCULAR = 2

class VisEquation(ABC):

    def __init__(self):
        # set what to apply and solve for
        self.apply_vis_jones = [] # Ordered list of VisJones to be applied
        self.upstream_vis_jones = [] # Describes which VisJones to apply before the solve
        self.downstream_vis_jones = [] # Describes which VisJones to apply after the solve
        self.solve_vis_jones = None # The VisJones to be used in solve
        self.solve_pivot = None

    # value function for sorting cal terms
    def __visEquationSortVal__(self, vis_cal):
        return vis_cal.type["value"]

    # add a VisJones to the apply list
    def setApply(self, apply_vis_jones=None):
        # Clear solve term and pivot if there is no set solve

        # Add the apply VisJones to the apply list

        # Sort the apply list based on ordering value
        # Each VisCal has a VEindex property?
        pass

    # arrange Vis Equation to solve (Component to solve for given jones)
    def setSolve(self, solve_vis_jones):
        self.solve_vis_jones = solve_vis_jones

        # if there are VisJones set to apply then find the pivot point
        if len(self.apply_vis_jones) > 0:
            # set the pivot to the first point where it's value is greatest
            bisect.insort(self.apply_vis_jones, self.solve_vis_jones, key=self.__visEquationSortVal__)
            self.solve_pivot = self.apply_vis_jones.index(self.solve_vis_jones)

            self.upstream_vis_jones = self.apply_vis_jones[self.solve_pivot:][::-1] #inward order
            self.downstream_vis_jones = self.apply_vis_jones[0:self.solve_pivot] #outward order
        else:
            self.solve_pivot = -1

        print("Arranging VisEquation to solve for " +
              self.solve_vis_jones.type["name"] + " " + 
              self.solve_vis_jones.name)


    # solve (currently using my probably slow scipy.optimize edition)
    def solve(self, plot):
        """
        Passes the solve_vis_jones to the solver
        NOTE: Currently using scipy on this branch. Will test with Josh's implimentation

        Returns: An xarray caltable
        """
        # test using Josh's solver one pol for now
        #vis_array = self.solve_vis_jones.matrix.data[:, 0]
        '''
        full_antenna_list = np.union1d(
            self.solve_vis_jones.matrix.baseline_antenna1_name.to_numpy(),
            self.solve_vis_jones.matrix.baseline_antenna2_name.to_numpy()
        )

        encoder, antennas = cv.math.tools.encode(full_antenna_list)
        index_a = encoder.transform(self.solve_vis_jones.matrix.baseline_antenna1_name.to_numpy())
        index_b = encoder.transform(self.solve_vis_jones.matrix.baseline_antenna2_name.to_numpy())

        vis_array = self.solve_vis_jones.matrix.data
        v_ = cv.math.tools.build_visibility_matrix(array=vis_array, index_a=index_a, index_b=index_b)

        solver = cv.math.solver.least_squares.LeastSquaresSolver()
        gain_solutions = solver.solve(
            vis = v_,
            iterations=50,
            optimizer=cv.math.optimizer.MeanSquaredError(alpha=0.25),
            stopping=1e-4
        )
        '''

        # do solve (currently assuming a dict on return for gains but probably an ndarray)
        solver = ScipySolverLeastSquares(self.solve_vis_jones)
        solution_dict = solver.solve(tracking=plot)
        tracked_vals = solver.get_tracked_vals()
        # just test plot
        if plot:
            solver.plot_solution()
            #t = np.linspace(1, len(solver.losses), len(solver.losses))

            #plt.scatter(solver.losses, t)
            #plt.show()

        # build table (type specific? ant x ant) with 4 pols for each ant soln
        xarr_out = solver.generate_cal_table(solution_dict)

        # return table
        return xarr_out, tracked_vals
        #cm = cv.factory.jones.CalibrationMatrix()
        #G = cm.create_jones("gain").empty_like(self.solve_vis_jones)
        #G.gain.empty()
        #return solver.parameter