import xarray as xr
from xradio.measurement_set import open_processing_set
from calviper.jones import GainJones

# Open the data and grab observation blocks for gaincal test data all have cal intents
# Should this take in a processing set? This way no xradio is needed here

# Turn gains from data into GJones matricies
# Setting par shape with GJones object (time, ant, channel par)
# create based on the xradio processing set
gain_matrix = GainJones()
# ADD SET PAR SHAPE FROM VIS DATA [X]
# --> nchanpar 1 is enforced for the example case, same with pol?
# JONES CALCULATE                 [X]
# INFO FUNTION                    [ ]

# Extracted the model visibilites for the vis equation [X]
# solve for the gains and end with a table of gain cal values

# Vis Equation construction
# VisEquation class implement (in base.py?) [ ]
# VisEquation setApply (NOT YET)            [ ]
# VisEquation setSolve                      [ ]
# VisEquation Solve                         [ ]