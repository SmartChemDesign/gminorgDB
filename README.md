# gminorgDB
Database of small organic molecules in the global minimum conformation
gminorg is a database of small organic molecules with a wide range of sizes (5-52 non-hydrogen atoms) and degrees of freedom (up to 13 rotatable bonds).
This database was created to provide the ground for benchmarking systems for global optimization or conformation analysis techniques in computational chemistry.
Molecular gometries were extracted from literature data and then optimized using quantum chemistry methods (level of theory: RI-MP2; see .csv files for info on used basis set)

The ./glob folder contains 68 .xyz structures in a global minimum configuration:
    28 molecules obtained from the CCDC database (https://www.ccdc.cam.ac.uk/)
    30 molecules obtained using the GED method
    10  molecules obtained with use of microwave spectroscopy methods
For more information see ./globdata.csv

The ./conf folder contains 40 .xyz structures in the global and one or several local minimum configurations:
    19 molecules obtained using the GED method
    21 structures obtained with use of microwave spectroscopy methods
For more information see  ./globdata.csv

TPE_global_optimization folder contains the final implementation of the global optimization routine using the Tree Parzen Estimator

A benchmarking program package is also presented, as well as a use case.
