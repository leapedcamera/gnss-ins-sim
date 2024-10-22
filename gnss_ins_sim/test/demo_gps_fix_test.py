"""
Test a modified gps_fix function
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import gnsstoolbox.orbits as orb
from gnss_ins_sim.attitude import attitude
from gnss_ins_sim.geoparams import geoparams

def gps_fix(orbit, time, prn, psr, iter=5):
    '''
    Solve pseudorange position and time solution
    '''   
    # Parse the input
    good_idx = np.argwhere(~np.isnan(psr))
    n = good_idx.size
    if n < 4:
        return None
    meas_psr = psr[good_idx]
    meas_psr = meas_psr[:,0]
    prn = prn[good_idx]
    prn = prn[:,0]
    mjd =  orbit.NAV_dataG[0][0].mjd

    # Initialize variables, G * m = d
    G = np.zeros([n, 4])
    G[:,3] = -1
    vehicle_pos = np.zeros(3)
    sat_pos = np.zeros([n,3])
    est_clk_bias = 0
    d = np.zeros(n)
    
    # Get all sat positions 
    for  i in range(n):
        X, Y, Z, dte = orbit.calcSatCoord("G", int(prn[i]), mjd )
        sat_pos[i,:] = np.array([X, Y, Z])

    # Iterate on solution
    for k in range(iter):
        # Load up the matrix with all sats
        for j in range(n):
            pred_pos_diff = vehicle_pos - sat_pos[j,:]
            pred_psr = np.linalg.norm(pred_pos_diff)
            G[j,0:3] = pred_pos_diff / pred_psr
            d[j] = meas_psr[j] - pred_psr

        m = np.matmul( np.matmul( np.linalg.inv( \
            np.matmul( np.transpose(G), G ) ), np.transpose(G)), d )
        est_clk_bias = est_clk_bias + m[3]
        vehicle_pos =  vehicle_pos + m[0:3] 
    return vehicle_pos, est_clk_bias

# Load ephemeris
gps_orbits = orb.orbit()
motion_def_path = os.path.abspath('./data/')
rinex=[ motion_def_path+"/AMC400USA_R_20230290000_15M_GN.rnx",\
                           motion_def_path+"/BREW00USA_R_20230290000_15M_GN.rnx",\
                           motion_def_path+"/USUD00JPN_R_20230290000_15M_GN.rnx",\
                           motion_def_path+"/WROC00POL_R_20230290000_15M_GN.rnx",\
                           motion_def_path+"/YEL200CAN_R_20230290000_15M_GN.rnx",\
                           motion_def_path+"/YKRO00CIV_R_20230290000_15M_GN.rnx",\
                           motion_def_path+"/ZAMB00ZMB_R_20230290000_15M_GN.rnx",\
                           motion_def_path+"/VACS00MUS_R_20230290000_15M_GN.rnx"     ]
for iRin in rinex:
    gps_orbits.loadRinexN(iRin)

# Initilize our truth position
true_pos_lla = np.array([42.35880941958805, -71.07224531905231, 0])
true_pos_ecef = geoparams.lla2ecef(true_pos_lla)
time = gps_orbits.NAV_dataG[0][0].mjd

# Set up the empty measurement vectors
prn = np.zeros(32)
prn[:] = np.nan
psr = np.zeros(32)
psr[:] = np.nan
psr_counter = 0
c = 299792458.0

# Take the measurement and fill in the vector
for i in range(32):
    X, Y, Z, dte = gps_orbits.calcSatCoord("G", i, time )
    if np.isnan( X ):
        continue
    u_pos_ecef = true_pos_ecef / np.linalg.norm(true_pos_ecef)
    sat_pos_ecef = np.array([X,Y,Z])
    u_sat_ecef = sat_pos_ecef / np.linalg.norm(sat_pos_ecef)
    theta = np.arccos( np.dot(u_pos_ecef,u_sat_ecef))
    # 60 degreee elevation angle
    if theta < 1.0471:
        pseudorange = np.linalg.norm( sat_pos_ecef - true_pos_ecef ) 
        prn[psr_counter] = i
        psr[psr_counter] = pseudorange
        psr_counter += 1

# Let's take a look at the error as a function of iterations...
pos_error = np.zeros([9,3])
clk_error = np.zeros(9)

for k in range(1, 10):
    est_pos_ecef, est_clk_bias = gps_fix(gps_orbits, time, prn, psr, iter=k)
    pos_error[k-1,:] = np.absolute(true_pos_ecef - est_pos_ecef)
    clk_error[k-1] = est_clk_bias


# Plot the result!
fig, axs = plt.subplots(4, figsize=(9, 9), sharey=True)
fig.suptitle('Pseudorange Solution Error', fontsize=16)
fig.supxlabel('# of Solver Iterations', fontsize=16)
iterations = np.arange(1,10)
axs[0].set_yscale('log')
axs[0].plot( iterations, pos_error[:,0])
axs[0].set_ylabel( 'ECEF X Error (m)')

axs[1].set_yscale('log')
axs[1].plot( iterations, pos_error[:,1])
axs[1].set_ylabel( 'ECEF Y Error (m)')

axs[2].set_yscale('log')
axs[2].plot( iterations, pos_error[:,2])
axs[2].set_ylabel( 'ECEF Z Error (m)')

axs[3].set_yscale('log')
axs[3].plot( iterations, np.linalg.norm(pos_error, axis=1))
axs[3].set_ylabel( 'Norm of ECEF Error (m)')

plt.figure()
plt.plot( iterations, clk_error)
plt.ylabel("Clock Bias Error (m)")
plt.ylabel("Iterations")
plt.show()
