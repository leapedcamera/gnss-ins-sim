# -*- coding: utf-8 -*-
# Fielname = ins_tight.py

"""
Tightly coupled INS algorithm.
Created on 2024-09-16
@author: leapedcamera
"""

# import
import math
import numpy as np
import gnsstoolbox.orbits as orb
from gnss_ins_sim.attitude import attitude
from gnss_ins_sim.geoparams import geoparams

UNINI = 0           # uninitialized
ATTITUDE_INI = 1    # attitude initialized
POS_INI = 2         # position and velocity initialized
                   
class InsTight(object):
    '''
    Tightly coupled INS algorithm.
    '''
    def __init__(self, orbit, init_from_truth=False):
        '''
        vars
        '''
        # algorithm description
        self.input = ['ref_pos', 'ref_vel', 'ref_att_euler', 'fs', 'gyro', 'accel', 'time', 'gps_time', 'gps_prn', 'gps_obs']
        self.output = ['att_euler', 'pos', 'vel']
        self.batch = True
        self.results = None

        # algorithm vars
        self.att = None
        self.pos = None
        self.vel = None
        self.vel_b = None
        self.ini = 0                                # indicate if attitude is initialized
        self.step = 0                               # track what step of the sim we're on
        self.dt = 10.0                               # sample period, sec
        self.q = np.array([1.0, 0.0, 0.0, 0.0])     # quaternion
        self.orbit = orbit
        self.init_from_truth = init_from_truth
        
    def run(self, set_of_input):
        '''
        main procedure of the algorithm
        Args:
            set_of_input is a tuple or list consistent with self.input
        '''
        # Truth used only for initialization
        self.init_state = None
        if self.init_from_truth:
            true_pos_0 = set_of_input[0][0,:]
            true_vel_0 = set_of_input[1][0,:]
            true_att_0 = set_of_input[2][0,:]
            self.init_state = np.concatenate([true_pos_0, true_vel_0, true_att_0])


        # Get sensor inputs
        fs = set_of_input[3]
        gyro = set_of_input[4]
        accel = set_of_input[5]
        time = set_of_input[6]
        gps_time = set_of_input[7]
        gps_prn = set_of_input[8]
        gps = set_of_input[9]

        # run the algorithm
        self.ins_tight(fs, time, gyro, accel, gps_time, gps_prn, gps)

    def ins_tight(self, fs, time, gyro, accel, gps_times, gps_prn, gps):
        '''
        main procedure of the tightly coupled INS algorithm.
        '''

        # Strapdown States
        n = time.shape[0]
        self.att = np.zeros((n, 3))
        self.pos = np.zeros((n, 3))
        self.vel = np.zeros((n, 3))     # NED vel
        self.vel_b = np.zeros((n, 3))   # body vel

        self.initialize_nav_state(fs, time, gyro, accel, gps_times, gps_prn, gps)

        # Kalman filter state and matrices
        self.P = np.zeros((n, 10, 10))
        self.Q = np.zeros((10, 10))
        self.H = np.eye(10)
        self.R = np.zeros(1)

        # Set Uncertainty
        self.P[0,0] = 10
        self.P[1,1] = 10
        self.P[2,2] = 10
        self.P[3,3] = 2
        self.P[4,4] = 2
        self.P[5,5] = 2
        self.P[6,6] = 0.0625
        self.P[7,7] = 0.0625
        self.P[8,8] = 0.0625
        self.P[9,9] = 0.0625

        # Set Process noise
        self.Q[0,0] = 10
        self.Q[1,1] = 10
        self.Q[2,2] = 10
        self.Q[3,3] = 2
        self.Q[4,4] = 2
        self.Q[5,5] = 2
        self.Q[6,6] = .01
        self.Q[7,7] = .01
        self.Q[8,8] = .01
        self.Q[9,9] = .01


        # KF loop variables
        n_accum = 0
        filt_time = 0
        avg_accel = np.zeros(3)
        avg_omega = np.zeros(3)
        gps_time = gps_times[0]
        gps_counter = 0
        
        for i in range(self.step, n):

            # Strapdown each time
            self.strapdown(gyro[i-1,:], accel[i-1,:], 1/fs)
            avg_accel += accel[i-1,:]
            avg_omega += gyro[i-1,:]

            # Propagate the filter at the proper rate
            if filt_time == time[i]:
                avg_accel = accel_accum / n_accum
                avg_omega = gyro_accum / n_accum
                self.prediction(avg_accel, avg_omega, time[i] - prev_event_time)
                n_accum = 0
                avg_accel = np.zeros(3)
                avg_omega = np.zeros(3)
                prev_event_time = time[i]
                filt_time = time[i] + self.dt

            # If a GPS measurement is available, run the filter
            if gps_time == time[i]:
                self.correction( gps_prn[gps_counter], gps[gps_counter,:] )
                gps_counter += 1
                gps_time = gps_times[gps_counter]
            
            self.step += 1

        # Save results
        self.results = [self.att, self.pos, self.vel]

    def initialize_nav_state(self, fs, time, gyro, accel, gps_time, gps_prn, gps):
        '''
        Initialization strategy:
            0. Let's just initialize from truth...
            1. Collect first n accel samples to initialize roll and pitch.
            2. If GPS is available during collecting the n samples, position and velocity
                are initialized.
            3. If GPS is not availabe during collecting the n samples, position and velocity
                will be initialzied when GPS becomes available. Before position and velocity
                are initialized, system runs in free integration mode.
        '''

        if self.init_state.any():
            self.pos[0,:] = self.init_state[0:3]
            self.vel[0,:] = self.init_state[3:6]
            self.att[0,:] = self.init_state[6:9]
            c_bn = attitude.euler2dcm(self.att[0, :])
            self.vel_b[0, :] = c_bn.dot(self.vel[0,:])
            earth_param = geoparams.geo_param(geoparams.ecef2lla(self.pos[0,:]))    # geo parameters
            self.g_n = np.array([0, 0, earth_param[2]])
            self.step = 1
            return
    

        dt = 1.0 / fs
        n = time.shape[0]
        samples_for_attitude_ini = 10
        average_accel = np.zeros((3,))
        i_gps_time = 0  # index of current GPS time, GPS data for this time are not used yet.

        for i in range(n):
            
            # None of attitude, position and velocity is initialized.
            if self.ini == UNINI:
                # average accel for initial pitch and roll
                average_accel += accel[i, :]
                if i == samples_for_attitude_ini - 1:
                    average_accel /= 10.0
                    accel_norm = math.sqrt(average_accel[0]*average_accel[0] +\
                                           average_accel[1]*average_accel[1] +\
                                           average_accel[2]*average_accel[2])
                    average_accel /= accel_norm
                    euler_zyx = np.zeros((3,))  # Euler angles, ZYX
                    euler_zyx[0] = 10.0 * attitude.D2R
                    euler_zyx[1] = math.asin(average_accel[0])
                    euler_zyx[2] = math.atan2(-average_accel[1], -average_accel[2])

                    # find if there is any valid GPS measurement during attitude initialization.
                    # if there is, using the latest one to initialize position and velocity.
                    while time[i] > gps_time[i_gps_time+1]:
                        i_gps_time += 1
                    if gps_time[i_gps_time] > time[i]:
                        self.ini = ATTITUDE_INI
                        pos = np.zeros((3,))
                    else:
                        self.ini = POS_INI
                        pos, clk_bias = self.gps_fix( self.orbit, \
                                                     gps_time[i_gps_time], \
                                                     gps_prn[i_gps_time, :], \
                                                     gps[i_gps_time, :] )
                        i_gps_time += 1 # this GPS measurement is used, index moves to next

            # Attitude is initialized, to initialize position and velocity
            elif self.ini == ATTITUDE_INI:
                # try to iniialize position and velocity
                if time[i] > gps_time[i_gps_time]:
                    # propagate to gps_time
                    self.prediction(gyro[i-1, :], accel[i-1, :], gps_time[i_gps_time]-time[i-1])
                    # initialize position via fix
                    pos, clk_bias = self.gps_fix( self.orbit, gps_time[i_gps_time], gps_prn[i_gps_time, :], gps[i_gps_time, :] )
                    # propagate to current time
                    self.prediction(gyro[i-1, :], accel[i-1, :], time[i]-gps_time[i_gps_time])
                    self.ini = POS_INI
                    i_gps_time += 1
                else:
                    self.prediction(gyro[i-1, :], accel[i-1, :], dt)

            # attitude, position and velocity are all initialized
            elif self.ini == POS_INI:
                self.step = i
                self.att[i,:] = euler_zyx
                self.pos[i,:] = pos
                earth_param = geoparams.geo_param(pos)    # geo parameters
                self.g_n = np.array([0, 0, earth_param[2]])
                return
            else:
                self.ini = UNINI

    def strapdown(self, gyro, accel, dt):
                
        #### propagate Euler angles
        i = self.step
        self.att[i, :] = attitude.euler_update_zyx(self.att[i-1, :], gyro, dt)
        c_bn = attitude.euler2dcm(self.att[i-1, :])

        self.vel_b[i, :] = self.vel_b[i-1, :] +\
                (accel + c_bn.dot(self.g_n)) * dt -\
                attitude.cross3(gyro, self.vel_b[i-1, :]) * dt
        
        c_bn = attitude.euler2dcm(self.att[i, :])
        self.vel[i, :] = c_bn.T.dot(self.vel_b[i, :])   # velocity in navigation frame
        self.pos[i, :] = self.pos[i-1, :] + self.vel[i-1, :] * dt


    def prediction(self, gyro, acc, dt):
        '''
        Kalman prediction
        '''
        i = self.step
        quat = self.att[i - 1, :]
        ## Form linearized state transition matrix

        # Position
        dfrdr = np.zeros(3)
        dfrdv = np.ones(3)
        dfrdq = np.zeros(4,3)
        
        # Velocity
        dfvdr = np.zeros(3)
        dfvdv = np.zeros(3)
        Q = np.array([[q[1],  q[0],  q[3],  q[2]],
                      [q[2],  q[3],  q[0], -q[1]],
                      [q[3], -q[2],  q[0],  q[0]]])
        A = np.array([0,        accel[0],  accel[1], accel[2]],
                     [accel[0],        0,  accel[2], -accel[1]],
                     [accel[1], -accel[2],        0,  accel[0]],
                     [accel[2], accel[1], -accel[0],        0])
        dfvdq = 2 * Q * A

        # Attitude (quaternion)
        dfqdr = np.zeros(3,4)
        dfqdv = np.zeros(3,4)
        
        W = np.array([0,       -gyro[0], -gyro[1], -gyro[2]],
                     [gyro[0],        0,  gyro[2], -gyro[1]],
                     [gyro[1], -gyro[2],        0,  gyro[0]],
                     [gyro[2],  gyro[1], -gyro[0],        0])
        dfqdq = dt/2 * W

        phi = np.eye(10) + np.array([[ dfrdr, dfrdv, dfrdq ],
                                     [ dfvdr, dfvdv, dfvdq ],
                                     [ dfqdr, dfqdv, dfqdq ]] ) * dt
        
        pass

    def correction(self, gps):
        '''
        Kalman correction
        '''
        pass

    def get_results(self):
        '''
        return algorithm results as specified in self.output
        '''
        return self.results

    def reset(self):
        '''
        Reset the fusion process to uninitialized state.
        '''
        self.ini = 0

    def gps_fix(self, orbit, time, prn, psr, iter=5):
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

        

