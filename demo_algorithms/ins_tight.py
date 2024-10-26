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
        self.input = ['ref_pos', 'ref_vel', 'ref_att_euler', 'fs', 'gyro', 'accel', 'time', 'gps_tov', 'gps_prn', 'gps_obs', 'gps_mjd']
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
        gps_tov = set_of_input[7]
        gps_prn = set_of_input[8]
        gps_psr = set_of_input[9]
        gps_mjd = set_of_input[10]
        gps_data = np.c_[gps_tov, gps_prn, gps_psr, gps_mjd]

        # run the algorithm
        self.ins_tight(fs, time, gyro, accel, gps_data)

    def ins_tight(self, fs, time, gyro, accel, gps_data):
        '''
        main procedure of the tightly coupled INS algorithm.
        '''

        # Strapdown States
        n = time.shape[0]
        self.att = np.zeros((n, 4))
        self.pos = np.zeros((n, 3))
        self.vel = np.zeros((n, 3))     # NED vel
        self.vel_b = np.zeros((n, 3))   # body vel

        self.initialize_nav_state(fs, time, gyro, accel, gps_data)

        # Kalman filter state and matrices
        self.P = np.zeros((n, 10, 10))
        self.Q = np.zeros((10, 10))
        self.H = np.eye(10)

        # Set Uncertainty
        self.P[0,0,0] = 10
        self.P[0,1,1] = 10
        self.P[0,2,2] = 10
        self.P[0,3,3] = 2
        self.P[0,4,4] = 2
        self.P[0,5,5] = 2
        self.P[0,6,6] = 0.001
        self.P[0,7,7] = 0.001
        self.P[0,8,8] = 0.001
        self.P[0,9,9] = 0.001

        self.psr_sig = 3

        # Set Process noise
        self.Q_q_sig = 0.0625
        self.Q_v_sig = 0.0009728161

        # KF loop variables
        n_accum = 0
        filt_time = 0
        accel_accum = np.zeros(3)
        gyro_accum = np.zeros(3)
        gps_tov = gps_data[0, 0]
        gps_counter = 0
        
        for i in range(self.step, n):

            # Strapdown each time
            self.strapdown(gyro[i-1,:], accel[i-1,:], 1/fs)
            accel_accum += accel[i-1,:]
            gyro_accum += gyro[i-1,:]

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
            else:
                self.P[i,:,:] = self.P[i-1,:,:]

            # If a GPS measurement is available, run the filter
            if gps_tov <= time[i]:
                if gps_tov == time[i]:
                    self.correction( gps_data[gps_counter , :] )
                gps_counter += 1
                if gps_counter < np.size(gps_data, 1):
                    gps_tov = gps_data[gps_counter, 0]
            
            self.step += 1

        # Save results
        self.results = [self.att, self.pos, self.vel]

    def initialize_nav_state(self, fs, time, gyro, accel, gps_data):
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
            self.att[0,:] = attitude.euler2quat(self.init_state[6:9])
            c_bn = attitude.quat2dcm(self.att[0, :])
            self.vel_b[0, :] = c_bn.dot(self.vel[0,:])
            earth_param = geoparams.geo_param(geoparams.ecef2lla(self.pos[0,:]))    # geo parameters
            self.g_n = np.array([0, 0, earth_param[2]])
            self.step = 1
            return
    

        dt = 1.0 / fs
        n = time.shape[0]
        samples_for_attitude_ini = 10
        average_accel = np.zeros((3,))
        i_gps_tov = 0  # index of current GPS time, GPS data for this time are not used yet.

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
                    quat = attitude.euler2quat(euler_zyx)

                    # find if there is any valid GPS measurement during attitude initialization.
                    # if there is, using the latest one to initialize position and velocity.
                    while time[i] > gps_data[i_gps_tov + 1, 0]:
                        i_gps_tov += 1
                    if gps_data[i_gps_tov, 0] > time[i]:
                        self.ini = ATTITUDE_INI
                        pos = np.zeros((3,))
                    else:
                        self.ini = POS_INI
                        pos, clk_bias = self.gps_fix(gps_data[i_gps_tov, 65], \
                                                     gps_data[i_gps_tov, 1:32], \
                                                     gps_data[i_gps_tov, 33:64] )
                        i_gps_tov += 1 # this GPS measurement is used, index moves to next

            # Attitude is initialized, to initialize position and velocity
            elif self.ini == ATTITUDE_INI:
                # try to iniialize position and velocity
                if time[i] > gps_data[i_gps_tov, 0]:
                    # propagate to gps_tov
                    self.prediction(gyro[i-1, :], accel[i-1, :], gps_data[i_gps_tov]-time[i-1], 0)
                    # initialize position via fix
                    pos, clk_bias = self.gps_fix(gps_data[i_gps_tov, 65], \
                                                     gps_data[i_gps_tov, 1:32], \
                                                     gps_data[i_gps_tov, 33:64] )
                    # propagate to current time
                    self.prediction(gyro[i-1, :], accel[i-1, :], time[i]-gps_data[i_gps_tov, 0])
                    self.ini = POS_INI
                    i_gps_tov += 1
                else:
                    self.prediction(gyro[i-1, :], accel[i-1, :], dt)

            # attitude, position and velocity are all initialized
            elif self.ini == POS_INI:
                self.step = i
                self.att[i,:] = quat
                self.pos[i,:] = pos
                earth_param = geoparams.geo_param(pos)    # geo parameters
                self.g_n = np.array([0, 0, earth_param[2]])
                return
            else:
                self.ini = UNINI

    def strapdown(self, gyro, accel, dt):
                
        # Propagate quaternion
        i = self.step
        self.att[i, :] = attitude.quat_update(self.att[i-1, :], gyro, dt)
        c_bn = attitude.quat2dcm(self.att[i-1, :])

        self.vel_b[i, :] = self.vel_b[i-1, :] +\
                (accel + c_bn.dot(self.g_n)) * dt -\
                attitude.cross3(gyro, self.vel_b[i-1, :]) * dt
        
        c_bn = attitude.quat2dcm(self.att[i, :])
        self.vel[i, :] = c_bn.T.dot(self.vel_b[i, :])   # velocity in navigation frame
        self.pos[i, :] = self.pos[i-1, :] + self.vel[i-1, :] * dt


    def prediction(self, gyro, accel, dt):
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
        

        E = np.array([1 -  q[0] *  q[0], -q[0] *  q[1], -q[0] *  q[2], -q[0] * q[3]],
                     [-q[0] *  q[1],        1 -  q[1] *  q[1],  -q[1] * q[2], -q[1] * q[3]],
                     [-q[0] *  q[2], -q[1] * q[2],        1 -  q[2] *  q[2],  -q[2] * q[3]],
                     [-q[0] * q[3], -q[1] * q[3], -q[2] * q[3],        1 -  q[3] *  q[3]])
        
        self.Q[0:2, 0:2] = ( self.Q_v_sig * self.Q_v_sig * dt * dt ) * np.eye(3)
        self.Q[3:5, 3:5] = ( self.Q_v_sig * self.Q_v_sig * dt * dt * dt * dt ) * np.eye(3)
        self.Q[6:9, 6:9] = ( self.Q_q_sig * self.Q_q_sig * dt * dt / 4 )
        
        self.P[i, :, :] = phi * self.P[i-1, :, :] * np.transpose(phi) + Q

    def correction(self, gps_data ):
        '''
        Kalman Update
        '''
        # Pull out State
        P = self.P[self.step, :, :] 
        x = np.concatenate([self.pos[self.step, :],\
            self.vel[self.step, :], \
            self.att[self.step, :]]).flatten()

        # Extract GPS data
        mjd = gps_data[65]
        prn = gps_data[1:32]
        psr = gps_data[33:64]

        # Create H Matrix
        good_idx = np.argwhere(~np.isnan(psr))
        n = good_idx.size
        meas_psr = psr[good_idx]
        meas_prn = prn[good_idx]
        H = np.zeros([n, 10])
        y = np.zeros(n)
        z = np.zeros(n)
        R = np.eye(n) * self.psr_sig

        for i in range(n):
            X, Y, Z, dte = self.orbit.calcSatCoord("G", int(meas_prn[i]), mjd )
            sat_pos = np.array([X, Y, Z])
            z[i] = np.linalg.norm(x[0:3] - sat_pos)
            y[i] =  meas_psr[i] - z[i]
            u_sat =  (x[0:3] - sat_pos) / z[i]
            H[i,0:3] = u_sat

        # Now the KF Equations
        S = np.matmul(H, np.matmul( P, np.transpose(H))) + R
        K = np.matmul(P, np.matmul(np.transpose(H), np.linalg.inv(S)))
        x = x + np.matmul(K, y)
        P = np.matmul(np.eye(10) - np.matmul(K,H),  P)
        y = z - np.matmul(H,x)

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

    def gps_fix(self, mjd, prn, psr, iter=5):
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

        # Initialize variables, G * m = d
        G = np.zeros([n, 4])
        G[:,3] = -1
        vehicle_pos = np.zeros(3)
        sat_pos = np.zeros([n,3])
        est_clk_bias = 0
        d = np.zeros(n)
        
        # Get all sat positions 
        for  i in range(n):
            X, Y, Z, dte = self.orbit.calcSatCoord("G", int(prn[i]), mjd )
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

        

