#!/usr/bin/env python3

import numpy as np
import scipy as sp
from itertools import combinations
import do_mpc
import ctypes
import ctypes.util
import do_mpc.data
from casadi import *
import sys

import matplotlib.pyplot as plt
import matplotlib as mpl

solver = 'ma97'

if __name__ == '__main__':

    if len(sys.argv) > 1:
        m = int(sys.argv[1])
        width = int(sys.argv[2])
    else:
        m = 6
        width = 10

    n_horizon = 20
    a_init = 10
    t_step = 0.25

    model_type = 'discrete'
    model = do_mpc.model.Model(model_type)
    awareness = model.set_variable(var_name='awareness', var_type='_x', shape=(width*width, 1))  # 5x5 in each direction

    uav_state = []
    uav_input = []
    for i in range(m):
        uav1_pos_x = model.set_variable(var_name='uav'+str(i)+'_pos_x', var_type='_x')
        uav1_pos_y = model.set_variable(var_name='uav'+str(i)+'_pos_y', var_type='_x')
        uav1_vel_x = model.set_variable(var_name='uav'+str(i)+'_vel_x', var_type='_x')
        uav1_vel_y = model.set_variable(var_name='uav'+str(i)+'_vel_y', var_type='_x')
        uav1_acc_x = model.set_variable(var_name='uav'+str(i)+'_acc_x', var_type='_u')
        uav1_acc_y = model.set_variable(var_name='uav'+str(i)+'_acc_y', var_type='_u')
        d_uav1_pos_x = uav1_pos_x + t_step * uav1_vel_x + t_step**2*uav1_acc_x/2.0
        d_uav1_pos_y = uav1_pos_y + t_step * uav1_vel_y + t_step**2*uav1_acc_y/2.0
        model.set_rhs('uav'+str(i)+'_pos_x', d_uav1_pos_x)
        model.set_rhs('uav'+str(i)+'_pos_y', d_uav1_pos_y)
        model.set_rhs('uav'+str(i)+'_vel_x', uav1_vel_x + t_step * uav1_acc_x)
        model.set_rhs('uav'+str(i)+'_vel_y', uav1_vel_y + t_step * uav1_acc_y)
        uav_state.append((uav1_pos_x,uav1_pos_y,uav1_vel_x,uav1_vel_y))
        uav_input.append((uav1_acc_x,uav1_acc_y))

    c = 0.25  # cutoff
    c = c**2
    n = 8
    n = n-2
    al = 1

    aw_max = 1

    awareness_next = SX(1,width**2)
    A_tot = 0
    for i in range(0, width):
        for j in range(0, width):
            uav_function = 0
            for k in range(0, m):
                uav_function += (1 / (1 + (((uav_state[k][0] - i) ** 2 + (uav_state[k][1] - j) ** 2) / c) ** n))
            awareness_next[i * width + j] = (al*t_step + awareness[i * width + j]) \
                  * (1 - casadi.fmin(1,uav_function))
            A_tot += awareness[i * width + j] ** 2

    model.set_rhs('awareness', awareness_next)

    model.set_expression('A_tot', A_tot)
    model.setup()
    mpc = do_mpc.controller.MPC(model)
    setup_mpc = {
        'n_horizon': n_horizon,
        't_step': t_step,
        'state_discretization': 'discrete',
        'store_full_solution': True,
        'nlpsol_opts': {'ipopt.linear_solver': solver,'ipopt.print_level': 5}
    }

    mpc.set_param(**setup_mpc)

    lterm = model.aux['A_tot']
    mterm = model.aux['A_tot']

    mpc.set_objective(lterm=lterm, mterm=mterm)
    rterm = 1e-2
    rterms = {}
    for i in range(m):
        rterms['uav'+str(i)+'_acc_x'] = rterm
        rterms['uav'+str(i)+'_acc_y'] = rterm
    mpc.set_rterm(**rterms)

    vmax = 1
    amax = 2
    
    for i in range(m):
        mpc.bounds['lower', '_x', 'uav'+str(i)+'_pos_x'] = -1
        mpc.bounds['upper', '_x', 'uav'+str(i)+'_pos_x'] = width
        mpc.bounds['lower', '_x', 'uav'+str(i)+'_vel_x'] = -vmax
        mpc.bounds['upper', '_x', 'uav'+str(i)+'_vel_x'] = vmax
        mpc.bounds['lower', '_u', 'uav'+str(i)+'_acc_x'] = -amax
        mpc.bounds['upper', '_u', 'uav'+str(i)+'_acc_x'] = amax
        mpc.bounds['lower', '_x', 'uav'+str(i)+'_pos_y'] = -1
        mpc.bounds['upper', '_x', 'uav'+str(i)+'_pos_y'] = width
        mpc.bounds['lower', '_x', 'uav'+str(i)+'_vel_y'] = -vmax
        mpc.bounds['upper', '_x', 'uav'+str(i)+'_vel_y'] = vmax
        mpc.bounds['lower', '_u', 'uav'+str(i)+'_acc_y'] = -amax
        mpc.bounds['upper', '_u', 'uav'+str(i)+'_acc_y'] = amax

    mpc.scaling['_x', 'awareness'] = 1e2

    for i in range(m):
        uav_vel_con = mpc.set_nl_cons('uav'+str(i)+'_vel', uav_state[i][2]**2+uav_state[i][3]**2,vmax**2)
    d_min = 1
    
    if m > 1:
        pairs = combinations(range(0,m),2)
        for p in pairs:
            mpc.set_nl_cons('col_'+str(p[0])+'_'+str(p[1]),-((uav_state[p[0]][0] - uav_state[p[1]][0])**2) - ((uav_state[p[0]][1] - uav_state[p[1]][1])**2), -(d_min**2))

    mpc.setup()
    simulator = do_mpc.simulator.Simulator(model)
    simulator.set_param(t_step=t_step)

    x0 = np.ones(width * width)*a_init

    offset = (width-1) / 2
    indices = np.arange(0, m, dtype=float) + 0.5
    r = np.sqrt(indices/m) * offset
    theta = np.pi * (1 + 5**0.5) * indices
    x0_x = r*np.cos(theta) + offset
    x0_y = r*np.sin(theta) + offset

    for i in range(m):
        x0 = np.append(x0, [x0_x[i],x0_y[i],0,0])
    simulator.x0 = x0
    simulator.setup()
    mpc.set_initial_guess()

    avgs = []
    for i in range(1200):
        u0 = mpc.make_step(x0)
        x0 = simulator.make_step(u0)
        print("AVG reward: " + str(np.average(x0[:-8])))
        avgs.append(np.average(x0[:-8]))
    print("AVG Overall reward: "+str(np.average(avgs)))

    sim_data = simulator.data
    data_name = 'monitoring_'+str(m)+'_'+str(width)
    np.savetxt(data_name+'.csv', sim_data['_x'])
    t_mpc = mpc.data['t_wall_total']
    print("AVG Overall time: "+str(np.average(t_mpc)))
    np.savetxt(data_name+'_times.csv', t_mpc)

    # uav1_pos_x_plot = sim_data['_x','uav1_pos_x']
    # uav1_pos_y_plot = sim_data['_x','uav1_pos_y']
    # plt.plot(uav1_pos_x_plot,uav1_pos_y_plot)
    # uav2_pos_x_plot = sim_data['_x','uav2_pos_x']
    # uav2_pos_y_plot = sim_data['_x','uav2_pos_y']
    # plt.plot(uav2_pos_x_plot,uav2_pos_y_plot)
    # plt.show()
