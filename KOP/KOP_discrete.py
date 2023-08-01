#!/usr/bin/env python3

import numpy as np
import csv
import ctypes
import ctypes.util
import do_mpc.data
import do_mpc
from casadi import *
import sys

import matplotlib.pyplot as plt
import matplotlib as mpl

solver = 'ma97'

# Customizing Matplotlib:
mpl.rcParams['font.size'] = 18
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['axes.grid'] = True

if __name__ == '__main__':

    if len(sys.argv) > 1:
        Cmax = int(sys.argv[1])
    else:
        Cmax = 40

    nodes = []
    with open('tsiligirides.txt', newline='') as instance:
        instance_reader = csv.reader(instance, delimiter='\t')
        start_node = next(instance_reader)
        end_node = next(instance_reader)
        for i in range(19):
            nodes.append(next(instance_reader))

    t_step = 0.1
    n_horizon = int(Cmax/t_step)

    model_type = 'discrete'
    model = do_mpc.model.Model(model_type)
    awareness = model.set_variable(var_name='awareness', var_type='_x', shape=(len(nodes), 1))  # 5x5 in each direction
    uav1_pos_x = model.set_variable(var_name='uav1_pos_x', var_type='_x')
    uav1_pos_y = model.set_variable(var_name='uav1_pos_y', var_type='_x')
    uav1_vel_x = model.set_variable(var_name='uav1_vel_x', var_type='_x')
    uav1_vel_y = model.set_variable(var_name='uav1_vel_y', var_type='_x')
    uav1_acc_x = model.set_variable(var_name='uav1_acc_x', var_type='_u')
    uav1_acc_y = model.set_variable(var_name='uav1_acc_y', var_type='_u')

    d_uav1_pos_x = uav1_pos_x + t_step * uav1_vel_x + t_step**2*uav1_acc_x/2.0
    d_uav1_pos_y = uav1_pos_y + t_step * uav1_vel_y + t_step**2*uav1_acc_y/2.0

    model.set_rhs('uav1_pos_x', d_uav1_pos_x)
    model.set_rhs('uav1_pos_y', d_uav1_pos_y)
    model.set_rhs('uav1_vel_x', uav1_vel_x + t_step * uav1_acc_x)
    model.set_rhs('uav1_vel_y', uav1_vel_y + t_step * uav1_acc_y)

    c = 0.05  # cutoff
    c = c**2

    x0 = []
    awareness_next = SX(1,len(nodes))
    A_tot = 0
    for k in range(len(nodes)):
        i = float(nodes[k][0])
        j = float(nodes[k][1])
        x0.append(float(nodes[k][2]))
        exp = awareness[k] \
                  * (1 - 1 / (1 + (((uav1_pos_x - i) ** 2 + (uav1_pos_y - j) ** 2) / c)))
        awareness_next[k] = exp
        A_tot += awareness[k]

    x0 = np.append(x0,[float(start_node[0]),float(start_node[1]),0,0])
    A_final = A_tot + 1000*(uav1_pos_x-float(end_node[0]))**2 + 1000*(uav1_pos_y-float(end_node[1]))**2

    model.set_rhs('awareness', awareness_next)
    model.set_expression('A_tot', A_tot)
    model.set_expression('A_final', A_final)
    model.setup()
    mpc = do_mpc.controller.MPC(model)
    setup_mpc = {
        'n_horizon': n_horizon,
        't_step': t_step,
        'store_full_solution': True,
        'nlpsol_opts': {'ipopt.linear_solver': solver,'ipopt.print_level': 5}
    }
    mpc.set_param(**setup_mpc)

    lterm = model.aux['A_tot']
    mterm = model.aux['A_final']

    mpc.set_objective(lterm=lterm, mterm=mterm)
    rterm = 1e-3
    mpc.set_rterm(uav1_acc_x=rterm, uav1_acc_y=rterm)

    vmax = 3
    amax = 1.5

    # mpc.bounds['lower', '_x', 'uav1_pos_x'] = 0
    # mpc.bounds['upper', '_x', 'uav1_pos_x'] = width
    mpc.bounds['lower', '_x', 'uav1_vel_x'] = -vmax
    mpc.bounds['upper', '_x', 'uav1_vel_x'] = vmax
    mpc.bounds['lower', '_u', 'uav1_acc_x'] = -amax
    mpc.bounds['upper', '_u', 'uav1_acc_x'] = amax
    # mpc.bounds['lower', '_x', 'uav1_pos_y'] = 0
    # mpc.bounds['upper', '_x', 'uav1_pos_y'] = width
    mpc.bounds['lower', '_x', 'uav1_vel_y'] = -vmax
    mpc.bounds['upper', '_x', 'uav1_vel_y'] = vmax
    mpc.bounds['lower', '_u', 'uav1_acc_y'] = -amax
    mpc.bounds['upper', '_u', 'uav1_acc_y'] = amax

    uav1_vel_con = mpc.set_nl_cons('uav1_vel', uav1_vel_x**2+uav1_vel_y**2,vmax**2)

    mpc.setup()
    x0 = np.array(x0)

    mpc.x0 = x0
    mpc.u0 = np.array([0,0])
    mpc.set_initial_guess()
    u0 = mpc.make_step(x0)

    data = mpc.data
    uav1_pos_x_plot = data.prediction(('_x', 'uav1_pos_x'))[0]
    uav1_pos_y_plot = data.prediction(('_x', 'uav1_pos_y'))[0]
    aw = data.prediction(('_x', 'awareness'))

    aw_collected = x0[:-4][(aw[:,-1].flatten()) < 0.1]
    print(aw[:,-1].flatten())

    print('final rewards ')
    print(aw[:,-1])
    print('final gain ' + str(sum(aw_collected)))


    fig, ax = plt.subplots()

    for k in range(len(nodes)):
        i = float(nodes[k][0])
        j = float(nodes[k][1])
        circ = plt.Circle((i, j), 0.1, color='r', fill=False, linestyle='--')
        ax.add_patch(circ)

    ax.plot(uav1_pos_x_plot,uav1_pos_y_plot)
    ax.plot(float(start_node[0]),float(start_node[1]),'rx')

    data_name = 'KOP_'+str(n_horizon)
    np.savetxt(data_name+'.csv', np.hstack((uav1_pos_x_plot, uav1_pos_y_plot)))
    t_mpc = mpc.data['t_wall_total']

    with open('eval.txt', 'a') as file:
        file.write(str(Cmax)+" "+str(sum(aw_collected))+" "+str(t_mpc)+"\n")
        file.close()

    # plt.show()
