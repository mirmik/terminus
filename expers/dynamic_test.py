#!/usr/bin/env python3

import time
from terminus.physics.indexed_matrix import IndexedVector
from terminus.physics.body import Body2
from terminus.solver import quadratic_problem_solver_indexes_array
from terminus.physics.world import World
from terminus.physics.force import Force
from terminus.ga201 import Screw2

body = Body2()
body2 = Body2()

world = World()
world.add_body(body)
world.add_body(body2)

body.set_right_velocity_global(Screw2(v=[0,0], m=0.01))

force = Force(v=[0,1])
body.add_right_force(force)

start_time = time.time()
planned_time = start_time
while True:
    current_time = time.time()
    world.iteration(0.01)
    
    if 1==1 or world.iteration_counter() % 10 == 0:
        print()
        print(body.translation())
        print(body2.translation())

#    if current_time - start_time > 3:
#        if force.is_binded():
#            force.unbind()

    planned_time += 0.01
    sleep_interval = planned_time - time.time()
    if sleep_interval > 0:
        time.sleep(sleep_interval)

    #if world.iteration_counter() == 3:
    #    break