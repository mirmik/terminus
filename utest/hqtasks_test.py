import numpy as np

from termin.robot.hqsolver import HQPSolver, Level
from termin.robot.hqtasks import (
    JointTrackingTask,
    CartesianTrackingTask,
    JointEqualityConstraint,
    CartesianEqualityConstraint,
    JointBoundsConstraint,
)


def test_joint_tracking_and_bounds_tasks():
    solver = HQPSolver(n_vars=2)

    lvl = Level(priority=0)
    lvl.add_task(JointTrackingTask(q_ref=[2.0, -1.0]))
    lvl.add_inequality(JointBoundsConstraint(lower=[-0.5, -0.5], upper=[1.0, 0.0]))
    solver.add_level(lvl)

    x = solver.solve()
    np.testing.assert_allclose(x, [1.0, -0.5], atol=1e-7)


def test_cartesian_tasks_with_equalities():
    solver = HQPSolver(n_vars=3)

    lvl0 = Level(priority=0)
    sel = np.array([[1.0, 0.0, 0.0]])
    lvl0.add_equality(JointEqualityConstraint(q_target=[1.0, 0.0, 0.0], selection=sel))
    solver.add_level(lvl0)

    lvl1 = Level(priority=1)
    J_cart = np.array([[0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0]])
    desired = np.array([2.0, -3.0])
    lvl1.add_task(CartesianTrackingTask(J_cart, desired))
    lvl1.add_equality(CartesianEqualityConstraint(np.array([[0.0, 1.0, 1.0]]), np.array([0.0])))
    solver.add_level(lvl1)

    x = solver.solve()
    np.testing.assert_allclose(x, [1.0, 2.5, -2.5], atol=1e-7)
