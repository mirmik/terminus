# #!/usr/bin/env python3

# import unittest
# import numpy as np
# from termin.fem.assembler import MatrixAssembler
# from termin.fem.multibody3d import (
#     RotationalInertia3D,
#     TorqueVector3D,
#     LinearMass3D,
#     RigidBody3D,
#     ForceVector3D,
#     SphericalJoint3D,
#     FixedPoint3D,
#     FixedRotation3D
# )


# class TestRotationalInertia3D(unittest.TestCase):
#     """Тесты для RotationalInertia3D"""
    
#     def test_free_rotation_with_torque(self):
#         """Свободное вращение с постоянным моментом"""
#         # Параметры
#         J = np.diag([1.0, 2.0, 3.0])  # тензор инерции [кг·м²]
#         tau = np.array([1.0, 0.0, 0.0])  # момент вокруг оси X [Н·м]
#         dt = 0.01  # [с]
        
#         # Сборка системы
#         assembler = MatrixAssembler()
#         omega = assembler.add_variable("omega", size=3)
        
#         assembler.add_contribution(RotationalInertia3D(omega, J, B=0.0, dt=dt))
#         assembler.add_contribution(TorqueVector3D(omega, tau))
        
#         # Решение
#         assembler.solve_and_set()
        
#         # Проверка: угловое ускорение dω/dt = J^(-1) * τ
#         # ω_new = ω_old + dt * J^(-1) * τ
#         # При ω_old = 0: ω_new = dt * J^(-1) * τ
#         expected_omega = dt * np.linalg.inv(J) @ tau
        
#         np.testing.assert_allclose(omega.value, expected_omega, rtol=1e-10)
        
#         # Проверка: только вращение вокруг оси X
#         self.assertGreater(omega.value[0], 0)
#         np.testing.assert_allclose(omega.value[1:], 0, atol=1e-10)
    
#     def test_damped_rotation(self):
#         """Вращение с демпфированием"""
#         # Параметры
#         J = np.diag([1.0, 1.0, 1.0])
#         B = 0.5  # демпфирование [Н·м·с]
#         dt = 0.01
#         omega_old = np.array([10.0, 0.0, 0.0])  # начальная скорость
        
#         # Сборка системы
#         assembler = MatrixAssembler()
#         omega = assembler.add_variable("omega", size=3)
#         omega.set_value(omega_old)
        
#         inertia = RotationalInertia3D(omega, J, B=B, dt=dt)
#         assembler.add_contribution(inertia)
        
#         # Решение
#         assembler.solve_and_set()
        
#         # Проверка: скорость должна уменьшиться из-за демпфирования
#         # (J/dt + B)*ω_new = J/dt*ω_old
#         # ω_new = J/dt*ω_old / (J/dt + B)
#         expected_omega_x = (J[0,0]/dt * omega_old[0]) / (J[0,0]/dt + B)
        
#         np.testing.assert_allclose(omega.value[0], expected_omega_x, rtol=1e-10)
#         self.assertLess(omega.value[0], omega_old[0])  # скорость уменьшилась
    
#     def test_kinetic_energy(self):
#         """Вычисление кинетической энергии вращения"""
#         J = np.diag([1.0, 2.0, 3.0])
#         omega_vec = np.array([1.0, 2.0, 3.0])
        
#         assembler = MatrixAssembler()
#         omega = assembler.add_variable("omega", size=3)
#         omega.set_value(omega_vec)
#         inertia = RotationalInertia3D(omega, J, dt=0.01)
        
#         # E = (1/2) * ω^T * J * ω
#         expected_energy = 0.5 * omega_vec @ J @ omega_vec
#         actual_energy = inertia.get_kinetic_energy(omega_vec)
        
#         self.assertAlmostEqual(actual_energy, expected_energy, places=10)
    
#     def test_angular_momentum(self):
#         """Вычисление углового момента"""
#         J = np.diag([1.0, 2.0, 3.0])
#         omega_vec = np.array([1.0, 2.0, 3.0])
        
#         assembler = MatrixAssembler()
#         omega = assembler.add_variable("omega", size=3)
#         inertia = RotationalInertia3D(omega, J, dt=0.01)
        
#         # L = J * ω
#         expected_L = J @ omega_vec
#         actual_L = inertia.get_angular_momentum(omega_vec)
        
#         np.testing.assert_allclose(actual_L, expected_L, rtol=1e-10)


# class TestLinearMass3D(unittest.TestCase):
#     """Тесты для LinearMass3D"""
    
#     def test_free_motion_with_force(self):
#         """Свободное движение под действием силы"""
#         # Параметры
#         m = 2.0  # масса [кг]
#         F = np.array([10.0, 0.0, 0.0])  # сила [Н]
#         dt = 0.01  # [с]
        
#         # Сборка системы
#         assembler = MatrixAssembler()
#         v = assembler.add_variable("v", size=3)
        
#         assembler.add_contribution(LinearMass3D(v, m, C=0.0, dt=dt))
#         assembler.add_contribution(ForceVector3D(v, F))
        
#         # Решение
#         assembler.solve_and_set()
        
#         # Проверка: ускорение a = F/m
#         # v_new = v_old + dt * a = 0 + dt * F/m
#         expected_v = dt * F / m
        
#         np.testing.assert_allclose(v.value, expected_v, rtol=1e-10)
    
#     def test_damped_motion(self):
#         """Движение с демпфированием"""
#         # Параметры
#         m = 1.0
#         C = 0.5  # демпфирование [Н·с/м]
#         dt = 0.01
#         v_old = np.array([10.0, 5.0, 0.0])  # начальная скорость
        
#         # Сборка системы
#         assembler = MatrixAssembler()
#         v = assembler.add_variable("v", size=3)
#         v.set_value(v_old)
        
#         mass = LinearMass3D(v, m, C=C, dt=dt)
#         assembler.add_contribution(mass)
        
#         # Решение
#         assembler.solve_and_set()
        
#         # Проверка: скорость должна уменьшиться
#         # (m/dt + C)*v_new = m/dt*v_old
#         expected_v = (m/dt * v_old) / (m/dt + C)
        
#         np.testing.assert_allclose(v.value, expected_v, rtol=1e-10)
#         # Каждая компонента скорости меньше начальной
#         for i in range(3):
#             if v_old[i] != 0:
#                 self.assertLess(abs(v.value[i]), abs(v_old[i]))
    
#     def test_kinetic_energy(self):
#         """Вычисление кинетической энергии"""
#         m = 2.0
#         v_vec = np.array([3.0, 4.0, 0.0])
        
#         assembler = MatrixAssembler()
#         v = assembler.add_variable("v", size=3)
#         v.set_value(v_vec)
#         mass = LinearMass3D(v, m, dt=0.01)
        
#         # E = (1/2) * m * v²
#         expected_energy = 0.5 * m * np.dot(v_vec, v_vec)
#         actual_energy = mass.get_kinetic_energy(v_vec)
        
#         self.assertAlmostEqual(actual_energy, expected_energy, places=10)


# class TestRigidBody3D(unittest.TestCase):
#     """Тесты для RigidBody3D"""
    
#     def test_free_rigid_body_with_force_and_torque(self):
#         """Свободное твердое тело под действием силы и момента"""
#         # Параметры
#         m = 2.0  # масса [кг]
#         J = np.diag([1.0, 2.0, 3.0])  # тензор инерции [кг·м²]
#         F = np.array([10.0, 0.0, 0.0])  # сила [Н]
#         tau = np.array([0.0, 5.0, 0.0])  # момент [Н·м]
#         dt = 0.01  # [с]
        
#         # Сборка системы
#         assembler = MatrixAssembler()
#         v = assembler.add_variable("v", size=3)
#         omega = assembler.add_variable("omega", size=3)
        
#         assembler.add_contribution(RigidBody3D(v, omega, m, J, C=0.0, B=0.0, dt=dt))
#         assembler.add_contribution(ForceVector3D(v, F))
#         assembler.add_contribution(TorqueVector3D(omega, tau))
        
#         # Решение
#         assembler.solve_and_set()
        
#         # Проверка поступательного движения
#         expected_v = dt * F / m
#         np.testing.assert_allclose(v.value, expected_v, rtol=1e-10)
        
#         # Проверка вращательного движения
#         expected_omega = dt * np.linalg.inv(J) @ tau
#         np.testing.assert_allclose(omega.value, expected_omega, rtol=1e-10)
    
#     def test_rigid_body_kinetic_energy(self):
#         """Полная кинетическая энергия твердого тела"""
#         m = 2.0
#         J = np.diag([1.0, 2.0, 3.0])
#         v_vec = np.array([1.0, 0.0, 0.0])
#         omega_vec = np.array([0.0, 2.0, 0.0])
        
#         assembler = MatrixAssembler()
#         v = assembler.add_variable("v", size=3)
#         omega = assembler.add_variable("omega", size=3)
#         v.set_value(v_vec)
#         omega.set_value(omega_vec)
        
#         body = RigidBody3D(v, omega, m, J, dt=0.01)
        
#         # E = (1/2)*m*v² + (1/2)*ω^T*J*ω
#         E_trans = 0.5 * m * np.dot(v_vec, v_vec)
#         E_rot = 0.5 * omega_vec @ J @ omega_vec
#         expected_energy = E_trans + E_rot
        
#         actual_energy = body.get_kinetic_energy(v_vec, omega_vec)
        
#         self.assertAlmostEqual(actual_energy, expected_energy, places=10)


# class TestSphericalJoint3D(unittest.TestCase):
#     """Тесты для SphericalJoint3D"""
    
#     def test_spherical_joint_constraint(self):
#         """Сферический шарнир фиксирует точку на теле"""
#         # Параметры
#         m = 1.0
#         J = np.diag([1.0, 1.0, 1.0])
#         r = np.array([1.0, 0.0, 0.0])  # точка на расстоянии 1м по оси X
#         F = np.array([0.0, 10.0, 0.0])  # сила по оси Y
#         dt = 0.01
        
#         # Сборка системы
#         assembler = MatrixAssembler()
#         v = assembler.add_variable("v", size=3)
#         omega = assembler.add_variable("omega", size=3)
        
#         assembler.add_contribution(RigidBody3D(v, omega, m, J, dt=dt))
#         assembler.add_contribution(ForceVector3D(v, F))
        
#         # Сферический шарнир: v + ω × r = 0
#         assembler.constraints = [SphericalJoint3D(v, omega, r)]
        
#         # Решение
#         assembler.solve_and_set()
        
#         # Проверка связи: v + ω × r должно быть близко к нулю
#         joint = SphericalJoint3D(v, omega, r)
#         violation = joint.get_constraint_violation(v.value, omega.value)
        
#         np.testing.assert_allclose(violation, 0, atol=1e-8)
        
#         # Тело должно вращаться (не может двигаться поступательно)
#         self.assertGreater(np.linalg.norm(omega.value), 1e-6)
    
#     def test_spherical_joint_rotation_around_point(self):
#         """Тело вращается вокруг зафиксированной точки"""
#         m = 1.0
#         J = np.diag([1.0, 1.0, 1.0])
#         r = np.array([0.5, 0.0, 0.0])
#         tau = np.array([0.0, 0.0, 1.0])  # момент вокруг оси Z
#         dt = 0.01
        
#         assembler = MatrixAssembler()
#         v = assembler.add_variable("v", size=3)
#         omega = assembler.add_variable("omega", size=3)
        
#         assembler.add_contribution(RigidBody3D(v, omega, m, J, dt=dt))
#         assembler.add_contribution(TorqueVector3D(omega, tau))
#         assembler.constraints = [SphericalJoint3D(v, omega, r)]
        
#         assembler.solve_and_set()
        
#         # Связь должна выполняться
#         joint = SphericalJoint3D(v, omega, r)
#         violation = joint.get_constraint_violation(v.value, omega.value)
#         np.testing.assert_allclose(violation, 0, atol=1e-8)
        
#         # Должно быть вращение вокруг оси Z
#         self.assertGreater(abs(omega.value[2]), 1e-6)


# class TestFixedPoint3D(unittest.TestCase):
#     """Тесты для FixedPoint3D"""
    
#     def test_fixed_point_zero_velocity(self):
#         """Фиксация точки: скорость должна быть нулевой"""
#         m = 1.0
#         F = np.array([10.0, 5.0, 3.0])  # сила
#         dt = 0.01
        
#         assembler = MatrixAssembler()
#         v = assembler.add_variable("v", size=3)
        
#         assembler.add_contribution(LinearMass3D(v, m, dt=dt))
#         assembler.add_contribution(ForceVector3D(v, F))
#         assembler.constraints = [FixedPoint3D(v)]  # v = 0
        
#         assembler.solve_and_set()
        
#         # Скорость должна быть нулевой несмотря на силу
#         np.testing.assert_allclose(v.value, 0, atol=1e-10)
    
#     def test_fixed_point_target_velocity(self):
#         """Фиксация точки с заданной скоростью"""
#         m = 1.0
#         target_v = np.array([1.0, 2.0, 3.0])
#         dt = 0.01
        
#         assembler = MatrixAssembler()
#         v = assembler.add_variable("v", size=3)
        
#         assembler.add_contribution(LinearMass3D(v, m, dt=dt))
#         assembler.constraints = [FixedPoint3D(v, target=target_v)]
        
#         assembler.solve_and_set()
        
#         # Скорость должна быть равна целевой
#         np.testing.assert_allclose(v.value, target_v, rtol=1e-10)


# class TestFixedRotation3D(unittest.TestCase):
#     """Тесты для FixedRotation3D"""
    
#     def test_fixed_rotation_zero_omega(self):
#         """Фиксация вращения: угловая скорость должна быть нулевой"""
#         J = np.diag([1.0, 2.0, 3.0])
#         tau = np.array([1.0, 2.0, 3.0])  # момент
#         dt = 0.01
        
#         assembler = MatrixAssembler()
#         omega = assembler.add_variable("omega", size=3)
        
#         assembler.add_contribution(RotationalInertia3D(omega, J, dt=dt))
#         assembler.add_contribution(TorqueVector3D(omega, tau))
#         assembler.constraints = [FixedRotation3D(omega)]  # ω = 0
        
#         assembler.solve_and_set()
        
#         # Угловая скорость должна быть нулевой несмотря на момент
#         np.testing.assert_allclose(omega.value, 0, atol=1e-10)
    
#     def test_fixed_rotation_target_omega(self):
#         """Фиксация вращения с заданной угловой скоростью"""
#         J = np.diag([1.0, 2.0, 3.0])
#         target_omega = np.array([1.0, 0.0, 2.0])
#         dt = 0.01
        
#         assembler = MatrixAssembler()
#         omega = assembler.add_variable("omega", size=3)
        
#         assembler.add_contribution(RotationalInertia3D(omega, J, dt=dt))
#         assembler.constraints = [FixedRotation3D(omega, target=target_omega)]
        
#         assembler.solve_and_set()
        
#         # Угловая скорость должна быть равна целевой
#         np.testing.assert_allclose(omega.value, target_omega, rtol=1e-10)


# class TestIntegrationMultibody3D(unittest.TestCase):
#     """Интеграционные тесты для 3D многотельной механики"""
    
#     def test_pendulum_3d(self):
#         """3D математический маятник под действием гравитации"""
#         # Параметры
#         m = 1.0  # масса [кг]
#         L = 1.0  # длина [м]
#         g = 9.81  # ускорение свободного падения [м/с²]
#         J = np.diag([1.0, 1.0, 1.0])  # тензор инерции
#         dt = 0.01
        
#         # Сила гравитации
#         F_gravity = np.array([0.0, -m*g, 0.0])
        
#         # Сборка системы
#         assembler = MatrixAssembler()
#         v = assembler.add_variable("v", size=3)
#         omega = assembler.add_variable("omega", size=3)
        
#         assembler.add_contribution(RigidBody3D(v, omega, m, J, dt=dt))
#         assembler.add_contribution(ForceVector3D(v, F_gravity))
        
#         # Точка подвеса на расстоянии L от ЦМ по оси Y
#         # Маятник висит вниз, точка подвеса выше ЦМ
#         r_pivot = np.array([0.0, L, 0.0])  # от ЦМ к точке подвеса
#         assembler.constraints = [SphericalJoint3D(v, omega, r_pivot)]
        
#         # Решение
#         assembler.solve_and_set()
        
#         # Проверка: связь должна выполняться
#         joint = SphericalJoint3D(v, omega, r_pivot)
#         violation = joint.get_constraint_violation(v.value, omega.value)
#         np.testing.assert_allclose(violation, 0, atol=1e-8)
        
#         # Маятник висит вертикально вниз - это положение равновесия
#         # Скорости и угловые скорости должны быть нулевыми (равновесие)
#         # Но реакция связи должна компенсировать гравитацию
#         # Это валидный результат для начального состояния в равновесии
    
#     def test_spinning_top(self):
#         """Вращающийся волчок с начальной угловой скоростью"""
#         # Параметры (симметричный волчок)
#         J = np.diag([1.0, 1.0, 2.0])  # Jz > Jx = Jy
#         omega_initial = np.array([0.1, 0.1, 10.0])  # быстрое вращение вокруг оси Z
#         B = 0.01  # малое демпфирование
#         dt = 0.001
        
#         assembler = MatrixAssembler()
#         omega = assembler.add_variable("omega", size=3)
#         omega.set_value(omega_initial)
        
#         inertia = RotationalInertia3D(omega, J, B=B, dt=dt, 
#                                       include_gyroscopic=False)  # без гироскопа для простоты
#         assembler.add_contribution(inertia)
        
#         # Решение
#         assembler.solve_and_set()
        
#         # Проверка: скорость должна немного уменьшиться из-за демпфирования
#         for i in range(3):
#             if omega_initial[i] != 0:
#                 self.assertLess(abs(omega.value[i]), abs(omega_initial[i]) + dt)
        
#         # Компонента вдоль оси Z должна оставаться доминирующей
#         self.assertGreater(abs(omega.value[2]), abs(omega.value[0]))
#         self.assertGreater(abs(omega.value[2]), abs(omega.value[1]))
    
#     def test_coupled_translation_rotation(self):
#         """Связанное поступательное и вращательное движение"""
#         # Тело с силой, приложенной не в центре масс
#         m = 2.0
#         J = np.diag([1.0, 1.0, 1.0])
#         dt = 0.01
        
#         # Сила приложена на расстоянии от ЦМ
#         F = np.array([10.0, 0.0, 0.0])
#         r_force = np.array([0.0, 0.5, 0.0])  # точка приложения силы
        
#         # Момент: τ = r × F
#         tau = np.cross(r_force, F)
        
#         assembler = MatrixAssembler()
#         v = assembler.add_variable("v", size=3)
#         omega = assembler.add_variable("omega", size=3)
        
#         assembler.add_contribution(RigidBody3D(v, omega, m, J, dt=dt))
#         assembler.add_contribution(ForceVector3D(v, F))
#         assembler.add_contribution(TorqueVector3D(omega, tau))
        
#         assembler.solve_and_set()
        
#         # Должно быть и поступательное и вращательное движение
#         self.assertGreater(np.linalg.norm(v.value), 1e-6)
#         self.assertGreater(np.linalg.norm(omega.value), 1e-6)
        
#         # Проверка направлений
#         self.assertGreater(v.value[0], 0)  # движение по X
#         self.assertGreater(abs(omega.value[2]), 1e-6)  # вращение вокруг Z


# if __name__ == '__main__':
#     unittest.main()
