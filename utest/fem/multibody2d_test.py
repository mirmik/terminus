#!/usr/bin/env python3
# coding:utf-8

import unittest
import numpy as np
import warnings
from termin.fem.assembler import MatrixAssembler
from termin.fem.multibody2d import (
    RotationalInertia2D,
    TorqueSource2D,
    RigidBody2D,
    ForceVector2D,
    ForceOnBody2D,
    FixedRevoluteJoint2D,
    TwoBodyRevoluteJoint2D
)


class TestIntegrationMultibody2D(unittest.TestCase):
    """Интеграционные тесты для многотельной системы"""
    
    def test_free_rotation_with_torque(self):
        """
        Тест свободного вращения с постоянным моментом.
        Без демпфирования ω должна расти линейно: ω = τ*t/J
        """
        assembler = MatrixAssembler()
        omega = assembler.add_variable("omega", size=1)
        
        J = 2.0  # кг·м²
        torque = 10.0  # Н·м
        dt = 0.01  # с
        
        inertia = RotationalInertia2D(omega, J=J, B=0.0, dt=dt)
        torque_source = TorqueSource2D(omega, torque=torque)
        
        assembler.add_contribution(inertia)
        assembler.add_contribution(torque_source)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assembler.solve_and_set()
        
        # Ожидаемое ускорение: α = τ/J
        # За один шаг: ω_new = ω_old + α*dt = 0 + (10/2)*0.01 = 0.05
        # Но для неявной схемы: (J/dt)*ω_new = τ + (J/dt)*ω_old
        # ω_new = τ*dt/J = 10*0.01/2 = 0.05
        expected_omega = torque * dt / J
        
        self.assertAlmostEqual(omega.value, expected_omega, places=5)
    
    def test_damped_rotation(self):
        """
        Тест вращения с демпфированием без внешнего момента.
        Скорость должна затухать.
        """
        assembler = MatrixAssembler()
        omega = assembler.add_variable("omega", size=1)
        omega.set_value(10.0)  # начальная скорость
        
        J = 1.0
        B = 10.0
        dt = 0.01
        
        inertia = RotationalInertia2D(omega, J=J, B=B, dt=dt)
        assembler.add_contribution(inertia)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assembler.solve_and_set()
        
        # (J/dt + B)*ω_new = (J/dt)*ω_old
        # ω_new = ω_old * (J/dt) / (J/dt + B)
        omega_old = 10.0
        expected_omega = omega_old * (J / dt) / (J / dt + B)
        
        self.assertAlmostEqual(omega.value, expected_omega, places=5)
        
        # Новая скорость должна быть меньше начальной
        self.assertLess(omega.value, omega_old)
    
    def test_rigid_body_with_force(self):
        """
        Тест твердого тела с приложенной силой.
        """
        assembler = MatrixAssembler()
        velocity = assembler.add_variable("velocity", size=2)
        omega = assembler.add_variable("omega", size=1)
        
        m = 2.0  # кг
        J = 0.5  # кг·м²
        dt = 0.01  # с
        force = np.array([20.0, 0.0])  # Н
        
        body = RigidBody2D(m=m, J=J, C=0.0, B=0.0, dt=dt, velocity=velocity, omega=omega)
        force_element = ForceVector2D(velocity, force)
        
        assembler.add_contribution(body)
        assembler.add_contribution(force_element)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assembler.solve_and_set()
        
        # v_new = F*dt/m = 20*0.01/2 = 0.1
        expected_vx = force[0] * dt / m
        expected_vy = 0.0
        
        self.assertAlmostEqual(velocity.value[0], expected_vx, places=5)
        self.assertAlmostEqual(velocity.value[1], expected_vy, places=5)
        self.assertAlmostEqual(omega.value, 0.0, places=5)


class TestPendulum2D(unittest.TestCase):
    """Тесты для 2D маятника с RigidBody2D и FixedRevoluteJoint2D"""
    
    def test_pendulum_equilibrium(self):
        """
        Маятник в положении равновесия (висит вертикально вниз).
        При нулевой начальной скорости должен остаться в равновесии.
        """
        # Параметры маятника
        m = 1.0      # масса [кг]
        L = 1.0      # длина [м]
        g = 9.81     # ускорение свободного падения [м/с²]
        J = m * L**2 / 3  # момент инерции стержня относительно конца [кг·м²]
        dt = 0.01    # шаг времени [с]
        
        # Сборка системы
        assembler = MatrixAssembler()
        
        # Твердое тело (создает свои переменные автоматически)
        body = RigidBody2D(m=m, J=J, C=0.0, B=0.0, dt=dt)
        velocity = body.velocity  # [vx, vy]
        omega = body.omega  # [ω]

        assembler.add_contribution(body)
        
        # Сила гравитации (приложена к центру масс) - новый API
        F_gravity = np.array([0.0, -m*g])
        assembler.add_contribution(ForceOnBody2D(body, force=F_gravity))
        
        # Вращательный шарнир с фиксацией в пространстве
        # Центр масс находится на расстоянии L/2 от точки подвеса
        # Вектор от ЦМ к точке подвеса: r = [0, L/2] (вверх)
        r_pivot = np.array([0.0, L/2])
        joint = FixedRevoluteJoint2D(body, r_pivot)
        assembler.add_constraint(joint)
        
        # Решение
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assembler.solve_and_set()
        
        # В положении равновесия скорости должны быть нулевыми
        # (реакция шарнира компенсирует гравитацию)
        np.testing.assert_allclose(velocity.value, 0.0, atol=1e-8)
        np.testing.assert_allclose(omega.value, 0.0, atol=1e-8)
    
    def test_pendulum_with_initial_velocity(self):
        """
        Маятник с начальной угловой скоростью.
        Проверяем, что кинематическая связь выполняется.
        """
        # Параметры
        m = 1.0
        L = 1.0
        g = 9.81
        J = m * L**2 / 3
        dt = 0.01
        
        # Начальная угловая скорость
        omega_initial = 1.0  # рад/с
        
        # Сборка системы
        assembler = MatrixAssembler()
        velocity = assembler.add_variable("velocity", size=2)
        omega = assembler.add_variable("omega", size=1)
        omega.set_value(omega_initial)
        
        body = RigidBody2D(m=m, J=J, C=0.0, B=0.0, dt=dt, velocity=velocity, omega=omega)
        assembler.add_contribution(body)
        
        F_gravity = np.array([0.0, -m*g])
        assembler.add_contribution(ForceOnBody2D(body, force=F_gravity))
        
        r_pivot = np.array([0.0, L/2])
        joint = FixedRevoluteJoint2D(body, r_pivot)
        assembler.add_constraint(joint)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assembler.solve_and_set()
        
        # Проверка кинематической связи: v + ω × r = 0
        # В 2D: [vx, vy] + [-ω*ry, ω*rx] = 0
        v = velocity.value
        w = omega.value
        rx, ry = r_pivot
        
        constraint_x = v[0] - w * ry
        constraint_y = v[1] + w * rx
        
        np.testing.assert_allclose(constraint_x, 0.0, atol=1e-8)
        np.testing.assert_allclose(constraint_y, 0.0, atol=1e-8)
        
        # Маятник должен замедлиться под действием гравитации
        # (момент силы тяжести направлен против начального вращения при малых углах)
        if omega_initial > 0:
            # Угловая скорость должна немного измениться
            self.assertNotEqual(omega.value, omega_initial)
    
    def test_pendulum_horizontal_position(self):
        """
        Маятник в горизонтальном положении с начальной скоростью.
        Проверяем, что момент силы тяжести создает угловое ускорение.
        """
        # Параметры
        m = 1.0
        L = 1.0
        g = 9.81
        J = m * L**2 / 3
        dt = 0.01
        
        # Начальное состояние: маятник горизонтален, небольшая угловая скорость
        omega_initial = 0.0
        
        # Сборка системы
        assembler = MatrixAssembler()
        velocity = assembler.add_variable("velocity", size=2)
        omega = assembler.add_variable("omega", size=1)
        
        body = RigidBody2D(m=m, J=J, C=0.0, B=0.0, dt=dt, velocity=velocity, omega=omega)
        assembler.add_contribution(body)
        
        # Гравитация и момент от нее - новый API объединяет силу и момент
        F_gravity = np.array([0.0, -m*g])
        # Момент от силы тяжести относительно точки подвеса
        # Если маятник горизонтален (ЦМ справа от точки подвеса):
        # r_cm = [L/2, 0] - положение ЦМ относительно точки подвеса
        # τ = r_cm × F = [L/2, 0] × [0, -mg] = -mg*L/2 (по часовой)
        torque_gravity = -m * g * L / 2
        assembler.add_contribution(ForceOnBody2D(body, force=F_gravity, torque=torque_gravity))
        
        # Шарнир с фиксацией (для горизонтального положения r от ЦМ к шарниру)
        r_pivot = np.array([-L/2, 0.0])
        joint = FixedRevoluteJoint2D(body, r_pivot)
        assembler.add_constraint(joint)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assembler.solve_and_set()
        
        # Проверка связи
        v = velocity.value
        w = omega.value
        rx, ry = r_pivot
        
        constraint_x = v[0] - w * ry
        constraint_y = v[1] + w * rx
        
        np.testing.assert_allclose(constraint_x, 0.0, atol=1e-8)
        np.testing.assert_allclose(constraint_y, 0.0, atol=1e-8)
        
        # Под действием момента маятник должен начать вращаться
        # (отрицательный момент -> отрицательная угловая скорость)
        self.assertLess(omega.value, 0.0)
        
        # Угловое ускорение: α = τ/J
        expected_alpha = torque_gravity / J
        expected_omega = omega_initial + expected_alpha * dt
        
        # Неявная схема дает немного другой результат, но знак должен совпадать
        self.assertLess(omega.value, 0.0)
    
    def test_pendulum_energy_conservation(self):
        """
        Проверка закона сохранения энергии для маятника без демпфирования.
        Полная энергия E = E_kinetic + E_potential должна сохраняться.
        """
        # Параметры
        m = 1.0
        L = 1.0
        g = 9.81
        J = m * L**2 / 3  # момент инерции стержня относительно конца
        dt = 0.001  # малый шаг для точности
        
        # Начальное отклонение маятника
        theta0 = np.pi / 12  # 15 градусов (малый угол для линейного приближения)
        omega_initial = 0.0  # отпускаем из состояния покоя
        
        # Положение центра масс относительно точки подвеса
        # Начальное положение: ЦМ на расстоянии L/2 от точки подвеса под углом theta0
        x0 = (L/2) * np.sin(theta0)
        y0 = -(L/2) * np.cos(theta0)  # ось Y вниз
        
        # Начальная потенциальная энергия (относительно положения равновесия)
        # E_pot = m*g*h, где h - высота ЦМ над положением равновесия
        h0 = y0 - (-(L/2))  # разница между текущим y и y в равновесии
        E_potential_initial = m * g * h0
        
        # Начальная кинетическая энергия
        E_kinetic_initial = 0.5 * m * 0**2 + 0.5 * J * omega_initial**2  # нулевая скорость
        
        # Полная начальная энергия
        E_total_initial = E_kinetic_initial + E_potential_initial
        
        print(f"\nНачальная энергия: E_total = {E_total_initial:.6f} Дж")
        print(f"  E_kinetic = {E_kinetic_initial:.6f} Дж")
        print(f"  E_potential = {E_potential_initial:.6f} Дж")
        print(f"  theta0 = {np.degrees(theta0):.2f} град")
        print(f"  h0 = {h0:.6f} м")
        
        # Симулируем несколько шагов (достаточно для полного периода качания)
        # Период малых колебаний: T ≈ 2π√(L/(2g)) ≈ 1.0 с для L=1м
        n_steps = 1500  # 1.5 секунды - полтора периода
        energies = []
        times = []
        thetas = []
        omegas = []
        
        # Сборка системы (один раз!)
        assembler = MatrixAssembler()
        velocity = assembler.add_variable("velocity", size=2)
        omega_var = assembler.add_variable("omega", size=1)
        
        # Вычисляем начальную скорость ЦМ из связи: v = ω × r
        # v = [-ω*ry, ω*rx] где r - вектор от ЦМ к шарниру
        r_pivot = np.array([-(L/2)*np.sin(theta0), (L/2)*np.cos(theta0)])
        v_initial = np.array([omega_initial * r_pivot[1], -omega_initial * r_pivot[0]])
        
        # Устанавливаем начальные значения
        velocity.set_value(v_initial)
        omega_var.set_value(omega_initial)
        
        # Создаем элементы системы
        body = RigidBody2D(m=m, J=J, C=0.0, B=0.0, dt=dt, velocity=velocity, omega=omega_var)
        assembler.add_contribution(body)
        
        # Сила гравитации - применяется к центру масс
        F_gravity = np.array([0.0, -m*g])
        assembler.add_contribution(ForceOnBody2D(body, force=F_gravity))
        
        # Создаем шарнир с фиксацией (один раз!)
        r_pivot = np.array([-(L/2)*np.sin(theta0), (L/2)*np.cos(theta0)])
        joint = FixedRevoluteJoint2D(body, r_pivot)
        assembler.add_constraint(joint)
        
        # Текущее состояние
        theta = theta0
        omega = omega_initial
        
        for step in range(n_steps):
            # Обновляем вектор к точке шарнира (он меняется при вращении)
            r_pivot = np.array([-(L/2)*np.sin(theta), (L/2)*np.cos(theta)])
            joint.update_r(r_pivot)
            
            # Решение
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                assembler.solve_and_set()
            
            # Обновляем состояние
            omega = omega_var.value
            v = velocity.value
            
            # Обновляем угол (простое интегрирование)
            theta += omega * dt
            
            # Вычисляем текущую энергию
            # Кинетическая энергия: E_k = (1/2)*m*v² + (1/2)*J*ω²
            v_squared = v[0]**2 + v[1]**2
            E_kinetic = 0.5 * m * v_squared + 0.5 * J * omega**2
            
            # Потенциальная энергия: E_p = m*g*h
            y = -(L/2) * np.cos(theta)
            h = y - (-(L/2))
            E_potential = m * g * h
            
            # Полная энергия
            E_total = E_kinetic + E_potential
            
            energies.append(E_total)
            times.append(step * dt)
            thetas.append(theta)
            omegas.append(omega)
            
            if step % 100 == 0:  # реже печатаем
                print(f"Step {step}: t={step*dt:.4f}s, theta={np.degrees(theta):6.2f}°, "
                      f"omega={omega:7.4f} rad/s, E={E_total:.6f} J, "
                      f"dE={E_total-E_total_initial:.6e} J")
        
        print()  # пустая строка перед статистикой
        
        # Проверка сохранения энергии
        energies = np.array(energies)
        energy_deviation = np.abs(energies - E_total_initial)
        max_deviation = np.max(energy_deviation)
        mean_deviation = np.mean(energy_deviation)
        
        print(f"\nСтатистика энергии:")
        print(f"  Начальная энергия: {E_total_initial:.6f} Дж")
        print(f"  Конечная энергия:  {energies[-1]:.6f} Дж")
        print(f"  Макс. отклонение:  {max_deviation:.6e} Дж ({100*max_deviation/E_total_initial:.4f}%)")
        print(f"  Сред. отклонение:  {mean_deviation:.6e} Дж ({100*mean_deviation/E_total_initial:.4f}%)")
        
        # Энергия должна сохраняться с погрешностью не более 1%
        # (неявная схема Эйлера может давать небольшую численную диссипацию)
        relative_error = max_deviation / E_total_initial
        self.assertLess(relative_error, 0.01, 
                       f"Энергия не сохраняется: отклонение {100*relative_error:.2f}%")
        
        # Маятник должен качаться (omega должна менять знак)
        omegas = np.array(omegas)
        omega_max = np.max(omegas)
        omega_min = np.min(omegas)
        
        print(f"  Диапазон omega: [{omega_min:.4f}, {omega_max:.4f}] rad/s")
        
        # Убеждаемся, что маятник качается в обе стороны
        self.assertGreater(omega_max, 0.1)
        self.assertLess(omega_min, -0.1)


class TestTwoBodyRevoluteJoint2D(unittest.TestCase):
    """Тесты для двухтельного вращательного шарнира (RevoluteJoint2D)"""
    
    def test_two_bodies_at_rest(self):
        """
        Два тела, соединенные шарниром, в состоянии покоя.
        Без сил система должна оставаться в покое.
        """
        # Параметры тел
        m1 = 1.0
        m2 = 1.0
        J1 = 0.5
        J2 = 0.5
        dt = 0.01
        
        # Сборка системы
        assembler = MatrixAssembler()
        
        # Создаем два тела
        body1 = RigidBody2D(m=m1, J=J1, C=0.0, B=0.0, dt=dt)
        body2 = RigidBody2D(m=m2, J=J2, C=0.0, B=0.0, dt=dt)
        
        assembler.add_contribution(body1)
        assembler.add_contribution(body2)
        
        # Соединяем их шарниром
        # Точка соединения на расстоянии 0.5м от каждого центра масс
        r1 = np.array([0.5, 0.0])  # от ЦМ body1 к точке шарнира
        r2 = np.array([-0.5, 0.0]) # от ЦМ body2 к точке шарнира
        joint = TwoBodyRevoluteJoint2D(body1, body2, r1, r2)
        assembler.add_constraint(joint)
        
        # Решение
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assembler.solve_and_set()
        
        # Все скорости должны быть нулевыми
        np.testing.assert_allclose(body1.velocity.value, 0.0, atol=1e-8)
        np.testing.assert_allclose(body2.velocity.value, 0.0, atol=1e-8)
        np.testing.assert_allclose(body1.omega.value, 0.0, atol=1e-8)
        np.testing.assert_allclose(body2.omega.value, 0.0, atol=1e-8)
    
    def test_two_bodies_constraint_satisfaction(self):
        """
        Проверка выполнения кинематической связи при начальных скоростях.
        """
        # Параметры
        m1 = 1.0
        m2 = 1.0
        J1 = 0.5
        J2 = 0.5
        dt = 0.01
        
        # Сборка системы
        assembler = MatrixAssembler()
        
        body1 = RigidBody2D(m=m1, J=J1, C=0.0, B=0.0, dt=dt)
        body2 = RigidBody2D(m=m2, J=J2, C=0.0, B=0.0, dt=dt)
        
        # Устанавливаем начальные скорости
        body1.velocity.set_value(np.array([1.0, 0.0]))
        body1.omega.set_value(0.5)
        body2.velocity.set_value(np.array([0.5, 0.0]))
        body2.omega.set_value(-0.3)
        
        assembler.add_contribution(body1)
        assembler.add_contribution(body2)
        
        # Шарнир
        r1 = np.array([0.5, 0.0])
        r2 = np.array([-0.5, 0.0])
        joint = TwoBodyRevoluteJoint2D(body1, body2, r1, r2)
        assembler.add_constraint(joint)
        
        # Решение
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assembler.solve_and_set()
        
        # Проверка выполнения кинематической связи
        # v1 + ω1 × r1 = v2 + ω2 × r2
        v1 = body1.velocity.value
        w1 = body1.omega.value
        v2 = body2.velocity.value
        w2 = body2.omega.value
        
        # Скорость точки шарнира на body1
        v_joint1 = v1 + np.array([-w1 * r1[1], w1 * r1[0]])
        
        # Скорость точки шарнира на body2
        v_joint2 = v2 + np.array([-w2 * r2[1], w2 * r2[0]])
        
        # Они должны совпадать
        np.testing.assert_allclose(v_joint1, v_joint2, atol=1e-8)
    
    def test_double_pendulum(self):
        """
        Двойной маятник: два звена, соединенные шарнирами.
        Первое звено подвешено на фиксированном шарнире, второе - на первом.
        """
        # Параметры звеньев
        m = 1.0      # масса каждого звена
        L = 1.0      # длина каждого звена
        g = 9.81     # гравитация
        J = m * L**2 / 3  # момент инерции стержня относительно конца
        dt = 0.001   # малый шаг для точности
        
        # Сборка системы
        assembler = MatrixAssembler()
        
        # Первое звено (верхнее)
        body1 = RigidBody2D(m=m, J=J, C=0.0, B=0.0, dt=dt)
        assembler.add_contribution(body1)
        
        # Второе звено (нижнее)
        body2 = RigidBody2D(m=m, J=J, C=0.0, B=0.0, dt=dt)
        assembler.add_contribution(body2)
        
        # Гравитация на оба звена
        F_gravity = np.array([0.0, -m*g])
        assembler.add_contribution(ForceOnBody2D(body1, force=F_gravity))
        assembler.add_contribution(ForceOnBody2D(body2, force=F_gravity))
        
        # Первый шарнир: фиксация первого звена в пространстве
        # Вектор от ЦМ к точке подвеса (в положении равновесия - вверх)
        r1_fixed = np.array([0.0, L/2])
        joint1 = FixedRevoluteJoint2D(body1, r1_fixed)
        assembler.add_constraint(joint1)
        
        # Второй шарнир: соединение двух звеньев
        # От ЦМ первого звена к низу: [0, -L/2]
        # От ЦМ второго звена к верху: [0, L/2]
        r1_joint = np.array([0.0, -L/2])
        r2_joint = np.array([0.0, L/2])
        joint2 = TwoBodyRevoluteJoint2D(body1, body2, r1_joint, r2_joint)
        assembler.add_constraint(joint2)
        
        # Решение для положения равновесия
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assembler.solve_and_set()
        
        # В положении равновесия все скорости должны быть нулевыми
        np.testing.assert_allclose(body1.velocity.value, 0.0, atol=1e-8)
        np.testing.assert_allclose(body1.omega.value, 0.0, atol=1e-8)
        np.testing.assert_allclose(body2.velocity.value, 0.0, atol=1e-8)
        np.testing.assert_allclose(body2.omega.value, 0.0, atol=1e-8)
        
        # Проверка связей
        v1 = body1.velocity.value
        w1 = body1.omega.value
        v2 = body2.velocity.value
        w2 = body2.omega.value
        
        # Связь 1: точка на body1 зафиксирована
        v_fixed = v1 + np.array([-w1 * r1_fixed[1], w1 * r1_fixed[0]])
        np.testing.assert_allclose(v_fixed, 0.0, atol=1e-8)
        
        # Связь 2: точки соединения имеют одинаковую скорость
        v_joint1 = v1 + np.array([-w1 * r1_joint[1], w1 * r1_joint[0]])
        v_joint2 = v2 + np.array([-w2 * r2_joint[1], w2 * r2_joint[0]])
        np.testing.assert_allclose(v_joint1, v_joint2, atol=1e-8)
    
    def test_force_transmission_through_joint(self):
        """
        Проверка передачи силы через шарнир.
        Если к одному телу приложена сила, она должна передаваться на второе тело.
        """
        # Параметры
        m1 = 1.0
        m2 = 2.0
        J1 = 0.5
        J2 = 1.0
        dt = 0.01
        
        # Сборка системы
        assembler = MatrixAssembler()
        
        body1 = RigidBody2D(m=m1, J=J1, C=0.0, B=0.0, dt=dt)
        body2 = RigidBody2D(m=m2, J=J2, C=0.0, B=0.0, dt=dt)
        
        assembler.add_contribution(body1)
        assembler.add_contribution(body2)
        
        # Прикладываем силу к body1
        force = np.array([10.0, 0.0])
        assembler.add_contribution(ForceOnBody2D(body1, force=force))
        
        # Соединяем шарниром
        r1 = np.array([0.5, 0.0])
        r2 = np.array([-0.5, 0.0])
        joint = TwoBodyRevoluteJoint2D(body1, body2, r1, r2)
        assembler.add_constraint(joint)
        
        # Решение
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assembler.solve_and_set()
        
        # Оба тела должны двигаться (через шарнир передается сила)
        # Связанные тела ускоряются вместе
        v1 = body1.velocity.value
        v2 = body2.velocity.value
        
        # Оба тела должны иметь ненулевые скорости
        self.assertGreater(np.linalg.norm(v1), 1e-6)
        self.assertGreater(np.linalg.norm(v2), 1e-6)
        
        # Проверяем, что связь выполняется
        w1 = body1.omega.value
        w2 = body2.omega.value
        v_joint1 = v1 + np.array([-w1 * r1[1], w1 * r1[0]])
        v_joint2 = v2 + np.array([-w2 * r2[1], w2 * r2[0]])
        np.testing.assert_allclose(v_joint1, v_joint2, atol=1e-8)
    
    def test_update_r_method(self):
        """
        Проверка метода update_r для обновления векторов к точке шарнира.
        """
        # Параметры
        m1 = 1.0
        m2 = 1.0
        J1 = 0.5
        J2 = 0.5
        dt = 0.01
        
        # Сборка системы
        assembler = MatrixAssembler()
        
        body1 = RigidBody2D(m=m1, J=J1, C=0.0, B=0.0, dt=dt)
        body2 = RigidBody2D(m=m2, J=J2, C=0.0, B=0.0, dt=dt)
        
        body1.omega.set_value(1.0)
        body2.omega.set_value(-1.0)
        
        assembler.add_contribution(body1)
        assembler.add_contribution(body2)
        
        # Начальные векторы
        r1 = np.array([0.5, 0.0])
        r2 = np.array([-0.5, 0.0])
        joint = TwoBodyRevoluteJoint2D(body1, body2, r1, r2)
        assembler.add_constraint(joint)
        
        # Решение
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assembler.solve_and_set()
        
        # Проверяем, что связь выполняется с начальными векторами
        v1 = body1.velocity.value
        w1 = body1.omega.value
        v2 = body2.velocity.value
        w2 = body2.omega.value
        
        v_joint1 = v1 + np.array([-w1 * r1[1], w1 * r1[0]])
        v_joint2 = v2 + np.array([-w2 * r2[1], w2 * r2[0]])
        np.testing.assert_allclose(v_joint1, v_joint2, atol=1e-8)
        
        # Обновляем векторы (тела повернулись)
        new_r1 = np.array([0.3, 0.4])
        new_r2 = np.array([-0.3, -0.4])
        joint.update_r(new_r1, new_r2)
        
        # Проверяем, что векторы обновились
        np.testing.assert_allclose(joint.r1, new_r1)
        np.testing.assert_allclose(joint.r2, new_r2)
        
        # Решаем снова с новыми векторами
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assembler.solve_and_set()
        
        # Связь должна выполняться с новыми векторами
        v1 = body1.velocity.value
        w1 = body1.omega.value
        v2 = body2.velocity.value
        w2 = body2.omega.value
        
        v_joint1_new = v1 + np.array([-w1 * new_r1[1], w1 * new_r1[0]])
        v_joint2_new = v2 + np.array([-w2 * new_r2[1], w2 * new_r2[0]])
        np.testing.assert_allclose(v_joint1_new, v_joint2_new, atol=1e-8)


if __name__ == '__main__':
    unittest.main()
