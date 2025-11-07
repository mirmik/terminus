"""
Вторая версия модели многотельной системы в 2D
"""

from typing import List, Dict
import numpy as np
from termin.fem.assembler import Variable, Contribution
from termin.geombase.pose2 import Pose2


class RigidBody2D(Contribution):
    """
    Твердое тело в плоскости (3 степени свободы: x, y, θ).
    """
    
    def __init__(self,
                 m: float,                # масса [кг]
                 J: float,                # момент инерции [кг·м²]
                 C: float = 0.0,          # коэффициент линейного демпфирования [Н·с/м]
                 B: float = 0.0,          # коэффициент углового демпфирования [Н·м·с]
                 dt: float = None,        # шаг по времени [с]
                 gravity: np.ndarray = None, # ускорение свободного падения [м/с²]
                 velocity: Variable = None,  # переменная линейной скорости [vx, vy]
                 omega: Variable = None,      # переменная угловой скорости ω
                 assembler=None):             # ассемблер для автоматической регистрации
        """
        Args:
            m: Масса [кг]
            J: Момент инерции [кг·м²]
            C: Коэффициент вязкого сопротивления для поступательного движения [Н·с/м]
            B: Коэффициент вязкого трения для вращательного движения [Н·м·с]
            dt: Шаг по времени [с]
            velocity: Переменная линейной скорости (размер 2)
            omega: Переменная угловой скорости (размер 1)
            assembler: MatrixAssembler для автоматической регистрации переменных
        """
        
        self.velocity = Variable("v_body", size=2, tag="acceleration")
        self.omega = Variable("omega_body", size=1, tag="acceleration")

        super().__init__([self.velocity, self.omega], assembler)
        
        self.m = m
        self.J = J
        #self.C = C
        #self.B = B
        self.dt = dt
        self.gravity = gravity  # ускорение свободного падения [м/с²]

    def pose(self):
        """Получить текущую позу тела (позиция и ориентация)"""
        return Pose2(lin=self.velocity.value_by_rank(2), ang=self.omega.value_by_rank(2)[0])

    def contribute(self, matrices, index_maps: Dict[str, Dict[Variable, List[int]]]):
        """
        Добавить вклад в матрицы
        """
        A = matrices["mass"]  # Матрица масс
        b = matrices["load"]  # Вектор нагрузок
        index_map = index_maps["acceleration"]

        v_indices = index_map[self.velocity]
        omega_idx = index_map[self.omega][0]

        # Вклад от массы и момента инерции
        A[v_indices[0], v_indices[0]] += self.m
        A[v_indices[1], v_indices[1]] += self.m
        A[omega_idx, omega_idx] += self.J

        # Вклад от гравитации
        b[v_indices[0]] += self.m * self.gravity[0]
        b[v_indices[1]] += self.m * self.gravity[1]


class ForceOnBody2D(Contribution):
    """
    Внешняя сила и момент, приложенные к твердому телу в 2D.
    """
    def __init__(self,
        body: RigidBody2D,
        force: np.ndarray = np.zeros(2),  
        torque: float = 0.0,
        assembler=None):            
        """
        Args:
            force: Внешняя сила [Fx, Fy] в Н
            torque: Внешний момент τ в Н·м
        """
        self.body = body
        self.velocity = body.velocity
        self.omega = body.omega
        self.force = force
        self.torque = torque
        super().__init__([], assembler)  # Нет переменных для этой нагрузки

    def contribute(self, matrices, index_maps: Dict[str, Dict[Variable, List[int]]]):
        """
        Добавить вклад в вектор нагрузок
        """
        b = matrices["load"]  # Вектор нагрузок

        index_map = index_maps["acceleration"]
        v_indices = index_map[self.velocity]
        omega_idx = index_map[self.omega][0]

        # Вклад от внешней силы и момента
        b[v_indices[0]] += self.force[0]
        b[v_indices[1]] += self.force[1]
        b[omega_idx] += self.torque


class FixedRotationJoint2D(Contribution):
    """
    Вращательный шарнир с фиксацией в пространстве (ground revolute joint).
    
    Фиксирует точку на теле в пространстве, разрешая только вращение вокруг этой точки.
    Эквивалентно присоединению тела к неподвижному основанию через шарнир.
    
    Кинематическая связь:
    - Скорость точки крепления должна быть нулевой: v_cm + ω × r = 0
    - В 2D: v_cm + [-ω*ry, ω*rx] = 0
    
    где:
    - v_cm: скорость центра масс тела [vx, vy]
    - ω: угловая скорость тела
    - r: вектор от центра масс к точке крепления [rx, ry]
    
    Реализуется через множители Лагранжа (точное удовлетворение связи).
    """
    def __init__(self,
        body: RigidBody2D,
        coords_of_joint: np.ndarray = np.zeros(2),
        assembler=None):
        """
        Args:
            body: Твердое тело, к которому применяется шарнир
            coords_of_joint: Вектор координат шарнира [x, y]
            coords_of_body_connection: Вектор координат точки соединения с телом [x, y]
        """
        self.body = body
        self.internal_force = Variable("F_joint", size=2, tag="holonomic_constraint_force")
        
        body_pose = self.body.pose()
        self.joint_in_body = body_pose.inverse_transform_point(coords_of_joint)
        self.coords_of_joint = coords_of_joint
        self.length_of_radius = np.linalg.norm(self.coords_of_joint - body_pose.lin)
        self.update_radius_to_body()
        super().__init__([body.velocity, self.internal_force], assembler)

    def update_radius_to_body(self):
        """Обновить радиус до тела"""
        radius_to_body = self.body.pose().lin - self.coords_of_joint
        radius_in_body_ort = radius_to_body / np.linalg.norm(radius_to_body)
        self.radius_to_body = radius_in_body_ort * self.length_of_radius 

    def contribute(self, matrices, index_maps: Dict[str, Dict[Variable, List[int]]]):
        """
        Добавить вклад в матрицы
        """
        H = matrices["holonomic"]  # Матрица ограничений
        h = matrices["holonomic_load"]    # Вектор ограничений
        b = matrices["load"]  # Вектор нагрузок
        C = matrices["damping"]  # Матрица демпфирования
        old_q_dot = matrices["old_q_dot"]
        old_q = matrices["old_q"]

        index_map = index_maps["acceleration"]
        constraint_map = index_maps["holonomic_constraint_force"]
        v_indices = index_map[self.body.velocity]
        F_indices = constraint_map[self.internal_force]

        dt = 0.01 # TODO: пробросить через параметры

        # Матрица коэффициентов связи
        # Связь: vx + ω*ry = 0
        #        vy - ω*rx = 0
        #
        # В матричной форме: C * [vx, vy, ω]^T = 0
        # H = [[1,  0,  ry],
        #      [0,  1,  -rx]]

        # Вклад в матрицу ограничений от связи шарнира
        H[F_indices[0], v_indices[0]] += 1.0
        H[F_indices[0], v_indices[1]] += 0.0
        H[F_indices[1], v_indices[0]] += 0.0
        H[F_indices[1], v_indices[1]] += 1.0

        # Вращательное влияние
        H[F_indices[0], index_map[self.body.omega][0]] += self.radius_to_body[1]
        H[F_indices[1], index_map[self.body.omega][0]] += -self.radius_to_body[0]

        # Baumgarte стабилизация для ограничения скорости
        ksi = 1.0
        omega_baumgarte = 0.01 / dt

        v_x = old_q_dot[v_indices[0]]
        v_y = old_q_dot[v_indices[1]]
        omega = old_q_dot[index_map[self.body.omega][0]]

        x = old_q[v_indices[0]]
        y = old_q[v_indices[1]]



        # e_vx = v_x + omega * ry
        # e_vy = v_y - omega * rx
        velocity_err = np.array([v_x + omega * self.radius_to_body[1],
                                 v_y - omega * self.radius_to_body[0]])

        coordinate_err = self.body.pose().lin - self.radius_to_body - self.coords_of_joint


        h_1 = np.array([
            omega*omega * self.radius_to_body[0],
            omega*omega * self.radius_to_body[1]
        ])

        h_2 = 2 * ksi * omega_baumgarte * velocity_err

        h_3 = omega_baumgarte * omega_baumgarte * coordinate_err

        print("h_1 =", h_1)
        print("h_2 =", h_2)
        print("h_3 =", h_3)
        print("coordinate_err =", coordinate_err)

        h[F_indices[0]] += -h_1[0] 
        h[F_indices[1]] += -h_1[1] 

        h[F_indices[0]] += -h_2[0] 
        h[F_indices[1]] += -h_2[1] 

        h[F_indices[0]] += -h_3[0] 
        h[F_indices[1]] += -h_3[1] 
