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
        self.contribute_mass(A, v_indices, omega_idx)

        # Вклад от гравитации
        b[v_indices[0]] += self.m * self.gravity[0]
        b[v_indices[1]] += self.m * self.gravity[1]

    def contribute_mass(self, A, v_indices, omega_idx):
        """Добавить вклад массы в матрицу масс"""
        A[v_indices[0], v_indices[0]] += self.m
        A[v_indices[1], v_indices[1]] += self.m
        A[omega_idx, omega_idx] += self.J

    def contribute_for_constraints_correction(self, matrices, index_maps: Dict[str, Dict[Variable, List[int]]]):
        """
        Добавить вклад в матрицы для коррекции ограничений на положения
        """
        A = matrices["mass"]  # Матрица масс
        index_map = index_maps["acceleration"]
        v_indices = index_map[self.velocity]
        omega_idx = index_map[self.omega][0]

        # Вклад от массы и момента инерции
        self.contribute_mass(A, v_indices, omega_idx)


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

    def contribute_for_constraints_correction(self, matrices, index_maps: Dict[str, Dict[Variable, List[int]]]):
        """
        Добавить вклад в матрицы для коррекции ограничений на положения
        """
        # Внешние силы не влияют на коррекцию ограничений на положения
        pass


class FixedRotationJoint2D(Contribution):
    """
    Вращательный шарнир с фиксацией в пространстве (ground revolute joint).
    
    Фиксирует точку на теле в пространстве, разрешая только вращение вокруг этой точки.
    Эквивалентно присоединению тела к неподвижному основанию через шарнир.
    
    Кинематическая связь:
    
    r_0 = j - p_0
    e = p + r_0 - j 

    de/dt = H * q_dot

    H = |de/dx| = [1, 0, -ry]
        |de/dy| = [0, 1,  rx]

    H * q_dot = | vx - ω*ry |
                | vy + ω*rx |

    dde/dt2 = H_dot * q_dot + H * q_ddot 

    dH/dq = | 0 0 0 | | 0 0 0 | | 0 0 -r_x |
            | 0 0 0 | | 0 0 0 | | 0 0 -r_y |

    dH/dq * q_dot = | 0 0 -ω*rx |
                    | 0 0 -ω*ry |
    
    H_dot * q_dot = dH/dq * q_dot * q_dot = | -ω*ω*rx |
                                            | -ω*ω*ry |

    Baumgarte стабилизация (не используется):
    dde/dt2 + 2*ξ*ω_n*de/dt + ω_n²*e = 0

    H_dot * q_dot + H * q_ddot + 2*ξ*ω_n*H*q_dot + ω_n²*e = 0

    H * q_ddot = -H_dot * q_dot - 2*ξ*ω_n*H*q_dot - ω_n²*e

    H * q_ddot = -| -ω*ω*rx | - 2*ξ*ω_n*| vx - ω*ry | - ω_n²*| x + rx - j_x |
                  | -ω*ω*ry |           | vy + ω*rx |        | y + ry - j_y |

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
        
        self.coords_of_joint = coords_of_joint
        self.radius_in_local = body_pose.inverse_transform_point(self.coords_of_joint)

        self.update_radius_to_body()
        super().__init__([body.velocity, self.internal_force], assembler)

    def update_radius_to_body(self):
        """Обновить радиус до тела"""
        body_pose = self.body.pose()
        self.radius = body_pose.transform_point(self.radius_in_local) - body_pose.lin

    def contribute(self, matrices, index_maps: Dict[str, Dict[Variable, List[int]]]):
        """
        Добавить вклад в матрицы
        """
        self.update_radius_to_body()

        H = matrices["holonomic"]  # Матрица ограничений
        h = matrices["holonomic_load"]    # Вектор ограничений
        b = matrices["load"]  # Вектор нагрузок
        old_q_dot = matrices["old_q_dot"]
        old_q = matrices["old_q"]
        poserr = matrices["position_error"]

        index_map = index_maps["acceleration"]
        constraint_map = index_maps["holonomic_constraint_force"]
        v_indices = index_map[self.body.velocity]
        F_indices = constraint_map[self.internal_force]

        # Вклад в матрицу ограничений от связи шарнира
        H[F_indices[0], v_indices[0]] += 1.0
        H[F_indices[0], v_indices[1]] += 0.0
        H[F_indices[1], v_indices[0]] += 0.0
        H[F_indices[1], v_indices[1]] += 1.0

        # Вращательное влияние
        H[F_indices[0], index_map[self.body.omega][0]] += -self.radius[1]
        H[F_indices[1], index_map[self.body.omega][0]] += self.radius[0]

        x = old_q[v_indices[0]]
        y = old_q[v_indices[1]]

        poserr[F_indices[0]] += x + self.radius[0] - self.coords_of_joint[0]
        poserr[F_indices[1]] += y + self.radius[1] - self.coords_of_joint[1]

    def contribute_for_constraints_correction(self, matrices, index_maps: Dict[str, Dict[Variable, List[int]]]):
        """
        Добавить вклад в матрицы для коррекции ограничений на положения
        """
        self.update_radius_to_body()

        H = matrices["holonomic"]  # Матрица ограничений
        poserr = matrices["position_error"]
        
        index_map = index_maps["acceleration"]
        constraint_map = index_maps["holonomic_constraint_force"]
        v_indices = index_map[self.body.velocity]
        F_indices = constraint_map[self.internal_force]
        omega_idx = index_map[self.body.omega][0]

        # Вклад в матрицу ограничений от связи шарнира
        H[F_indices[0], v_indices[0]] += 1.0
        H[F_indices[0], v_indices[1]] += 0.0
        H[F_indices[1], v_indices[0]] += 0.0
        H[F_indices[1], v_indices[1]] += 1.0
        # Вращательное влияние
        H[F_indices[0], omega_idx] += -self.radius[1]
        H[F_indices[1], omega_idx] += self.radius[0]

        # Ошибка положения
        x = self.body.pose().lin[0]
        y = self.body.pose().lin[1]
        poserr[F_indices[0]] += x + self.radius[0] - self.coords_of_joint[0]
        poserr[F_indices[1]] += y + self.radius[1] - self.coords_of_joint[1]
