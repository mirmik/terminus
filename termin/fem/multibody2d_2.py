"""
Вторая версия модели многотельной системы в 2D
"""

from typing import List, Dict
import numpy as np
from termin.fem.assembler import Variable, Contribution
from termin.geombase.pose2 import Pose2
from termin.geombase.screw import Screw2
from termin.fem.inertia2d import SpatialInertia2D


class RigidBody2D(Contribution):
    """
    Твердое тело в плоскости (3 СС: x, y, θ).
    Неизвестные: [vx, vy, ω] — ускорения (VW-порядок).
    Поддерживает внецентренную инерцию (смещённый ЦМ).
    """


    def __init__(self,
                 inertia: SpatialInertia2D,
                 gravity: np.ndarray = None,  # g в ГЛОБАЛЬНОЙ СК, shape=(2,)
                 assembler=None,
                 name="rbody2d"):
        self.acceleration_var = Variable(name+"_acc", size=3, tag="acceleration")  # [vx, vy, ω] в глобальной СК
        self.velocity_var = Variable(name+"_vel", size=3, tag="velocity")  # [vx, vy, ω] в глобальной СК
        self.pose_var = Variable(name+"_pos", size=3, tag="position")  # [x, y, θ] в глобальной СК
        self.inertia = inertia
        self.gravity = np.array([0.0, -9.81]) if gravity is None else np.asarray(gravity, float).reshape(2)
        super().__init__([self.acceleration_var, 
        #    self.velocity, self._pose
        ], assembler=assembler)

    def pose(self):
        return Pose2(lin=self.acceleration_var.value[0:2], ang=float(self.acceleration_var.value[2]))
        #return Pose2(lin=self.pose.value[0:2], ang=float(self.pose.value[2]))
        
    # ---------- ВКЛАД В СИСТЕМУ ----------
    def contribute(self, matrices, index_maps):
        self.contribute_to_mass_matrix(matrices, index_maps)

        # ---------- НАГРУЗКА (гравитация в ГЛОБАЛЬНОЙ СК) ----------
        a_idx = index_maps["acceleration"][self.acceleration_var]
        b = matrices["load"]
        b[a_idx[0]:a_idx[2]+1] += self.inertia.gravity_wrench(self.gravity).to_vector_vw_order()

    def contribute_for_constraints_correction(self, matrices, index_maps):
        self.contribute_to_mass_matrix(matrices, index_maps)
    
    def contribute_to_mass_matrix(self, matrices, index_maps):
        A = matrices["mass"]
        amap = index_maps["acceleration"]
        a_idx = amap[self.acceleration_var]
        IM = self.inertia.to_matrix_vw_order()
        A[a_idx[0]:a_idx[2]+1, a_idx[0]:a_idx[2]+1] += IM



class ForceOnBody2D(Contribution):
    """
    Внешняя сила и момент, приложенные к твердому телу в 2D.
    """
    def __init__(self,
        body: RigidBody2D,
        wrench: Screw2,
        in_local_frame: bool = False,
        assembler=None):            
        """
        Args:
            force_local: Внешняя сила [Fx, Fy] в ЛОКАЛЬНОЙ СК тела
            torque_local: Внешний момент τ в ЛОКАЛЬНОЙ СК
        """
        self.body = body
        self.acceleration = body.acceleration_var
        self.wrench = wrench
        self.in_local_frame = in_local_frame
        super().__init__([], assembler=assembler)  # Нет переменных для этой нагрузки

    def contribute(self, matrices, index_maps: Dict[str, Dict[Variable, List[int]]] ):
        """
        Добавить вклад в вектор нагрузок (в локальной СК тела)
        """
        b = matrices["load"]  # Вектор нагрузок

        index_map = index_maps["acceleration"]
        v_indices = index_map[self.acceleration]
        
        wrench = self.wrench if self.in_local_frame else self.wrench.rotated_by(self.body.pose())

        b[v_indices[0]] += wrench.lin[0]
        b[v_indices[1]] += wrench.lin[1]
        b[v_indices[2]] += wrench.ang


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
        self.internal_force = Variable("F_joint", size=2, tag="force")
        
        body_pose = self.body.pose()
        
        self.coords_of_joint = coords_of_joint
        self.radius_in_local = body_pose.inverse_transform_point(self.coords_of_joint)

        self.update_radius_to_body()
        super().__init__([body.acceleration_var, self.internal_force], assembler=assembler)

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

        index_map = index_maps["acceleration"]
        constraint_map = index_maps["force"]
        v_indices = index_map[self.body.acceleration_var]
        F_indices = constraint_map[self.internal_force]

        # Вклад в матрицу ограничений от связи шарнира
        H[F_indices[0], v_indices[0]] += 1.0
        H[F_indices[0], v_indices[1]] += 0.0
        H[F_indices[1], v_indices[0]] += 0.0
        H[F_indices[1], v_indices[1]] += 1.0

        # Вращательное влияние
        H[F_indices[0], index_map[self.body.acceleration_var][2]] += -self.radius[1]
        H[F_indices[1], index_map[self.body.acceleration_var][2]] += self.radius[0]

    def contribute_for_constraints_correction(self, matrices, index_maps: Dict[str, Dict[Variable, List[int]]]):
        """
        Добавить вклад в матрицы для коррекции ограничений на положения
        """
        self.update_radius_to_body()
        self.contribute(matrices, index_maps)

        constraint_map = index_maps["force"]
        poserr = matrices["position_error"]
        F_indices = constraint_map[self.internal_force]

        x = self.body.pose().lin[0]
        y = self.body.pose().lin[1]

        poserr[F_indices[0]] += x + self.radius[0] - self.coords_of_joint[0]
        poserr[F_indices[1]] += y + self.radius[1] - self.coords_of_joint[1]

class RevoluteJoint2D(Contribution):
    """
    Двухтелый вращательный шарнир (revolute joint).
    Связывает две точки на двух телах: точка A должна совпадать с точкой B.
    """

    def __init__(self,
        bodyA: RigidBody2D,
        bodyB: RigidBody2D,
        coords_of_joint: np.ndarray,
        assembler=None):

        self.bodyA = bodyA
        self.bodyB = bodyB

        # переменная внутренней силы (двухкомпонентная)
        self.internal_force = Variable("F_rev", size=2, tag="force")

        # вычисляем локальные точки для обоих тел
        poseA = self.bodyA.pose()
        poseB = self.bodyB.pose()

        self.rA_local = poseA.inverse_transform_point(coords_of_joint)
        self.rB_local = poseB.inverse_transform_point(coords_of_joint)

        # актуализируем глобальные вектор-радиусы
        self.update_radii()

        super().__init__([bodyA.acceleration_var, bodyB.acceleration_var, self.internal_force], assembler=assembler)

    def update_radii(self):
        """Пересчитать глобальные радиусы до опорных точек"""
        poseA = self.bodyA.pose()
        poseB = self.bodyB.pose()

        self.rA = poseA.transform_point(self.rA_local) - poseA.lin
        self.rB = poseB.transform_point(self.rB_local) - poseB.lin


    def contribute(self, matrices, index_maps: Dict[str, Dict[Variable, List[int]]]):
        """Добавляет вклад в матрицы для ускорений"""

        # радиусы актуализируем каждый вызов
        self.update_radii()

        H = matrices["holonomic"]

        amap = index_maps["acceleration"]
        cmap = index_maps["force"]

        # индексы вектора скоростей
        vA = amap[self.bodyA.acceleration_var]
        vB = amap[self.bodyB.acceleration_var]

        F = cmap[self.internal_force]  # 2 строки ограничений

        # ---------- Якобиан по A ----------
        # dφ/dx_A = +1
        H[F[0], vA[0]] += 1.0
        H[F[1], vA[1]] += 1.0

        # dφ/dθ_A = [-rAy, +rAx]
        H[F[0], vA[2]] += -self.rA[1]
        H[F[1], vA[2]] +=  self.rA[0]

        # ---------- Якобиан по B ----------
        # dφ/dx_B = -1
        H[F[0], vB[0]] += -1.0
        H[F[1], vB[1]] += -1.0

        # dφ/dθ_B = [+rBy, -rBx]
        H[F[0], vB[2]] +=  self.rB[1]
        H[F[1], vB[2]] += -self.rB[0]


    def contribute_for_constraints_correction(self, matrices, index_maps):
        """Для позиционной и скоростной проекции"""
        self.update_radii()
        self.contribute(matrices, index_maps)
        poserr = matrices["position_error"]
        cmap = index_maps["force"]
        F = cmap[self.internal_force]  # 2 строки ограничений

        # ---------- позиционная ошибка ----------
        # φ = cA - cB = (pA + rA) - (pB + rB)
        pA = self.bodyA.pose().lin
        pB = self.bodyB.pose().lin

        poserr[F[0]] += (pA[0] + self.rA[0]) - (pB[0] + self.rB[0])
        poserr[F[1]] += (pA[1] + self.rA[1]) - (pB[1] + self.rB[1])

