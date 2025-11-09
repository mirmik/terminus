
from typing import List, Dict
import numpy as np
from termin.fem.assembler import Variable, PoseVariable, Contribution
from termin.geombase.pose3 import Pose3
from termin.geombase.screw import Screw3
from termin.fem.inertia3d import SpatialInertia3D

def skew(v: np.ndarray) -> np.ndarray:
    """Возвращает кососимметричную матрицу для вектора v."""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

# class RigidBody3D(Contribution):
#     """
#     Твердое тело в 3D (6 СС: x,y,z и ориентация q).
#     Скорости: v=[vx,vy,vz], ω=[ωx,ωy,ωz].
#     Поза хранится как (lin, quat).
#     """

#     def __init__(self,
#                  inertia: SpatialInertia3D,
#                  gravity: np.ndarray = np.array([0.0, 0.0, -9.81]),
#                  assembler=None, name: str = "rbody3d"):

#         self.velocity = PoseVariable(name+"_acc", tag="acceleration")
#         super().__init__([self.velocity], assembler=assembler)

#         self.gravity = np.asarray(gravity, float)

#         # Базовая (локальная) пространственная инерция тела
#         self.spatial_local = inertia

#     # ---------- Геометрия/поза ----------

#     def pose(self) -> Pose3:
#         """
#         Возвращает текущую позу.
#         Позиция берётся из rank=2 у self.velocity (аналог твоего 2D),
#         ориентация — из self.quat (ранг «позиции»).
#         """
#         return self.velocity.pose()

#     def contribute(self, matrices, index_maps: Dict[str, Dict[Variable, List[int]]]):
#         """
#         Вклады в mass (A) и load (b).
#         Массовая матрица — полный 6x6 блок с учётом COM и поворота (через SpatialInertia3D).
#         Порядок неизвестных: [v, ω] — как в твоём 2D.
#         """
        

#         pose = self.pose()
#         spatial_world = self.spatial_local.rotated_spatial_inertia(pose)
#         print(spatial_world.I_com)
#         print(spatial_world.c)

#         self.contribute_mass_matrix(spatial_world, matrices, index_maps)

#         b = matrices["load"]
#         amap = index_maps["acceleration"]
#         v_idx = amap[self.velocity]
        
#         gravity_wrench = spatial_world.gravity_wrench(pose, self.gravity)
#         Fg = gravity_wrench.lin
#         tau_g = gravity_wrench.ang

#         b[v_idx[0]] += Fg[0]
#         b[v_idx[1]] += Fg[1]
#         b[v_idx[2]] += Fg[2]

#         b[v_idx[3]] += tau_g[0]
#         b[v_idx[4]] += tau_g[1]
#         b[v_idx[5]] += tau_g[2]

#         # TODO: добавить гироскопические силы 

#     def contribute_mass_matrix(self, spatial_world, matrices, index_maps: Dict[str, Dict[Variable, List[int]]]):
#         """
#         Для позиционной коррекции нужна только «масса» (как метрика).
#         Кладём тот же 6x6 блок, что и в contribute(), но без сил.
#         """
#         A = matrices["mass"]
#         amap = index_maps["acceleration"]

#         v_idx = amap[self.velocity]

#         S_vw = spatial_world.to_matrix_vw_order()
#         print(S_vw)
#         A[np.ix_(v_idx, v_idx)] += S_vw

#     def contribute_for_constraints_correction(self, matrices, index_maps: Dict[str, Dict[Variable, List[int]]]):
#         """
#         Вклад в массовую матрицу для коррекции ограничений.
#         """
#         pose = self.pose()
#         spatial_world = self.spatial_local.transform_by(pose)
#         self.contribute_mass_matrix(spatial_world, matrices, index_maps)

class RigidBody3D(Contribution):

    def __init__(self, inertia: SpatialInertia3D,
                 gravity=np.array([0,0,-9.81]),
                 assembler=None, name="rbody3d"):

        print("Creating RigidBody3D:", name)

        # spatial acceleration variable: 6 components
        self.acc = PoseVariable(name+"_acc", tag="acceleration")

        super().__init__([self.acc], assembler=assembler)

        self.gravity = gravity
        self.spatial_local = inertia   # spatial inertia in body frame

    def pose(self):
        return self.acc.pose()

    def contribute(self, matrices, index_maps):
        print("CONTRIBUTE")

        pose = self.pose()

        Iw=self.contribute_mass_matrix(matrices, index_maps)
        idx = index_maps["acceleration"][self.acc]

        # 3) Spatial gravity
        #gravity_local = pose.inverse_transform_vector(self.gravity)
        #Fg_screw = self.spatial_local.gravity_wrench(gravity_local)   # 6×1 vector
        Fg_screw = Iw.gravity_wrench(self.gravity)   # 6×1 vector

        Fg = Fg_screw.to_vw_array()  # in world frame
        print("Gravity wrench:", Fg)

        b = matrices["load"]
        for i in range(6):
            b[idx[i]] += Fg[i]

    def contribute_mass_matrix(self, matrices, index_maps):
        pose = self.pose()
        I_origin = self.spatial_local.at_body_origin()
        Iw = I_origin.rotate_by(pose)

        A = matrices["mass"]
        idx = index_maps["acceleration"][self.acc]
        A[np.ix_(idx, idx)] += Iw.to_matrix_vw_order()
        return Iw

    def contribute_for_constraints_correction(self, matrices, index_maps):
        self.contribute_mass_matrix(matrices, index_maps)
        

class ForceOnBody3D(Contribution):
    """
    Внешняя сила и момент, приложенные к твердому телу в 3D.
    """
    def __init__(self,
                 body: RigidBody3D,
                 force: np.ndarray = np.zeros(3),     # Fx, Fy, Fz
                 torque: np.ndarray = np.zeros(3),    # τx, τy, τz
                 assembler=None):
        """
        Args:
            force: Внешняя сила (3,)
            torque: Внешний момент (3,)
        """
        self.body = body
        self.velocity = body.velocity  # PoseVariable
        self.force = np.asarray(force, float)
        self.torque = np.asarray(torque, float)

        super().__init__([], assembler=assembler)  # переменных нет


    def contribute(self, matrices, index_maps):
        """
        Добавить вклад в вектор нагрузок b.
        """
        b = matrices["load"]
        amap = index_maps["acceleration"]

        # v_idx: три индекса линейной части
        # w_idx: три индекса угловой части
        v_idx = amap[self.velocity][0:3]
        w_idx = amap[self.velocity][3:6]

        # Линейная сила
        b[v_idx[0]] += self.force[0]
        b[v_idx[1]] += self.force[1]
        b[v_idx[2]] += self.force[2]

        # Момент
        b[w_idx[0]] += self.torque[0]
        b[w_idx[1]] += self.torque[1]
        b[w_idx[2]] += self.torque[2]


    def contribute_for_constraints_correction(self, matrices, index_maps):
        """
        Внешние силы не участвуют в позиционной коррекции.
        """
        pass

class FixedRotationJoint3D(Contribution):
    """
    3D фиксированная точка (ground spherical joint).
    
    Условие:
        p + R * r_local = joint_world
    
    Скоростная связь:
        v + ω × r = 0
    """

    def __init__(self, 
                 body,                      # RigidBody3D
                 joint_point: np.ndarray,   # мировая точка (3,)
                 assembler=None):

        self.body = body
        self.joint_point = np.asarray(joint_point, float)

        # внутренняя сила — 3 компоненты
        self.internal_force = Variable(
            "F_fixed3d",
            size=3,
            tag="force"
        )

        # вычисляем локальную точку (обратное преобразование)
        pose = self.body.pose()
        self.r_local = pose.inverse_transform_point(self.joint_point)

        # актуализируем r в мировых
        self.update_radius()

        super().__init__([self.body.acc, self.internal_force], assembler=assembler)

    # -----------------------------------------------------------

    def update_radius(self):
        """Обновить мировой радиус r = R * r_local."""
        pose = self.body.pose()
        self.r = pose.transform_vector(self.r_local)
        self.radius = self.r

    # -----------------------------------------------------------

    def contribute(self, matrices, index_maps):
        """
        Вклад в матрицу holonomic.
        Ограничение: e = p + r - j = 0.
        Якобиан: [ I3 | -skew(r) ].
        """
        self.update_radius()

        H = matrices["holonomic"]

        amap = index_maps["acceleration"]
        cmap = index_maps["force"]

        # индексы скоростей тела
        v_idx = amap[self.body.acc]      # 6 индексов
        # индексы внутренних сил
        f_idx = cmap[self.internal_force]     # 3 индексов

        # --- Заполняем якобиан ---
        # H[f, v]
        # линейные скорости: +I
        H[f_idx[0], v_idx[0]] += 1
        H[f_idx[1], v_idx[1]] += 1
        H[f_idx[2], v_idx[2]] += 1

        # угловые скорости: -[r]_×
        S = skew(self.r)
        # порядок v_idx[3:6] — (wx, wy, wz)
        H[np.ix_(f_idx, v_idx[3:6])] -= S


    # -----------------------------------------------------------

    def contribute_for_constraints_correction(self, matrices, index_maps):
        """
        Для коррекции ограничений делаем то же самое.
        """
        self.update_radius()
        self.contribute(matrices, index_maps)
        
        poserr = matrices["position_error"]
        f_idx = index_maps["force"][self.internal_force]


        # --- Позиционная ошибка ---
        pose = self.body.pose()
        p = pose.lin
        e = p + self.r - self.joint_point

        poserr[f_idx[0]] += e[0]
        poserr[f_idx[1]] += e[1]
        poserr[f_idx[2]] += e[2]


class RevoluteJoint3D(Contribution):
    """
    3D револьвентный шарнир в смысле 2D-версии:
    совпадение двух точек на двух телах.
    
    Ограничения: (pA + rA) - (pB + rB) = 0   (3 eq)
    Не ограничивает ориентацию!
    Даёт 3 степени свободы на вращение.
    """

    def __init__(self,
                 bodyA,
                 bodyB,
                 joint_point_world: np.ndarray,
                 assembler=None):

        self.bodyA = bodyA
        self.bodyB = bodyB

        # Внутренняя реакция — вектор из 3 компонент
        self.internal_force = Variable("F_rev3d", size=3,
                                       tag="force")

        # локальные точки крепления
        poseA = self.bodyA.pose()
        poseB = self.bodyB.pose()

        self.rA_local = poseA.inverse_transform_point(joint_point_world)
        self.rB_local = poseB.inverse_transform_point(joint_point_world)

        # обновляем мировую геометрию
        self.update_kinematics()

        super().__init__([bodyA.velocity, bodyB.velocity, self.internal_force],
                         assembler=assembler)

    # --------------------------------------------------------------

    def update_kinematics(self):
        poseA = self.bodyA.pose()
        poseB = self.bodyB.pose()

        self.pA = poseA.lin
        self.pB = poseB.lin

        self.rA = poseA.transform_vector(self.rA_local)
        self.rB = poseB.transform_vector(self.rB_local)

    # --------------------------------------------------------------

    def contribute(self, matrices, index_maps):
        self.update_kinematics()

        H = matrices["holonomic"]

        amap = index_maps["acceleration"]
        cmap = index_maps["force"]

        vA = amap[self.bodyA.velocity]
        vB = amap[self.bodyB.velocity]

        F = cmap[self.internal_force]  # 3 строки

        # Матрицы скосов радиусов
        SA = skew(self.rA)
        SB = skew(self.rB)

        # dφ/dvA_lin = +I
        H[np.ix_(F, vA[0:3])] += np.eye(3)

        # dφ/dvA_ang = -skew(rA)
        H[np.ix_(F, vA[3:6])] += -SA

        # dφ/dvB_lin = -I
        H[np.ix_(F, vB[0:3])] += -np.eye(3)

        # dφ/dvB_ang = +skew(rB)
        H[np.ix_(F, vB[3:6])] += SB


    # --------------------------------------------------------------

    def contribute_for_constraints_correction(self, matrices, index_maps):
        self.update_kinematics()
        self.contribute(matrices, index_maps)
        cmap = index_maps["force"]
        F = cmap[self.internal_force]  # 3 строки
        poserr = matrices["position_error"]
        # позиционная ошибка: φ = (pA+rA) - (pB+rB)
        err = (self.pA + self.rA) - (self.pB + self.rB)

        poserr[F[0]] += err[0]
        poserr[F[1]] += err[1]
        poserr[F[2]] += err[2]