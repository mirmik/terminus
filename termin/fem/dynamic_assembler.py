from termin.fem.assembler import MatrixAssembler, Variable, Contribution
from typing import Dict, List, Tuple
import numpy as np
from termin.linalg.subspaces import project_onto_affine

class DynamicMatrixAssembler(MatrixAssembler):
    def _build_index_maps(self) -> Dict[Variable, List[int]]:
        """
        Построить отображение: Variable -> глобальные индексы DOF
        
        Назначает каждой компоненте каждой переменной уникальный
        глобальный индекс в системе.
        """
        acceleration_vars = [var for var in self.variables if var.tag == "acceleration"]
        self._index_map = self._build_index_map(acceleration_vars)

        holonomic_vars = [var for var in self.variables if var.tag == "holonomic_constraint_force"]
        self._holonomic_index_map = self._build_index_map(holonomic_vars)

        #self.set_old_q(self.collect_current_q())
        #self.set_old_q_dot(self.collect_current_q_dot())
        self._dirty_index_map = False

    def collect_current_q(self, index_map: Dict[Variable, List[int]]):
        """Собрать текущее значение q из всех переменных"""
        old_q = np.zeros(self.total_variables_by_tag("acceleration"))
        for var in self.variables:
            if var.tag == "acceleration":
                indices = index_map[var]
                old_q[indices] = var.value_by_rank(2)  # текущее значение
        return old_q

    def collect_current_q_dot(self, index_map: Dict[Variable, List[int]]):
        """Собрать текущее значение q_dot из всех переменных"""
        old_q_dot = np.zeros(self.total_variables_by_tag("acceleration"))
        for var in self.variables:
            if var.tag == "acceleration":
                indices = index_map[var]
                old_q_dot[indices] = var.value_by_rank(1)  # текущее значение скорости
        return old_q_dot

    # def set_old_q(self, q: np.ndarray):
    #     """Установить старое значение q"""
    #     self.old_q = np.array(q)

    # def set_old_q_dot(self, q_dot: np.ndarray):
    #     """Установить старое значение q_dot"""
    #     self.old_q_dot = np.array(q_dot)

    def index_maps(self) -> Dict[str, Dict[Variable, List[int]]]:
        """
        Получить текущее отображение Variable -> глобальные индексы DOF
        для разных типов переменных
        """
        if self._dirty_index_map:
            self._build_index_maps()
        return {
            "acceleration": self._index_map,
            "holonomic_constraint_force": self._holonomic_index_map
        }

    def assemble(self):
        # Построить карту индексов
        index_maps = self.index_maps()

        # Создать глобальные матрицы и вектор
        n_dofs = self.total_variables_by_tag(tag="acceleration")
        n_constraints = self.total_variables_by_tag(tag="holonomic_constraint_force")

        A = np.zeros((n_dofs, n_dofs))
        C = np.zeros((n_dofs, n_dofs))
        K = np.zeros((n_dofs, n_dofs))
        b = np.zeros(n_dofs)
        H = np.zeros((n_constraints, n_dofs))
        h = np.zeros(n_constraints)
        old_q = self.collect_current_q(index_maps["acceleration"])
        old_q_dot = self.collect_current_q_dot(index_maps["acceleration"])

        matrices = {
            "mass": A,
            "damping": C,
            "stiffness": K,
            "load": b,
            "holonomic": H,
            "holonomic_load": h,
            "old_q": old_q,
            "old_q_dot": old_q_dot
        }

        for contribution in self.contributions:
            contribution.contribute(matrices, index_maps)

        return matrices

    def assemble_extended_system(self, matrices: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        A = matrices["mass"]
        C = matrices["damping"]
        K = matrices["stiffness"]
        b = matrices["load"]
        old_q = matrices["old_q"]
        old_q_dot = matrices["old_q_dot"]
        H = matrices["holonomic"]
        h = matrices["holonomic_load"]

        size = self.total_variables_by_tag(tag="acceleration") + self.total_variables_by_tag(tag="holonomic_constraint_force")
        n_dofs = self.total_variables_by_tag(tag="acceleration")
        n_holonomic = self.total_variables_by_tag(tag="holonomic_constraint_force")

        # Расширенная система
        A_ext = np.zeros((size, size))
        b_ext = np.zeros(size)

        r0 = A.shape[0]
        r1 = A.shape[0] + n_holonomic

        c0 = A.shape[1]
        c1 = A.shape[1] + n_holonomic

        A_ext[0:r0, 0:c0] = A
        A_ext[0:r0, c0:c1] = H.T
        A_ext[r0:r1, 0:c0] = H
        b_ext[0:r0] = b - C @ old_q_dot - K @ old_q
        b_ext[r0:r1] = h

        return A_ext, b_ext

    def sort_results(self, x_ext: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Разделить расширенное решение на ускорения и множители Лагранжа"""
        n_dofs = self.total_variables_by_tag(tag="acceleration")
        n_holonomic = self.total_variables_by_tag(tag="holonomic_constraint_force")

        q_ddot = x_ext[:n_dofs]
        holonomic_lambdas = x_ext[n_dofs:n_dofs + n_holonomic]

        nonholonomic_lambdas = []

        return q_ddot, holonomic_lambdas, nonholonomic_lambdas

    def integrate_velocities(self, old_q_dot: np.ndarray, q_ddot: np.ndarray, dt: float) -> np.ndarray:
        """Интегрировать ускорения для получения новых скоростей"""
        return old_q_dot + q_ddot * dt

    def restore_velocity_constraints(self, q_dot: np.ndarray, HN: np.ndarray, hn: np.ndarray) -> np.ndarray:
        """Восстановить ограничения на скорости (например, для закрепленных тел)

            HN - матрица ограничений на скорости, объединение H и N
        """
        return project_onto_affine(q_dot, HN, hn)

    def integrate_positions(self, old_q: np.ndarray, q_dot: np.ndarray, q_ddot: np.ndarray, dt: float) -> np.ndarray:
        """Интегрировать скорости для получения новых положений"""
        return old_q + q_dot * dt + 0.5 * q_ddot * dt**2

    def restore_position_constraints(self, q: np.ndarray, H: np.ndarray, h: np.ndarray) -> np.ndarray:
        """Восстановить ограничения на положения (например, для закрепленных тел)"""
        return project_onto_affine(q, H, h)

    def upload_results(self, q_ddot: np.ndarray, q_dot: np.ndarray, q: np.ndarray):
        """Загрузить результаты обратно в переменные"""
        index_map = self.index_maps()["acceleration"]
        for var in self.variables:
            if var.tag == "acceleration":
                indices = index_map[var]
                var.set_values(q_ddot[indices], q_dot[indices], q[indices])

    def integrate_nonlinear(self, dt: float):
        """Интегрировать нелинейные переменные (если есть)"""
        index_map = self.index_maps()["acceleration"]
        for var in self.variables:
            if var.tag == "acceleration":
                var.integrate_nonlinear(dt)
