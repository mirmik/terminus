import numpy as np
from typing import Optional

from .hqsolver import QuadraticTask, EqualityConstraint, InequalityConstraint


def _prepare_weight(weight: Optional[np.ndarray], rows: int) -> Optional[np.ndarray]:
    if weight is None:
        return None
    W = np.asarray(weight, dtype=float)
    if W.ndim == 1:
        if W.size != rows:
            raise ValueError("Weight vector size must match the task dimension.")
        return np.diag(W)
    if W.shape != (rows, rows):
        raise ValueError("Weight matrix must be square with size equal to the task dimension.")
    return W


class JointTrackingTask(QuadraticTask):
    """Следит за желаемыми суставными координатами.

    Минимизируется функционал
        min_x || S x - S q_ref ||_W^2,
    где x — искомые обобщённые скорости/приращения, q_ref — желаемый вектор,
    а S — матрица выбора (по умолчанию S = I). При W = diag(w) это простая
    взвешенная подстройка отдельных координат.
    """

    def __init__(
        self,
        q_ref: np.ndarray,
        selection: Optional[np.ndarray] = None,
        weight: Optional[np.ndarray] = None,
    ):
        q_ref = np.asarray(q_ref, dtype=float)
        n = q_ref.size
        if selection is None:
            J = np.eye(n)
            target = q_ref
        else:
            selection = np.asarray(selection, dtype=float)
            if selection.shape[1] != n:
                raise ValueError("Selection matrix must have the same number of columns as len(q_ref).")
            J = selection
            target = selection @ q_ref

        W = _prepare_weight(weight, J.shape[0])
        super().__init__(J, target, W)


class CartesianTrackingTask(QuadraticTask):
    """Типовая задача: совместить линейную/угловую скорость с целью.

    Функционал:
        min_x || J_cart x - v_des ||_W^2,
    где J_cart — пространственный Якобиан, v_des = v_ref + K e — желаемая
    скорость/скользящий вектор. Вызов принимает уже сформированные (J_cart, v_des).
    """

    def __init__(
        self,
        jacobian: np.ndarray,
        desired_twist: np.ndarray,
        weight: Optional[np.ndarray] = None,
    ):
        jacobian = np.asarray(jacobian, dtype=float)
        desired_twist = np.asarray(desired_twist, dtype=float)
        if jacobian.shape[0] != desired_twist.size:
            raise ValueError("Jacobian rows must match the size of desired_twist.")
        W = _prepare_weight(weight, jacobian.shape[0])
        super().__init__(jacobian, desired_twist, W)


class JointEqualityConstraint(EqualityConstraint):
    """Фиксирует отдельные суставы: S x = S q_target.

    Это линейное равенство с матрицей S (по умолчанию I).
    """

    def __init__(self, q_target: np.ndarray, selection: Optional[np.ndarray] = None):
        q_target = np.asarray(q_target, dtype=float)
        n = q_target.size
        if selection is None:
            A = np.eye(n)
            b = q_target
        else:
            selection = np.asarray(selection, dtype=float)
            if selection.shape[1] != n:
                raise ValueError("Selection matrix must have the same number of columns as len(q_target).")
            A = selection
            b = selection @ q_target

        super().__init__(A, b)


class CartesianEqualityConstraint(EqualityConstraint):
    """Жёсткая задача J x = rhs для контактов или привязки инструмента."""

    def __init__(self, jacobian: np.ndarray, rhs: np.ndarray):
        super().__init__(jacobian, rhs)


class JointBoundsConstraint(InequalityConstraint):
    """Задаёт ограничения вида lower ≤ x ≤ upper (возможно односторонние).

    Преобразуется к стандартной форме C x ≤ d:
        x ≤ upper  →  [ I] x ≤ upper
       -x ≤ -lower → [-I] x ≤ -lower.
    """

    def __init__(self, lower: Optional[np.ndarray] = None, upper: Optional[np.ndarray] = None):
        if lower is None and upper is None:
            raise ValueError("At least one of lower/upper bounds must be provided.")

        rows = []
        rhs_parts = []

        n: Optional[int] = None

        if upper is not None:
            upper = np.asarray(upper, dtype=float)
            n = upper.size
            rows.append(np.eye(n))
            rhs_parts.append(upper)

        if lower is not None:
            lower = np.asarray(lower, dtype=float)
            if n is None:
                n = lower.size
            if lower.size != n:
                raise ValueError("Lower/upper bounds must have the same length.")
            rows.append(-np.eye(n))
            rhs_parts.append(-lower)

        assert n is not None
        C = np.vstack(rows)
        d = np.concatenate(rhs_parts)
        super().__init__(C, d)
