import numpy as np
from scipy.linalg import cho_factor, cho_solve

def solve_qp_equalities(H, g, A, b):
    """
    Решает задачу:
        min  1/2 x^T H x + g^T x
        s.t. A x = b

    Академический вывод системы Шура:

      ККТ:
        [ H   A^T ] [ x ] = [ -g ]
        [ A    0  ] [ λ ]   [  b ]

      Из первой строки:
        x = -H^{-1}(g + A^T λ)

      Подставляем в A x = b:
        A[-H^{-1}(g + A^T λ)] = b

      Получаем:
        -A H^{-1} g - A H^{-1} A^T λ = b

      Система Шура:
        (A H^{-1} A^T) λ = -b - A H^{-1} g

      После нахождения λ:
        H x = -g - A^T λ
    """

    # --- 1) Cholesky разложение: H = L L^T ---
    L, lower = cho_factor(H)

    # --- 2) Вычислить H^{-1} g и H^{-1} A^T ---
    # здесь считаем два объекта, входящие в формулы Шура:
    # H^{-1} g  и  H^{-1} A^T
    y_g = cho_solve((L, lower), g)     # = H^{-1} g
    Y   = cho_solve((L, lower), A.T)   # = H^{-1} A^T

    # --- 3) Построить систему Шура ---
    # S = A H^{-1} A^T
    S = A @ Y
    # r_λ = -b - A H^{-1} g
    r_lambda = -b - A @ y_g

    # --- 4) Решить систему Шура: S λ = r_λ ---
    λ = np.linalg.solve(S, r_lambda)

    # --- 5) Восстановить x: H x = -g - A^T λ ---
    w = -g - A.T @ λ
    x = cho_solve((L, lower), w)

    return x, λ