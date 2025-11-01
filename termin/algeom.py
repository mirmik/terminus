"""Алгебраическая геометрия: квадратичные формы, эллипсоиды, коники."""
import numpy


def fit_ellipsoid(points, center=None):
    """Строит квадратичную форму эллипсоида по набору точек методом наименьших квадратов.
    
    Эллипсоид задаётся уравнением: (x-c)ᵀ A (x-c) = 1, где:
    - A - положительно определённая матрица (задаёт форму и ориентацию)
    - c - центр эллипсоида
    
    Альтернативная форма: xᵀAx + bᵀx + d = 0 (общая квадратичная форма)
    
    Args:
        points: Массив точек размера (n_points, n_dim).
                Точки должны приблизительно лежать на поверхности эллипсоида.
                Минимум n_dim*(n_dim+3)/2 точек для определённости.
        center: Центр эллипсоида размера (n_dim,).
                Если None, центр определяется автоматически.
                Если задан, строится эллипсоид с фиксированным центром.
    
    Returns:
        A: Матрица квадратичной формы размера (n_dim, n_dim)
        center: Центр эллипсоида размера (n_dim,)
        radii: Полуоси эллипсоида (собственные значения A⁻¹)
        axes: Направления осей (собственные векторы A)
    
    Notes:
        Решает задачу наименьших квадратов для общей квадратичной формы:
        x² + B₁₁xy + B₁₂xz + B₂₂y² + B₂₃yz + B₃₃z² + C₁x + C₂y + C₃z + D = 0
        
        Для 3D это 9 неизвестных (при нормировке), для nD: n(n+3)/2 параметров.
        
        Метод:
        1. Составляем матрицу дизайна для квадратичной формы
        2. Решаем переопределённую систему методом SVD
        3. Извлекаем матрицу A и вектор b из коэффициентов
        4. Находим центр: c = -½A⁻¹b
        5. Нормализуем к канонической форме (x-c)ᵀA(x-c) = 1
    
    Examples:
        >>> # Точки на сфере радиуса 2
        >>> theta = np.linspace(0, 2*np.pi, 50)
        >>> phi = np.linspace(0, np.pi, 50)
        >>> THETA, PHI = np.meshgrid(theta, phi)
        >>> X = 2 * np.sin(PHI) * np.cos(THETA)
        >>> Y = 2 * np.sin(PHI) * np.sin(THETA)
        >>> Z = 2 * np.cos(PHI)
        >>> points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        >>> A, center, radii, axes = fit_ellipsoid(points)
        >>> radii  # [2, 2, 2]
    """
    points = numpy.asarray(points, dtype=float)
    
    if points.ndim != 2:
        raise ValueError(f"points должен быть 2D массивом, получен {points.ndim}D")
    
    n_points, n_dim = points.shape
    
    # Минимальное количество точек для определения эллипсоида
    min_points = n_dim * (n_dim + 3) // 2
    if n_points < min_points:
        raise ValueError(f"Недостаточно точек: нужно минимум {min_points}, получено {n_points}")
    
    # Решаем задачу подгонки
    if center is not None:
        center = numpy.asarray(center, dtype=float)
        if center.shape != (n_dim,):
            raise ValueError(f"center должен иметь размер ({n_dim},), получен {center.shape}")
        A = _fit_ellipsoid_fixed_center(points, center)
    else:
        A, center = _fit_ellipsoid_auto_center(points)
    
    # Вычисляем полуоси и направления через собственные значения/векторы
    eigvals, eigvecs = numpy.linalg.eigh(A)
    
    # Полуоси = sqrt(1/λᵢ), так как (x-c)ᵀA(x-c) = 1 и A = VΛV^T
    radii = 1.0 / numpy.sqrt(eigvals)
    
    # Сортируем по убыванию полуосей (a ≥ b ≥ c)
    sort_idx = numpy.argsort(radii)[::-1]
    radii = radii[sort_idx]
    axes = eigvecs[:, sort_idx]
    
    return A, center, radii, axes


def _fit_ellipsoid_fixed_center(points, center):
    """Подгоняет эллипсоид с заданным центром.
    
    Args:
        points: Массив точек размера (n_points, n_dim)
        center: Центр эллипсоида размера (n_dim,)
    
    Returns:
        Матрица A размера (n_dim, n_dim)
    """
    n_points, n_dim = points.shape
    
    # Сдвигаем точки к центру
    points_centered = points - center
    
    # Строим матрицу дизайна для квадратичной формы xᵀAx = 1
    design_matrix = _build_quadratic_design_matrix(points_centered)
    
    # Решаем систему: design_matrix @ coeffs = 1
    coeffs, residuals, rank, s = numpy.linalg.lstsq(
        design_matrix, numpy.ones(n_points), rcond=None
    )
    
    # Восстанавливаем симметричную матрицу A из коэффициентов
    A = _coeffs_to_matrix(coeffs, n_dim)
    
    # Проверяем положительную определённость
    eigvals = numpy.linalg.eigvalsh(A)
    if numpy.any(eigvals <= 0):
        raise ValueError(
            "Получена не положительно определённая матрица. "
            "Точки не лежат на эллипсоиде."
        )
    
    return A


def _fit_ellipsoid_auto_center(points):
    """Подгоняет эллипсоид с автоматическим определением центра.
    
    Args:
        points: Массив точек размера (n_points, n_dim)
    
    Returns:
        A: Матрица размера (n_dim, n_dim)
        center: Центр размера (n_dim,)
    """
    n_points, n_dim = points.shape
    
    # Строим полную матрицу дизайна: [квадратичные, линейные, константа]
    design_matrix = _build_full_design_matrix(points)
    
    # Решаем однородную систему с ограничением ||coeffs|| = 1
    # Используем SVD: решение = правый сингулярный вектор для минимального σ
    u, s, vh = numpy.linalg.svd(design_matrix, full_matrices=True)
    coeffs = vh[-1, :]
    
    # Извлекаем компоненты
    n_quad = n_dim * (n_dim + 1) // 2
    A = _coeffs_to_matrix(coeffs[:n_quad], n_dim)
    b = coeffs[n_quad:n_quad + n_dim]
    d = coeffs[n_quad + n_dim]
    
    # SVD может дать решение с произвольным знаком
    # Проверяем знак: для эллипсоида нужно A > 0
    eigvals = numpy.linalg.eigvalsh(A)
    if numpy.all(eigvals < 0):
        # Инвертируем знак
        A = -A
        b = -b
        d = -d
        eigvals = -eigvals
    elif not numpy.all(eigvals > 0):
        raise ValueError(
            "Получена не положительно определённая матрица. "
            "Точки не лежат на эллипсоиде."
        )
    
    # Находим центр: c = -½A⁻¹b
    A_inv = numpy.linalg.inv(A)
    center = -0.5 * A_inv @ b
    
    # Нормализуем к канонической форме (x-c)ᵀA(x-c) = 1
    k = -(center @ A @ center + b @ center + d)
    
    if k <= 0:
        raise ValueError("Некорректная нормализация. Проверьте входные данные.")
    
    A = A / k
    
    return A, center


def _build_quadratic_design_matrix(points):
    """Строит матрицу дизайна для квадратичных членов.
    
    Для каждой точки: [x², xy, xz, y², yz, z², ...] (верхний треугольник).
    
    Args:
        points: Массив точек размера (n_points, n_dim)
    
    Returns:
        Матрица дизайна размера (n_points, n_dim*(n_dim+1)/2)
    """
    n_points, n_dim = points.shape
    columns = []
    
    for i in range(n_dim):
        for j in range(i, n_dim):
            if i == j:
                # Диагональные элементы: x², y², z²
                columns.append(points[:, i] ** 2)
            else:
                # Внедиагональные: 2*xy, 2*xz, 2*yz (множитель 2 для симметрии)
                columns.append(2 * points[:, i] * points[:, j])
    
    return numpy.column_stack(columns)


def _build_full_design_matrix(points):
    """Строит полную матрицу дизайна: квадратичные + линейные + константа.
    
    Args:
        points: Массив точек размера (n_points, n_dim)
    
    Returns:
        Матрица дизайна размера (n_points, n_dim*(n_dim+3)/2 + 1)
    """
    n_points, n_dim = points.shape
    
    # Квадратичные члены
    quad_matrix = _build_quadratic_design_matrix(points)
    
    # Линейные члены
    linear_columns = [points[:, i] for i in range(n_dim)]
    
    # Константный член
    const_column = [numpy.ones(n_points)]
    
    return numpy.column_stack([quad_matrix] + linear_columns + const_column)


def _coeffs_to_matrix(coeffs, n_dim):
    """Восстанавливает симметричную матрицу из коэффициентов верхнего треугольника.
    
    Args:
        coeffs: Коэффициенты размера (n_dim*(n_dim+1)/2,)
        n_dim: Размерность матрицы
    
    Returns:
        Симметричная матрица размера (n_dim, n_dim)
    """
    A = numpy.zeros((n_dim, n_dim))
    idx = 0
    
    for i in range(n_dim):
        for j in range(i, n_dim):
            if i == j:
                A[i, j] = coeffs[idx]
            else:
                A[i, j] = coeffs[idx]
                A[j, i] = coeffs[idx]
            idx += 1
    
    return A


def ellipsoid_equation(A, center):
    """Форматирует уравнение эллипсоида в читаемую строку.
    
    Args:
        A: Матрица квадратичной формы размера (n_dim, n_dim)
        center: Центр эллипсоида размера (n_dim,)
    
    Returns:
        Строковое представление уравнения
    """
    n_dim = len(center)
    coord_names = ['x', 'y', 'z', 'w'] + [f'x_{i}' for i in range(4, n_dim)]
    
    terms = []
    for i in range(n_dim):
        ci = center[i]
        coord = coord_names[i]
        if abs(ci) > 1e-10:
            terms.append(f"({coord} - {ci:.3g})")
        else:
            terms.append(coord)
    
    if n_dim <= 3:
        return f"(x-c)ᵀ A (x-c) = 1, где c = {center}"
    else:
        return f"Эллипсоид в R^{n_dim} с центром {center}"


def ellipsoid_volume(radii):
    """Вычисляет объём эллипсоида по полуосям.
    
    Args:
        radii: Полуоси эллипсоида размера (n_dim,)
    
    Returns:
        Объём эллипсоида
    
    Notes:
        V = (π^(n/2) / Γ(n/2 + 1)) * ∏rᵢ
        
        Для малых размерностей:
        - 1D: 2r (длина отрезка)
        - 2D: πab (площадь эллипса)
        - 3D: (4/3)πabc (объём эллипсоида)
    """
    radii = numpy.asarray(radii)
    n_dim = len(radii)
    
    # Вычисляем Γ(n/2 + 1) без scipy
    # Для целых и полуцелых значений можем использовать формулы
    half_n = n_dim / 2.0
    
    if n_dim % 2 == 0:
        # n чётное: Γ(k+1) = k! для k = n/2
        k = n_dim // 2
        gamma_val = float(numpy.math.factorial(k))
    else:
        # n нечётное: Γ(k+0.5) = (2k)! / (4^k * k!) * sqrt(π) для k = (n-1)/2
        k = (n_dim - 1) // 2
        numerator = numpy.math.factorial(2 * k)
        denominator = (4 ** k) * numpy.math.factorial(k)
        gamma_val = (numerator / denominator) * numpy.sqrt(numpy.pi)
    
    volume = (numpy.pi ** half_n / gamma_val) * numpy.prod(radii)
    
    return volume


def evaluate_ellipsoid(points, A, center):
    """Вычисляет значения квадратичной формы эллипсоида в точках.
    
    Для эллипсоида (x-c)ᵀA(x-c) = 1 вычисляет (x-c)ᵀA(x-c) для каждой точки.
    Значения:
    - < 1: точка внутри эллипсоида
    - = 1: точка на поверхности
    - > 1: точка снаружи
    
    Args:
        points: Массив точек размера (n_points, n_dim)
        A: Матрица квадратичной формы размера (n_dim, n_dim)
        center: Центр эллипсоида размера (n_dim,)
    
    Returns:
        Массив значений размера (n_points,)
    """
    points = numpy.asarray(points, dtype=float)
    center = numpy.asarray(center, dtype=float)
    A = numpy.asarray(A, dtype=float)
    
    # Центрируем точки
    points_centered = points - center
    
    # Вычисляем квадратичную форму: (x-c)ᵀA(x-c)
    # Эффективно: sum((A @ p) * p) для каждой точки
    values = numpy.sum((points_centered @ A) * points_centered, axis=1)
    
    return values


def ellipsoid_contains(points, A, center, tol=1e-10):
    """Проверяет, лежат ли точки внутри или на эллипсоиде.
    
    Args:
        points: Массив точек размера (n_points, n_dim)
        A: Матрица квадратичной формы размера (n_dim, n_dim)
        center: Центр эллипсоида размера (n_dim,)
        tol: Допуск для точек на границе
    
    Returns:
        Булев массив размера (n_points,): True если точка внутри/на эллипсоиде
    """
    values = evaluate_ellipsoid(points, A, center)
    return values <= (1.0 + tol)

