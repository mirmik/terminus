#!/usr/bin/env python3
"""
Универсальная система сборки матриц для МКЭ и других задач.

Основная идея: 
- Система уравнений собирается из вкладов (contributions)
- Каждый вклад знает, какие переменные он затрагивает
- Итоговая система: A*x = b
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


class Variable:
    """
    Переменная (неизвестная величина) в системе уравнений.
    
    Может представлять:
    - Перемещение узла в механике
    - Напряжение в узле электрической цепи
    - Температуру в узле тепловой задачи
    - И т.д.
    """
    
    def __init__(self, name: str, size: int = 1):
        """
        Args:
            name: Имя переменной (для отладки)
            size: Размерность (1 для скаляра, 2/3 для вектора)
        """
        self.name = name
        self.size = size
        self.global_indices = []  # будет заполнено при сборке
        self.value = None  # будет заполнено после решения системы
        
    def __repr__(self):
        return f"Variable({self.name}, size={self.size})"


class Contribution:
    """
    Базовый класс для вклада в систему уравнений.
    
    Вклад - это что-то, что добавляет элементы в матрицу A и/или вектор b.
    Примеры:
    - Уравнение стержня в механике
    - Резистор в электрической цепи
    - Граничное условие
    - Уравнение связи между переменными
    """
    
    def get_variables(self) -> List[Variable]:
        """Возвращает список переменных, которые затрагивает этот вклад"""
        raise NotImplementedError
    
    def contribute_to_A(self, A: np.ndarray, index_map: Dict[Variable, List[int]]):
        """
        Добавить вклад в матрицу A
        
        Args:
            A: Глобальная матрица (изменяется in-place)
            index_map: Отображение Variable -> список глобальных индексов
        """
        raise NotImplementedError
    
    def contribute_to_b(self, b: np.ndarray, index_map: Dict[Variable, List[int]]):
        """
        Добавить вклад в вектор правой части b
        
        Args:
            b: Глобальный вектор (изменяется in-place)
            index_map: Отображение Variable -> список глобальных индексов
        """
        raise NotImplementedError


class MatrixAssembler:
    """
    Сборщик матриц из вкладов.
    
    Основной класс системы - собирает глобальную матрицу A и вектор b
    из множества локальных вкладов.
    """
    
    def __init__(self):
        self.variables: List[Variable] = []
        self.contributions: List[Contribution] = []
        self._index_map: Optional[Dict[Variable, List[int]]] = None
        
    def add_variable(self, name: str, size: int = 1) -> Variable:
        """
        Добавить переменную в систему
        
        Args:
            name: Имя переменной
            size: Размерность (1 для скаляра, 2/3 для вектора)
            
        Returns:
            Созданная переменная
        """
        var = Variable(name, size)
        self.variables.append(var)
        return var
    
    def add_contribution(self, contribution: Contribution):
        """
        Добавить вклад в систему
        
        Args:
            contribution: Вклад (уравнение, граничное условие, и т.д.)
        """
        self.contributions.append(contribution)
    
    def _build_index_map(self) -> Dict[Variable, List[int]]:
        """
        Построить отображение: Variable -> глобальные индексы DOF
        
        Назначает каждой компоненте каждой переменной уникальный
        глобальный индекс в системе.
        """
        index_map = {}
        current_index = 0
        
        for var in self.variables:
            indices = list(range(current_index, current_index + var.size))
            index_map[var] = indices
            var.global_indices = indices
            current_index += var.size
            
        return index_map
    
    def total_dofs(self) -> int:
        """Общее количество степеней свободы в системе"""
        return sum(var.size for var in self.variables)
    
    def assemble(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Собрать глобальную систему A*x = b
        
        Returns:
            (A, b): Матрица и вектор правой части
        """
        # Построить карту индексов
        self._index_map = self._build_index_map()
        
        # Создать глобальные матрицу и вектор
        n_dofs = self.total_dofs()
        A = np.zeros((n_dofs, n_dofs))
        b = np.zeros(n_dofs)
        
        # Собрать вклады
        for contribution in self.contributions:
            contribution.contribute_to_A(A, self._index_map)
            contribution.contribute_to_b(b, self._index_map)
        
        return A, b
    
    def solve(self, check_conditioning: bool = True, 
              use_least_squares: bool = False) -> np.ndarray:
        """
        Собрать и решить систему A*x = b
        
        Args:
            check_conditioning: Проверить обусловленность матрицы и выдать предупреждение
            use_least_squares: Использовать lstsq вместо solve (робастнее, но медленнее)
        
        Returns:
            x: Вектор решения (все переменные подряд)
        """
        A, b = self.assemble()
        
        # Проверка обусловленности
        if check_conditioning:
            cond_number = np.linalg.cond(A)
            if cond_number > 1e10:
                import warnings
                warnings.warn(
                    f"Матрица плохо обусловлена: cond(A) = {cond_number:.2e}. "
                    f"Это может быть из-за penalty method в граничных условиях. "
                    f"Рассмотрите использование use_least_squares=True",
                    RuntimeWarning
                )
            elif cond_number > 1e6:
                import warnings
                warnings.warn(
                    f"Матрица имеет высокое число обусловленности: cond(A) = {cond_number:.2e}",
                    RuntimeWarning
                )
        
        # Решение системы
        if use_least_squares:
            # Метод наименьших квадратов - более робастный
            x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            if check_conditioning and rank < len(b):
                import warnings
                warnings.warn(
                    f"Матрица вырожденная или близка к вырожденной: "
                    f"rank(A) = {rank}, expected {len(b)}",
                    RuntimeWarning
                )
        else:
            # Прямое решение - быстрее, но менее робастное
            try:
                x = np.linalg.solve(A, b)
            except np.linalg.LinAlgError as e:
                raise RuntimeError(
                    f"Не удалось решить систему: {e}. "
                    f"Возможно, матрица вырожденная (не хватает граничных условий?) "
                    f"или плохо обусловлена. Попробуйте use_least_squares=True"
                ) from e
        
        return x
    
    def get_solution_dict(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Преобразовать вектор решения в словарь {имя_переменной: значение}
        
        Args:
            x: Вектор решения
            
        Returns:
            Словарь с решениями для каждой переменной
        """
        if self._index_map is None:
            raise RuntimeError("Система не собрана. Вызовите assemble() или solve()")
        
        result = {}
        for var in self.variables:
            indices = self._index_map[var]
            result[var.name] = x[indices] if len(indices) > 1 else x[indices[0]]
        
        return result
    
    def set_solution_to_variables(self, x: np.ndarray):
        """
        Сохранить решение в объекты Variable
        
        После вызова этого метода каждая переменная будет иметь атрибут value
        с решением (скаляр или numpy array).
        
        Args:
            x: Вектор решения
        """
        if self._index_map is None:
            raise RuntimeError("Система не собрана. Вызовите assemble() или solve()")
        
        for var in self.variables:
            indices = self._index_map[var]
            if len(indices) > 1:
                var.value = x[indices]
            else:
                var.value = x[indices[0]]
    
    def solve_and_set(self, check_conditioning: bool = True, 
                      use_least_squares: bool = False) -> np.ndarray:
        """
        Решить систему и сохранить результат в переменные
        
        Удобный метод, который объединяет solve() и set_solution_to_variables().
        
        Args:
            check_conditioning: Проверить обусловленность матрицы
            use_least_squares: Использовать lstsq вместо solve
        
        Returns:
            x: Вектор решения (также сохранен в переменных)
        """
        x = self.solve(check_conditioning=check_conditioning, 
                       use_least_squares=use_least_squares)
        self.set_solution_to_variables(x)
        return x
    
    def diagnose_matrix(self) -> Dict[str, any]:
        """
        Диагностика собранной матрицы системы
        
        Returns:
            Словарь с информацией о матрице:
            - condition_number: число обусловленности
            - is_symmetric: симметричность
            - is_positive_definite: положительная определённость
            - rank: ранг матрицы
            - min_eigenvalue: минимальное собственное значение
            - max_eigenvalue: максимальное собственное значение
        """
        A, b = self.assemble()
        
        info = {}
        
        # Число обусловленности
        info['condition_number'] = np.linalg.cond(A)
        
        # Симметричность
        info['is_symmetric'] = np.allclose(A, A.T)
        
        # Ранг
        info['rank'] = np.linalg.matrix_rank(A)
        info['expected_rank'] = len(A)
        info['is_full_rank'] = info['rank'] == info['expected_rank']
        
        # Собственные значения (только для небольших матриц)
        if len(A) <= 100:
            eigenvalues = np.linalg.eigvalsh(A) if info['is_symmetric'] else np.linalg.eigvals(A)
            eigenvalues = np.real(eigenvalues)
            info['min_eigenvalue'] = np.min(eigenvalues)
            info['max_eigenvalue'] = np.max(eigenvalues)
            info['is_positive_definite'] = np.all(eigenvalues > 0)
        else:
            info['eigenvalues'] = 'Skipped (matrix too large)'
            info['is_positive_definite'] = None
        
        # Оценка качества
        cond = info['condition_number']
        if cond < 100:
            info['quality'] = 'excellent'
        elif cond < 1e4:
            info['quality'] = 'good'
        elif cond < 1e8:
            info['quality'] = 'acceptable'
        elif cond < 1e12:
            info['quality'] = 'poor'
        else:
            info['quality'] = 'very_poor'
        
        return info
    
    def print_diagnose(self):
        """
        Print human-readable matrix diagnostics
        """
        info = self.diagnose_matrix()
        
        print("=" * 70)
        print("MATRIX SYSTEM DIAGNOSTICS")
        print("=" * 70)
        
        # Dimensions
        print(f"\nSystem dimensions:")
        print(f"  Number of variables: {len(self.variables)}")
        print(f"  Degrees of freedom (DOF): {self.total_dofs()}")
        
        # Matrix rank
        print(f"\nMatrix rank:")
        print(f"  Current rank: {info['rank']}")
        print(f"  Expected rank: {info['expected_rank']}")
        if info['is_full_rank']:
            print(f"  [OK] Matrix has full rank")
        else:
            print(f"  [PROBLEM] Matrix is singular (rank deficient)")
            print(f"    Possibly missing boundary conditions")
        
        # Symmetry
        print(f"\nSymmetry:")
        if info['is_symmetric']:
            print(f"  [OK] Matrix is symmetric")
        else:
            print(f"  [PROBLEM] Matrix is not symmetric")
            print(f"    This may indicate an error in contributions")
        
        # Conditioning
        print(f"\nConditioning:")
        print(f"  Condition number: {info['condition_number']:.2e}")
        print(f"  Quality assessment: {info['quality']}")
        
        quality_desc = {
            'excellent': '[OK] Excellent - matrix is very well conditioned',
            'good': '[OK] Good - matrix is well conditioned',
            'acceptable': '[WARNING] Acceptable - may have small numerical errors',
            'poor': '[PROBLEM] Poor - high risk of numerical errors',
            'very_poor': '[PROBLEM] Very poor - solution may be inaccurate'
        }
        print(f"  {quality_desc.get(info['quality'], '')}")
        
        if info['quality'] in ['poor', 'very_poor']:
            print(f"\n  Recommendations:")
            print(f"    - Reduce penalty in boundary conditions (try 1e8)")
            print(f"    - Use use_least_squares=True when solving")
            print(f"    - Check the scales of quantities in the problem")
        
        # Eigenvalues
        if info.get('min_eigenvalue') is not None:
            print(f"\nEigenvalues:")
            print(f"  Minimum: {info['min_eigenvalue']:.2e}")
            print(f"  Maximum: {info['max_eigenvalue']:.2e}")
            
            if info.get('is_positive_definite'):
                print(f"  [OK] Matrix is positive definite")
            else:
                print(f"  [PROBLEM] Matrix is not positive definite")
                if info['min_eigenvalue'] <= 0:
                    print(f"    Has non-positive eigenvalues")
        
        # Final recommendation
        print(f"\n" + "=" * 70)
        if info['is_full_rank'] and info['is_symmetric'] and info['quality'] in ['excellent', 'good', 'acceptable']:
            print("SUMMARY: [OK] System is ready to solve")
        else:
            print("SUMMARY: [WARNING] Problems detected, attention required")
        print("=" * 70)


# ============================================================================
# Примеры конкретных вкладов
# ============================================================================

class BilinearContribution(Contribution):
    """
    Билинейный вклад: связь двух переменных через локальную матрицу
    
    Пример: стержень, пружина, резистор
    Вклад в A: A[i,j] += K_local[i,j] для пар индексов переменных
    """
    
    def __init__(self, variables: List[Variable], K_local: np.ndarray):
        """
        Args:
            variables: Список переменных (например, [u1, u2] для стержня)
            K_local: Локальная матрица вклада
        """
        self.variables = variables
        self.K_local = np.array(K_local)
        
        # Проверка размерности
        expected_size = sum(v.size for v in variables)
        if self.K_local.shape != (expected_size, expected_size):
            raise ValueError(f"Размер K_local {self.K_local.shape} не соответствует "
                           f"суммарному размеру переменных {expected_size}")
    
    def get_variables(self) -> List[Variable]:
        return self.variables
    
    def contribute_to_A(self, A: np.ndarray, index_map: Dict[Variable, List[int]]):
        # Собрать глобальные индексы всех переменных
        global_indices = []
        for var in self.variables:
            global_indices.extend(index_map[var])
        
        # Добавить локальную матрицу в глобальную
        for i, gi in enumerate(global_indices):
            for j, gj in enumerate(global_indices):
                A[gi, gj] += self.K_local[i, j]
    
    def contribute_to_b(self, b: np.ndarray, index_map: Dict[Variable, List[int]]):
        # Этот тип вклада не влияет на правую часть
        pass


class LoadContribution(Contribution):
    """
    Вклад нагрузки/источника в правую часть
    
    Пример: приложенная сила, источник тока, тепловой источник
    Вклад в b: b[i] += F[i]
    """
    
    def __init__(self, variable: Variable, load: np.ndarray):
        """
        Args:
            variable: Переменная, к которой приложена нагрузка
            load: Вектор нагрузки (размера variable.size)
        """
        self.variable = variable
        self.load = np.atleast_1d(load)
        
        if len(self.load) != variable.size:
            raise ValueError(f"Размер нагрузки {len(self.load)} не соответствует "
                           f"размеру переменной {variable.size}")
    
    def get_variables(self) -> List[Variable]:
        return [self.variable]
    
    def contribute_to_A(self, A: np.ndarray, index_map: Dict[Variable, List[int]]):
        # Не влияет на матрицу
        pass
    
    def contribute_to_b(self, b: np.ndarray, index_map: Dict[Variable, List[int]]):
        indices = index_map[self.variable]
        for i, idx in enumerate(indices):
            b[idx] += self.load[i]


class ConstraintContribution(Contribution):
    """
    Граничное условие: фиксированное значение переменной
    
    Пример: u1 = 0 (закрепленный узел), V1 = 5 (источник напряжения)
    
    Реализовано через penalty method:
    A[i,i] += penalty
    b[i] += penalty * prescribed_value
    """
    
    def __init__(self, variable: Variable, value: float, 
                 component: int = 0, penalty: float = 1e10):
        """
        Args:
            variable: Переменная для ограничения
            value: Предписанное значение
            component: Компонента переменной (0 для скаляра)
            penalty: Штрафной коэффициент (большое число)
        """
        self.variable = variable
        self.value = value
        self.component = component
        self.penalty = penalty
        
        if component >= variable.size:
            raise ValueError(f"Компонента {component} вне диапазона для переменной "
                           f"размера {variable.size}")
    
    def get_variables(self) -> List[Variable]:
        return [self.variable]
    
    def contribute_to_A(self, A: np.ndarray, index_map: Dict[Variable, List[int]]):
        indices = index_map[self.variable]
        idx = indices[self.component]
        A[idx, idx] += self.penalty
    
    def contribute_to_b(self, b: np.ndarray, index_map: Dict[Variable, List[int]]):
        indices = index_map[self.variable]
        idx = indices[self.component]
        b[idx] += self.penalty * self.value


# ============================================================================
# Вспомогательные функции для удобства
# ============================================================================

def spring_element(u1: Variable, u2: Variable, stiffness: float) -> BilinearContribution:
    """
    Создать вклад пружины/стержня между двумя скалярными переменными
    
    Уравнение: F = k*(u2-u1)
    Матрица:  [[k, -k],
               [-k, k]]
    """
    K = stiffness * np.array([
        [ 1, -1],
        [-1,  1]
    ])
    return BilinearContribution([u1, u2], K)


def conductance_element(V1: Variable, V2: Variable, conductance: float) -> BilinearContribution:
    """
    Создать вклад проводимости (резистор) между двумя узлами
    
    То же самое что spring_element, но с другим физическим смыслом
    """
    return spring_element(V1, V2, conductance)
