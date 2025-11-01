"""Тесты для модуля алгебраической геометрии."""
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from termin.algeom import (
    fit_ellipsoid,
    evaluate_ellipsoid,
    ellipsoid_contains,
    ellipsoid_equation,
    ellipsoid_volume
)


class TestFitEllipsoid(unittest.TestCase):
    """Тесты для подгонки эллипсоида по точкам."""
    
    def test_sphere_basic(self):
        """Сфера радиуса 2"""
        # Генерируем точки на сфере
        n = 30
        theta = np.linspace(0, 2*np.pi, n)
        phi = np.linspace(0, np.pi, n // 2)
        
        points = []
        for p in phi:
            for t in theta:
                x = 2 * np.sin(p) * np.cos(t)
                y = 2 * np.sin(p) * np.sin(t)
                z = 2 * np.cos(p)
                points.append([x, y, z])
        
        points = np.array(points)
        A, center, radii, axes = fit_ellipsoid(points)
        
        # Проверяем, что все полуоси примерно равны 2
        np.testing.assert_allclose(radii, [2, 2, 2], rtol=0.01)
        
        # Проверяем, что центр близок к нулю
        np.testing.assert_allclose(center, [0, 0, 0], atol=0.1)
    
    def test_ellipsoid_3d(self):
        """Эллипсоид с полуосями a=3, b=2, c=1"""
        # Генерируем точки на эллипсоиде
        n = 30
        theta = np.linspace(0, 2*np.pi, n)
        phi = np.linspace(0, np.pi, n // 2)
        
        a, b, c = 3.0, 2.0, 1.0
        
        points = []
        for p in phi:
            for t in theta:
                x = a * np.sin(p) * np.cos(t)
                y = b * np.sin(p) * np.sin(t)
                z = c * np.cos(p)
                points.append([x, y, z])
        
        points = np.array(points)
        A, center, radii, axes = fit_ellipsoid(points)
        
        # Проверяем полуоси (с небольшой погрешностью из-за дискретизации)
        expected_radii = np.array([a, b, c])
        np.testing.assert_allclose(radii, expected_radii, rtol=0.01)
        
        # Центр должен быть близок к нулю
        np.testing.assert_allclose(center, [0, 0, 0], atol=0.1)
    
    def test_ellipsoid_2d(self):
        """Эллипс на плоскости с полуосями a=5, b=3"""
        # Генерируем точки на эллипсе
        t = np.linspace(0, 2*np.pi, 100)
        a, b = 5.0, 3.0
        
        x = a * np.cos(t)
        y = b * np.sin(t)
        points = np.column_stack([x, y])
        
        A, center, radii, axes = fit_ellipsoid(points)
        
        # Проверяем полуоси
        np.testing.assert_allclose(radii, [a, b], rtol=0.01)
        
        # Центр близок к нулю
        np.testing.assert_allclose(center, [0, 0], atol=0.1)
    
    def test_ellipsoid_with_offset(self):
        """Эллипсоид со смещённым центром"""
        # Эллипсоид с центром в (1, 2, 3)
        center_true = np.array([1.0, 2.0, 3.0])
        a, b, c = 4.0, 3.0, 2.0
        
        n = 30
        theta = np.linspace(0, 2*np.pi, n)
        phi = np.linspace(0, np.pi, n // 2)
        
        points = []
        for p in phi:
            for t in theta:
                x = center_true[0] + a * np.sin(p) * np.cos(t)
                y = center_true[1] + b * np.sin(p) * np.sin(t)
                z = center_true[2] + c * np.cos(p)
                points.append([x, y, z])
        
        points = np.array(points)
        A, center, radii, axes = fit_ellipsoid(points)
        
        # Проверяем центр
        np.testing.assert_allclose(center, center_true, atol=0.1)
        
        # Проверяем полуоси
        np.testing.assert_allclose(radii, [a, b, c], rtol=0.01)
    
    def test_ellipsoid_with_fixed_center(self):
        """Подгонка эллипсоида с заданным центром"""
        center_fixed = np.array([0.0, 0.0, 0.0])
        a, b, c = 2.0, 1.5, 1.0
        
        # Генерируем точки
        n = 30
        theta = np.linspace(0, 2*np.pi, n)
        phi = np.linspace(0, np.pi, n // 2)
        
        points = []
        for p in phi:
            for t in theta:
                x = a * np.sin(p) * np.cos(t)
                y = b * np.sin(p) * np.sin(t)
                z = c * np.cos(p)
                points.append([x, y, z])
        
        points = np.array(points)
        A, center, radii, axes = fit_ellipsoid(points, center=center_fixed)
        
        # Центр должен быть точно равен заданному
        np.testing.assert_array_equal(center, center_fixed)
        
        # Полуоси должны совпадать
        np.testing.assert_allclose(radii, [a, b, c], rtol=0.01)
    
    def test_rotated_ellipsoid(self):
        """Эллипсоид с поворотом"""
        # Создаём эллипсоид, повёрнутый на 45° вокруг оси Z
        angle = np.pi / 4
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        
        a, b, c = 3.0, 2.0, 1.0
        
        n = 30
        theta = np.linspace(0, 2*np.pi, n)
        phi = np.linspace(0, np.pi, n // 2)
        
        points = []
        for p in phi:
            for t in theta:
                # Точка на стандартном эллипсоиде
                point = np.array([
                    a * np.sin(p) * np.cos(t),
                    b * np.sin(p) * np.sin(t),
                    c * np.cos(p)
                ])
                # Поворачиваем
                rotated = R @ point
                points.append(rotated)
        
        points = np.array(points)
        A, center, radii, axes = fit_ellipsoid(points)
        
        # Полуоси должны совпадать (инвариантны к повороту)
        np.testing.assert_allclose(radii, [a, b, c], rtol=0.01)
        
        # Центр близок к нулю
        np.testing.assert_allclose(center, [0, 0, 0], atol=0.1)
    
    def test_insufficient_points(self):
        """Ошибка при недостаточном количестве точек"""
        # Для 3D нужно минимум 9 точек
        points = np.random.randn(5, 3)
        
        with self.assertRaises(ValueError) as ctx:
            fit_ellipsoid(points)
        
        self.assertIn("Недостаточно точек", str(ctx.exception))
    
    def test_noisy_data(self):
        """Подгонка с зашумлёнными данными"""
        a, b, c = 2.0, 1.5, 1.0
        
        # Генерируем точки с шумом
        n = 50
        theta = np.linspace(0, 2*np.pi, n)
        phi = np.linspace(0, np.pi, n // 2)
        
        points = []
        for p in phi:
            for t in theta:
                x = a * np.sin(p) * np.cos(t)
                y = b * np.sin(p) * np.sin(t)
                z = c * np.cos(p)
                # Добавляем шум
                noise = np.random.randn(3) * 0.05
                points.append([x, y, z] + noise)
        
        points = np.array(points)
        A, center, radii, axes = fit_ellipsoid(points)
        
        # С шумом точность снижается, но должна быть разумной
        np.testing.assert_allclose(radii, [a, b, c], rtol=0.1)


class TestEvaluateEllipsoid(unittest.TestCase):
    """Тесты для вычисления значений квадратичной формы."""
    
    def test_points_on_surface(self):
        """Точки на поверхности дают значение ≈ 1"""
        # Создаём единичную сферу
        A = np.eye(3)
        center = np.zeros(3)
        
        # Точки на единичной сфере
        points = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1/np.sqrt(2), 1/np.sqrt(2), 0]
        ])
        
        values = evaluate_ellipsoid(points, A, center)
        
        np.testing.assert_allclose(values, [1, 1, 1, 1], atol=1e-10)
    
    def test_points_inside(self):
        """Точки внутри дают значение < 1"""
        A = np.eye(3)
        center = np.zeros(3)
        
        # Точки внутри единичной сферы
        points = np.array([
            [0, 0, 0],  # Центр
            [0.5, 0, 0],
            [0.3, 0.3, 0.3]
        ])
        
        values = evaluate_ellipsoid(points, A, center)
        
        self.assertTrue(np.all(values < 1))
        self.assertAlmostEqual(values[0], 0)  # Центр
    
    def test_points_outside(self):
        """Точки снаружи дают значение > 1"""
        A = np.eye(3)
        center = np.zeros(3)
        
        # Точки снаружи единичной сферы
        points = np.array([
            [2, 0, 0],
            [1.5, 1.5, 0]
        ])
        
        values = evaluate_ellipsoid(points, A, center)
        
        self.assertTrue(np.all(values > 1))


class TestEllipsoidContains(unittest.TestCase):
    """Тесты для проверки принадлежности точек эллипсоиду."""
    
    def test_sphere_contains(self):
        """Проверка точек относительно единичной сферы"""
        A = np.eye(3)
        center = np.zeros(3)
        
        points = np.array([
            [0, 0, 0],      # Внутри (центр)
            [0.5, 0, 0],    # Внутри
            [1, 0, 0],      # На поверхности
            [2, 0, 0]       # Снаружи
        ])
        
        result = ellipsoid_contains(points, A, center)
        
        expected = np.array([True, True, True, False])
        np.testing.assert_array_equal(result, expected)
    
    def test_ellipsoid_contains(self):
        """Проверка для эллипсоида"""
        # Эллипсоид с полуосями 2, 1.5, 1
        radii = np.array([2, 1.5, 1])
        A = np.diag(1 / radii**2)
        center = np.zeros(3)
        
        points = np.array([
            [0, 0, 0],          # Центр - внутри
            [2, 0, 0],          # На границе по X
            [0, 1.5, 0],        # На границе по Y
            [0, 0, 1],          # На границе по Z
            [3, 0, 0],          # Снаружи
            [1, 0.5, 0.3]       # Внутри
        ])
        
        result = ellipsoid_contains(points, A, center)
        
        # Первые 4 точки внутри/на границе, 5-я снаружи, 6-я внутри
        self.assertTrue(result[0])
        self.assertTrue(result[1])
        self.assertTrue(result[2])
        self.assertTrue(result[3])
        self.assertFalse(result[4])
        self.assertTrue(result[5])


if __name__ == '__main__':
    unittest.main()
