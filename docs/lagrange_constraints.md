# Связи через множители Лагранжа

## Описание

Добавлена поддержка **точных голономных связей** через **множители Лагранжа** вместо приближенного penalty method.

## Преимущества

### Penalty Method (старый подход)
- ❌ Приближенное выполнение связей (зависит от K)
- ❌ Плохая обусловленность матрицы при больших K
- ❌ Численная неустойчивость
- ❌ Накопление ошибок в динамике

### Lagrange Multipliers (новый подход)
- ✅ **Точное** выполнение связей (машинная точность)
- ✅ Хорошая обусловленность системы
- ✅ Множители = физические силы реакции
- ✅ Лучшее сохранение энергии в динамике

## Новые классы

### `LagrangeConstraint`

Голономная связь вида **C·x = d**.

```python
from termin.fem.assembler import LagrangeConstraint

# Связь: u1 - u2 = 0 (равенство двух переменных)
constraint = LagrangeConstraint(
    variables=[u1, u2],
    coefficients=[
        np.array([[1.0]]),   # коэффициент при u1
        np.array([[-1.0]])   # коэффициент при u2
    ],
    rhs=np.array([0.0])
)

assembler.constraints = [constraint]
assembler.solve_and_set(use_constraints=True)

# Получить силы реакции
lagrange_multipliers = assembler.get_lagrange_multipliers()
```

### `create_revolute_constraint()`

Создает точную шарнирную связь для `RigidBody2D`.

**Связь:** v_cm + ω × r = 0 (скорость точки крепления = 0)

```python
from termin.fem.multibody2d import RigidBody2D, create_revolute_constraint

assembler = MatrixAssembler()

velocity = Variable("v", 2)  # [vx, vy]
omega = Variable("omega", 1)
assembler.variables = [velocity, omega]

# Твердое тело
body = RigidBody2D(velocity, omega, m=1.0, J=0.1)
assembler.contributions = [body]

# Шарнирная связь: фиксирует точку на расстоянии r от центра масс
r = np.array([0.0, 1.0])  # вектор от ЦМ к точке шарнира
constraint = create_revolute_constraint(velocity, omega, r)
assembler.constraints = [constraint]

# Решение с использованием множителей Лагранжа
assembler.solve_and_set(use_constraints=True)

# Силы реакции в шарнире
reaction_forces = assembler.get_lagrange_multipliers()
```

## Математика

### Расширенная система

Вместо **A·x = b** решается:

```
[ A   C^T ] [ x ]   [ b ]
[ C    0  ] [ λ ] = [ d ]
```

где:
- **A** - матрица системы (n_dofs × n_dofs)
- **C** - матрица связей (n_constraints × n_dofs)
- **x** - вектор переменных
- **λ** - множители Лагранжа (силы реакции)
- **b** - правая часть системы
- **d** - правая часть связей (обычно 0)

### Шарнирная связь (revolute joint)

Кинематика: точка крепления неподвижна.

**v_точки = v_cm + ω × r = 0**

В 2D компонентах:
- vx - ω·ry = 0
- vy + ω·rx = 0

Матричная форма:
```
[ 1  0  -ry ] [ vx ]   [ 0 ]
[ 0  1   rx ] [ vy ] = [ 0 ]
              [ ω  ]
```

## Пример: маятник

### С penalty method (старый способ)
```python
# Приближенная связь
revolute = RevoluteJoint2D(velocity, omega, r, K=1e8)
assembler.contributions = [body, revolute, torque]
assembler.solve_and_set()  # обычное решение
# Ошибка энергии: ~15%
```

### С множителями Лагранжа (новый способ)
```python
# Точная связь
constraint = create_revolute_constraint(velocity, omega, r)
assembler.contributions = [body, torque]
assembler.constraints = [constraint]
assembler.solve_and_set(use_constraints=True)
# Ошибка энергии: ~0.016% (в 1000 раз лучше!)
```

## Результаты тестов

### Энергия маятника за 300 шагов (θ₀ = 30°)

| Метод | Ошибка энергии | Размах угла |
|-------|----------------|-------------|
| Penalty (K=1e8) | 15% | 0.5° |
| Lagrange | **0.016%** | **6.1°** |

### Точность связи

| Метод | ||v_точки|| |
|-------|-----------|
| Penalty (K=1e8) | ~1e-6 м/с |
| Lagrange | **~1e-14 м/с** |

## API Changes

### MatrixAssembler

**Новые атрибуты:**
- `constraints: List[LagrangeConstraint]` - список связей

**Новые методы:**
- `assemble_with_constraints()` - сборка расширенной системы
- `get_lagrange_multipliers()` - получить силы реакции

**Изменено:**
- `solve(use_constraints=True)` - использовать множители Лагранжа
- `solve_and_set(use_constraints=True)` - решить с связями

## Совместимость

✅ Все старые тесты проходят (79 passed, 1 skipped)
✅ Обратная совместимость сохранена
✅ Penalty method всё ещё работает (через `RevoluteJoint2D`)

## Ограничения

- Поддерживаются только **линейные голономные связи** (C·x = d)
- Для нелинейных связей нужна линеаризация
- Расширенная система больше по размеру (n_dofs + n_constraints)

## Дальнейшее развитие

Возможные улучшения:
- [ ] Контактные связи (неравенства)
- [ ] Трение в шарнирах
- [ ] Автоматическая стабилизация связей (Baumgarte)
- [ ] Разреженное хранение матриц для больших систем
