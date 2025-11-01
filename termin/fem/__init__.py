"""
FEM (Finite Element Method) модуль для мультифизического моделирования.

Содержит инструменты для:
- Сборки и решения систем линейных уравнений методом конечных элементов
- Моделирования механических систем (стержни, балки, треугольные элементы)
- Моделирования электрических цепей (резисторы, конденсаторы, индуктивности)
- Моделирования многотельной динамики (инерции, пружины, демпферы)
- Моделирования электромеханических систем (двигатели постоянного тока)
"""

# Базовые классы для сборки систем
from .assembler import (
    Variable,
    Contribution,
    MatrixAssembler,
    BilinearContribution,
    LoadContribution,
    ConstraintContribution,
)

# Механические элементы
from .mechanic import (
    BarElement,
    BeamElement2D,
    DistributedLoad,
    Triangle3Node,
    BodyForce,
)

# Электрические элементы
from .electrical import (
    Resistor,
    Capacitor,
    Inductor,
    VoltageSource,
    CurrentSource,
    Ground,
)

# Многотельная динамика
from .multibody import (
    RotationalInertia,
    TorqueSource,
    RotationalSpring,
    RotationalDamper,
    FixedRotation,
    LinearMass,
    ForceSource,
    LinearSpring,
)

# Электромеханика
from .electromechanical import (
    DCMotor,
)

__all__ = [
    # Assembler
    'Variable',
    'Contribution',
    'MatrixAssembler',
    'BilinearContribution',
    'LoadContribution',
    'ConstraintContribution',
    
    # Mechanic
    'BarElement',
    'BeamElement2D',
    'DistributedLoad',
    'Triangle3Node',
    'BodyForce',
    
    # Electrical
    'Resistor',
    'Capacitor',
    'Inductor',
    'VoltageSource',
    'CurrentSource',
    'Ground',
    
    # Multibody
    'RotationalInertia',
    'TorqueSource',
    'RotationalSpring',
    'RotationalDamper',
    'FixedRotation',
    'LinearMass',
    'ForceSource',
    'LinearSpring',
    
    # Electromechanical
    'DCMotor',
]
