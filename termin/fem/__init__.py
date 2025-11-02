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
    Constraint,
    MatrixAssembler,
    BilinearContribution,
    LoadContribution,
    ConstraintContribution,
    LagrangeConstraint,
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

# Многотельная динамика (2D - планарное движение)
from .multibody2d import (
    RotationalInertia2D,
    TorqueSource2D,
    RigidBody2D,
    ForceVector2D,
    RevoluteJoint2D,
    FixedPoint2D,
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
