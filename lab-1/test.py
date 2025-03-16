import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Вспомогательные функции для визуализации
def plot_triangle(ax, points, color='blue', alpha=0.5, label=None):
    triangle = Polygon(points, closed=True, alpha=alpha, color=color, label=label)
    ax.add_patch(triangle)
    
def plot_quadrilateral(ax, points, color='blue', alpha=0.5, label=None):
    quad = Polygon(points, closed=True, alpha=alpha, color=color, label=label)
    ax.add_patch(quad)

def apply_transform(points, transform_matrix):
    """Применяет матрицу преобразования к точкам"""
    # Конвертация в однородные координаты
    homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
    # Применение преобразования
    transformed_homogeneous = np.dot(homogeneous_points, transform_matrix.T)
    # Конвертация обратно в декартовы координаты
    return transformed_homogeneous[:, :2]

# ----- Задача 1: Преобразование треугольника -----
def triangle_transformations():
    # 1. Задаем треугольник
    triangle_points = np.array([
        [0, 0],    # Точка A
        [4, 0],    # Точка B
        [2, 3]     # Точка C
    ])
    
    # Матрица однородных координат (добавляем 1 для каждой точки)
    homogeneous_coords = np.hstack((triangle_points, np.ones((3, 1))))
    
    # Создаем фигуру для отображения
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Отображаем исходный треугольник на всех графиках
    for ax in axes:
        plot_triangle(ax, triangle_points, color='blue', label='Исходный')
        ax.set_xlim(-6, 8)
        ax.set_ylim(-6, 8)
        ax.grid(True)
        ax.set_aspect('equal')
        ax.plot(triangle_points[:, 0], triangle_points[:, 1], 'bo')  # точки треугольника
    
    # а. Перенос T_vec{a} - перенос на вектор (2, 2)
    translation_vector = np.array([2, 2])
    translation_matrix = np.array([
        [1, 0, translation_vector[0]],
        [0, 1, translation_vector[1]],
        [0, 0, 1]
    ])
    
    translated_points = apply_transform(triangle_points, translation_matrix)
    plot_triangle(axes[0], translated_points, color='red', alpha=0.5, label='Перенос')
    axes[0].set_title('Перенос на вектор (2, 2)')
    axes[0].plot(translated_points[:, 0], translated_points[:, 1], 'ro')
    
    # б. Поворот R_0^ф - поворот на 45 градусов вокруг начала координат
    angle = np.radians(45)
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    
    rotated_points = apply_transform(triangle_points, rotation_matrix)
    plot_triangle(axes[1], rotated_points, color='green', alpha=0.5, label='Поворот')
    axes[1].set_title('Поворот на 45° вокруг (0,0)')
    axes[1].plot(rotated_points[:, 0], rotated_points[:, 1], 'go')
    
    # в. Отражение S_l - отражение относительно оси x
    reflection_matrix = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ])
    
    reflected_points = apply_transform(triangle_points, reflection_matrix)
    plot_triangle(axes[2], reflected_points, color='purple', alpha=0.5, label='Отражение')
    axes[2].set_title('Отражение относительно оси X')
    axes[2].plot(reflected_points[:, 0], reflected_points[:, 1], 'mo')
    
    # г. Гомотетия H_0^k - масштабирование с коэффициентом 1.5 относительно начала координат
    scale_factor = 1.5
    scaling_matrix = np.array([
        [scale_factor, 0, 0],
        [0, scale_factor, 0],
        [0, 0, 1]
    ])
    
    scaled_points = apply_transform(triangle_points, scaling_matrix)
    plot_triangle(axes[3], scaled_points, color='orange', alpha=0.5, label='Гомотетия')
    axes[3].set_title(f'Гомотетия с k={scale_factor} относительно (0,0)')
    axes[3].plot(scaled_points[:, 0], scaled_points[:, 1], 'yo')
    
    # д. Композиция гомотетии и поворота на 180гр
    # Найдем середину стороны AB
    midpoint = (triangle_points[0] + triangle_points[1]) / 2
    
    # Гомотетия относительно середины стороны AB с коэффициентом 1.5
    # 1. Сдвиг центра в начало координат
    to_origin = np.array([
        [1, 0, -midpoint[0]],
        [0, 1, -midpoint[1]],
        [0, 0, 1]
    ])
    
    # 2. Масштабирование
    scale = np.array([
        [scale_factor, 0, 0],
        [0, scale_factor, 0],
        [0, 0, 1]
    ])
    
    # 3. Возврат обратно
    from_origin = np.array([
        [1, 0, midpoint[0]],
        [0, 1, midpoint[1]],
        [0, 0, 1]
    ])
    
    # 4. Поворот на 180 градусов относительно начала координат
    angle_180 = np.radians(180)
    rotation_180 = np.array([
        [np.cos(angle_180), -np.sin(angle_180), 0],
        [np.sin(angle_180), np.cos(angle_180), 0],
        [0, 0, 1]
    ])
    
    # Применяем композицию преобразований
    composite_matrix = from_origin @ scale @ to_origin @ rotation_180
    
    composite_points = apply_transform(triangle_points, composite_matrix)
    plot_triangle(axes[4], composite_points, color='cyan', alpha=0.5, label='Композиция')
    axes[4].set_title('Гомотетия + поворот на 180°')
    axes[4].plot(composite_points[:, 0], composite_points[:, 1], 'co')
    axes[4].plot(midpoint[0], midpoint[1], 'rx', markersize=10, label='Центр гомотетии')
    
    # Выводим сводный график
    for ax in axes:
        ax.legend()
    
    axes[5].axis('off')  # Не используем последний график
    plt.tight_layout()
    plt.savefig('triangle_transformations.png')
    plt.show()
    
    return {
        "original": triangle_points,
        "homogeneous": homogeneous_coords,
        "translated": translated_points,
        "rotated": rotated_points,
        "reflected": reflected_points,
        "scaled": scaled_points,
        "composite": composite_points,
        "matrices": {
            "translation": translation_matrix,
            "rotation": rotation_matrix,
            "reflection": reflection_matrix,
            "scaling": scaling_matrix,
            "composite": composite_matrix
        }
    }

# ----- Задача 2: Преобразование квадрата -----
def square_to_parallelogram():
    # 1. Задаем квадрат
    square_points = np.array([
        [0, 0],  # Нижний левый
        [2, 0],  # Нижний правый
        [2, 2],  # Верхний правый
        [0, 2]   # Верхний левый
    ])
    
    # Матрица однородных координат
    square_homogeneous = np.hstack((square_points, np.ones((4, 1))))
    
    # 2. Преобразование квадрата в параллелограмм
    # Сдвиг вправо и растяжение по x
    transform_matrix = np.array([
        [1.5, 0.5, 4],  # Сжатие по x, скос и сдвиг по x
        [0, 1, 3],      # Сдвиг по y
        [0, 0, 1]
    ])
    
    # Применяем преобразование
    parallelogram_points = apply_transform(square_points, transform_matrix)
    
    # 3. Находим обратное преобразование
    inverse_matrix = np.linalg.inv(transform_matrix)
    
    # Проверяем обратное преобразование
    restored_points = apply_transform(parallelogram_points, inverse_matrix)
    
    # Визуализация
    fig, ax = plt.subplots(figsize=(10, 8))
    
    plot_quadrilateral(ax, square_points, color='blue', alpha=0.5, label='Квадрат')
    plot_quadrilateral(ax, parallelogram_points, color='red', alpha=0.5, label='Параллелограмм')
    plot_quadrilateral(ax, restored_points, color='green', alpha=0.3, label='Восстановленный квадрат')
    
    ax.plot(square_points[:, 0], square_points[:, 1], 'bo')
    ax.plot(parallelogram_points[:, 0], parallelogram_points[:, 1], 'ro')
    
    ax.set_xlim(-1, 10)
    ax.set_ylim(-1, 8)
    ax.grid(True)
    ax.set_aspect('equal')
    ax.set_title('Преобразование квадрата в параллелограмм')
    ax.legend()
    
    plt.savefig('square_to_parallelogram.png')
    plt.show()
    
    print("Матрица преобразования:")
    print(transform_matrix)
    print("\nМатрица обратного преобразования:")
    print(inverse_matrix)
    
    return {
        "square": square_points,
        "parallelogram": parallelogram_points,
        "restored_square": restored_points,
        "transform_matrix": transform_matrix,
        "inverse_matrix": inverse_matrix
    }

# Выполнить все задачи
if __name__ == "__main__":
    print("Задача 1: Преобразование треугольника")
    triangle_results = triangle_transformations()
    
    print("\nЗадача 2: Преобразование квадрата")
    square_results = square_to_parallelogram()