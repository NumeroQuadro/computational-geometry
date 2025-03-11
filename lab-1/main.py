import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def plot_triangle(ax, points, color='blue', alpha=0.5, label=None, marker='o'):
    """Отображение треугольника"""
    triangle = Polygon(points, closed=True, alpha=alpha, color=color, label=label)
    ax.add_patch(triangle)
    ax.plot(points[:, 0], points[:, 1], marker, color=color)
    
def plot_polygon(ax, points, color='blue', alpha=0.5, label=None, marker='o'):
    """Отображение многоугольника"""
    polygon = Polygon(points, closed=True, alpha=alpha, color=color, label=label)
    ax.add_patch(polygon)
    ax.plot(points[:, 0], points[:, 1], marker, color=color)
    # Добавляем метки вершин
    for i, point in enumerate(points):
        ax.text(point[0], point[1], f'{i+1}', fontsize=10)

def apply_transform(points, transform_matrix):
    """Применение матрицы преобразования к точкам"""
    # Конвертация в однородные координаты
    homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
    # Применение преобразования
    transformed_homogeneous = np.dot(homogeneous_points, transform_matrix.T)
    # Конвертация обратно в декартовы координаты
    return transformed_homogeneous[:, :2]

# ----- Задание 1: Преобразование треугольника -----
def triangle_transformations():
    print("Задание 1: Преобразование треугольника")
    
    # a) Задаем треугольник декартовыми координатами
    triangle_points = np.array([
        [0, 0],    # Точка A
        [4, 0],    # Точка B
        [2, 3]     # Точка C
    ])
    
    # Матрица однородных координат вершин треугольника
    homogeneous_coords = np.hstack((triangle_points, np.ones((3, 1))))
    print("a) Матрица однородных координат вершин треугольника:")
    print(homogeneous_coords)
    
    # b) Задаем параметры преобразований
    # Вычисляем центр треугольника (центроид)
    center = np.mean(triangle_points, axis=0)
    
    # Находим наименьшую сторону и её середину
    sides = [
        np.linalg.norm(triangle_points[1] - triangle_points[0]),  # AB
        np.linalg.norm(triangle_points[2] - triangle_points[1]),  # BC
        np.linalg.norm(triangle_points[0] - triangle_points[2])   # CA
    ]
    min_side_idx = np.argmin(sides)
    
    # Определяем индексы вершин для наименьшей стороны
    if min_side_idx == 0:  # AB
        idx1, idx2 = 0, 1
    elif min_side_idx == 1:  # BC
        idx1, idx2 = 1, 2
    else:  # CA
        idx1, idx2 = 2, 0
        
    # Середина наименьшей стороны
    midpoint_min_side = (triangle_points[idx1] + triangle_points[idx2]) / 2
    
    # Параметры преобразований
    translation_vector = np.array([2, 1])  # Вектор для переноса
    rotation_angle = np.radians(60)  # Угол поворота 60 градусов
    line_point = np.array([0, 0])  # Точка на прямой l (ось X)
    line_direction = np.array([1, 0])  # Направление прямой l (ось X)
    scale_factor_origin = 1.5  # Коэффициент для гомотетии относительно начала координат
    scale_factor_midpoint = 2  # Коэффициент для гомотетии относительно середины наименьшей стороны
    
    print(f"b) Параметры преобразований:")
    print(f"   - Вектор переноса: {translation_vector}")
    print(f"   - Угол поворота: {np.degrees(rotation_angle)} градусов")
    print(f"   - Прямая для осевой симметрии: точка {line_point}, направление {line_direction}")
    print(f"   - Коэффициент гомотетии H_O^k: {scale_factor_origin}")
    print(f"   - Центр треугольника: {center}")
    print(f"   - Наименьшая сторона: {min_side_idx+1}-я сторона, длина {sides[min_side_idx]:.2f}")
    print(f"   - Середина наименьшей стороны: {midpoint_min_side}")
    print(f"   - Коэффициент гомотетии H_M^m: {scale_factor_midpoint}")
    
    # Матрицы преобразований в однородных координатах
    
    # 1. Перенос T_a
    translation_matrix = np.array([
        [1, 0, translation_vector[0]],
        [0, 1, translation_vector[1]],
        [0, 0, 1]
    ])
    
    # 2. Поворот R_C^φ относительно центра треугольника
    # Сначала переносим центр в начало координат, затем поворачиваем, затем возвращаем обратно
    to_origin = np.array([
        [1, 0, -center[0]],
        [0, 1, -center[1]],
        [0, 0, 1]
    ])
    
    rotation = np.array([
        [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
        [np.sin(rotation_angle), np.cos(rotation_angle), 0],
        [0, 0, 1]
    ])
    
    from_origin = np.array([
        [1, 0, center[0]],
        [0, 1, center[1]],
        [0, 0, 1]
    ])
    
    rotation_matrix = from_origin @ rotation @ to_origin
    
    # 3. Осевая симметрия S_l относительно оси X
    reflection_matrix = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ])
    
    # 4. Гомотетия H_O^k относительно начала координат
    scaling_matrix_origin = np.array([
        [scale_factor_origin, 0, 0],
        [0, scale_factor_origin, 0],
        [0, 0, 1]
    ])
    
    # 5. Гомотетия H_M^m относительно середины наименьшей стороны
    to_midpoint = np.array([
        [1, 0, -midpoint_min_side[0]],
        [0, 1, -midpoint_min_side[1]],
        [0, 0, 1]
    ])
    
    scaling = np.array([
        [scale_factor_midpoint, 0, 0],
        [0, scale_factor_midpoint, 0],
        [0, 0, 1]
    ])
    
    from_midpoint = np.array([
        [1, 0, midpoint_min_side[0]],
        [0, 1, midpoint_min_side[1]],
        [0, 0, 1]
    ])
    
    scaling_matrix_midpoint = from_midpoint @ scaling @ to_midpoint
    
    # 6. Композиция H_M^m ∘ R_M^π (гомотетия относительно середины наименьшей стороны и поворот на 180°)
    # Поворот на 180 градусов
    rotation_180 = np.array([
        [np.cos(np.pi), -np.sin(np.pi), 0],
        [np.sin(np.pi), np.cos(np.pi), 0],
        [0, 0, 1]
    ])
    
    # Поворот относительно середины наименьшей стороны
    rotation_matrix_midpoint = from_midpoint @ rotation_180 @ to_midpoint
    
    # Композиция
    composite_matrix = scaling_matrix_midpoint @ rotation_matrix_midpoint
    
    print("\nМатрицы преобразований в однородных координатах:")
    print("1. Перенос T_a:")
    print(translation_matrix)
    print("\n2. Поворот R_C^φ относительно центра треугольника:")
    print(rotation_matrix)
    print("\n3. Осевая симметрия S_l относительно оси X:")
    print(reflection_matrix)
    print("\n4. Гомотетия H_O^k относительно начала координат:")
    print(scaling_matrix_origin)
    print("\n5. Гомотетия H_M^m относительно середины наименьшей стороны:")
    print(scaling_matrix_midpoint)
    print("\n6. Композиция H_M^m ∘ R_M^π:")
    print(composite_matrix)
    
    # Применяем преобразования и находим образы треугольника
    translated_points = apply_transform(triangle_points, translation_matrix)
    rotated_points = apply_transform(triangle_points, rotation_matrix)
    reflected_points = apply_transform(triangle_points, reflection_matrix)
    scaled_points_origin = apply_transform(triangle_points, scaling_matrix_origin)
    composite_points = apply_transform(triangle_points, composite_matrix)
    
    print("\nКоординаты образов треугольника:")
    print("Исходный треугольник:")
    print(triangle_points)
    print("\nПосле переноса T_a:")
    print(translated_points)
    print("\nПосле поворота R_C^φ:")
    print(rotated_points)
    print("\nПосле осевой симметрии S_l:")
    print(reflected_points)
    print("\nПосле гомотетии H_O^k:")
    print(scaled_points_origin)
    print("\nПосле композиции H_M^m ∘ R_M^π:")
    print(composite_points)
    
    # c) Визуализация
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Отображаем исходный треугольник
    plot_triangle(ax, triangle_points, color='blue', label='Исходный треугольник')
    
    # Отображаем образы треугольника
    plot_triangle(ax, translated_points, color='red', label='После переноса T_a')
    plot_triangle(ax, rotated_points, color='green', label='После поворота R_C^φ')
    plot_triangle(ax, reflected_points, color='purple', label='После симметрии S_l')
    plot_triangle(ax, scaled_points_origin, color='orange', label='После гомотетии H_O^k')
    plot_triangle(ax, composite_points, color='cyan', label='После H_M^m ∘ R_M^π')
    
    # Отмечаем особые точки
    ax.plot(center[0], center[1], 'ro', markersize=8, label='Центр треугольника C')
    ax.plot(midpoint_min_side[0], midpoint_min_side[1], 'go', markersize=8, label='Середина мин. стороны M')
    ax.plot(0, 0, 'ko', markersize=8, label='Начало координат O')
    
    # Настройка графика
    ax.set_xlim(-10, 10)
    ax.set_ylim(-6, 6)
    ax.grid(True)
    ax.set_aspect('equal')
    ax.set_title('Преобразования треугольника')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.savefig('triangle_transformations.png')
    
    return {
        "triangle_points": triangle_points,
        "homogeneous_coords": homogeneous_coords,
        "center": center,
        "midpoint_min_side": midpoint_min_side,
        "translated_points": translated_points,
        "rotated_points": rotated_points,
        "reflected_points": reflected_points,
        "scaled_points_origin": scaled_points_origin,
        "composite_points": composite_points,
        "matrices": {
            "translation": translation_matrix,
            "rotation": rotation_matrix,
            "reflection": reflection_matrix,
            "scaling_origin": scaling_matrix_origin,
            "scaling_midpoint": scaling_matrix_midpoint,
            "composite": composite_matrix
        }
    }

# ----- Задание 2: Преобразование квадрата -----
def square_transformation():
    print("\n\nЗадание 2: Преобразование квадрата")
    
    # Задаем квадрат KLMN
    square_points = np.array([
        [0, 0],  # K - нижний левый
        [2, 0],  # L - нижний правый
        [2, 2],  # M - верхний правый
        [0, 2]   # N - верхний левый
    ])
    
    # Матрица однородных координат вершин квадрата
    square_homogeneous = np.hstack((square_points, np.ones((4, 1))))
    print("Матрица однородных координат вершин квадрата KLMN:")
    print(square_homogeneous)
    
    # a) Найдем аффинное преобразование по заданным условиям
    
    # Вершина A лежит на луче KM и находится в 3 раза дальше от K, чем точка M
    # Значит A = K + 3*(M-K) = K + 3*[2,2] = [0,0] + 3*[2,2] = [6,6]
    A = np.array([6, 6])
    
    # Образ стороны KN (вертикальной) параллелен ей и его длина в 2 раза больше KN
    # Значит длина B-A = 2*длина N-K = 2*2 = 4
    # И поскольку она параллельна KN, то B = A + [0,-4] = [6,2]
    B = np.array([6, 2])
    
    # Угол параллелограмма при вершине A равен π/3
    # Это угол между векторами AB и AD
    # AB = [0,-4], AD нужно найти
    # Угол π/3 между AB и AD означает, что угол между осью x и AD равен π/3
    # Значит D = A + [длина_AD*cos(π/3), длина_AD*sin(π/3)]
    
    # Высота параллелограмма BH, H ∈ AD равна двум сторонам квадрата, т.е. 4
    # Высота BH перпендикулярна AD и равна 4
    # Если |AD| = x, то 4 = x*sin(π/3) => x = 4/sin(π/3) = 4/(sqrt(3)/2) = 8/sqrt(3)
    
    angle_A = np.pi/3
    length_AD = 4 / np.sin(angle_A)
    D = A + np.array([length_AD * np.cos(angle_A), length_AD * np.sin(angle_A)])
    
    # C = B + (D - A) для замыкания параллелограмма
    C = B + (D - A)
    
    # Координаты параллелограмма ABCD
    parallelogram_points = np.array([A, B, C, D])
    
    print("\na) Найденный параллелограмм ABCD:")
    print(f"A = {A}")
    print(f"B = {B}")
    print(f"C = {C}")
    print(f"D = {D}")
    
    # b) Составляем матрицу преобразования F
    # Нам нужно найти матрицу F такую, что:
    # F(K) = A, F(L) = B, F(M) = C, F(N) = D
    
    # Для аффинного преобразования F в форме матрицы 3x3 в однородных координатах
    # нам нужно решить систему линейных уравнений:
    # F * [K; 1] = [A; 1]
    # F * [L; 1] = [B; 1]
    # F * [M; 1] = [C; 1]
    
    # Мы можем использовать только 3 точки, так как 4-я определяется однозначно
    # Формируем матрицы для решения системы AX = B
    X = np.column_stack((square_points[:3], np.ones(3)))
    Y = np.column_stack((parallelogram_points[:3], np.ones(3)))
    
    # Решаем систему уравнений для нахождения матрицы преобразования
    F_matrix = np.linalg.solve(X, Y)
    
    # Транспонируем, чтобы получить матрицу 3x3
    F_matrix = F_matrix.T
    
    print("\nb) Матрица преобразования F в однородных координатах:")
    print(F_matrix)
    
    # Находим матрицу обратного преобразования
    F_inverse = np.linalg.inv(F_matrix)
    print("\nМатрица обратного преобразования F^(-1):")
    print(F_inverse)
    
    # c) Находим матрицы однородных координат образов
    # Применяем преобразование F к квадрату KLMN
    F_KLMN = apply_transform(square_points, F_matrix)
    
    # Применяем обратное преобразование F^(-1) к параллелограмму ABCD
    F_inv_ABCD = apply_transform(parallelogram_points, F_inverse)
    
    print("\nc) Матрица однородных координат образа F(KLMN):")
    print(np.hstack((F_KLMN, np.ones((4, 1)))))
    print("\nМатрица однородных координат образа F^(-1)(ABCD):")
    print(np.hstack((F_inv_ABCD, np.ones((4, 1)))))
    
    # d) Визуализация
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Отображаем квадрат KLMN
    plot_polygon(ax, square_points, color='blue', alpha=0.6, label='Квадрат KLMN')
    
    # Отображаем параллелограмм ABCD
    plot_polygon(ax, parallelogram_points, color='red', alpha=0.6, label='Параллелограмм ABCD = F(KLMN)')
    
    # Отображаем образ F^(-1)(ABCD)
    plot_polygon(ax, F_inv_ABCD, color='green', alpha=0.4, label='F^(-1)(ABCD)')
    
    # Проверяем, что образы соответствуют оригиналам
    error_F_KLMN = np.linalg.norm(F_KLMN - parallelogram_points)
    error_F_inv_ABCD = np.linalg.norm(F_inv_ABCD - square_points)
    
    print(f"\ne) Проверка F(KLMN) = ABCD:")
    print(f"Ошибка: {error_F_KLMN}")
    print(f"Проверка F^(-1)(ABCD) = KLMN:")
    print(f"Ошибка: {error_F_inv_ABCD}")
    
    # Настройка графика
    ax.set_xlim(-1, 10)
    ax.set_ylim(-1, 8)
    ax.grid(True)
    ax.set_aspect('equal')
    ax.set_title('Преобразование квадрата в параллелограмм')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('square_transformation.png')
    
    return {
        "square_points": square_points,
        "parallelogram_points": parallelogram_points,
        "F_matrix": F_matrix,
        "F_inverse": F_inverse,
        "F_KLMN": F_KLMN,
        "F_inv_ABCD": F_inv_ABCD
    }

# Выполнить все задачи
if __name__ == "__main__":
    print("Лабораторная №1. Аффинные преобразования. Однородные координаты.")
    
    triangle_results = triangle_transformations()
    square_results = square_transformation()
    
    plt.show()