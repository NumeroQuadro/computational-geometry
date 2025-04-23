import numpy as np
import matplotlib.pyplot as plt

# ====================================
# Задание 1. Линейный отрезок
# ====================================

def bresenham_line(x0, y0, x1, y1):
    """
    Алгоритм Брезенхейма для растрового изображения отрезка.
    Возвращает список координат пикселей, которые должны быть закрашены.
    """
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    x, y = x0, y0
    
    if dx > dy:  # Горизонтальные линии
        err = dx // 2
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:  # Вертикальные линии
        err = dy // 2
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
            
    points.append((x1, y1))
    return points

def natural_line(x0, y0, x1, y1):
    """
    Естественный алгоритм растеризации отрезка по уравнению прямой y = mx + b.
    Возвращает список координат пикселей, которые должны быть закрашены.
    """
    points = []
    
    if x0 == x1:  # Вертикальная линия
        y_start, y_end = min(y0, y1), max(y0, y1)
        for y in range(y_start, y_end + 1):
            points.append((x0, y))
    else:
        m = (y1 - y0) / (x1 - x0)
        b = y0 - m * x0
        x_start, x_end = min(x0, x1), max(x0, x1)
        
        for x in range(x_start, x_end + 1):
            y = m * x + b
            points.append((x, int(round(y))))
            
    return points

def plot_line_comparison(A, B, marker_size=50):
    """
    Параметры:
    - A, B: кортежи (x, y) - начальная и конечная точки отрезка
    - marker_size: размер маркеров для точек (можно менять для наглядности)
    """
    bres_pts = bresenham_line(*A, *B)
    nat_pts = natural_line(*A, *B)
    
    print(f"Алгоритм Брезенхейма создал {len(bres_pts)} пикселей")
    print(f"Естественный алгоритм создал {len(nat_pts)} пикселей")
    
    # Создаем два подграфика (1 строка, 2 столбца)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    # Подграфик 1: Естественный алгоритм
    ax1.scatter(*zip(*nat_pts), marker='s', label='Естественный алгоритм', alpha=0.4, s=marker_size, color='blue')
    ax1.plot([A[0], B[0]], [A[1], B[1]], 'r--', alpha=0.7, label='Исходный отрезок')
    ax1.set_aspect('equal')
    ax1.grid(True, which='both', linestyle=':')
    ax1.set_title('Естественный алгоритм')
    ax1.legend()
    
    # Подграфик 2: Алгоритм Брезенхейма
    ax2.scatter(*zip(*bres_pts), marker='s', label='Алгоритм Брезенхейма', edgecolors='k', s=marker_size, color='green')
    ax2.plot([A[0], B[0]], [A[1], B[1]], 'r--', alpha=0.7, label='Исходный отрезок')
    ax2.set_aspect('equal')
    ax2.grid(True, which='both', linestyle=':')
    ax2.set_title('Алгоритм Брезенхейма')
    ax2.legend()
    
    # Общий заголовок для всей фигуры
    fig.suptitle('Сравнение алгоритмов растеризации отрезка', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Настройка отступов, чтобы заголовок не перекрывался
    plt.show()

# ====================================
# Задание 2. Кривые Безье
# ====================================

# --- 2a. Квадратичная кривая Безье с аффинными преобразованиями ---

def quadratic_bezier(P0, P1, P2, n=200):
    """
    Вычисляет n точек на квадратичной кривой Безье, заданной тремя контрольными точками.
    
    Параметры:
    - P0, P1, P2: контрольные точки (массивы или кортежи (x, y))
    - n: количество точек на кривой (чем больше, тем плавнее кривая)
    """
    t = np.linspace(0, 1, n)
    P0 = np.array(P0)
    P1 = np.array(P1)
    P2 = np.array(P2)
    
    coeff0 = (1 - t)**2
    coeff1 = 2 * (1 - t) * t
    coeff2 = t**2
    
    coeff0 = coeff0[:, None]
    coeff1 = coeff1[:, None]
    coeff2 = coeff2[:, None]
    
    curve = coeff0 * P0 + coeff1 * P1 + coeff2 * P2
    return curve

def affine_matrix_translation(a):
    """Матрица переноса на вектор a"""
    T = np.eye(3)
    T[:2, 2] = a[:2]
    return T

def affine_matrix_shear_x(k):
    """Матрица горизонтального сдвига с коэффициентом k"""
    Sh = np.eye(3)
    Sh[0, 1] = k
    return Sh

def affine_matrix_rotation(theta):
    """Матрица поворота на угол theta вокруг начала координат"""
    R = np.eye(3)
    c, s = np.cos(theta), np.sin(theta)
    R[:2, :2] = [[c, -s],
                 [s,  c]]
    return R

def affine_matrix_scale(s):
    """Матрица однородного масштабирования с коэффициентом s"""
    S = np.eye(3)
    S[0, 0] = S[1, 1] = s
    return S

def draw_quadratic_bezier_with_transforms(P0=None, P1=None, P2=None, n=200, translation_vector=None, shear_k=2, rotation_angle=None, scale_factor=0.8):
    """
    Строит квадратичную кривую Безье и её аффинное преобразование
    
    Параметры (можно менять для тестирования):
    - P0, P1, P2: контрольные точки (по умолчанию: [0, 0], [4, 6], [10, 0])
    - n: количество точек на кривой (по умолчанию 200)
    - translation_vector: вектор переноса (по умолчанию [2, -1])
    - shear_k: коэффициент горизонтального сдвига (по умолчанию 2)
    - rotation_angle: угол поворота в радианах (по умолчанию pi/4)
    - scale_factor: коэффициент масштабирования (по умолчанию 0.8)
    """
    # Установка значений по умолчанию
    P0 = np.array([0, 0, 1]) if P0 is None else np.array([*P0, 1])
    P1 = np.array([4, 6, 1]) if P1 is None else np.array([*P1, 1])
    P2 = np.array([10, 0, 1]) if P2 is None else np.array([*P2, 1])
    translation_vector = np.array([2, -1, 0]) if translation_vector is None else np.array([*translation_vector, 0])
    rotation_angle = np.pi/4 if rotation_angle is None else rotation_angle
    
    orig_curve = quadratic_bezier(P0[:2], P1[:2], P2[:2], n=n)
    
    T = affine_matrix_translation(translation_vector)
    Sh = affine_matrix_shear_x(shear_k)
    R = affine_matrix_rotation(rotation_angle)
    S = affine_matrix_scale(scale_factor)
    
    M = S @ R @ Sh @ T
    
    P0_t = M @ P0
    P1_t = M @ P1
    P2_t = M @ P2
    
    trans_curve = quadratic_bezier(P0_t[:2], P1_t[:2], P2_t[:2], n=n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(orig_curve[:,0], orig_curve[:,1], label='Исходная кривая')
    ax.plot(trans_curve[:,0], trans_curve[:,1], label='Преобразованная кривая')
    
    ax.scatter([P0[0], P1[0], P2[0]], [P0[1], P1[1], P2[1]], 
               marker='o', c='black', s=40, label='Исходные контрольные точки')
    
    ax.scatter([P0_t[0], P1_t[0], P2_t[0]], [P0_t[1], P1_t[1], P2_t[1]], 
               marker='x', c='red', s=50, label='Преобразованные контрольные точки')
    
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':')
    ax.legend()
    ax.set_title('Квадратичная кривая Безье и её аффинное преобразование')
    plt.show()

# --- 2b. Кубическая кривая Безье ---

def cubic_bezier(P0, P1, P2, P3, n=400):
    """
    Вычисляет n точек на кубической кривой Безье, заданной четырьмя контрольными точками.
    
    Параметры:
    - P0, P1, P2, P3: контрольные точки (массивы или кортежи (x, y))
    - n: количество точек на кривой (чем больше, тем плавнее кривая)
    """
    t = np.linspace(0, 1, n)
    P0 = np.array(P0)
    P1 = np.array(P1)
    P2 = np.array(P2)
    P3 = np.array(P3)
    
    coeff0 = (1 - t)**3
    coeff1 = 3 * (1 - t)**2 * t
    coeff2 = 3 * (1 - t) * t**2
    coeff3 = t**3
    
    coeff0 = coeff0[:, None]
    coeff1 = coeff1[:, None]
    coeff2 = coeff2[:, None]
    coeff3 = coeff3[:, None]
    
    curve = coeff0 * P0 + coeff1 * P1 + coeff2 * P2 + coeff3 * P3
    return curve

def draw_cubic_bezier(P0=None, P1=None, P2=None, P3=None, n=400):
    """
    Строит кубическую кривую Безье
    
    Параметры:
    - P0, P1, P2, P3: контрольные точки (по умолчанию: [0, 0], [2, 5], [5, 5], [7, 0])
    - n: количество точек на кривой (по умолчанию 400)
    """
    # Установка значений по умолчанию
    P0 = np.array([0, 0]) if P0 is None else np.array(P0)
    P1 = np.array([2, 5]) if P1 is None else np.array(P1)
    P2 = np.array([5, 5]) if P2 is None else np.array(P2)
    P3 = np.array([7, 0]) if P3 is None else np.array(P3)
    
    curve = cubic_bezier(P0, P1, P2, P3, n=n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(curve[:,0], curve[:,1], label='Кубическая кривая Безье')
    
    control_points = [P0, P1, P2, P3]
    colors = ['red', 'green', 'blue', 'purple']
    labels = ['P0', 'P1', 'P2', 'P3']
    
    for i, (p, c, l) in enumerate(zip(control_points, colors, labels)):
        ax.scatter(p[0], p[1], marker='o', c=c, s=50, label=f'Точка {l}')
        
    ax.plot([p[0] for p in control_points], [p[1] for p in control_points], 
            'k--', alpha=0.5, label='Многоугольник управления')
    
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':')
    ax.legend()
    ax.set_title('Кубическая кривая Безье')
    plt.show()

# --- 2c. Сплайн из нескольких сегментов кривых Безье ---

def draw_bezier_spline(segments=None, n=400):
    """
    Строит сплайн из нескольких кубических кривых Безье
    
    Параметры:
    - segments: список сегментов, каждый сегмент - кортеж из 4 контрольных точек
      (по умолчанию: 3 сегмента с заданными точками)
    - n: количество точек на каждом сегменте (по умолчанию 400)
    """
    # Установка значений по умолчанию для сегментов
    if segments is None:
        segments = [
            (np.array([0, 0]), np.array([2, 4]), np.array([4, 4]), np.array([6, 0])),
            (np.array([6, 0]), np.array([8, -4]), np.array([10, -4]), np.array([12, 0])),
            (np.array([12, 0]), np.array([14, 4]), np.array([16, 4]), np.array([18, 0])),
        ]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for i, (P0, P1, P2, P3) in enumerate(segments):
        seg_curve = cubic_bezier(P0, P1, P2, P3, n=n)
        ax.plot(seg_curve[:,0], seg_curve[:,1], label=f'Сегмент {i+1}')
        
        ax.scatter([P0[0], P1[0], P2[0], P3[0]], [P0[1], P1[1], P2[1], P3[1]], s=30, alpha=0.7)
        
        ax.plot([P0[0], P1[0], P2[0], P3[0]], [P0[1], P1[1], P2[1], P3[1]], 
                'k--', alpha=0.3)

    ax.set_aspect('equal')
    ax.grid(True, linestyle=':')
    ax.legend()
    ax.set_title('Сплайн из кубических кривых Безье (C⁰-непрерывный)')
    plt.show()

# Функции для демонстрации выполнения всех заданий

def demo_task1(A=None, B=None, marker_size=50):
    """
    Демонстрация задания 1 - растеризация отрезка
    
    Параметры:
    - A, B: начальная и конечная точки отрезка (по умолчанию: (-10, -7), (5, 23))
    - marker_size: размер маркеров для точек (по умолчанию 50)
    """
    A = (-10, -7) if A is None else A
    B = (5, 23) if B is None else B
    plot_line_comparison(A, B, marker_size=marker_size)

def demo_task2a():
    """
    Демонстрация задания 2a - квадратичная кривая с преобразованиями
    """
    draw_quadratic_bezier_with_transforms(
        P0=[0, 0],
        P1=[4, 6],
        P2=[10, 0],
        n=200,
        translation_vector=[2, -1],
        shear_k=2,
        rotation_angle=np.pi/4,
        scale_factor=0.8
    )

def demo_task2b():
    """
    Демонстрация задания 2b - кубическая кривая Безье
    """
    draw_cubic_bezier(
        P0=[0, 0],
        P1=[5, 5], #[2,5]
        P2=[2, 5], # [5,5]
        P3=[7, 0],
        n=400
    )

def demo_task2c():
    """
    Демонстрация задания 2c - сплайн из нескольких кривых Безье
    """
    custom_segments = [
        (np.array([0, 0]), np.array([2, 4]), np.array([4, 4]), np.array([6, 0])),
        (np.array([6, 0]), np.array([8, -4]), np.array([10, -4]), np.array([12, 0])),
        (np.array([12, 0]), np.array([14, 4]), np.array([16, 4]), np.array([18, 0])),
    ]
    draw_bezier_spline(segments=custom_segments, n=400)

# Запуск всех демонстраций
if __name__ == "__main__":
    print("Запуск всех демонстраций...")
    demo_task1()     # Задание 1
    demo_task2a()    # Задание 2a
    demo_task2b()    # Задание 2b
    demo_task2c()    # Задание 2c