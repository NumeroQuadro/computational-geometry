import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import sympy as sp
from sympy import symbols, solve, diff, sqrt, cosh, sinh

# ************************************
# исправления
# 1. показать где подсчитывается полярный радиус
# 2. поменять точки относительно которых подсчитывается
# ************************************

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12

def task1():
    print("Задание 1: Конические сечения")
    
    # Параметры для эллипса и гиперболы
    a, b = 4, 2  # полуоси для обеих кривых
    t_max_hyperbola = 2  # ограничение параметра t для гиперболы

    # a) Параметрические уравнения кривых
    print("\na) Параметрические уравнения кривых:")
    # Эллипс
    print(f"Эллипс: x = {a}*cos(t), y = {b}*sin(t), t ∈ [0; 2π]")
    # Гипербола
    print(f"Гипербола: x = {a}*cosh(t), y = {b}*sinh(t), t ∈ [-{t_max_hyperbola}; {t_max_hyperbola}]")

    # b) Уравнения касательных в заданной точке
    # Для эллипса
    t0_ellipse = np.pi/4
    x0_ellipse = a * np.cos(t0_ellipse)
    y0_ellipse = b * np.sin(t0_ellipse)
    dx_dt_ellipse = -a * np.sin(t0_ellipse)
    dy_dt_ellipse = b * np.cos(t0_ellipse)
    if abs(dx_dt_ellipse) > 1e-10:
        k_ellipse = dy_dt_ellipse / dx_dt_ellipse
        b_ellipse = y0_ellipse - k_ellipse * x0_ellipse
        print(f"\nb) Уравнение касательной к эллипсу в точке ({x0_ellipse:.2f}, {y0_ellipse:.2f}):")
        print(f"y = {k_ellipse:.4f}*x + {b_ellipse:.4f}")
    else:
        print(f"\nb) Уравнение касательной к эллипсу в точке ({x0_ellipse:.2f}, {y0_ellipse:.2f}):")
        print(f"x = {x0_ellipse:.4f}")

    # Для гиперболы
    t0_hyperbola = 0.5
    x0_hyperbola = a * np.cosh(t0_hyperbola)
    y0_hyperbola = b * np.sinh(t0_hyperbola)
    dx_dt_hyperbola = a * np.sinh(t0_hyperbola)
    dy_dt_hyperbola = b * np.cosh(t0_hyperbola)
    if abs(dx_dt_hyperbola) > 1e-10:
        k_hyperbola = dy_dt_hyperbola / dx_dt_hyperbola
        b_hyperbola = y0_hyperbola - k_hyperbola * x0_hyperbola
        print(f"Уравнение касательной к гиперболе в точке ({x0_hyperbola:.2f}, {y0_hyperbola:.2f}):")
        print(f"y = {k_hyperbola:.4f}*x + {b_hyperbola:.4f}")
    else:
        print(f"Уравнение касательной к гиперболе в точке ({x0_hyperbola:.2f}, {y0_hyperbola:.2f}):")
        print(f"x = {x0_hyperbola:.4f}")

    # c) Разбиение промежутков на n участков
    n = 10
    # Эллипс: [0; 2π]
    t_points_ellipse = np.linspace(0, 2*np.pi, n+1)[:-1]
    # Гипербола: [-t_max; t_max]
    t_points_hyperbola = np.linspace(-t_max_hyperbola, t_max_hyperbola, n+1)

    print(f"\nc) Разбиение промежутков на {n} участков:")
    print("Эллипс:")
    for i in range(n):
        interval_start = t_points_ellipse[i]
        interval_end = t_points_ellipse[(i+1)%n]
        print(f"Участок {i+1}: [{interval_start:.2f}; {interval_end:.2f}]")
    print("Гипербола:")
    for i in range(n):
        interval_start = t_points_hyperbola[i]
        interval_end = t_points_hyperbola[i+1]
        print(f"Участок {i+1}: [{interval_start:.2f}; {interval_end:.2f}]")

    # d) Построение многоугольников из касательных
    # Функция для нахождения точки пересечения двух прямых
    def find_intersection(p1, v1, p2, v2):
        A = np.array([v1, [-v for v in v2]]).T
        b = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        try:
            t1, t2 = np.linalg.solve(A, b)
            return [p1[0] + t1*v1[0], p1[1] + t1*v1[1]]
        except np.linalg.LinAlgError:
            print(f"Пересечение не найдено для касательных")
            return None

    # Эллипс
    tangent_points_x_ellipse = [a * np.cos(t) for t in t_points_ellipse]
    tangent_points_y_ellipse = [b * np.sin(t) for t in t_points_ellipse]
    dx_dt_values_ellipse = [-a * np.sin(t) for t in t_points_ellipse]
    dy_dt_values_ellipse = [b * np.cos(t) for t in t_points_ellipse]

    polygon_points_ellipse = []
    for i in range(n):
        p1 = [tangent_points_x_ellipse[i], tangent_points_y_ellipse[i]]
        v1 = [dx_dt_values_ellipse[i], dy_dt_values_ellipse[i]]
        p2 = [tangent_points_x_ellipse[(i+1)%n], tangent_points_y_ellipse[(i+1)%n]]
        v2 = [dx_dt_values_ellipse[(i+1)%n], dy_dt_values_ellipse[(i+1)%n]]
        intersection = find_intersection(p1, v1, p2, v2)
        if intersection:
            polygon_points_ellipse.append(intersection)

    # Гипербола
    tangent_points_x_hyperbola = [a * np.cosh(t) for t in t_points_hyperbola]
    tangent_points_y_hyperbola = [b * np.sinh(t) for t in t_points_hyperbola]
    dx_dt_values_hyperbola = [a * np.sinh(t) for t in t_points_hyperbola]
    dy_dt_values_hyperbola = [b * np.cosh(t) for t in t_points_hyperbola]

    polygon_points_hyperbola = []
    for i in range(n):
        p1 = [tangent_points_x_hyperbola[i], tangent_points_y_hyperbola[i]]
        v1 = [dx_dt_values_hyperbola[i], dy_dt_values_hyperbola[i]]
        p2 = [tangent_points_x_hyperbola[i+1], tangent_points_y_hyperbola[i+1]]
        v2 = [dx_dt_values_hyperbola[i+1], dy_dt_values_hyperbola[i+1]]
        intersection = find_intersection(p1, v1, p2, v2)
        if intersection:
            polygon_points_hyperbola.append(intersection)

    print(f"\nd) Координаты вершин многоугольников:")
    print("Эллипс:")
    for i, point in enumerate(polygon_points_ellipse):
        print(f"Вершина {i+1}: ({point[0]:.2f}, {point[1]:.2f})")
    print("Гипербола:")
    for i, point in enumerate(polygon_points_hyperbola):
        print(f"Вершина {i+1}: ({point[0]:.2f}, {point[1]:.2f})")

    # e) Уравнения эволют
    print("\ne) Уравнения эволют:")
    # Эллипс
    print(f"Эволюта эллипса: x = {(a**2 - b**2):.2f}*cos(t)^3/{a:.2f}, y = {(a**2 - b**2):.2f}*sin(t)^3/{b:.2f}, t ∈ [0; 2π]")
    # Гипербола
    print(f"Эволюта гиперболы: x = {(a**2 + b**2):.2f}*cosh(t)^3/{a:.2f}, y = -{(a**2 + b**2):.2f}*sinh(t)^3/{b:.2f}, t ∈ [-{t_max_hyperbola}; {t_max_hyperbola}]")

    plt.figure(figsize=(15, 12))

    # Эллипс
    t_ellipse = np.linspace(0, 2*np.pi, 1000)
    x_ellipse = a * np.cos(t_ellipse)
    y_ellipse = b * np.sin(t_ellipse)
    plt.plot(x_ellipse, y_ellipse, 'b-', linewidth=2.5, label='Эллипс')

    plt.scatter(tangent_points_x_ellipse, tangent_points_y_ellipse, color='red', s=80, label='Точки касания (эллипс)')
    for i in range(len(polygon_points_ellipse)):
        p1 = polygon_points_ellipse[i]
        p2 = polygon_points_ellipse[(i+1)%len(polygon_points_ellipse)]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r--', linewidth=1.5)
    if polygon_points_ellipse:
        polygon_ellipse = Polygon(polygon_points_ellipse, fill=True, alpha=0.3, edgecolor='purple', linewidth=2.5, label='Многоугольник (эллипс)')
        plt.gca().add_patch(polygon_ellipse)

    t_evolute_ellipse = np.linspace(0, 2*np.pi, 1000)
    x_evolute_ellipse = (a**2 - b**2) * np.cos(t_evolute_ellipse)**3 / a
    y_evolute_ellipse = (a**2 - b**2) * np.sin(t_evolute_ellipse)**3 / b
    plt.plot(x_evolute_ellipse, y_evolute_ellipse, 'm-', linewidth=2, label='Эволюта эллипса')

    # Гипербола
    t_hyperbola = np.linspace(-t_max_hyperbola, t_max_hyperbola, 1000)
    x_hyperbola = a * np.cosh(t_hyperbola)
    y_hyperbola = b * np.sinh(t_hyperbola)
    plt.plot(x_hyperbola, y_hyperbola, 'g-', linewidth=2.5, label='Гипербола')

    plt.scatter(tangent_points_x_hyperbola, tangent_points_y_hyperbola, color='orange', s=80, label='Точки касания (гиппербола)')
    for i in range(len(polygon_points_hyperbola)):
        p1 = polygon_points_hyperbola[i]
        p2 = polygon_points_hyperbola[(i+1)%len(polygon_points_hyperbola)]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'orange', linestyle='--', linewidth=1.5)
    if polygon_points_hyperbola:
        polygon_hyperbola = Polygon(polygon_points_hyperbola, fill=True, alpha=0.3, edgecolor='darkgreen', linewidth=2.5, label='Многоугольник (гиппербола)')
        plt.gca().add_patch(polygon_hyperbola)

    t_evolute_hyperbola = np.linspace(-t_max_hyperbola, t_max_hyperbola, 1000)
    x_evolute_hyperbola = (a**2 + b**2) * np.cosh(t_evolute_hyperbola)**3 / a
    y_evolute_hyperbola = -(a**2 + b**2) * np.sinh(t_evolute_hyperbola)**3 / b
    plt.plot(x_evolute_hyperbola, y_evolute_hyperbola, 'c-', linewidth=2, label='Эволюта гиперболы')

    plt.grid(True)
    plt.axis('equal')
    limit = max(a, b) * 2
    plt.xlim(-limit, limit)
    plt.ylim(-limit, limit)
    plt.title('Задание 1: Конические сечения (эллипс и гипербола)', fontsize=16)
    plt.legend(fontsize=12)
    plt.savefig('task1_ellipse_hyperbola.png', dpi=300, bbox_inches='tight')
    plt.show()

# Задание 2: Специальные кривые
def task2():
    print("\nЗадание 2: Специальные кривые")
    
    # a) Выберем две кривые из разных семейств
    # 1) Логарифмическая спираль
    print("\na) Параметрические уравнения выбранных кривых:")
    a_spiral = 0.2  # параметр спирали
    print(f"1) Логарифмическая спираль (полярное уравнение): r = e^({a_spiral}*θ)")
    print(f"   Параметрические уравнения: x = e^({a_spiral}*t)*cos(t), y = e^({a_spiral}*t)*sin(t), t ∈ [0; 8π)")
    
    # 3) Лемниската Бернулли
    a_lemniscate = 2  # параметр лемнискаты
    print(f"2) Лемниската Бернулли (полярное уравнение): r² = {a_lemniscate**2}*cos(2θ)")
    print(f"   Параметрические уравнения: x = {a_lemniscate}*cos(t)/(1+sin²(t)), y = {a_lemniscate}*sin(t)*cos(t)/(1+sin²(t)), t ∈ [0; 2π)")
    
    # b) Найдем уравнения касательных и нормалей к кривым
    t0_spiral =  np.pi / 2 # точка для спирали !!!
    r0_spiral = np.exp(a_spiral * t0_spiral) # полярный радиус
    x0_spiral = r0_spiral * np.cos(t0_spiral)
    y0_spiral = r0_spiral * np.sin(t0_spiral)
    
    # Производные параметрических уравнений спирали
    dx_dt_spiral = r0_spiral * (-np.sin(t0_spiral) + a_spiral * np.cos(t0_spiral))
    dy_dt_spiral = r0_spiral * (np.cos(t0_spiral) + a_spiral * np.sin(t0_spiral))
    
    # Уравнение касательной к спирали
    k_spiral = dy_dt_spiral / dx_dt_spiral
    b_spiral = y0_spiral - k_spiral * x0_spiral
    
    print(f"\nb) Уравнения касательной и нормали к логарифмической спирали в точке ({x0_spiral:.2f}, {y0_spiral:.2f}):")
    print(f"   Касательная: y = {k_spiral:.4f}*x + {b_spiral:.4f}")
    
    # Уравнение нормали к спирали
    k_normal_spiral = -1 / k_spiral
    b_normal_spiral = y0_spiral - k_normal_spiral * x0_spiral
    print(f"   Нормаль: y = {k_normal_spiral:.4f}*x + {b_normal_spiral:.4f}")
    
    # Точка для лемнискаты
    t0_lemniscate = np.pi/6 # !!!
    
    # Вычисляем координаты точки на лемнискате
    r0_lemniscate = np.sqrt(a_lemniscate**2 * np.cos(2*t0_lemniscate))
    x0_lemniscate = r0_lemniscate * np.cos(t0_lemniscate)
    y0_lemniscate = r0_lemniscate * np.sin(t0_lemniscate)
    
    # Для параметрического представления лемнискаты
    # x = a*cos(t)/(1+sin²(t)), y = a*sin(t)*cos(t)/(1+sin²(t))
    # Вычисляем производные
    t, a = symbols('t a')
    x_lemniscate = a*sp.cos(t)/(1+sp.sin(t)**2)
    y_lemniscate = a*sp.sin(t)*sp.cos(t)/(1+sp.sin(t)**2)
    
    dx_dt_lemniscate = sp.diff(x_lemniscate, t)
    dy_dt_lemniscate = sp.diff(y_lemniscate, t)
    
    # Вычисляем численные значения производных
    dx_dt_lemniscate_val = float(dx_dt_lemniscate.subs([(t, t0_lemniscate), (a, a_lemniscate)]))
    dy_dt_lemniscate_val = float(dy_dt_lemniscate.subs([(t, t0_lemniscate), (a, a_lemniscate)]))
    
    # Уравнение касательной к лемнискате
    k_lemniscate = dy_dt_lemniscate_val / dx_dt_lemniscate_val
    b_lemniscate = y0_lemniscate - k_lemniscate * x0_lemniscate
    
    print(f"\n   Уравнения касательной и нормали к лемнискате Бернулли в точке ({x0_lemniscate:.2f}, {y0_lemniscate:.2f}):")
    print(f"   Касательная: y = {k_lemniscate:.4f}*x + {b_lemniscate:.4f}")
    
    # Уравнение нормали к лемнискате
    k_normal_lemniscate = -1 / k_lemniscate
    b_normal_lemniscate = y0_lemniscate - k_normal_lemniscate * x0_lemniscate
    print(f"   Нормаль: y = {k_normal_lemniscate:.4f}*x + {b_normal_lemniscate:.4f}")
    
    # Построение касательного и нормального векторов
    # Для спирали
    tangent_vector_spiral = [dx_dt_spiral, dy_dt_spiral]
    length_tangent_spiral = np.sqrt(tangent_vector_spiral[0]**2 + tangent_vector_spiral[1]**2)
    tangent_vector_spiral = [v/length_tangent_spiral for v in tangent_vector_spiral]
    
    normal_vector_spiral = [-tangent_vector_spiral[1], tangent_vector_spiral[0]] # поворот касат вектора на 90 гр
    
    # Для лемнискаты
    tangent_vector_lemniscate = [dx_dt_lemniscate_val, dy_dt_lemniscate_val]
    length_tangent_lemniscate = np.sqrt(tangent_vector_lemniscate[0]**2 + tangent_vector_lemniscate[1]**2)
    tangent_vector_lemniscate = [v/length_tangent_lemniscate for v in tangent_vector_lemniscate]
    
    normal_vector_lemniscate = [-tangent_vector_lemniscate[1], tangent_vector_lemniscate[0]]
    
    print("\n   Касательные и нормальные векторы:")
    print(f"   Спираль - касательный вектор: ({tangent_vector_spiral[0]:.4f}, {tangent_vector_spiral[1]:.4f})")
    print(f"   Спираль - нормальный вектор: ({normal_vector_spiral[0]:.4f}, {normal_vector_spiral[1]:.4f})")
    print(f"   Лемниската - касательный вектор: ({tangent_vector_lemniscate[0]:.4f}, {tangent_vector_lemniscate[1]:.4f})")
    print(f"   Лемниската - нормальный вектор: ({normal_vector_lemniscate[0]:.4f}, {normal_vector_lemniscate[1]:.4f})")
    
    # c) Найдем радиус кривизны кривых
    # Для спирали
    # Радиус кривизны логарифмической спирали: ρ = r*sqrt(1+a²)
    radius_spiral = r0_spiral * np.sqrt(1 + a_spiral**2)
    
    # Для лемнискаты
    # Используем формулу для радиуса кривизны в полярных координатах
    # ρ = r²/|r²+2(dr/dθ)² - r*d²r/dθ²|
    
    # Для лемнискаты в параметрическом виде используем формулу
    # ρ = ||(dx/dt)² + (dy/dt)²||^(3/2) / |dx/dt * d²y/dt² - dy/dt * d²x/dt²|
    
    d2x_dt2_lemniscate = sp.diff(x_lemniscate, t, 2)
    d2y_dt2_lemniscate = sp.diff(y_lemniscate, t, 2)
    
    d2x_dt2_lemniscate_val = float(d2x_dt2_lemniscate.subs([(t, t0_lemniscate), (a, a_lemniscate)]))
    d2y_dt2_lemniscate_val = float(d2y_dt2_lemniscate.subs([(t, t0_lemniscate), (a, a_lemniscate)]))
    
    numerator_lemniscate = (dx_dt_lemniscate_val**2 + dy_dt_lemniscate_val**2)**(3/2)
    denominator_lemniscate = abs(dx_dt_lemniscate_val*d2y_dt2_lemniscate_val - dy_dt_lemniscate_val*d2x_dt2_lemniscate_val)
    
    radius_lemniscate = numerator_lemniscate / denominator_lemniscate
    
    print(f"\nc) Радиусы кривизны кривых:")
    print(f"   Радиус кривизны логарифмической спирали в точке ({x0_spiral:.2f}, {y0_spiral:.2f}): {radius_spiral:.4f}")
    print(f"   Радиус кривизны лемнискаты Бернулли в точке ({x0_lemniscate:.2f}, {y0_lemniscate:.2f}): {radius_lemniscate:.4f}")
    
    plt.figure(figsize=(15, 12))
    
    plt.subplot(2, 2, 1)
    
    # отрисовываем логарифмическую спираль
    t_spiral = np.linspace(0, 8*np.pi, 1000)
    r_spiral = np.exp(a_spiral * t_spiral)
    x_spiral = r_spiral * np.cos(t_spiral)
    y_spiral = r_spiral * np.sin(t_spiral)
    plt.plot(x_spiral, y_spiral, 'b-', label='Логарифмическая спираль')
    
    # отрисовываем точку на спирали
    plt.scatter(x0_spiral, y0_spiral, color='red', s=50)
    
    # отрисовываем касательную к спирали
    t_tangent = np.linspace(-10, 10, 2)
    x_tangent_spiral = x0_spiral + t_tangent * tangent_vector_spiral[0]
    y_tangent_spiral = y0_spiral + t_tangent * tangent_vector_spiral[1]
    plt.plot(x_tangent_spiral, y_tangent_spiral, 'r--', label='Касательная')
    
    # отрисовываем нормаль к спирали
    t_normal = np.linspace(-5, 5, 2)
    x_normal_spiral = x0_spiral + t_normal * normal_vector_spiral[0]
    y_normal_spiral = y0_spiral + t_normal * normal_vector_spiral[1]
    plt.plot(x_normal_spiral, y_normal_spiral, 'g--', label='Нормаль')
    
    # отрисовываем окружность кривизны
    theta_circle = np.linspace(0, 2*np.pi, 100)
    # Центр окружности кривизны находится на нормальном векторе на расстоянии radius_spiral
    center_x_spiral = x0_spiral + radius_spiral * normal_vector_spiral[0]
    center_y_spiral = y0_spiral + radius_spiral * normal_vector_spiral[1]
    x_circle_spiral = center_x_spiral + radius_spiral * np.cos(theta_circle)
    y_circle_spiral = center_y_spiral + radius_spiral * np.sin(theta_circle)
    plt.plot(x_circle_spiral, y_circle_spiral, 'm:', label='Окружность кривизны')
    
    plt.grid(True)
    plt.axis('equal')
    plt.title('Логарифмическая спираль')
    plt.legend()
    
    # отрисовываем лемнискату Бернулли
    plt.subplot(2, 2, 2)
    
    # Лемниската в параметрическом виде
    def lemniscate_x(t, a):
        return a * np.cos(t) / (1 + np.sin(t)**2)
    
    def lemniscate_y(t, a):
        return a * np.sin(t) * np.cos(t) / (1 + np.sin(t)**2)
    
    t_lemniscate = np.linspace(0, 2*np.pi, 1000)
    x_lemniscate_plot = lemniscate_x(t_lemniscate, a_lemniscate)
    y_lemniscate_plot = lemniscate_y(t_lemniscate, a_lemniscate)
    plt.plot(x_lemniscate_plot, y_lemniscate_plot, 'b-', label='Лемниската Бернулли')
    
    # отрисовка точки на лемнискате
    plt.scatter(x0_lemniscate, y0_lemniscate, color='red', s=50)
    
    # отрисовываем касательную к лемнискате
    t_tangent = np.linspace(-3, 3, 2)
    x_tangent_lemniscate = x0_lemniscate + t_tangent * tangent_vector_lemniscate[0]
    y_tangent_lemniscate = y0_lemniscate + t_tangent * tangent_vector_lemniscate[1]
    plt.plot(x_tangent_lemniscate, y_tangent_lemniscate, 'r--', label='Касательная')
    
    # отрисовываем нормаль к лемнискате
    t_normal = np.linspace(-3, 3, 2)
    x_normal_lemniscate = x0_lemniscate + t_normal * normal_vector_lemniscate[0]
    y_normal_lemniscate = y0_lemniscate + t_normal * normal_vector_lemniscate[1]
    plt.plot(x_normal_lemniscate, y_normal_lemniscate, 'g--', label='Нормаль')
    
    # отрисовываем окружность кривизны
    theta_circle = np.linspace(0, 2*np.pi, 100)
    # Центр окружности кривизны находится на нормальном векторе на расстоянии radius_lemniscate
    center_x_lemniscate = x0_lemniscate + radius_lemniscate * normal_vector_lemniscate[0]
    center_y_lemniscate = y0_lemniscate + radius_lemniscate * normal_vector_lemniscate[1]
    x_circle_lemniscate = center_x_lemniscate + radius_lemniscate * np.cos(theta_circle)
    y_circle_lemniscate = center_y_lemniscate + radius_lemniscate * np.sin(theta_circle)
    plt.plot(x_circle_lemniscate, y_circle_lemniscate, 'm:', label='Окружность кривизны')
    
    plt.grid(True)
    plt.axis('equal')
    plt.title('Лемниската Бернулли')
    plt.legend()
    
    # Увеличенный вид окрестности точки на спирали
    plt.subplot(2, 2, 3)
    plt.plot(x_spiral, y_spiral, 'b-', label='Логарифмическая спираль')
    plt.scatter(x0_spiral, y0_spiral, color='red', s=50)
    plt.plot(x_tangent_spiral, y_tangent_spiral, 'r--', label='Касательная')
    plt.plot(x_normal_spiral, y_normal_spiral, 'g--', label='Нормаль')
    plt.plot(x_circle_spiral, y_circle_spiral, 'm:', label='Окружность кривизны')
    plt.grid(True)
    plt.axis('equal')
    # Ограничиваем область вокруг точки
    plt.xlim(x0_spiral - 2*radius_spiral, x0_spiral + 2*radius_spiral)
    plt.ylim(y0_spiral - 2*radius_spiral, y0_spiral + 2*radius_spiral)
    plt.title('Увеличенный вид - Логарифмическая спираль')
    plt.legend()
    
    # Увеличенный вид окрестности точки на лемнискате
    plt.subplot(2, 2, 4)
    plt.plot(x_lemniscate_plot, y_lemniscate_plot, 'b-', label='Лемниската Бернулли')
    plt.scatter(x0_lemniscate, y0_lemniscate, color='red', s=50)
    plt.plot(x_tangent_lemniscate, y_tangent_lemniscate, 'r--', label='Касательная')
    plt.plot(x_normal_lemniscate, y_normal_lemniscate, 'g--', label='Нормаль')
    plt.plot(x_circle_lemniscate, y_circle_lemniscate, 'm:', label='Окружность кривизны')
    plt.grid(True)
    plt.axis('equal')
    # Ограничиваем область вокруг точки
    plt.xlim(x0_lemniscate - 2*radius_lemniscate, x0_lemniscate + 2*radius_lemniscate)
    plt.ylim(y0_lemniscate - 2*radius_lemniscate, y0_lemniscate + 2*radius_lemniscate)
    plt.title('Увеличенный вид - Лемниската Бернулли')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('task2_special_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return



if __name__ == "__main__":
    task1()
    task2()