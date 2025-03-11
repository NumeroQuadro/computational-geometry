import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.optimize import fsolve
import sympy as sp
from sympy import symbols, solve, diff, sqrt, cos, sin, ln, exp, Matrix

# Настраиваем параметры графиков для корректного отображения русского текста
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12

# Задание 1: Конические сечения
def task1():
    print("Задание 1: Конические сечения")
    
    # a) Задаем эллипс и гиперболу полуосями a, b
    a, b = 4, 2  # произвольные значения полуосей
    
    # Параметрические уравнения эллипса: x = a*cos(t), y = b*sin(t)
    print("\na) Параметрические уравнения кривых:")
    print(f"Эллипс: x = {a}*cos(t), y = {b}*sin(t), t ∈ [0; 2π)")
    
    # Параметрические уравнения гиперболы: x = a*cosh(t), y = b*sinh(t)
    print(f"Гипербола: x = {a}*cosh(t), y = {b}*sinh(t), t ∈ (-∞; +∞)")
    
    # b) Найдем уравнения касательных к кривым в точке
    t0 = np.pi/4  # произвольное значение параметра
    x0_ellipse = a * np.cos(t0)
    y0_ellipse = b * np.sin(t0)
    
    # Производные параметрических уравнений эллипса
    dx_dt_ellipse = -a * np.sin(t0)
    dy_dt_ellipse = b * np.cos(t0)
    
    # Уравнение касательной к эллипсу
    if abs(dx_dt_ellipse) > 1e-10:  # проверяем, что производная не равна нулю
        k_ellipse = dy_dt_ellipse / dx_dt_ellipse
        b_ellipse = y0_ellipse - k_ellipse * x0_ellipse
        print(f"\nb) Уравнение касательной к эллипсу в точке ({x0_ellipse:.2f}, {y0_ellipse:.2f}):")
        print(f"y = {k_ellipse:.4f}*x + {b_ellipse:.4f}")
    else:
        print(f"\nb) Уравнение касательной к эллипсу в точке ({x0_ellipse:.2f}, {y0_ellipse:.2f}):")
        print(f"x = {x0_ellipse:.4f}")
    
    # Для гиперболы выбираем другой параметр
    t0_hyp = 0.5
    x0_hyperbola = a * np.cosh(t0_hyp)
    y0_hyperbola = b * np.sinh(t0_hyp)
    
    # Производные параметрических уравнений гиперболы
    dx_dt_hyperbola = a * np.sinh(t0_hyp)
    dy_dt_hyperbola = b * np.cosh(t0_hyp)
    
    # Уравнение касательной к гиперболе
    k_hyperbola = dy_dt_hyperbola / dx_dt_hyperbola
    b_hyperbola = y0_hyperbola - k_hyperbola * x0_hyperbola
    print(f"Уравнение касательной к гиперболе в точке ({x0_hyperbola:.2f}, {y0_hyperbola:.2f}):")
    print(f"y = {k_hyperbola:.4f}*x + {b_hyperbola:.4f}")
    
    # c) Разбиваем промежуток [0; 2π) на n участков
    n = 8  # количество участков
    t_points = np.linspace(0, 2*np.pi, n+1)[:-1]  # точки разбиения без последней (она совпадает с первой)
    t_mid_points = [(t_points[i] + t_points[(i+1)%n])/2 for i in range(n)]  # середины участков
    
    print(f"\nc) Разбиение промежутка [0; 2π) на {n} участков:")
    for i in range(n):
        interval_start = t_points[i]
        interval_end = t_points[(i+1)%n]
        mid_point = t_mid_points[i]
        print(f"Участок {i+1}: [{interval_start:.2f}; {interval_end:.2f}], середина: {mid_point:.2f}")
    
    # d) Из отрезков касательных составим многоугольник
    # Найдем точки касания и направляющие векторы касательных
    tangent_points_x = [a * np.cos(t) for t in t_mid_points]
    tangent_points_y = [b * np.sin(t) for t in t_mid_points]
    
    tangent_vectors_x = [-a * np.sin(t) for t in t_mid_points]
    tangent_vectors_y = [b * np.cos(t) for t in t_mid_points]
    
    # Функция для нахождения точки пересечения двух прямых
    def find_intersection(p1, v1, p2, v2):
        # p1, p2 - точки на прямых
        # v1, v2 - направляющие векторы прямых
        # Решаем систему уравнений: p1 + t1*v1 = p2 + t2*v2
        A = np.array([v1, [-v for v in v2]]).T  # Negate each element in v2
        b = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        try:
            t1, t2 = np.linalg.solve(A, b)
            return [p1[0] + t1*v1[0], p1[1] + t1*v1[1]]
        except np.linalg.LinAlgError:
            # Прямые параллельны или совпадают
            return None
    
    # Находим точки пересечения касательных
    polygon_points = []
    for i in range(n):
        p1 = [tangent_points_x[i], tangent_points_y[i]]
        v1 = [tangent_vectors_x[i], tangent_vectors_y[i]]
        
        p2 = [tangent_points_x[(i+1)%n], tangent_points_y[(i+1)%n]]
        v2 = [tangent_vectors_x[(i+1)%n], tangent_vectors_y[(i+1)%n]]
        
        intersection = find_intersection(p1, v1, p2, v2)
        if intersection:
            polygon_points.append(intersection)
    
    print(f"\nd) Координаты вершин многоугольника:")
    for i, point in enumerate(polygon_points):
        print(f"Вершина {i+1}: ({point[0]:.2f}, {point[1]:.2f})")
    
    # e) Найдем уравнение эволюты эллипса
    print("\ne) Уравнение эволюты эллипса:")
    print(f"x = {(a**2 - b**2):.2f}*cos(t)^3/{a:.2f}, y = {(a**2 - b**2):.2f}*sin(t)^3/{b:.2f}, t ∈ [0; 2π)")
    
    # Визуализация результатов
    plt.figure(figsize=(15, 12))  # Увеличиваем размер фигуры

    # Рисуем эллипс
    t = np.linspace(0, 2*np.pi, 1000)
    x_ellipse = a * np.cos(t)
    y_ellipse = b * np.sin(t)
    plt.plot(x_ellipse, y_ellipse, 'b-', linewidth=2.5, label='Эллипс')

    # Рисуем гиперболу
    t_hyp = np.linspace(-2, 2, 1000)
    x_hyperbola = a * np.cosh(t_hyp)
    y_hyperbola = b * np.sinh(t_hyp)
    plt.plot(x_hyperbola, y_hyperbola, 'g-', linewidth=2.5, label='Гипербола')

    # Рисуем точки касания
    plt.scatter(tangent_points_x, tangent_points_y, color='red', s=80, label='Точки касания')

    # Рисуем касательные с увеличенным диапазоном для лучшей видимости
    for i in range(n):
        # Увеличиваем длину отображаемых касательных
        t = np.linspace(-100, 100, 2)
        p = [tangent_points_x[i], tangent_points_y[i]]
        v = [tangent_vectors_x[i], tangent_vectors_y[i]]
        # Нормализуем направляющие векторы для лучшего отображения
        length = np.sqrt(v[0]**2 + v[1]**2)
        v_norm = [v[0]/length, v[1]/length]
        
        # Рисуем касательную через точку касания
        plt.plot([p[0] - t[0]*v_norm[1], p[0] + t[1]*v_norm[1]], 
                [p[1] + t[0]*v_norm[0], p[1] - t[1]*v_norm[0]], 
                'r--', linewidth=1.5, alpha=0.7)

    # Рисуем многоугольник
    if polygon_points:
        polygon = Polygon(polygon_points, fill=True, alpha=0.3, edgecolor='purple', 
                        linewidth=2.5, label='Многоугольник')
        plt.gca().add_patch(polygon)

    # Рисуем эволюту эллипса
    t_evolute = np.linspace(0, 2*np.pi, 1000)
    x_evolute = (a**2 - b**2) * np.cos(t_evolute)**3 / a
    y_evolute = (a**2 - b**2) * np.sin(t_evolute)**3 / b
    plt.plot(x_evolute, y_evolute, 'm-', linewidth=2, label='Эволюта эллипса')

    plt.grid(True)
    plt.axis('equal')
    # Устанавливаем более удобный масштаб для просмотра
    plt.xlim(-8, 8)
    plt.ylim(-8, 8)
    plt.title('Задание 1: Конические сечения', fontsize=16)
    plt.legend(fontsize=12)
    plt.savefig('task1_conic_sections.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return

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
    t0_spiral = np.pi/2  # точка для спирали
    r0_spiral = np.exp(a_spiral * t0_spiral)
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
    t0_lemniscate = np.pi/6
    
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
    
    normal_vector_spiral = [-tangent_vector_spiral[1], tangent_vector_spiral[0]]
    
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
    
    # Визуализация результатов
    plt.figure(figsize=(15, 12))
    
    # Создаем подграфики
    plt.subplot(2, 2, 1)
    
    # Рисуем логарифмическую спираль
    t_spiral = np.linspace(0, 8*np.pi, 1000)
    r_spiral = np.exp(a_spiral * t_spiral)
    x_spiral = r_spiral * np.cos(t_spiral)
    y_spiral = r_spiral * np.sin(t_spiral)
    plt.plot(x_spiral, y_spiral, 'b-', label='Логарифмическая спираль')
    
    # Рисуем точку на спирали
    plt.scatter(x0_spiral, y0_spiral, color='red', s=50)
    
    # Рисуем касательную к спирали
    t_tangent = np.linspace(-10, 10, 2)
    x_tangent_spiral = x0_spiral + t_tangent * tangent_vector_spiral[0]
    y_tangent_spiral = y0_spiral + t_tangent * tangent_vector_spiral[1]
    plt.plot(x_tangent_spiral, y_tangent_spiral, 'r--', label='Касательная')
    
    # Рисуем нормаль к спирали
    t_normal = np.linspace(-5, 5, 2)
    x_normal_spiral = x0_spiral + t_normal * normal_vector_spiral[0]
    y_normal_spiral = y0_spiral + t_normal * normal_vector_spiral[1]
    plt.plot(x_normal_spiral, y_normal_spiral, 'g--', label='Нормаль')
    
    # Рисуем окружность кривизны
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
    
    # Рисуем лемнискату Бернулли
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
    
    # Рисуем точку на лемнискате
    plt.scatter(x0_lemniscate, y0_lemniscate, color='red', s=50)
    
    # Рисуем касательную к лемнискате
    t_tangent = np.linspace(-3, 3, 2)
    x_tangent_lemniscate = x0_lemniscate + t_tangent * tangent_vector_lemniscate[0]
    y_tangent_lemniscate = y0_lemniscate + t_tangent * tangent_vector_lemniscate[1]
    plt.plot(x_tangent_lemniscate, y_tangent_lemniscate, 'r--', label='Касательная')
    
    # Рисуем нормаль к лемнискате
    t_normal = np.linspace(-3, 3, 2)
    x_normal_lemniscate = x0_lemniscate + t_normal * normal_vector_lemniscate[0]
    y_normal_lemniscate = y0_lemniscate + t_normal * normal_vector_lemniscate[1]
    plt.plot(x_normal_lemniscate, y_normal_lemniscate, 'g--', label='Нормаль')
    
    # Рисуем окружность кривизны
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

# Основная функция
def main():
    task1()
    task2()

if __name__ == "__main__":
    main()