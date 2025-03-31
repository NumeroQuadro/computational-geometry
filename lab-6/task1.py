import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.patches import Polygon

# Функция для расчета ориентации трех точек (по часовой, против часовой или коллинеарные)
def orientation(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0  # коллинеарные
    return 1 if val > 0 else 2  # по часовой или против часовой

# Функция для вычисления расстояния между двумя точками
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Алгоритм Джарвиса (построение выпуклой оболочки)
def jarvis_march(points):
    n = len(points)
    if n < 3:
        return points  # выпуклая оболочка невозможна для менее 3 точек
    
    # Находим самую левую точку
    leftmost = 0
    for i in range(1, n):
        if points[i][0] < points[leftmost][0]:
            leftmost = i
    
    # Начинаем с самой левой точки и двигаемся против часовой стрелки
    hull = []
    p = leftmost
    q = 0
    
    while True:
        # Добавляем текущую точку в оболочку
        hull.append(points[p])
        
        # Ищем следующую точку
        q = (p + 1) % n
        
        for i in range(n):
            # Если i лежит против часовой стрелки от линии p-q
            if orientation(points[p], points[i], points[q]) == 2:
                q = i
        
        p = q
        
        # Если мы вернулись к первой точке, то выходим из цикла
        if p == leftmost:
            break
    
    return hull

# Алгоритм Грэхема (построение выпуклой оболочки)
def graham_scan(points):
    n = len(points)
    if n < 3:
        return points  # выпуклая оболочка невозможна для менее 3 точек
    
    # Находим точку с минимальной y-координатой
    # и если таких несколько - то самую левую из них
    bottom_y = min(range(n), key=lambda i: (points[i][1], points[i][0]))
    
    # Функция для сортировки точек по полярному углу относительно bottom_y
    def polar_angle_sort(p1, p2):
        o = orientation(points[bottom_y], p1, p2)
        if o == 0:  # если точки коллинеарны, берём ближайшую
            return -1 if distance(points[bottom_y], p1) <= distance(points[bottom_y], p2) else 1
        return -1 if o == 2 else 1  # против часовой (2) идёт первым
    
    # Копируем список точек и сортируем его
    sorted_points = points.copy()
    sorted_points[0], sorted_points[bottom_y] = sorted_points[bottom_y], sorted_points[0]
    
    # Сортировка точек по полярному углу
    # Используем лямбда-функцию для сортировки с помощью key
    pivot = sorted_points[0]
    sorted_points[1:] = sorted(sorted_points[1:], 
                              key=lambda point: (math.atan2(point[1]-pivot[1], point[0]-pivot[0]), # вычисляет угол между положительной осью Х и лучом проведенным из опорной точки к текущей
                                               distance(pivot, point)))
    
    # Начинаем с трех точек
    stack = [sorted_points[0], sorted_points[1]]
    
    # Проходим по всем точкам
    for i in range(2, n):
        # Убираем точки, пока не получим поворот против часовой стрелки
        while len(stack) > 1 and orientation(stack[-2], stack[-1], sorted_points[i]) != 2:
            stack.pop()
        stack.append(sorted_points[i])
    
    return stack

# Функция для вычисления периметра выпуклой оболочки
def perimeter(hull):
    n = len(hull)
    p = 0
    for i in range(n):
        p += distance(hull[i], hull[(i + 1) % n])
    return p

# Функция для вычисления площади выпуклой оболочки (формула Гаусса)
def area(hull):
    n = len(hull)
    a = 0
    for i in range(n):
        a += hull[i][0] * hull[(i + 1) % n][1] - hull[(i + 1) % n][0] * hull[i][1]
    return abs(a) / 2

# Функция для проверки, находится ли точка внутри выпуклой оболочки
def is_inside(point, hull):
    n = len(hull)
    # Для всех точек выпуклой оболочки, точка должна быть на той же стороне от любого ребра
    for i in range(n):
        if orientation(hull[i], hull[(i + 1) % n], point) != 2:
            return False
    return True

# Генерация случайных точек
np.random.seed(42)  # для воспроизводимости
points_n1 = np.random.rand(10, 2) * 20  # 10 точек
points_n2 = np.random.rand(50, 2) * 20  # 50 точек

# Генерация тестовых точек для проверки принадлежности
test_points = np.random.rand(5, 2) * 20  # 5 точек для проверки

# Построение выпуклой оболочки с помощью алгоритма Джарвиса
hull_jarvis_n1 = jarvis_march(points_n1.tolist())
hull_jarvis_n2 = jarvis_march(points_n2.tolist())

# Построение выпуклой оболочки с помощью алгоритма Грэхема
hull_graham_n1 = graham_scan(points_n1.tolist())
hull_graham_n2 = graham_scan(points_n2.tolist())

# Вычисление периметра и площади для n1=10
perimeter_jarvis_n1 = perimeter(hull_jarvis_n1)
area_jarvis_n1 = area(hull_jarvis_n1)

# Вычисление периметра и площади для n2=50
perimeter_jarvis_n2 = perimeter(hull_jarvis_n2)
area_jarvis_n2 = area(hull_jarvis_n2)

# Проверка принадлежности тестовых точек выпуклой оболочке для n2=50
is_inside_points = [is_inside(point, hull_jarvis_n2) for point in test_points]

# Вывод результатов
print(f"Для n1=10 точек (алгоритм Джарвиса):")
print(f"  Периметр выпуклой оболочки: {perimeter_jarvis_n1:.2f}")
print(f"  Площадь выпуклой оболочки: {area_jarvis_n1:.2f}")

print(f"\nДля n2=50 точек (алгоритм Джарвиса):")
print(f"  Периметр выпуклой оболочки: {perimeter_jarvis_n2:.2f}")
print(f"  Площадь выпуклой оболочки: {area_jarvis_n2:.2f}")

print("\nПроверка принадлежности тестовых точек выпуклой оболочке для n2=50:")
for i, point in enumerate(test_points):
    print(f"  Точка {i+1} {point}: {'внутри' if is_inside_points[i] else 'снаружи'}")

# Визуализация
plt.figure(figsize=(15, 7))

# Для n1=10
plt.subplot(1, 2, 1)
plt.scatter(points_n1[:, 0], points_n1[:, 1], color='blue', label='Точки множества')
hull_jarvis_n1_array = np.array(hull_jarvis_n1)
plt.fill(hull_jarvis_n1_array[:, 0], hull_jarvis_n1_array[:, 1], alpha=0.3, color='green', label='Выпуклая оболочка (Джарвис)')
plt.plot(hull_jarvis_n1_array[:, 0], hull_jarvis_n1_array[:, 1], 'g-')
plt.title(f'n1=10 точек\nПериметр: {perimeter_jarvis_n1:.2f}, Площадь: {area_jarvis_n1:.2f}')
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.legend()
plt.grid(True)

# Для n2=50
plt.subplot(1, 2, 2)
plt.scatter(points_n2[:, 0], points_n2[:, 1], color='blue', label='Точки множества')
hull_jarvis_n2_array = np.array(hull_jarvis_n2)
plt.fill(hull_jarvis_n2_array[:, 0], hull_jarvis_n2_array[:, 1], alpha=0.3, color='green', label='Выпуклая оболочка (Джарвис)')
plt.plot(hull_jarvis_n2_array[:, 0], hull_jarvis_n2_array[:, 1], 'g-')

# Отображение тестовых точек с разными цветами в зависимости от принадлежности
for i, point in enumerate(test_points):
    color = 'red' if is_inside_points[i] else 'orange'
    marker = 'o' if is_inside_points[i] else 'x'
    plt.scatter(point[0], point[1], color=color, marker=marker, s=100)
    plt.text(point[0]+0.2, point[1]+0.2, f'T{i+1}', fontsize=12)

plt.title(f'n2=50 точек\nПериметр: {perimeter_jarvis_n2:.2f}, Площадь: {area_jarvis_n2:.2f}')
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Сравнение алгоритмов
print("\nСравнение алгоритмов (должны давать одинаковый результат):")
print(f"Для n1=10 точек:")
print(f"  Алгоритм Джарвиса: периметр={perimeter_jarvis_n1:.2f}, площадь={area_jarvis_n1:.2f}")
perimeter_graham_n1 = perimeter(hull_graham_n1)
area_graham_n1 = area(hull_graham_n1)
print(f"  Алгоритм Грэхема: периметр={perimeter_graham_n1:.2f}, площадь={area_graham_n1:.2f}")

print(f"\nДля n2=50 точек:")
print(f"  Алгоритм Джарвиса: периметр={perimeter_jarvis_n2:.2f}, площадь={area_jarvis_n2:.2f}")
perimeter_graham_n2 = perimeter(hull_graham_n2)
area_graham_n2 = area(hull_graham_n2)
print(f"  Алгоритм Грэхема: периметр={perimeter_graham_n2:.2f}, площадь={area_graham_n2:.2f}")