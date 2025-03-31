import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.patches import Polygon

# Функция для расчета ориентации трех точек (по часовой, против часовой или коллинеарные)
def orientation(point1, point2, point3):
    val = (point2[1] - point1[1]) * (point3[0] - point2[0]) - (point2[0] - point1[0]) * (point3[1] - point2[1])
    if val == 0:
        return 0  # коллинеарные
    return 1 if val > 0 else 2  # по часовой или против часовой

# Функция для вычисления расстояния между двумя точками
def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Алгоритм Джарвиса (построение выпуклой оболочки)
def jarvis_march(points):
    num_points = len(points)
    if num_points < 3:
        return points  # выпуклая оболочка невозможна для менее 3 точек
    
    # Находим самую левую точку
    leftmost_idx = 0
    for i in range(1, num_points):
        if points[i][0] < points[leftmost_idx][0]:
            leftmost_idx = i
    
    # Начинаем с самой левой точки и двигаемся против часовой стрелки
    hull = []
    current_idx = leftmost_idx
    next_idx = 0
    
    while True:
        # Добавляем текущую точку в оболочку
        hull.append(points[current_idx])
        
        # Ищем следующую точку
        next_idx = (current_idx + 1) % num_points
        
        for i in range(num_points):
            # Если i лежит против часовой стрелки от линии current_idx-next_idx
            if orientation(points[current_idx], points[i], points[next_idx]) == 2:
                next_idx = i
        
        current_idx = next_idx
        
        # Если мы вернулись к первой точке, то выходим из цикла
        if current_idx == leftmost_idx:
            break
    
    return hull

def graham_scan(points):
    num_points = len(points)
    if num_points < 3: # на всякий случай :)
        return points
    
    # Находим точку с минимальной y-координатой
    # и если таких несколько - то самую левую из них
    bottom_point_idx = min(range(num_points), key=lambda i: (points[i][1], points[i][0]))
    
    # Функция для сортировки точек по полярному углу относительно bottom_point_idx
    def polar_angle_sort(point1, point2):
        orientation_result = orientation(points[bottom_point_idx], point1, point2)
        if orientation_result == 0:  # если точки коллинеарны, берём ближайшую
            return -1 if distance(points[bottom_point_idx], point1) <= distance(points[bottom_point_idx], point2) else 1
        return -1 if orientation_result == 2 else 1  # против часовой (2) идёт первым
    
    # Копируем список точек и сортируем его
    sorted_points = points.copy()
    sorted_points[0], sorted_points[bottom_point_idx] = sorted_points[bottom_point_idx], sorted_points[0]
    
    # Сортировка точек по полярному углу
    # Используем лямбда-функцию для сортировки с помощью key
    pivot_point = sorted_points[0]
    sorted_points[1:] = sorted(sorted_points[1:], 
                              key=lambda point: (math.atan2(point[1]-pivot_point[1], point[0]-pivot_point[0]), # вычисляет угол между положительной осью Х и лучом проведенным из опорной точки к текущей
                                               distance(pivot_point, point)))
    
    # Начинаем с трех точек
    hull_stack = [sorted_points[0], sorted_points[1]]
    
    # Проходим по всем точкам
    for i in range(2, num_points):
        # Убираем точки, пока не получим поворот против часовой стрелки
        while len(hull_stack) > 1 and orientation(hull_stack[-2], hull_stack[-1], sorted_points[i]) != 2:
            hull_stack.pop()
        hull_stack.append(sorted_points[i])
    
    return hull_stack

# Функция для вычисления периметра выпуклой оболочки
def perimeter(hull):
    num_vertices = len(hull)
    total_perimeter = 0
    for i in range(num_vertices):
        total_perimeter += distance(hull[i], hull[(i + 1) % num_vertices])
    return total_perimeter

# Функция для вычисления площади выпуклой оболочки (формула Гаусса)
def area(hull):
    num_vertices = len(hull)
    total_area = 0
    for i in range(num_vertices):
        total_area += hull[i][0] * hull[(i + 1) % num_vertices][1] - hull[(i + 1) % num_vertices][0] * hull[i][1]
    return abs(total_area) / 2

# Функция для проверки, находится ли точка внутри выпуклой оболочки
def is_inside(test_point, hull):
    num_vertices = len(hull)
    # Для всех точек выпуклой оболочки, точка должна быть на той же стороне от любого ребра
    for i in range(num_vertices):
        if orientation(hull[i], hull[(i + 1) % num_vertices], test_point) != 2:
            return False
    return True

# Генерация случайных точек
np.random.seed(42)  # для воспроизводимости
points_n1 = np.random.rand(10, 2) * 20  # 10 точек
points_n2 = np.random.rand(50, 2) * 20  # 50 точек

test_points = np.random.rand(5, 2) * 20

# Построение выпуклой оболочки с помощью алгоритма Джарвиса
hull_jarvis_n1 = jarvis_march(points_n1.tolist())
hull_jarvis_n2 = jarvis_march(points_n2.tolist())

hull_graham_n1 = graham_scan(points_n1.tolist())
hull_graham_n2 = graham_scan(points_n2.tolist())

# Вычисление периметра и площади для n1=10
perimeter_jarvis_n1 = perimeter(hull_jarvis_n1)
area_jarvis_n1 = area(hull_jarvis_n1)

# Вычисление периметра и площади для n2=50
perimeter_jarvis_n2 = perimeter(hull_jarvis_n2)
area_jarvis_n2 = area(hull_jarvis_n2)

is_inside_points = [is_inside(point, hull_jarvis_n2) for point in test_points]

print(f"Для n1=10 точек (алгоритм Джарвиса):")
print(f"  Периметр выпуклой оболочки: {perimeter_jarvis_n1:.2f}")
print(f"  Площадь выпуклой оболочки: {area_jarvis_n1:.2f}")

print(f"\nДля n2=50 точек (алгоритм Джарвиса):")
print(f"  Периметр выпуклой оболочки: {perimeter_jarvis_n2:.2f}")
print(f"  Площадь выпуклой оболочки: {area_jarvis_n2:.2f}")

print("\nПроверка принадлежности тестовых точек выпуклой оболочке для n2=50:")
for i, point in enumerate(test_points):
    print(f"  Точка {i+1} {point}: {'внутри' if is_inside_points[i] else 'снаружи'}")

plt.figure(figsize=(15, 7))

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