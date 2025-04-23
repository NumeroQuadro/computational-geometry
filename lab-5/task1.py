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

def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def graham_scan(points):
    num_points = len(points)
    if num_points < 3: # на всякий случай :)
        return points
    
    # находим самую нижнюю левую точку 
    bottom_point_idx = min(range(num_points), key=lambda i: (points[i][1], points[i][0]))
    
    # Функция для сортировки точек по полярному углу относительно bottom_point_idx
    def polar_angle_sort(point1, point2):
        orientation_result = orientation(points[bottom_point_idx], point1, point2)
        if orientation_result == 0:  # если точки коллинеарны, берём ближайшую
            return -1 if distance(points[bottom_point_idx], point1) <= distance(points[bottom_point_idx], point2) else 1
        return -1 if orientation_result == 2 else 1  # против часовой (2) идёт первым
    
    sorted_points = points.copy()
    sorted_points[0], sorted_points[bottom_point_idx] = sorted_points[bottom_point_idx], sorted_points[0]
    
    # Сортировка точек по полярному углу
    pivot_point = sorted_points[0]
    sorted_points[1:] = sorted(sorted_points[1:], 
                              key=lambda point: (math.atan2(point[1]-pivot_point[1], point[0]-pivot_point[0]), # вычисляет угол между положительной осью Х и лучом проведенным из опорной точки к текущей
                                               distance(pivot_point, point)))
    
    hull_stack = [sorted_points[0], sorted_points[1]]
    
    # Проходим по всем точкам
    for i in range(2, num_points):
        # Убираем точки, пока не получим поворот против часовой стрелки
        while len(hull_stack) > 1 and orientation(hull_stack[-2], hull_stack[-1], sorted_points[i]) != 2:
            hull_stack.pop()
        hull_stack.append(sorted_points[i])
    
    return hull_stack

def perimeter(hull):
    num_vertices = len(hull)
    total_perimeter = 0
    for i in range(num_vertices):
        total_perimeter += distance(hull[i], hull[(i + 1) % num_vertices])
    return total_perimeter

# S = (1/2) | Σ(x_i * y_(i+1) - x_(i+1) * y_i
# Функция для вычисления площади выпуклой оболочки (формула Гаусса)
def area(hull):
    num_vertices = len(hull)
    total_area = 0
    for i in range(num_vertices):
        total_area += hull[i][0] * hull[(i + 1) % num_vertices][1] - hull[(i + 1) % num_vertices][0] * hull[i][1]
    return abs(total_area) / 2

def is_inside(test_point, hull):
    num_vertices = len(hull)
    for i in range(num_vertices):
        if orientation(hull[i], hull[(i + 1) % num_vertices], test_point) != 2:
            return False
    return True

# np.random.seed(42)
points_n1 = np.random.rand(10, 2) * 20  # 10 точек
points_n2 = np.random.rand(50, 2) * 20  # 50 точек

test_points = np.random.rand(5, 2) * 60

hull_graham_n1 = graham_scan(points_n1.tolist())
hull_graham_n2 = graham_scan(points_n2.tolist())

perimeter_graham_n1 = perimeter(hull_graham_n1)
area_graham_n1 = area(hull_graham_n1)

perimeter_graham_n2 = perimeter(hull_graham_n2)
area_graham_n2 = area(hull_graham_n2)

is_inside_points = [is_inside(point, hull_graham_n2) for point in test_points]

print(f"Для n1=10 точек (алгоритм Грэхема):")
print(f"  Периметр выпуклой оболочки: {perimeter_graham_n1:.2f}")
print(f"  Площадь выпуклой оболочки: {area_graham_n1:.2f}")

print(f"\nДля n2=50 точек (алгоритм Грэхема):")
print(f"  Периметр выпуклой оболочки: {perimeter_graham_n2:.2f}")
print(f"  Площадь выпуклой оболочки: {area_graham_n2:.2f}")

print("\nПроверка принадлежности тестовых точек выпуклой оболочке для n2=50:")
for i, point in enumerate(test_points):
    print(f"  Точка {i+1} {point}: {'внутри' if is_inside_points[i] else 'снаружи'}")

plt.figure(figsize=(15, 7))

plt.subplot(1, 2, 1)
plt.scatter(points_n1[:, 0], points_n1[:, 1], color='blue', label='Точки множества')
hull_graham_n1_array = np.array(hull_graham_n1)
plt.fill(hull_graham_n1_array[:, 0], hull_graham_n1_array[:, 1], alpha=0.3, color='green', label='Выпуклая оболочка (Грэхем)')
plt.plot(hull_graham_n1_array[:, 0], hull_graham_n1_array[:, 1], 'g-')
plt.title(f'n1=10 точек\nПериметр: {perimeter_graham_n1:.2f}, Площадь: {area_graham_n1:.2f}')
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(points_n2[:, 0], points_n2[:, 1], color='blue', label='Точки множества')
hull_graham_n2_array = np.array(hull_graham_n2)
plt.fill(hull_graham_n2_array[:, 0], hull_graham_n2_array[:, 1], alpha=0.3, color='green', label='Выпуклая оболочка (Грэхем)')
plt.plot(hull_graham_n2_array[:, 0], hull_graham_n2_array[:, 1], 'g-')

for i, point in enumerate(test_points):
    color = 'red' if is_inside_points[i] else 'orange'
    marker = 'o' if is_inside_points[i] else 'x'
    plt.scatter(point[0], point[1], color=color, marker=marker, s=100)
    plt.text(point[0]+0.2, point[1]+0.2, f'T{i+1}', fontsize=12)

plt.title(f'n2=50 точек\nПериметр: {perimeter_graham_n2:.2f}, Площадь: {area_graham_n2:.2f}')
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()