import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random

def euclidean_distance(p1, p2):
    """Вычисляет евклидово расстояние между двумя точками"""
    return np.sqrt(np.sum((p1 - p2) ** 2))

def find_closest_pair_brute_force(points):
    """Находит ближайшую пару точек методом перебора"""
    n = len(points)
    min_dist = float('inf')
    closest_pair = None
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = euclidean_distance(points[i], points[j])
            if dist < min_dist:
                min_dist = dist
                closest_pair = (i, j)
    
    return closest_pair, min_dist

def find_closest_pair_divide_conquer(points):
    """Находит ближайшую пару точек методом 'разделяй и властвуй'"""
    # Сортируем точки по x-координате
    sorted_indices = np.argsort(points[:, 0])
    sorted_points = points[sorted_indices]
    
    # Определяем вспомогательную функцию для рекурсивного поиска
    def closest_pair_recursive(start, end):
        if end - start <= 3:
            # Для маленького количества точек используем метод перебора
            local_min_dist = float('inf')
            local_closest_pair = None
            for i in range(start, end):
                for j in range(i + 1, end):
                    dist = euclidean_distance(sorted_points[i], sorted_points[j])
                    if dist < local_min_dist:
                        local_min_dist = dist
                        local_closest_pair = (sorted_indices[i], sorted_indices[j])
            return local_closest_pair, local_min_dist
        
        # Разделяем на две части
        mid = (start + end) // 2
        mid_x = sorted_points[mid][0]
        
        # Рекурсивно находим ближайшие пары в левой и правой частях
        left_pair, left_dist = closest_pair_recursive(start, mid)
        right_pair, right_dist = closest_pair_recursive(mid, end)
        
        # Определяем минимальное расстояние и соответствующую пару
        if left_dist <= right_dist:
            min_pair = left_pair
            min_dist = left_dist
        else:
            min_pair = right_pair
            min_dist = right_dist
        
        # Находим точки в полосе шириной 2*min_dist вокруг средней линии
        strip_indices = []
        for i in range(start, end):
            if abs(sorted_points[i][0] - mid_x) < min_dist:
                strip_indices.append(i)
        
        # Сортируем точки в полосе по y-координате
        strip_indices.sort(key=lambda i: sorted_points[i][1])
        
        # Проверяем пары в полосе
        for i in range(len(strip_indices)):
            for j in range(i + 1, min(i + 7, len(strip_indices))):  # Проверяем только ближайшие 6 точек
                if sorted_points[strip_indices[j]][1] - sorted_points[strip_indices[i]][1] >= min_dist:
                    break
                
                dist = euclidean_distance(sorted_points[strip_indices[i]], sorted_points[strip_indices[j]])
                if dist < min_dist:
                    min_dist = dist
                    min_pair = (sorted_indices[strip_indices[i]], sorted_indices[strip_indices[j]])
        
        return min_pair, min_dist
    
    return closest_pair_recursive(0, len(sorted_points))

def circumcenter(p1, p2, p3):
    """Вычисляет центр описанной окружности для треугольника"""
    D = 2 * (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
    
    if abs(D) < 1e-10:  # Проверка на коллинеарность
        return None
    
    Ux = ((p1[0]**2 + p1[1]**2) * (p2[1] - p3[1]) + 
          (p2[0]**2 + p2[1]**2) * (p3[1] - p1[1]) + 
          (p3[0]**2 + p3[1]**2) * (p1[1] - p2[1])) / D
    
    Uy = ((p1[0]**2 + p1[1]**2) * (p3[0] - p2[0]) + 
          (p2[0]**2 + p2[1]**2) * (p1[0] - p3[0]) + 
          (p3[0]**2 + p3[1]**2) * (p2[0] - p1[0])) / D
    
    return np.array([Ux, Uy])

def in_circle_test(p1, p2, p3, p4):
    """Проверяет, находится ли точка p4 внутри описанной окружности треугольника p1,p2,p3"""
    center = circumcenter(p1, p2, p3)
    if center is None:
        return False
    
    radius = euclidean_distance(center, p1)
    dist_to_p4 = euclidean_distance(center, p4)
    
    return dist_to_p4 < radius

def compute_voronoi_naive(points):
    """
    Наивный алгоритм построения диаграммы 
    """
    n = len(points)
    
    vertices = []
    edges = []
    
    vertex_dict = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                
                center = circumcenter(p1, p2, p3)
                if center is not None:  # Исправленная проверка
                    r = euclidean_distance(center, p1)
                    is_valid = all(
                        euclidean_distance(center, points[l]) > r - 1e-10  # Небольшой допуск для численной стабильности
                        for l in range(n) if l != i and l != j and l != k
                    )
                    
                    if is_valid:
                        center_tuple = (round(center[0], 10), round(center[1], 10))
                        if center_tuple not in vertex_dict:
                            vertex_dict[center_tuple] = len(vertices)
                            vertices.append(center)
    
    if not vertices:  # Если вершин нет, возвращаем пустые списки
        return np.array([]).reshape(0, 2), []
    
    # Находим рёбра диаграммы Вороного
    for i in range(n):
        for j in range(i + 1, n):
            p1, p2 = points[i], points[j]
            
            # Находим все вершины, равноудаленные от p1 и p2
            perp_vertices = []
            for v_idx, vertex in enumerate(vertices):
                dist1 = euclidean_distance(vertex, p1)
                dist2 = euclidean_distance(vertex, p2)
                if abs(dist1 - dist2) < 1e-10:
                    perp_vertices.append(v_idx)
            
            # Создаем рёбра между парами вершин
            if len(perp_vertices) >= 2:
                for v1 in range(len(perp_vertices)):
                    for v2 in range(v1 + 1, len(perp_vertices)):
                        edges.append((perp_vertices[v1], perp_vertices[v2]))
    
    return np.array(vertices), edges

def compute_voronoi_divide_conquer(points):
    """
    Строит диаграмму Вороного методом 'разделяй и властвуй'
    """
    n = len(points)
    
    # Для малого количества точек используем наивный алгоритм
    if n <= 3:
        return compute_voronoi_naive(points)
    
    # Сортируем точки по x-координате
    sorted_indices = np.argsort(points[:, 0])
    sorted_points = points[sorted_indices]
    
    # Разделяем на две части
    mid = n // 2
    left_points = sorted_points[:mid]
    right_points = sorted_points[mid:]
    
    # Рекурсивно строим диаграммы для левой и правой частей
    left_vertices, left_edges = compute_voronoi_divide_conquer(left_points)
    right_vertices, right_edges = compute_voronoi_divide_conquer(right_points)
    
    # Для простоты мы будем использовать наивный подход для всего набора точек
    # В полноценной реализации здесь должно быть слияние левой и правой диаграмм
    return compute_voronoi_naive(points)

def compute_voronoi_diagram(points):
    """
    Вычисляет диаграмму Вороного для набора точек
    """
    vertices, edges = compute_voronoi_naive(points)
    
    infinite_edges = []
    
    if len(vertices) == 0:
        return vertices, edges, infinite_edges
    
    # Вычисляем "центр" множества точек для определения направления бесконечных рёбер
    center = np.mean(points, axis=0)
    
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            # Находим середину отрезка между точками
            midpoint = (points[i] + points[j]) / 2
            # Перпендикулярный вектор
            perp_vector = np.array([-(points[j][1] - points[i][1]), points[j][0] - points[i][0]])
            perp_vector = perp_vector / np.linalg.norm(perp_vector)
            
            # Проверяем, есть ли конечное ребро для этой пары точек
            has_finite_edge = False
            for edge in edges:
                v1, v2 = vertices[edge[0]], vertices[edge[1]]
                # Проверяем, лежит ли середина ребра на перпендикуляре
                edge_midpoint = (v1 + v2) / 2
                if abs(np.dot(edge_midpoint - midpoint, points[j] - points[i])) < 1e-10:
                    has_finite_edge = True
                    break
            
            if not has_finite_edge:
                # Находим ближайшую вершину к этой паре точек
                closest_vertex_idx = None
                min_dist = float('inf')
                for v_idx, vertex in enumerate(vertices):
                    dist_i = euclidean_distance(vertex, points[i])
                    dist_j = euclidean_distance(vertex, points[j])
                    if abs(dist_i - dist_j) < 1e-10:
                        dist_to_midpoint = euclidean_distance(vertex, midpoint)
                        if dist_to_midpoint < min_dist:
                            min_dist = dist_to_midpoint
                            closest_vertex_idx = v_idx
                
                if closest_vertex_idx is not None:
                    # Определяем направление бесконечного ребра
                    vertex = vertices[closest_vertex_idx]
                    direction = np.sign(np.dot(midpoint - center, perp_vector))
                    if direction == 0:
                        direction = 1
                    infinite_edges.append((closest_vertex_idx, midpoint, direction * perp_vector))
    
    return vertices, edges, infinite_edges

def build_delaunay_from_voronoi_complete(points, voronoi_vertices, voronoi_edges, infinite_edges):
    """
    Строит полную триангуляцию Делоне на основе диаграммы Вороного,
    учитывая также бесконечные рёбра для построения выпуклой оболочки
    """
    n = len(points)
    delaunay_edges = []
    
    # Если вершин Вороного нет, строим минимальный базовый граф
    if len(voronoi_vertices) == 0:
        if n <= 1:
            return []
        if n == 2:
            return [(0, 1)]
        if n == 3:
            return [(0, 1), (1, 2), (2, 0)]
    
    # Для каждого ребра диаграммы Вороного находим соответствующие точки
    for v1_idx, v2_idx in voronoi_edges:
        v1 = voronoi_vertices[v1_idx]
        v2 = voronoi_vertices[v2_idx]
        
        # Находим точки, для которых это ребро является границей ячеек
        equidistant_points = []
        for i in range(n):
            dist_i_v1 = euclidean_distance(points[i], v1)
            dist_i_v2 = euclidean_distance(points[i], v2)
            
            for j in range(i + 1, n):
                dist_j_v1 = euclidean_distance(points[j], v1)
                dist_j_v2 = euclidean_distance(points[j], v2)
                
                if (abs(dist_i_v1 - dist_j_v1) < 1e-10 and 
                    abs(dist_i_v2 - dist_j_v2) < 1e-10):
                    equidistant_points.append((i, j))
        
        # Добавляем найденные пары точек как рёбра триангуляции Делоне
        for point_pair in equidistant_points:
            if point_pair not in delaunay_edges and (point_pair[1], point_pair[0]) not in delaunay_edges:
                delaunay_edges.append(point_pair)
    
    # Обрабатываем бесконечные рёбра для выпуклой оболочки
    point_neighbors = defaultdict(set)
    
    # Заполняем соседей для каждой точки
    for i, j in delaunay_edges:
        point_neighbors[i].add(j)
        point_neighbors[j].add(i)
    
    # Обрабатываем бесконечные рёбра
    for vertex_idx, midpoint, direction in infinite_edges:
        vertex = voronoi_vertices[vertex_idx]
        
        # Находим точки, для которых это бесконечное ребро является границей ячеек
        candidates = []
        for i in range(n):
            dist_i = euclidean_distance(points[i], vertex)
            
            for j in range(i + 1, n):
                dist_j = euclidean_distance(points[j], vertex)
                
                if abs(dist_i - dist_j) < 1e-10:
                    # Проверяем, что это ребро действительно соответствует точкам i и j
                    midpoint_ij = (points[i] + points[j]) / 2
                    perp_vector = np.array([-(points[j][1] - points[i][1]), points[j][0] - points[i][0]])
                    perp_vector = perp_vector / np.linalg.norm(perp_vector)
                    
                    # Если направления примерно параллельны, это наше ребро
                    dot_product = abs(np.dot(perp_vector, direction))
                    if dot_product > 0.99:  # Почти параллельны
                        candidates.append((i, j))
        
        # Добавляем найденные пары точек как рёбра триангуляции Делоне
        for point_pair in candidates:
            if point_pair not in delaunay_edges and (point_pair[1], point_pair[0]) not in delaunay_edges:
                delaunay_edges.append(point_pair)
    
    # Дополнительно: находим выпуклую оболочку и добавляем её рёбра
    hull_edges = compute_convex_hull(points)
    for edge in hull_edges:
        if edge not in delaunay_edges and (edge[1], edge[0]) not in delaunay_edges:
            delaunay_edges.append(edge)
    
    return delaunay_edges

def compute_convex_hull(points):
    """
    Вычисляет выпуклую оболочку с помощью алгоритма Джарвиса (метод заворачивания подарка)
    Возвращает список рёбер выпуклой оболочки
    """
    n = len(points)
    if n <= 2:
        if n == 2:
            return [(0, 1)]
        return []
    
    # Находим самую левую точку
    leftmost = min(range(n), key=lambda i: points[i][0])
    
    hull = []
    p = leftmost
    q = 0
    
    while True:
        hull.append(p)
        
        q = (p + 1) % n
        for i in range(n):
            # Если точка i слева от линии p-q
            if orientation(points[p], points[i], points[q]) == 2:
                q = i
        
        p = q
        
        # Завершаем, если вернулись к началу
        if p == leftmost:
            break
    
    # Строим рёбра выпуклой оболочки
    hull_edges = []
    for i in range(len(hull)):
        hull_edges.append((hull[i], hull[(i + 1) % len(hull)]))
    
    return hull_edges

def orientation(p, q, r):
    """
    Определяет ориентацию тройки точек (p, q, r)
    Возвращает:
    0 - коллинеарны
    1 - по часовой стрелке
    2 - против часовой стрелки
    """
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    
    if abs(val) < 1e-10:
        return 0  # коллинеарны
    
    return 2 if val > 0 else 1  # 2 для против часовой, 1 для по часовой

def plot_solution(points):
    """
    Строит две диаграммы: Вороного и Делоне на разных картинках
    с улучшенной визуализацией и без пунктирных линий
    """
    # Ищем ближайшую пару точек
    closest_pair, min_dist = find_closest_pair_divide_conquer(points)
    
    # Строим диаграмму Вороного
    voronoi_vertices, voronoi_edges, infinite_edges = compute_voronoi_diagram(points)
    
    # Строим триангуляцию Делоне с полным замыканием
    delaunay_edges = build_delaunay_from_voronoi_complete(points, voronoi_vertices, voronoi_edges, infinite_edges)
    
    # Настройка параметров графиков
    margin = 1.5  # Увеличиваем отступ для лучшей видимости
    x_min, y_min = np.min(points, axis=0) - margin
    x_max, y_max = np.max(points, axis=0) + margin
    
    # Диаграмма Вороного
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    
    # Настройка сетки - делаем её менее заметной
    ax1.grid(True, linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Рисуем точки
    ax1.scatter(points[:, 0], points[:, 1], c='black', s=70, zorder=3)
    
    # Рисуем конечные рёбра диаграммы Вороного
    for v1_idx, v2_idx in voronoi_edges:
        v1 = voronoi_vertices[v1_idx]
        v2 = voronoi_vertices[v2_idx]
        ax1.plot([v1[0], v2[0]], [v1[1], v2[1]], 
                 'purple', lw=2, zorder=1, solid_capstyle='round')
    
    # Рисуем бесконечные рёбра диаграммы Вороного - сплошные вместо пунктирных
    for vertex_idx, midpoint, direction in infinite_edges:
        vertex = voronoi_vertices[vertex_idx]
        scale = max(abs(x_max - x_min), abs(y_max - y_min)) * 2
        far_point = vertex + direction * scale
        ax1.plot([vertex[0], far_point[0]], [vertex[1], far_point[1]], 
                 'purple', linestyle='-', lw=2, zorder=1, solid_capstyle='round', alpha=0.7)
    
    # Рисуем вершины диаграммы Вороного
    if len(voronoi_vertices) > 0:
        ax1.scatter(voronoi_vertices[:, 0], voronoi_vertices[:, 1], 
                   c='purple', s=40, alpha=0.7, zorder=2, edgecolor='white')
    
    # Выделяем ближайшую пару точек на диаграмме Вороного
    ax1.plot([points[closest_pair[0]][0], points[closest_pair[1]][0]],
             [points[closest_pair[0]][1], points[closest_pair[1]][1]], 
             'lime', lw=3, zorder=4, solid_capstyle='round')
    ax1.scatter(points[closest_pair, 0], points[closest_pair, 1], 
               c='lime', s=120, zorder=5, edgecolor='black', linewidth=1)
    
    # Подписываем точки
    for i, point in enumerate(points):
        ax1.annotate(f"P{i+1}", (point[0]+0.1, point[1]+0.1), fontsize=10)
    
    ax1.set_title('Диаграмма Вороного', fontsize=14)
    ax1.set_aspect('equal')
    
    # Триангуляция Делоне
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    
    # Настройка сетки - делаем её менее заметной
    ax2.grid(True, linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Рисуем точки
    ax2.scatter(points[:, 0], points[:, 1], c='black', s=70, zorder=3)
    
    # Рисуем рёбра триангуляции Делоне
    for i, j in delaunay_edges:
        ax2.plot([points[i][0], points[j][0]], [points[i][1], points[j][1]], 
                'orange', lw=2, zorder=1, solid_capstyle='round')
    
    # Выделяем ближайшую пару точек на триангуляции Делоне
    ax2.plot([points[closest_pair[0]][0], points[closest_pair[1]][0]],
             [points[closest_pair[0]][1], points[closest_pair[1]][1]], 
             'lime', lw=3, zorder=4, solid_capstyle='round')
    ax2.scatter(points[closest_pair, 0], points[closest_pair, 1], 
               c='lime', s=120, zorder=5, edgecolor='black', linewidth=1)
    
    # Подписываем точки
    for i, point in enumerate(points):
        ax2.annotate(f"P{i+1}", (point[0]+0.1, point[1]+0.1), fontsize=10)
    
    ax2.set_title('Триангуляция Делоне', fontsize=14)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
    
    return closest_pair, min_dist, voronoi_vertices, voronoi_edges, delaunay_edges

# Основная функция
def main():
    # Генерируем случайные точки
    random.seed()  # Использование текущего времени для инициализации генератора случайных чисел
    num_points = 10
    points = np.random.rand(num_points, 2) * 10  # Точки в диапазоне от 0 до 10
    
    # Выполняем расчеты и визуализацию
    closest_pair, min_dist, voronoi_vertices, voronoi_edges, delaunay_edges = plot_solution(points)
    
    # Выводим информацию о точках
    print(f"Множество точек E:")
    for i, point in enumerate(points):
        print(f"P{i+1}: ({point[0]:.4f}, {point[1]:.4f})")
    
    print(f"\nБлижайшая пара точек: P{closest_pair[0]+1} и P{closest_pair[1]+1}")
    print(f"Расстояние между ними: {min_dist:.4f}")

if __name__ == "__main__":
    main()