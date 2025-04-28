import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Входные точки
points = [
    (1, 1),   # P1
    (2, 4),   # P2
    (5, 2),   # P3
    (6, 6),   # P4
    (8, 3),   # P5
    (3, 7),   # P6
    (9, 8),   # P7
    (4, 9),   # P8
    (10, 1),  # P9
    (7, 5)    # P10
]

# Класс для представления ребра диаграммы Вороного
class Edge:
    def __init__(self, start, end, left_point, right_point):
        self.start = start
        self.end = end
        self.left_point = left_point
        self.right_point = right_point

    def __repr__(self):
        return f"Edge(start={self.start}, end={self.end}, left={self.left_point}, right={self.right_point})"

# Вычисление перпендикулярного бисектора для двух точек
def perpendicular_bisector(p1, p2):
    mid_x = (p1[0] + p2[0]) / 2
    mid_y = (p1[1] + p2[1]) / 2

    if p1[0] == p2[0]:  # Вертикальная линия
        return mid_x, None, mid_y, 'vertical'
    else:
        slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
        if slope == 0:  # Горизонтальная линия
            return None, mid_x, mid_y, 'horizontal'
        perp_slope = -1 / slope
        return perp_slope, mid_x, mid_y, 'normal'

# Нахождение пересечения двух бисекторов
def intersect_bisectors(b1, b2):
    slope1, mid_x1, mid_y1, type1 = b1
    slope2, mid_x2, mid_y2, type2 = b2

    if type1 == 'vertical' and type2 == 'vertical':
        return None
    elif type1 == 'vertical':
        x = mid_x1
        if type2 == 'horizontal':
            y = mid_y2
        else:
            y = slope2 * (x - mid_x2) + mid_y2
        return (x, y) if -10 <= x <= 20 and -10 <= y <= 20 else None
    elif type2 == 'vertical':
        x = mid_x2
        if type1 == 'horizontal':
            y = mid_y1
        else:
            y = slope1 * (x - mid_x1) + mid_y1
        return (x, y) if -10 <= x <= 20 and -10 <= y <= 20 else None
    elif type1 == 'horizontal' and type2 == 'horizontal':
        return None
    elif type1 == 'horizontal':
        y = mid_y1
        x = (y - mid_y2) / slope2 + mid_x2
        return (x, y) if -10 <= x <= 20 and -10 <= y <= 20 else None
    elif type2 == 'horizontal':
        y = mid_y2
        x = (y - mid_y1) / slope1 + mid_x1
        return (x, y) if -10 <= x <= 20 and -10 <= y <= 20 else None
    else:
        if abs(slope1 - slope2) < 1e-6:
            return None
        x = (slope1 * mid_x1 - slope2 * mid_x2 + mid_y2 - mid_y1) / (slope1 - slope2)
        y = slope1 * (x - mid_x1) + mid_y1
        return (x, y) if -10 <= x <= 20 and -10 <= y <= 20 else None

# Построение диаграммы Вороного методом Divide and Conquer
def voronoi_diagram(points):
    if len(points) <= 1:
        return [], []

    # Сортируем точки по x-координате
    points = sorted(points, key=lambda p: p[0])

    # Базовый случай: 2 или 3 точки
    if len(points) <= 3:
        return voronoi_base_case(points)

    # Разделяем на две половины
    mid = len(points) // 2
    left_points = points[:mid]
    right_points = points[mid:]

    # Рекурсивно строим диаграммы
    left_edges, left_vertices = voronoi_diagram(left_points)
    right_edges, right_vertices = voronoi_diagram(right_points)

    # Сливаем диаграммы
    edges, vertices = merge_voronoi(left_edges, right_edges, left_vertices, right_vertices, left_points, right_points)

    # Обрезаем рёбра до границ области
    bounds = (-2, 12, -2, 12)  # (xmin, xmax, ymin, ymax)
    clipped_edges = clip_edges(edges, bounds)

    return clipped_edges, vertices

# Базовый случай: диаграмма Вороного для 2 или 3 точек
def voronoi_base_case(points):
    edges = []
    vertices = []

    if len(points) == 2:
        p1, p2 = points
        bisector = perpendicular_bisector(p1, p2)
        if bisector[3] == 'vertical':
            edge = Edge((bisector[0], -2), (bisector[0], 12), p1, p2)
        elif bisector[3] == 'horizontal':
            edge = Edge((-2, bisector[2]), (12, bisector[2]), p1, p2)
        else:
            y1 = bisector[0] * (-2 - bisector[1]) + bisector[2]
            y2 = bisector[0] * (12 - bisector[1]) + bisector[2]
            edge = Edge((-2, y1), (12, y2), p1, p2)
        edges.append(edge)

    elif len(points) == 3:
        p1, p2, p3 = points
        bisector12 = perpendicular_bisector(p1, p2)
        bisector23 = perpendicular_bisector(p2, p3)
        bisector13 = perpendicular_bisector(p1, p3)

        v = intersect_bisectors(bisector12, bisector23)
        if v:
            vertices.append(v)
            # Обрезаем бисекторы до вершины
            for p_pair, bisector in [((p1, p2), bisector12), ((p2, p3), bisector23), ((p1, p3), bisector13)]:
                p1, p2 = p_pair
                if bisector[3] == 'vertical':
                    edge = Edge((bisector[0], v[1]), (bisector[0], -2 if p1[1] < v[1] else 12), p1, p2)
                elif bisector[3] == 'horizontal':
                    edge = Edge((v[0], bisector[2]), (-2 if p1[0] < v[0] else 12, bisector[2]), p1, p2)
                else:
                    y1 = bisector[0] * (-2 - bisector[1]) + bisector[2]
                    y2 = bisector[0] * (12 - bisector[1]) + bisector[2]
                    if (p1[0] + p2[0]) / 2 < v[0]:
                        edge = Edge((v[0], v[1]), (-2, y1), p1, p2)
                    else:
                        edge = Edge((v[0], v[1]), (12, y2), p1, p2)
                edges.append(edge)

    return edges, vertices

# Слияние двух диаграмм Вороного
def merge_voronoi(left_edges, right_edges, left_vertices, right_vertices, left_points, right_points):
    edges = left_edges + right_edges
    vertices = left_vertices + right_vertices

    # Находим ближайшую пару между левой и правой половинами
    min_dist = float('inf')
    closest_pair = None
    for lp in left_points:
        for rp in right_points:
            dist = ((lp[0] - rp[0])**2 + (lp[1] - rp[1])**2)**0.5
            if dist < min_dist:
                min_dist = dist
                closest_pair = (lp, rp)

    # Строим бисектор для ближайшей пары
    p1, p2 = closest_pair
    bisector = perpendicular_bisector(p1, p2)
    if bisector[3] == 'vertical':
        edge = Edge((bisector[0], -2), (bisector[0], 12), p1, p2)
    elif bisector[3] == 'horizontal':
        edge = Edge((-2, bisector[2]), (12, bisector[2]), p1, p2)
    else:
        y1 = bisector[0] * (-2 - bisector[1]) + bisector[2]
        y2 = bisector[0] * (12 - bisector[1]) + bisector[2]
        edge = Edge((-2, y1), (12, y2), p1, p2)
    edges.append(edge)

    # Находим новые вершины (пересечения бисекторов)
    for i, e1 in enumerate(edges):
        for e2 in edges[i+1:]:
            b1 = perpendicular_bisector(e1.left_point, e1.right_point)
            b2 = perpendicular_bisector(e2.left_point, e2.right_point)
            v = intersect_bisectors(b1, b2)
            if v and v not in vertices:
                vertices.append(v)

    return edges, vertices

# Обрезка рёбер до границ области или вершин
def clip_edges(edges, bounds):
    xmin, xmax, ymin, ymax = bounds
    clipped_edges = []
    vertices = set()

    # Собираем все вершины (пересечения рёбер)
    for i, e1 in enumerate(edges):
        for e2 in edges[i+1:]:
            b1 = perpendicular_bisector(e1.left_point, e1.right_point)
            b2 = perpendicular_bisector(e2.left_point, e2.right_point)
            v = intersect_bisectors(b1, b2)
            if v:
                vertices.add(v)

    for edge in edges:
        start, end = edge.start, edge.end
        if not start or not end:
            continue

        # Находим ближайшие вершины к концам ребра
        start_closest = min(vertices, key=lambda v: (v[0] - start[0])**2 + (v[1] - start[1])**2) if vertices else start
        end_closest = min(vertices, key=lambda v: (v[0] - end[0])**2 + (v[1] - end[1])**2) if vertices else end

        # Обрезаем до ближайших вершин или границ
        new_start = start_closest if (xmin <= start_closest[0] <= xmax and ymin <= start_closest[1] <= ymax) else start
        new_end = end_closest if (xmin <= end_closest[0] <= xmax and ymin <= end_closest[1] <= ymax) else end

        # Проверяем, попадает ли ребро в область
        if (xmin <= new_start[0] <= xmax and ymin <= new_start[1] <= ymax) or (xmin <= new_end[0] <= xmax and ymin <= new_end[1] <= ymax):
            clipped_edges.append(Edge(new_start, new_end, edge.left_point, edge.right_point))

    return clipped_edges

# Нахождение ближайшей пары точек
def closest_pair(voronoi_edges, points):
    min_dist = float('inf')
    closest_pair = None

    for edge in voronoi_edges:
        p1 = edge.left_point
        p2 = edge.right_point
        if p1 and p2:
            dist = ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
            if dist < min_dist:
                min_dist = dist
                closest_pair = (p1, p2)

    return closest_pair, min_dist

# Построение триангуляции Делоне
def delaunay_triangulation(voronoi_edges, points):
    delaunay_edges = set()
    for edge in voronoi_edges:
        p1 = edge.left_point
        p2 = edge.right_point
        if p1 and p2:
            delaunay_edges.add((min(p1, p2), max(p1, p2)))

    # Формируем граф соседства
    adj = defaultdict(set)
    for p1, p2 in delaunay_edges:
        adj[p1].add(p2)
        adj[p2].add(p1)

    # Формируем треугольники
    triangles = []
    for p1 in points:
        neighbors = sorted(list(adj[p1]), key=lambda p: (p[0] - p1[0], p[1] - p1[1]))
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                p2, p3 = neighbors[i], neighbors[j]
                if p3 in adj[p2]:  # Проверяем, замкнут ли треугольник
                    triangle = tuple(sorted([p1, p2, p3]))
                    if triangle not in triangles:
                        triangles.append(triangle)

    return list(delaunay_edges), triangles

# Визуализация
def visualize(points, voronoi_edges, closest_pair, delaunay_edges, delaunay_triangles):
    plt.figure(figsize=(10, 10))

    # 1. Рисуем точки
    x, y = zip(*points)
    plt.scatter(x, y, color='blue', label='Points', zorder=3)
    for i, (px, py) in enumerate(points):
        plt.text(px + 0.3, py, f"P{i+1}", fontsize=12, zorder=4)

    # 2. Рисуем диаграмму Вороного
    for edge in voronoi_edges:
        if edge.start and edge.end:
            plt.plot([edge.start[0], edge.end[0]], [edge.start[1], edge.end[1]], 'g--', label='Voronoi Edges' if 'Voronoi Edges' not in plt.gca().get_legend_handles_labels()[1] else "")

    # 3. Рисуем треугольники Делоне
    for triangle in delaunay_triangles:
        p1, p2, p3 = triangle
        x = [p1[0], p2[0], p3[0], p1[0]]
        y = [p1[1], p2[1], p3[1], p1[1]]
        plt.fill(x, y, 'cyan', alpha=0.3, label='Delaunay Triangles' if 'Delaunay Triangles' not in plt.gca().get_legend_handles_labels()[1] else "")

    # 4. Рисуем рёбра Делоне
    for p1, p2 in delaunay_edges:
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', label='Delaunay Edges' if 'Delaunay Edges' not in plt.gca().get_legend_handles_labels()[1] else "")

    # 5. Выделяем ближайшую пару
    if closest_pair:
        p1, p2 = closest_pair
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=2, label='Closest Pair')

    plt.xlim(-2, 12)
    plt.ylim(-2, 12)
    plt.grid(True)
    plt.legend()
    plt.title("Voronoi Diagram, Closest Pair, and Delaunay Triangulation")
    plt.savefig("voronoi_diagram.png")

# Основная программа
def main():
    # Построение диаграммы Вороного
    voronoi_edges, voronoi_vertices = voronoi_diagram(points)

    # Нахождение ближайшей пары
    pair, distance = closest_pair(voronoi_edges, points)
    print("Ближайшая пара точек:", pair, "Расстояние:", distance)

    # Построение триангуляции Делоне
    delaunay_edges, delaunay_triangles = delaunay_triangulation(voronoi_edges, points)
    print("Триангуляция Делоне (треугольники):", delaunay_triangles)

    # Визуализация
    visualize(points, voronoi_edges, pair, delaunay_edges, delaunay_triangles)

if __name__ == "__main__":
    main()