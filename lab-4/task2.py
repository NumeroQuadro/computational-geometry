import math
import matplotlib.pyplot as plt

heptagon = [(0, 0), (2, -1), (4, 0), (5, 2), (4, 4), (2, 5), (0, 4)]
test_points = [(6, 2), (3, 3), (10, 10), (2, 7)]

def vector_angle(p, v1, v2):
    vec1 = (v1[0] - p[0], v1[1] - p[1])
    vec2 = (v2[0] - p[0], v2[1] - p[1])
    
    dot = vec1[0] * vec2[0] + vec1[1] * vec2[1]
    mag1 = math.sqrt(vec1[0]**2 + vec1[1]**2)
    mag2 = math.sqrt(vec2[0]**2 + vec2[1]**2)
    
    if mag1 == 0 or mag2 == 0:
        return 0
    
    cos_theta = max(-1, min(1, dot / (mag1 * mag2)))
    angle = math.degrees(math.acos(cos_theta))
    
    cross = vec1[0] * vec2[1] - vec1[1] * vec2[0]
    if cross < 0:
        angle = -angle
    
    return angle

def is_point_in_convex_polygon_angle_method(point, vertices):
    n = len(vertices)
    total_angle = 0
    
    for i in range(n):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % n]
        angle = vector_angle(point, v1, v2)
        total_angle += angle
    
    epsilon = 1e-10
    return abs(abs(total_angle) - 360) < epsilon

def plot_polygon_and_points(vertices, points, title):
    plt.figure()
    x, y = zip(*(vertices + [vertices[0]]))
    plt.plot(x, y, 'b-', label='Heptagon')
    
    if points:
        px, py = zip(*points)
        plt.scatter(px, py, color='red', label='Test points')
    else:
        print("нет подходящий точек")
    
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

new_test_points = []
for point in test_points:
    is_on_vertex = False
    for v in heptagon:
        if point[0] == v[0] and point[1] == v[1]:
            is_on_vertex = True
            break
    if not is_on_vertex:
        new_test_points.append(point)

print("Task 1: Семишник методом углов")
print("Вершины:", heptagon)
for point in new_test_points:
    result = is_point_in_convex_polygon_angle_method(point, heptagon)
    print(f"Точка {point} {'внутри' if result else 'снаружи'} семиугольника")

plot_polygon_and_points(heptagon, new_test_points, ":)")