from matplotlib import pyplot as plt
import numpy as np
from collections import defaultdict


def naive_voronoi(points):
    n = len(points)

    vertices = []
    ridge_vertices = []

    vertex_dict = {}
    cells = defaultdict(list)

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]

                if center := circumcenter(p1, p2, p3):
                    r = np.sqrt((center[0] - p1[0]) ** 2 + (center[1] - p1[1]) ** 2)
                    is_valid = all(
                        np.sqrt((center[0] - points[l][0]) ** 2 + (center[1] - points[l][1]) ** 2) > r
                        for l in range(n) if l != i and l != j and l != k)

                    if is_valid:
                        center_tuple = (round(center[0], 10), round(center[1], 10))
                        if center_tuple not in vertex_dict:
                            vertex_dict[center_tuple] = len(vertices)
                            vertices.append(center)

    ridge_points = []

    for i in range(n):
        for j in range(i + 1, n):
            p1, p2 = points[i], points[j]

            perp_vertices = []
            for v_idx, vertex in enumerate(vertices):
                dist1 = np.sqrt((vertex[0] - p1[0]) ** 2 + (vertex[1] - p1[1]) ** 2)
                dist2 = np.sqrt((vertex[0] - p2[0]) ** 2 + (vertex[1] - p2[1]) ** 2)
                if abs(dist1 - dist2) < 1e-10:
                    perp_vertices.append(v_idx)

            if len(perp_vertices) == 1:
                ridge_points.append([i, j])
                ridge_vertices.append([perp_vertices[0], -1])
            else:
                for v1_idx in range(len(perp_vertices)):
                    for v2_idx in range(v1_idx + 1, len(perp_vertices)):
                        v1, v2 = perp_vertices[v1_idx], perp_vertices[v2_idx]

                        is_valid_edge = True

                        if is_valid_edge:
                            ridge_points.append([i, j])
                            ridge_vertices.append([v1, v2])
                            cells[i].append(len(ridge_vertices) - 1)
                            cells[j].append(len(ridge_vertices) - 1)

    return points, np.array(vertices), ridge_vertices, ridge_points


def circumcenter(p1, p2, p3):
    D = 2 * (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))

    if abs(D) < 1e-10:
        return None

    Ux = ((p1[0] ** 2 + p1[1] ** 2) * (p2[1] - p3[1]) +
          (p2[0] ** 2 + p2[1] ** 2) * (p3[1] - p1[1]) +
          (p3[0] ** 2 + p3[1] ** 2) * (p1[1] - p2[1])) / D

    Uy = ((p1[0] ** 2 + p1[1] ** 2) * (p3[0] - p2[0]) +
          (p2[0] ** 2 + p2[1] ** 2) * (p1[0] - p3[0]) +
          (p3[0] ** 2 + p3[1] ** 2) * (p2[0] - p1[0])) / D

    return Ux, Uy


np.random.seed(0)
points = np.random.rand(15, 2)

points, vertices, ridge_vertices, ridge_points = naive_voronoi(points)

plt.figure(figsize=(10, 10))
for pointidx, simplex in zip(ridge_points, ridge_vertices):
    simplex = np.asarray(simplex)
    if np.all(simplex >= 0):
        plt.plot(vertices[simplex, 0], vertices[simplex, 1], 'b')
        continue
    i = simplex[simplex >= 0][0]
    t = points[pointidx[1]] - points[pointidx[0]]
    n = np.array([-t[1], t[0]])
    midpoint = points[pointidx].mean(axis=0)

    direction = np.sign(np.dot(midpoint - points.mean(axis=0), n))
    if direction == 0:
        direction = 1

    far_point = vertices[i] + direction * n * 100
    plt.plot([vertices[i, 0], far_point[0]], [vertices[i, 1], far_point[1]], 'b--')

plt.scatter(vertices[:, 0], vertices[:, 1], c='blue', s=30, alpha=0.7)

for i, j in ridge_points:
    plt.plot([points[i][0], points[j][0]], [points[i][1], points[j][1]], 'r-')

plt.scatter(points[:, 0], points[:, 1], c='k')

plt.scatter(vertices[:, 0], vertices[:, 1], c='blue', s=30, alpha=0.7)

plt.title('Диаграмма Вороного (синий) и триангуляция Делоне (красный)')
plt.axis('equal')
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.show()