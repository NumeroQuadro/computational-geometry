import random
import matplotlib.pyplot as plt

def generate_points(n, xmin=0, xmax=20, ymin=0, ymax=20, seed=None):
    """Generate n random integer points in the given box (no duplicates)."""
    if seed is not None:
        random.seed(seed)
    pts = set()
    while len(pts) < n:
        pts.add((random.randint(xmin, xmax), random.randint(ymin, ymax)))
    return list(pts)

def cross(o, a, b):
    """2D cross product of OA × OB."""
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

def graham_scan(points):
    """Graham’s scan: возвращает вершины выпуклой оболочки в порядке обхода."""
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts

    # 1) Найти опорную точку с минимальным y (и минимальным x при равенстве)
    pivot = min(pts, key=lambda p: (p[1], p[0]))
    pts.remove(pivot)

    # 2) Отсортировать остальные по полярному углу от pivot (и по расстоянию при равных углах)
    def polar_key(p):
        dx, dy = p[0] - pivot[0], p[1] - pivot[1]
        import math
        angle = math.atan2(dy, dx)
        dist = dx*dx + dy*dy
        return (angle, dist)

    pts.sort(key=polar_key)

    # 3) Стек: добавляем pivot и первые два
    hull = [pivot, pts[0], pts[1]]

    # 4) Проходим по остальным и поддерживаем правый поворот
    for p in pts[2:]:
        while len(hull) >= 2 and cross(hull[-2], hull[-1], p) <= 0:
            hull.pop() # убираем точки до тех пор пока не вернемся к правому повороту
        hull.append(p)

    return hull

def sutherland_hodgman(subject, clipper):
    """обрезок многоугольник `subject` по многоугольнику `clipper`."""
    def inside(p, a, b):
        return cross(a, b, p) >= 0

    def intersect(s, e, a, b):
        ds = (e[0]-s[0], e[1]-s[1])
        dc = (b[0]-a[0], b[1]-a[1])
        denom = ds[0]*dc[1] - ds[1]*dc[0]
        if abs(denom) < 1e-9:
            return None
        t = ((a[0]-s[0])*dc[1] - (a[1]-s[1])*dc[0]) / denom
        return (s[0]+t*ds[0], s[1]+t*ds[1])

    output = subject
    for i in range(len(clipper)):
        a = clipper[i-1]
        b = clipper[i]
        input_list = output
        output = []
        if not input_list:
            break
        s = input_list[-1]
        for e in input_list:
            if inside(e, a, b):
                if not inside(s, a, b):
                    ip = intersect(s, e, a, b)
                    if ip:
                        output.append(ip)
                output.append(e)
            elif inside(s, a, b):
                ip = intersect(s, e, a, b)
                if ip:
                    output.append(ip)
            s = e
    return output

def is_strictly_inside(pt, poly):
    """True если точка строго внутри (не на границе тоже)"""
    x, y = pt
    n = len(poly)
    # boundary check
    for i in range(n):
        a, b = poly[i], poly[(i+1)%n]
        if abs(cross(a, b, pt)) < 1e-9 \
           and min(a[0],b[0]) -1e-9 <= x <= max(a[0],b[0]) +1e-9 \
           and min(a[1],b[1]) -1e-9 <= y <= max(a[1],b[1]) +1e-9:
            return False
    # ray‐casting
    inside = False
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[(i+1)%n]
        if (yi > y) != (yj > y):
            xint = (xj - xi) * (y - yi) / (yj - yi) + xi
            if xint > x:
                inside = not inside
    return inside

E1 = generate_points(9, seed=22)
E2 = generate_points(8, seed=7)

F1 = graham_scan(E1)
F2 = graham_scan(E2)

P = sutherland_hodgman(F1, F2)  # conv(E1) ∩ conv(E2)

internal_E1 = [p for p in E1 if is_strictly_inside(p, P)]
internal_E2 = [p for p in E2 if is_strictly_inside(p, P)]

# ———————— Plot —
plt.figure(figsize=(8,8))

# original points
for i,p in enumerate(E1):
    plt.scatter(*p, c='blue')
    plt.text(p[0], p[1]+0.3, f"E1_{i}", ha='center', color='blue')
for i,p in enumerate(E2):
    plt.scatter(*p, c='red')
    plt.text(p[0], p[1]+0.3, f"E2_{i}", ha='center', color='red')

# hulls
h1 = F1 + [F1[0]]
h2 = F2 + [F2[0]]
plt.plot([x for x,y in h1],[y for x,y in h1], 'b-', label='conv(E1)')
plt.plot([x for x,y in h2],[y for x,y in h2], 'r-', label='conv(E2)')

# intersection polygon
if P:
    Pi = P + [P[0]]
    plt.fill([x for x,y in Pi],[y for x,y in Pi], alpha=0.4, color='purple', label='conv(E1)∩conv(E2)')

# internal points
for p in internal_E1:
    plt.scatter(*p, c='cyan', s=100, marker='*', label='int(pt) in E1' if p==internal_E1[0] else "")
for p in internal_E2:
    plt.scatter(*p, c='orange', s=100, marker='*', label='int(pt) in E2' if p==internal_E2[0] else "")

plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()

print("E1:", E1)
print("E2:", E2)
print("conv(E1) vertices (F1):", F1)
print("conv(E2) vertices (F2):", F2)
print("Intersection P has vertices:", P)
print("Points of E1 strictly inside P:", internal_E1)
print("Points of E2 strictly inside P:", internal_E2)
