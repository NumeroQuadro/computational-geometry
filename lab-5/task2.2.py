import random
import math
import matplotlib.pyplot as plt

def generate_points(n, xmin=0, xmax=20, ymin=0, ymax=20, seed=None):
    if seed is not None:
        random.seed(seed)
    pts = set()
    while len(pts) < n:
        pts.add((random.randint(xmin, xmax), random.randint(ymin, ymax)))
    return list(pts)

def cross(o, a, b):
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

def graham_scan(points):
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts
    # опорная точка
    pivot = min(pts, key=lambda p: (p[1], p[0]))
    pts.remove(pivot)
    # сортировка по полярному углу от pivot
    def key(p):
        dx, dy = p[0]-pivot[0], p[1]-pivot[1]
        return (math.atan2(dy, dx), dx*dx+dy*dy)
    pts.sort(key=key)
    # строим hull
    hull = [pivot, pts[0], pts[1]]
    for p in pts[2:]:
        while len(hull) >= 2 and cross(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)
    return hull

def find_intersection(a1, a2, b1, b2):
    x1,y1 = a1; x2,y2 = a2
    x3,y3 = b1; x4,y4 = b2
    den = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
    if abs(den) < 1e-9:
        return None
    ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / den
    ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / den
    if 0 <= ua <= 1 and 0 <= ub <= 1:
        return (x1+ua*(x2-x1), y1+ua*(y2-y1))
    return None

def is_on_edge(p, poly):
    x,y = p
    n = len(poly)
    for i in range(n):
        a = poly[i]; b = poly[(i+1)%n]
        if abs(cross(a,b,p))<1e-9 \
           and min(a[0],b[0])<=x<=max(a[0],b[0]) \
           and min(a[1],b[1])<=y<=max(a[1],b[1]):
            return True
    return False

def is_strictly_inside(p, poly):
    # исключаем границу
    if is_on_edge(p, poly):
        return False
    # обычный ray-casting
    x,y = p; inside = False
    n = len(poly)
    for i in range(n):
        x1,y1 = poly[i]
        x2,y2 = poly[(i+1)%n]
        if (y1>y) != (y2>y):
            xinters = (x2-x1)*(y-y1)/(y2-y1) + x1
            if xinters > x:
                inside = not inside
    return inside

def polygon_intersection(F1, F2):
    cands = []
    # 1) вершины F1 внутри или на границе F2
    for p in F1:
        if is_strictly_inside(p, F2) or is_on_edge(p, F2):
            cands.append(p)
    # 2) вершины F2 внутри или на границе F1
    for p in F2:
        if is_strictly_inside(p, F1) or is_on_edge(p, F1):
            cands.append(p)
    # 3) пересечения рёбер
    for i in range(len(F1)):
        a1,a2 = F1[i], F1[(i+1)%len(F1)]
        for j in range(len(F2)):
            b1,b2 = F2[j], F2[(j+1)%len(F2)]
            ip = find_intersection(a1,a2,b1,b2)
            if ip:
                cands.append(ip)
    if not cands:
        return []
    # убираем почти-совпадения
    uniq = []
    for p in cands:
        if not any(abs(p[0]-q[0])<1e-6 and abs(p[1]-q[1])<1e-6 for q in uniq):
            uniq.append(p)
    # окончательный hull из кандидатов
    return graham_scan(uniq)

# — main —
E1 = generate_points(9, seed=22)
E2 = generate_points(8, seed=7)
F1 = graham_scan(E1)
F2 = graham_scan(E2)
P  = polygon_intersection(F1, F2)

internal_E1 = [p for p in E1 if is_strictly_inside(p, P)]
internal_E2 = [p for p in E2 if is_strictly_inside(p, P)]

# — plotting —
plt.figure(figsize=(8,8))
for i,p in enumerate(E1):
    plt.scatter(*p, c='blue');  plt.text(p[0],p[1]+.3,f"E1_{i}",color='blue')
for i,p in enumerate(E2):
    plt.scatter(*p, c='red');   plt.text(p[0],p[1]+.3,f"E2_{i}",color='red')

h1 = F1+[F1[0]]; plt.plot([x for x,y in h1],[y for x,y in h1],'b-')
h2 = F2+[F2[0]]; plt.plot([x for x,y in h2],[y for x,y in h2],'r-')

if P:
    pC = P+[P[0]]
    plt.fill([x for x,y in pC],[y for x,y in pC],alpha=0.3,color='purple')

for p in internal_E1:
    plt.scatter(*p,c='cyan',marker='*',s=100)
for p in internal_E2:
    plt.scatter(*p,c='orange',marker='*',s=100)

plt.axis('equal'); plt.grid(True)
plt.show()

# — вывод в консоль —
print("E1:", E1)
print("E2:", E2)
print("conv(E1):", F1)
print("conv(E2):", F2)
print("P = conv(E1)∩conv(E2):", P)
print("Внутр. E1 в P:", internal_E1)
print("Внутр. E2 в P:", internal_E2)
