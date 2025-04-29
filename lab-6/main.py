import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random

#------------------------------------------------------------------------
# 1) вспомогательные функции: bisector и Sutherland–Hodgman clipper
#------------------------------------------------------------------------
def bisector(p, q):
    """Коэффициенты A,B,C серед.-перпендикуляра (A x + B y = C)"""
    mid = (p + q) / 2
    dx, dy = q - p
    A, B = dx, dy
    C = A*mid[0] + B*mid[1]
    return A, B, C

def clip_polygon(poly, A, B, C):
    """Обрезаем выпуклый poly по полуплоскости A*x + B*y <= C"""
    out = []
    n = len(poly)
    for i in range(n):
        P, Q = poly[i], poly[(i+1)%n]
        inP = (A*P[0] + B*P[1]) <= C
        inQ = (A*Q[0] + B*Q[1]) <= C
        if inP:
            out.append(P)
        if inP ^ inQ:
            # найдём t пересечения
            t = (C - (A*P[0] + B*P[1])) / (A*(Q[0]-P[0]) + B*(Q[1]-P[1]))
            out.append(P + t*(Q-P))
    return np.array(out)

class VoronoiCell:
    def __init__(self, site, polygon):
        self.site = site
        self.poly = polygon

#------------------------------------------------------------------------
# 2) рекурсивный алгоритм «разделяй и властвуй» с настоящим merge
#------------------------------------------------------------------------
def voronoi_dnc(sites, bbox):
    """
    Строит Voronoi-диаграмму для массива sites (n×2) внутри bbox (4×2).
    Возвращает список VoronoiCell(site, poly).
    """
    n = len(sites)
    if n == 1:
        return [VoronoiCell(sites[0], bbox.copy())]
    if n == 2:
        A,B,C = bisector(sites[0], sites[1])
        cell0 = VoronoiCell(sites[0], clip_polygon(bbox.copy(),  A,  B,  C))
        cell1 = VoronoiCell(sites[1], clip_polygon(bbox.copy(), -A, -B, -C))
        return [cell0, cell1]

    # 2.1) split по x
    idx = np.argsort(sites[:,0])
    left = sites[idx[:n//2]]
    right= sites[idx[n//2:]]

    # 2.2) рекурсивно строим
    left_cells  = voronoi_dnc(left,  bbox)
    right_cells = voronoi_dnc(right, bbox)

    # 2.3) merge: отсекаем каждую левую ячейку бисекторами всех правых сайтов,
    #             и каждую правую — бисекторами всех левых сайтов
    merged = []
    left_sites  = [c.site for c in left_cells]
    right_sites = [c.site for c in right_cells]

    for cell in left_cells + right_cells:
        poly = cell.poly.copy()
        # если сайт в левой части — отсекаем правыми, иначе — левыми
        others = right_sites if any((cell.site==x).all() for x in left_sites) else left_sites
        for s in others:
            A,B,C = bisector(cell.site, s)
            # убедиться, что мы оставляем ту сторону, где наша точка
            if (A*s[0] + B*s[1]) < (A*cell.site[0] + B*cell.site[1]):
                A,B,C = -A,-B,-C
            poly = clip_polygon(poly, A, B, C)
            if poly.size == 0:
                break
        merged.append(VoronoiCell(cell.site, poly))

    return merged

#------------------------------------------------------------------------
# 3) вспомогалки для Delaunay & Closest Pair
#------------------------------------------------------------------------
def extract_delaunay(cells, bbox):
    from collections import defaultdict
    import numpy as np

    # 1) вспомогательная функция для оболочки (Jarvis / Gift wrapping)
    def convex_hull_edges(points):
        n = len(points)
        if n <= 1:
            return []
        # найти самую левую
        left = min(range(n), key=lambda i: points[i,0])
        hull = []
        p = left
        while True:
            hull.append(p)
            q = (p+1) % n
            for r in range(n):
                # если r левее линии p->q
                cross = ((points[q,0]-points[p,0])*(points[r,1]-points[p,1])
                         - (points[q,1]-points[p,1])*(points[r,0]-points[p,0]))
                if cross < 0:
                    q = r
            p = q
            if p == left:
                break
        # собираем пары
        edges = []
        for i in range(len(hull)):
            a = hull[i]
            b = hull[(i+1)%len(hull)]
            edges.append(tuple(sorted((a,b))))
        return edges

    # 2) маппинг из site→индекс
    index_of = {tuple(c.site): i for i,c in enumerate(cells)}

    # границы ящика
    min_x, min_y = bbox.min(axis=0)
    max_x, max_y = bbox.max(axis=0)
    eps = 1e-6

    # 3) соберём «внутренние» ребра по граням Вороного
    edge_map = defaultdict(set)
    for c in cells:
        V = c.poly
        m = len(V)
        for k in range(m):
            v1 = tuple(np.round(V[k],   6))
            v2 = tuple(np.round(V[(k+1)%m], 6))
            edge_map[frozenset((v1,v2))].add(tuple(c.site))

    delaunay = set()
    for verts, sites in edge_map.items():
        if len(sites) != 2:
            continue
        (x1,y1),(x2,y2) = verts
        # пропускаем бесконечные грани (те, что на бокс-границе)
        if (x1 < min_x+eps or x1 > max_x-eps or y1 < min_y+eps or y1 > max_y-eps or
            x2 < min_x+eps or x2 > max_x-eps or y2 < min_y+eps or y2 > max_y-eps):
            continue
        i,j = sorted(index_of[s] for s in sites)
        delaunay.add((i,j))

    # 4) теперь дополняем рёбрами выпуклой оболочки
    sites_arr = np.array([c.site for c in cells])
    for i,j in convex_hull_edges(sites_arr):
        delaunay.add((i,j))

    return list(delaunay)


def find_closest_pair(points):
    """Простейший brute-force (можно заменить вашим D&C)"""
    n=len(points)
    best=float('inf')
    pair=(0,1)
    for i in range(n):
        for j in range(i+1,n):
            d=np.linalg.norm(points[i]-points[j])
            if d<best:
                best, pair = d, (i,j)
    return pair, best

#------------------------------------------------------------------------
# 4) пример использования & визуализация
#------------------------------------------------------------------------
def main():
    random.seed(1)
    # генерим точки и bounding box
    pts = np.random.rand(10,2)*10
    min_xy = pts.min(axis=0)-1
    max_xy = pts.max(axis=0)+1
    bbox = np.array([[min_xy[0],min_xy[1]],
                     [min_xy[0],max_xy[1]],
                     [max_xy[0],max_xy[1]],
                     [max_xy[0],min_xy[1]]])

    # Voronoi D&C
    cells = voronoi_dnc(pts, bbox)

    # Delaunay из Voronoi
    edges = extract_delaunay(cells, bbox)

    # Closest pair
    (i0,i1), dmin = find_closest_pair(pts)

    # --- рисуем Voronoi ---
    plt.figure(figsize=(6,6))
    for c in cells:
        if len(c.poly)>0:
            plt.fill(c.poly[:,0], c.poly[:,1], alpha=0.3)
    plt.scatter(pts[:,0], pts[:,1], c='k', zorder=5)
    # выделим пару
    plt.plot([pts[i0,0],pts[i1,0]],[pts[i0,1],pts[i1,1]], 'lime', lw=3)
    plt.title("Voronoi")
    plt.grid(True)
    plt.axis('equal')
    plt.show()

    # --- рисуем Delaunay ---
    plt.figure(figsize=(6,6))
    plt.scatter(pts[:,0], pts[:,1], c='k', zorder=5)
    for i,j in edges:
        plt.plot([pts[i,0],pts[j,0]],[pts[i,1],pts[j,1]], 'r-')
    plt.title("Delaunay")
    plt.grid(True)
    plt.axis('equal')
    plt.show()

    print("Closest pair:", (i0,i1), "distance=", dmin)

if __name__ == "__main__":
    main()
