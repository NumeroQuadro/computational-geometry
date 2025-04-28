import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d
import random

def euclidean_distance(p1, p2):
    """Вычисляет евклидово расстояние между двумя точками"""
    return np.sqrt(np.sum((p1 - p2) ** 2))

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

def plot_solution(points):
    """
    Строит две диаграммы: Вороного и Делоне на разных картинках
    с использованием библиотеки scipy для более точных вычислений
    """
    # Ищем ближайшую пару точек
    closest_pair, min_dist = find_closest_pair_divide_conquer(points)
    
    # Используем scipy для построения диаграммы Вороного
    vor = Voronoi(points)
    
    # Используем scipy для построения триангуляции Делоне
    tri = Delaunay(points)
    
    # Настройка параметров графиков
    margin = 1.0
    x_min, y_min = np.min(points, axis=0) - margin
    x_max, y_max = np.max(points, axis=0) + margin
    
    # Диаграмма Вороного
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    
    # Рисуем точки
    ax1.scatter(points[:, 0], points[:, 1], c='black', s=70, zorder=3)
    
    # Рисуем диаграмму Вороного с помощью scipy
    voronoi_plot_2d(vor, ax=ax1, show_points=False, show_vertices=True,
                  line_colors='purple', line_width=2, line_alpha=0.8,
                  point_size=5)
    
    # Выделяем ближайшую пару точек на диаграмме Вороного
    ax1.plot([points[closest_pair[0]][0], points[closest_pair[1]][0]],
             [points[closest_pair[0]][1], points[closest_pair[1]][1]], 'lime', lw=3, zorder=4)
    ax1.scatter(points[closest_pair, 0], points[closest_pair, 1], c='lime', s=120, zorder=5)
    
    # Подписываем точки
    for i, point in enumerate(points):
        ax1.annotate(f"P{i+1}", (point[0]+0.1, point[1]+0.1), fontsize=10)
    
    ax1.set_title('Диаграмма Вороного')
    ax1.grid(True)
    ax1.set_aspect('equal')
    
    # Триангуляция Делоне
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    
    # Рисуем точки
    ax2.scatter(points[:, 0], points[:, 1], c='black', s=70, zorder=3)
    
    # Рисуем триангуляцию Делоне
    for simplex in tri.simplices:
        # Рисуем каждый треугольник
        for i in range(3):
            j = (i + 1) % 3
            ax2.plot([points[simplex[i], 0], points[simplex[j], 0]],
                    [points[simplex[i], 1], points[simplex[j], 1]], 'orange', lw=2, zorder=1)
    
    # Проверяем, что триангуляция образует выпуклую оболочку
    # Получаем выпуклую оболочку точек
    hull = tri.convex_hull
    for simplex in hull:
        ax2.plot([points[simplex[0], 0], points[simplex[1], 0]],
                [points[simplex[0], 1], points[simplex[1], 1]], 'orange', lw=3, zorder=2)
    
    # Выделяем ближайшую пару точек на триангуляции Делоне
    ax2.plot([points[closest_pair[0]][0], points[closest_pair[1]][0]],
             [points[closest_pair[0]][1], points[closest_pair[1]][1]], 'lime', lw=3, zorder=4)
    ax2.scatter(points[closest_pair, 0], points[closest_pair, 1], c='lime', s=120, zorder=5)
    
    # Подписываем точки
    for i, point in enumerate(points):
        ax2.annotate(f"P{i+1}", (point[0]+0.1, point[1]+0.1), fontsize=10)
    
    ax2.set_title('Триангуляция Делоне')
    ax2.grid(True)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
    
    return closest_pair, min_dist

# Основная функция
def main():
    # Генерируем случайные точки
    random.seed()  # Использование текущего времени для инициализации генератора случайных чисел
    num_points = 10
    points = np.random.rand(num_points, 2) * 10  # Точки в диапазоне от 0 до 10
    
    # Выполняем расчеты и визуализацию
    closest_pair, min_dist = plot_solution(points)
    
    # Выводим информацию о точках
    print(f"Множество точек E:")
    for i, point in enumerate(points):
        print(f"P{i+1}: ({point[0]:.4f}, {point[1]:.4f})")
    
    print(f"\nБлижайшая пара точек: P{closest_pair[0]+1} и P{closest_pair[1]+1}")
    print(f"Расстояние между ними: {min_dist:.4f}")

if __name__ == "__main__":
    main()