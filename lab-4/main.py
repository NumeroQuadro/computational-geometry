# Вершины выпуклого 7-угольника (по часовой стрелке)
polygon = [
    (0, 0),   # P1
    (2, 3),   # P2
    (5, 4),   # P3
    (7, 2),   # P4
    (6, -1),  # P5
    (3, -3),  # P6
    (1, -2)   # P7
]

# Вспомогательная точка O
O = (3, 1)

# Тестовая точка Q
Q = (4, 2)

def is_point_inside_polygon(polygon, point):
    x, y = point
    n = len(polygon)
    inside = False
    
    # Проходим по всем рёбрам многоугольника
    for i in range(n):
        j = (i + 1) % n  # Следующая вершина (с зацикливанием)
        
        # Координаты вершин текущего ребра
        x1, y1 = polygon[i]
        x2, y2 = polygon[j]
        
        # Проверяем пересечение горизонтального луча из точки с ребром
        # Условие 1: ребро пересекает уровень y-координаты точки
        # Условие 2: точка левее x-координаты пересечения
        if ((y1 > y) != (y2 > y)) and \
           (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
            inside = not inside
    
    return inside

# Функция для вычисления положения точки
def determine_point_position(polygon, O, Q):
    # Проверяем положение точки Q
    q_inside = is_point_inside_polygon(polygon, Q)
    
    result = {
        "Point Q": {
            "coordinates": Q,
            "position": "inside" if q_inside else "outside"
        },
        "Auxiliary point O": {
            "coordinates": O,
            "position": "given as reference"
        }
    }
    
    return result

# Тестирование
result = determine_point_position(polygon, O, Q)
print("Polygon vertices:", polygon)
print("Analysis result:", result)