import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D

# 1. Зададим квадрат KLMN с произвольными координатами
# Пусть квадрат будет иметь сторону длиной 2 и центр в начале координат
K = np.array([-1, -1])
L = np.array([1, -1])
M = np.array([1, 1])
N = np.array([-1, 1])

# Функции для работы с однородными координатами
def to_homogeneous(point):
    """Преобразование из декартовых координат в однородные"""
    return np.append(point, 1)

def from_homogeneous(point):
    """Преобразование из однородных координат в декартовы"""
    return point[0:2] / point[2]

# Составим матрицу однородных координат вершин квадрата
square_vertices = np.array([K, L, M, N])
square_homogeneous = np.column_stack((square_vertices, np.ones(4)))
print("Матрица однородных координат вершин квадрата:")
print(square_homogeneous)

# 2. Найдем аффинное преобразование F, при котором образом квадрата будет параллелограмм ABCD
# По условию: F(K)=A, F(L)=B, F(M)=C, F(N)=D

# Условия:
# - вершина A лежит на луче KM и находится в три раза дальше от вершины K, чем точка M
# Значит, A = K + 3*(M-K) = K + 3*M - 3*K = 3*M - 2*K
A = K + 3 * (M - K)
print(f"Координаты точки A: {A}")

# - образ стороны KN ей параллелен и его длина в два раза больше KN
# Поскольку KN вертикальна, B должна быть на той же вертикали что и A,
# и быть выше на расстояние 2*|KN| = 2*2 = 4
B = A + np.array([0, 4])
print(f"Координаты точки B: {B}")

# - угол параллелограмма при вершине A равен π/4
# - высота параллелограмма BH, H ∈ AD, равна двум сторонам квадрата
# Длина стороны квадрата = 2, поэтому высота = 4
# Из этого можно вычислить координаты точки C

# Вычислим направление AD под углом π/4 к горизонтали
angle = np.pi/4
direction_AD = np.array([np.cos(angle), np.sin(angle)])
direction_AD = direction_AD / np.linalg.norm(direction_AD)

# Найдем длину AD такую, чтобы высота была равна 4
# Высота = |AD| * sin(π/4) = 4
# |AD| = 4 / sin(π/4) = 4 / (1/√2) = 4√2
length_AD = 4 / np.sin(angle)
D = A + length_AD * direction_AD
print(f"Координаты точки D: {D}")

# C = B + (D - A)
C = B + (D - A)
print(f"Координаты точки C: {C}")

# Проверим, что высота BH действительно равна 4
# Высота BH — это проекция вектора BA на перпендикуляр к AD
perpendicular_AD = np.array([-direction_AD[1], direction_AD[0]])  # Перпендикуляр к AD
height = abs(np.dot(B - A, perpendicular_AD))
print(f"Высота параллелограмма: {height}")

# 3. Составим матрицу преобразования F в однородных координатах
# Нам известны координаты точек K, L, M, N и их образов A, B, C, D
# Можем составить систему уравнений и решить её

# Параллелограмм в однородных координатах
parallelogram_vertices = np.array([A, B, C, D])
parallelogram_homogeneous = np.column_stack((parallelogram_vertices, np.ones(4)))

# Для нахождения матрицы преобразования F, мы должны решить уравнение:
# F * square_homogeneous^T = parallelogram_homogeneous^T
# где ^T означает транспонирование

# Решаем систему уравнений для нахождения матрицы F
F = np.dot(parallelogram_homogeneous.T, np.linalg.pinv(square_homogeneous.T))
print("\nМатрица преобразования F:")
print(F)

# 4. Проверим, что F действительно преобразует квадрат в параллелограмм
transformed_square = np.dot(square_homogeneous, F.T)
transformed_square = transformed_square[:, :2] / transformed_square[:, 2:]
print("\nКоординаты преобразованного квадрата:")
print(transformed_square)
print("\nКоординаты целевого параллелограмма:")
print(parallelogram_vertices)

# 5. Найдем матрицу обратного преобразования F^(-1)
F_inverse = np.linalg.inv(F)
print("\nМатрица обратного преобразования F^(-1):")
print(F_inverse)

# 6. Проверим, что F^(-1) преобразует параллелограмм обратно в квадрат
inverse_transformed_parallelogram = np.dot(parallelogram_homogeneous, F_inverse.T)
inverse_transformed_parallelogram = inverse_transformed_parallelogram[:, :2] / inverse_transformed_parallelogram[:, 2:]
print("\nКоординаты обратно преобразованного параллелограмма:")
print(inverse_transformed_parallelogram)
print("\nКоординаты исходного квадрата:")
print(square_vertices)

# 7. Визуализация
plt.figure(figsize=(12, 10))

# Рисуем исходный квадрат
ax1 = plt.subplot(2, 2, 1)
square = Polygon(square_vertices, fill=False, edgecolor='blue', linewidth=2)
ax1.add_patch(square)
ax1.scatter(square_vertices[:, 0], square_vertices[:, 1], color='blue')
ax1.text(K[0]-0.2, K[1]-0.2, 'K', fontsize=12)
ax1.text(L[0]+0.1, L[1]-0.2, 'L', fontsize=12)
ax1.text(M[0]+0.1, M[1]+0.1, 'M', fontsize=12)
ax1.text(N[0]-0.2, N[1]+0.1, 'N', fontsize=12)
ax1.set_xlim(-5, 5)
ax1.set_ylim(-5, 5)
ax1.set_aspect('equal')
ax1.grid(True)
ax1.set_title('Исходный квадрат KLMN')

# Рисуем параллелограмм (образ квадрата)
ax2 = plt.subplot(2, 2, 2)
parallelogram = Polygon(parallelogram_vertices, fill=False, edgecolor='red', linewidth=2)
ax2.add_patch(parallelogram)
ax2.scatter(parallelogram_vertices[:, 0], parallelogram_vertices[:, 1], color='red')
ax2.text(A[0]-0.2, A[1]-0.2, 'A', fontsize=12)
ax2.text(B[0]+0.1, B[1]-0.2, 'B', fontsize=12)
ax2.text(C[0]+0.1, C[1]+0.1, 'C', fontsize=12)
ax2.text(D[0]-0.2, D[1]+0.1, 'D', fontsize=12)
ax2.set_xlim(-5, 12)
ax2.set_ylim(-5, 12)
ax2.set_aspect('equal')
ax2.grid(True)
ax2.set_title('Параллелограмм ABCD = F(KLMN)')

# Рисуем оба объекта и преобразование
ax3 = plt.subplot(2, 2, 3)
# Рисуем исходный квадрат
square = Polygon(square_vertices, fill=False, edgecolor='blue', linewidth=2)
ax3.add_patch(square)
ax3.scatter(square_vertices[:, 0], square_vertices[:, 1], color='blue')
ax3.text(K[0]-0.2, K[1]-0.2, 'K', fontsize=12)
ax3.text(L[0]+0.1, L[1]-0.2, 'L', fontsize=12)
ax3.text(M[0]+0.1, M[1]+0.1, 'M', fontsize=12)
ax3.text(N[0]-0.2, N[1]+0.1, 'N', fontsize=12)

# Рисуем параллелограмм
parallelogram = Polygon(parallelogram_vertices, fill=False, edgecolor='red', linewidth=2)
ax3.add_patch(parallelogram)
ax3.scatter(parallelogram_vertices[:, 0], parallelogram_vertices[:, 1], color='red')
ax3.text(A[0]-0.2, A[1]-0.2, 'A', fontsize=12)
ax3.text(B[0]+0.1, B[1]-0.2, 'B', fontsize=12)
ax3.text(C[0]+0.1, C[1]+0.1, 'C', fontsize=12)
ax3.text(D[0]-0.2, D[1]+0.1, 'D', fontsize=12)

# Рисуем стрелки преобразования
for i in range(4):
    ax3.arrow(square_vertices[i, 0], square_vertices[i, 1], 
              parallelogram_vertices[i, 0] - square_vertices[i, 0], 
              parallelogram_vertices[i, 1] - square_vertices[i, 1],
              head_width=0.2, head_length=0.3, fc='green', ec='green', alpha=0.5)

ax3.set_xlim(-5, 12)
ax3.set_ylim(-5, 12)
ax3.set_aspect('equal')
ax3.grid(True)
ax3.set_title('Преобразование F: KLMN → ABCD')

# Рисуем обратное преобразование
ax4 = plt.subplot(2, 2, 4)
# Рисуем параллелограмм
parallelogram = Polygon(parallelogram_vertices, fill=False, edgecolor='red', linewidth=2)
ax4.add_patch(parallelogram)
ax4.scatter(parallelogram_vertices[:, 0], parallelogram_vertices[:, 1], color='red')
ax4.text(A[0]-0.2, A[1]-0.2, 'A', fontsize=12)
ax4.text(B[0]+0.1, B[1]-0.2, 'B', fontsize=12)
ax4.text(C[0]+0.1, C[1]+0.1, 'C', fontsize=12)
ax4.text(D[0]-0.2, D[1]+0.1, 'D', fontsize=12)

# Рисуем исходный квадрат
square = Polygon(square_vertices, fill=False, edgecolor='blue', linewidth=2)
ax4.add_patch(square)
ax4.scatter(square_vertices[:, 0], square_vertices[:, 1], color='blue')
ax4.text(K[0]-0.2, K[1]-0.2, 'K', fontsize=12)
ax4.text(L[0]+0.1, L[1]-0.2, 'L', fontsize=12)
ax4.text(M[0]+0.1, M[1]+0.1, 'M', fontsize=12)
ax4.text(N[0]-0.2, N[1]+0.1, 'N', fontsize=12)

# Рисуем стрелки преобразования
for i in range(4):
    ax4.arrow(parallelogram_vertices[i, 0], parallelogram_vertices[i, 1], 
              square_vertices[i, 0] - parallelogram_vertices[i, 0], 
              square_vertices[i, 1] - parallelogram_vertices[i, 1],
              head_width=0.2, head_length=0.3, fc='purple', ec='purple', alpha=0.5)

ax4.set_xlim(-5, 12)
ax4.set_ylim(-5, 12)
ax4.set_aspect('equal')
ax4.grid(True)
ax4.set_title('Обратное преобразование F^(-1): ABCD → KLMN')

# Добавление легенды
legend_elements = [
    Line2D([0], [0], color='blue', lw=2, label='Квадрат KLMN'),
    Line2D([0], [0], color='red', lw=2, label='Параллелограмм ABCD'),
    Line2D([0], [0], color='green', lw=2, label='Преобразование F'),
    Line2D([0], [0], color='purple', lw=2, label='Обратное преобразование F^(-1)')
]
ax4.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.show()