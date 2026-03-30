import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Определение оптимальных параметров (можно менять) ---

# Идеальная длина (L) и обхват (W) в условных единицах
L_opt = 17.0
W_opt = 5.5

# Идеальный коэффициент пропорции R = L/W
R_opt = L_opt / W_opt

# Идеальная абсолютная величина M = L * W (для оценки "слишком много, тоже плохо")
M_opt = L_opt * W_opt

# Параметры допуска (насколько вы терпимы к отклонениям)
sigma_R = 1.0  # Допуск по пропорции (больше -> шире допуск)
sigma_M = 30.0  # Допуск по размеру (больше -> шире допуск)


def utility_function_LW(L, W, L_opt, W_opt, sigma_R, sigma_M):
    """
    Непрерывная функция полезности U(L, W) для заданных L (длина) и W (обхват).
    Возвращает значение от 0 до 1.
    """

    # Расчет текущих коэффициентов
    R = L / W  # Текущий коэффициент пропорции (L/W)
    M = L * W  # Текущий коэффициент величины (L*W)

    # --- 1. Полезность Пропорции (U_R) ---
    # Штраф за отклонение R от R_opt (решает "широко/коротко vs длинно/узко")
    R_opt_val = L_opt / W_opt
    U_R = np.exp(-((R - R_opt_val) ** 2) / (2 * sigma_R ** 2))

    # --- 2. Полезность Величины (U_M) ---
    # Штраф за отклонение M от M_opt (решает "слишком много, тоже плохо")
    M_opt_val = L_opt * W_opt
    U_M = np.exp(-((M - M_opt_val) ** 2) / (2 * sigma_M ** 2))

    # Итоговая полезность - произведение двух независимых оценок
    U_LW = U_R * U_M

    return U_LW


# --- Демонстрация и построение 3D графика ---

# Определяем диапазон для обхвата и длины (например, от 3 до 25)
L_range = np.linspace(3, 25, 50)
W_range = np.linspace(3, 8, 50)

L_grid, W_grid = np.meshgrid(L_range, W_range)
Z_utility = utility_function_LW(L_grid, W_grid, L_opt, W_opt, sigma_R, sigma_M)

# Построение 3D поверхности (непрерывной функции)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Поверхностный график (Surface Plot)
surf = ax.plot_surface(L_grid, W_grid, Z_utility, cmap='viridis',
                       linewidth=0.2, antialiased=True, alpha=0.8)

ax.set_xlabel(r'L (Длина, $\alpha$)', fontsize=12)
ax.set_ylabel(r'W (Обхват, $\beta$)', fontsize=12)
ax.set_zlabel('U(L, W) (Полезность от 0 до 1)', fontsize=12)
ax.set_title(f'Функция Полезности U(L, W). Пик при L={L_opt:.1f}, W={W_opt:.1f}', fontsize=14)

fig.colorbar(surf, shrink=0.5, aspect=5, label='Уровень Полезности')
plt.show()

# --- Примеры расчета ---

print(f"\n--- Примеры расчета Полезности ---")
print(
    f"Оптимальные параметры (L={L_opt:.1f}, W={W_opt:.1f}): U = {utility_function_LW(L_opt, W_opt, L_opt, W_opt, sigma_R, sigma_M):.4f}")




L_wide_short = 14.
W_wide_short = 4.8
print(
    f" (L={L_wide_short:.1f}, W={W_wide_short:.1f}, R={L_wide_short / W_wide_short:.1f}): U = {utility_function_LW(L_wide_short, W_wide_short, L_opt, W_opt, sigma_R, sigma_M):.4f}")

