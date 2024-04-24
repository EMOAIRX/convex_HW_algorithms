from sympy import symbols, diff

# 定义变量
x, y, z = symbols('x y z')

# 定义函数
f = (3 - x)**2 + 7*(y - x**2)**2 + 9*(z - x - y**2)**2

# 计算偏导数
partial_x = diff(f, x)
partial_y = diff(f, y)
partial_z = diff(f, z)

# 将偏导数转换为 Python 函数
def df_dx(x_val, y_val, z_val):
    return partial_x.subs({x: x_val, y: y_val, z: z_val})

def df_dy(x_val, y_val, z_val):
    return partial_y.subs({x: x_val, y: y_val, z: z_val})

def df_dz(x_val, y_val, z_val):
    return partial_z.subs({x: x_val, y: y_val, z: z_val})

# 示例：计算某一点的偏导数值
# x_val, y_val, z_val = 1, 2, 3  # 示例点的坐标
# print(df_dx(x_val, y_val, z_val))  # 输出 df/dx 在该点的值
# print(df_dy(x_val, y_val, z_val))  # 输出 df/dy 在该点的值
# print(df_dz(x_val, y_val, z_val))  # 输出 df/dz 在该点的值
