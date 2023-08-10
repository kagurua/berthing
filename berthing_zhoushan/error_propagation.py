import sympy
import math

import numpy as np

import dill
dill.settings['recurse'] = True


# # a naive sample of calculating error propagation
# # 计算C_分的误差
# x = Symbol('x')
# y = Symbol('y')
# f = Function('f')(x,y)
#
#
# #填对应变量的标准误差的平方
# sigma_2_x = 6**2
# sigma_2_y = 0.02**2
#
# f = x/(y-math.log(101.325))#填入间接计算的方程
# f = x/(y-math.log(101.325))#填入间接计算的方程
#
# sigma_f = diff(f,x)*diff(f,x)*sigma_2_x + diff(f,y)*diff(f,y)*sigma_2_y
# print(sqrt(sigma_f))
# print(sqrt(sigma_f.evalf(subs={"x":-3781,"y":15.42})))#替换对应变量的值


# calculate error propagation of my own scene(radar-rtp-to-image-uv)
def calculate_function(filename2save):
    # model xyz
    r = sympy.Symbol('r')
    t = sympy.Symbol('t')
    p = sympy.Symbol('p')
    # x = sympy.Function('x')(r, t, p)
    # y = sympy.Function('y')(r, t, p)
    # z = sympy.Function('z')(r, t, p)

    # set original sigma
    sigma_2_r = 0.6 ** 2  # 0.6m
    sigma_2_t = (math.radians(1.4) / sympy.cos(t)) ** 2  # 1.4 degree on 0 degree
    sigma_2_p = (math.radians(18) / sympy.cos(p)) ** 2  # 18 degree on 0 degree

    # function to calculate
    x = r * sympy.cos(p) * sympy.sin(t)
    y = r * sympy.cos(p) * sympy.cos(t)
    z = r * sympy.sin(p)

    # # calculate and output xyz's sigma
    # sigma_2_x = sympy.diff(x, r) * sympy.diff(x, r) * sigma_2_r \
    #             + sympy.diff(x, t) * sympy.diff(x, t) * sigma_2_t \
    #             + sympy.diff(x, p) * sympy.diff(x, p) * sigma_2_p
    # sigma_2_y = sympy.diff(y, r) * sympy.diff(y, r) * sigma_2_r \
    #             + sympy.diff(y, t) * sympy.diff(y, t) * sigma_2_t \
    #             + sympy.diff(y, p) * sympy.diff(y, p) * sigma_2_p
    # sigma_2_z = sympy.diff(z, r) * sympy.diff(z, r) * sigma_2_r \
    #             + sympy.diff(z, t) * sympy.diff(z, t) * sigma_2_t \
    #             + sympy.diff(z, p) * sympy.diff(z, p) * sigma_2_p

    # print(sympy.sqrt(sigma_2_x.evalf(subs={"r": 10, "t": 0, "p": 0})))  # 替换对应变量的值
    # print(sympy.sqrt(sigma_2_y.evalf(subs={"r": 10, "t": 0, "p": 0})))  # 替换对应变量的值
    # print(sympy.sqrt(sigma_2_z.evalf(subs={"r": 10, "t": 0, "p": 0})))  # 替换对应变量的值
    #
    # print(sympy.sqrt(sigma_2_x.evalf(subs={"r": 10, "t": 0, "p": 1/3})))  # 替换对应变量的值
    # print(sympy.sqrt(sigma_2_y.evalf(subs={"r": 10, "t": 0, "p": 1/3})))  # 替换对应变量的值
    # print(sympy.sqrt(sigma_2_z.evalf(subs={"r": 10, "t": 0, "p": 1/3})))  # 替换对应变量的值

    # todo: radar-to-camera external propagation
    a = sympy.Symbol('a')
    b = sympy.Symbol('b')
    g = sympy.Symbol('g')
    t1 = sympy.Symbol('t1')
    t2 = sympy.Symbol('t2')
    t3 = sympy.Symbol('t3')

    # set original sigma
    sigma_2_a = 0.001 ** 2  # alpha
    sigma_2_b = 0.001 ** 2  # beta
    sigma_2_g = 0.001 ** 2  # gamma
    sigma_2_t1 = 0.01 ** 2  # tx
    sigma_2_t2 = 0.01 ** 2  # ty
    sigma_2_t3 = 0.01 ** 2  # tz

    # calculate translation
    A = sympy.Matrix([[964.0495477017739, 0, 613.3463725824778], [0, 964.7504668505122, 377.1040664564536], [0, 0, 1]])
    Rx = sympy.Matrix([[1, 0, 0], [0, sympy.cos(a), sympy.sin(a)], [0, -sympy.sin(a), sympy.cos(a)]])
    Ry = sympy.Matrix([[sympy.cos(b), 0, -sympy.sin(b)], [0, 1, 0], [sympy.sin(b), 0, sympy.cos(b)]])
    Rz = sympy.Matrix([[sympy.cos(g), -sympy.sin(g), 0], [sympy.sin(g), sympy.cos(g), 0], [0, 0, 1]])
    R = Rx * Ry * Rz
    T = sympy.Matrix([t1, t2, t3])
    B = R.col_insert(-1, T)
    H = A * B
    # calculate projection
    udvdd = H * sympy.Matrix([[x], [y], [z], [1]])
    # print(udvdd.shape)
    uv1 = udvdd / udvdd[2, 0]

    # function
    u = uv1[0, 0]
    v = uv1[1, 0]
    d = udvdd[2, 0]

    sigma_2_u = sympy.diff(u, r) * sympy.diff(u, r) * sigma_2_r \
                + sympy.diff(u, t) * sympy.diff(u, t) * sigma_2_t \
                + sympy.diff(u, p) * sympy.diff(u, p) * sigma_2_p \
                + sympy.diff(u, a) * sympy.diff(u, a) * sigma_2_a \
                + sympy.diff(u, b) * sympy.diff(u, b) * sigma_2_b \
                + sympy.diff(u, g) * sympy.diff(u, g) * sigma_2_g \
                + sympy.diff(u, t1) * sympy.diff(u, t1) * sigma_2_t1 \
                + sympy.diff(u, t2) * sympy.diff(u, t2) * sigma_2_t2 \
                + sympy.diff(u, t3) * sympy.diff(u, t3) * sigma_2_t3
    sigma_2_v = sympy.diff(v, r) * sympy.diff(v, r) * sigma_2_r \
                + sympy.diff(v, t) * sympy.diff(v, t) * sigma_2_t \
                + sympy.diff(v, p) * sympy.diff(v, p) * sigma_2_p \
                + sympy.diff(v, a) * sympy.diff(v, a) * sigma_2_a \
                + sympy.diff(v, b) * sympy.diff(v, b) * sigma_2_b \
                + sympy.diff(v, g) * sympy.diff(v, g) * sigma_2_g \
                + sympy.diff(v, t1) * sympy.diff(v, t1) * sigma_2_t1 \
                + sympy.diff(v, t2) * sympy.diff(v, t2) * sigma_2_t2 \
                + sympy.diff(v, t3) * sympy.diff(v, t3) * sigma_2_t3
    sigma_2_d = sympy.diff(d, r) * sympy.diff(d, r) * sigma_2_r \
                + sympy.diff(d, t) * sympy.diff(d, t) * sigma_2_t \
                + sympy.diff(d, p) * sympy.diff(d, p) * sigma_2_p \
                + sympy.diff(d, a) * sympy.diff(d, a) * sigma_2_a \
                + sympy.diff(d, b) * sympy.diff(d, b) * sigma_2_b \
                + sympy.diff(d, g) * sympy.diff(d, g) * sigma_2_g \
                + sympy.diff(d, t1) * sympy.diff(d, t1) * sigma_2_t1 \
                + sympy.diff(d, t2) * sympy.diff(d, t2) * sigma_2_t2 \
                + sympy.diff(d, t3) * sympy.diff(d, t3) * sigma_2_t3

    # fix external params
    sigma_2_u_final = sigma_2_u.subs([(a, -1.34639980735600), (b, 0.0104533536651000), (g, 0.00294198274867599),
                                      (t1, 0.00356628356156506), (t2, -0.114767810602876), (t3, -0.119002343660951)])
    sigma_2_v_final = sigma_2_v.subs([(a, -1.34639980735600), (b, 0.0104533536651000), (g, 0.00294198274867599),
                                      (t1, 0.00356628356156506), (t2, -0.114767810602876), (t3, -0.119002343660951)])
    sigma_2_d_final = sigma_2_d.subs([(a, -1.34639980735600), (b, 0.0104533536651000), (g, 0.00294198274867599),
                                      (t1, 0.00356628356156506), (t2, -0.114767810602876), (t3, -0.119002343660951)])

    # # print function itself
    # print(sigma_2_u_final)
    # print(sigma_2_v_final)
    # print(sigma_2_d_final)

    # # output uvd's sigma
    # print(sympy.sqrt(sigma_2_u_final.evalf(subs={"r": 10, "t": 0, "p": 0})))  # 替换对应变量的值
    # print(sympy.sqrt(sigma_2_v_final.evalf(subs={"r": 10, "t": 0, "p": 0})))  # 替换对应变量的值
    # print(sympy.sqrt(sigma_2_d_final.evalf(subs={"r": 10, "t": 0, "p": 0})))  # 替换对应变量的值
    #
    # print(sympy.sqrt(sigma_2_u_final.evalf(subs={"r": 20, "t": 1 / 18, "p": 1 / 6})))  # 替换对应变量的值
    # print(sympy.sqrt(sigma_2_v_final.evalf(subs={"r": 20, "t": 1 / 18, "p": 1 / 6})))  # 替换对应变量的值
    # print(sympy.sqrt(sigma_2_d_final.evalf(subs={"r": 20, "t": 1 / 18, "p": 1 / 6})))  # 替换对应变量的值

    # todo: transfer to numpy formed function to increase speed
    func_sigma_u = sympy.lambdify((r, t, p), sympy.sqrt(sigma_2_u_final), 'numpy')
    func_sigma_v = sympy.lambdify((r, t, p), sympy.sqrt(sigma_2_v_final), 'numpy')
    func_sigma_d = sympy.lambdify((r, t, p), sympy.sqrt(sigma_2_d_final), 'numpy')
    # min fov on h/v of radar/camera as input
    # sigma_u_array = func_sigma_u(np.array([10, 20]), np.array([0, 20 / 180]), np.array([0, 45 / 180]))
    # sigma_v_array = func_sigma_v(np.array([10, 20]), np.array([0, 20 / 180]), np.array([0, 45 / 180]))
    # sigma_d_array = func_sigma_d(np.array([10, 20]), np.array([0, 20 / 180]), np.array([0, 45 / 180]))
    #
    # print(sigma_u_array)
    # print(sigma_v_array)
    # print(sigma_d_array)

    dill.dump((func_sigma_u, func_sigma_v, func_sigma_d), open(filename2save, 'wb'))

    # # todo: try to simplify the sigma expression  __very slow cannot get result after one night
    # print(sympy.simplify(sympy.sqrt(sigma_2_u_final)))
    # print(sympy.simplify(sympy.sqrt(sigma_2_v_final)))
    # print(sympy.simplify(sympy.sqrt(sigma_2_d_final)))


if __name__ == "__main__":
    saved_function_file = './error_function_saved.bin'
    # todo: calculate&save function
    calculate_function(saved_function_file)
    # todo: laod pre-calculated function
    f_sigma_u, f_sigma_v, f_sigma_d = dill.load(open(saved_function_file, 'rb'))

    sigma_u_array = f_sigma_u(np.array([10, 20]), np.array([0, 20 / 180]), np.array([0, 45 / 180]))
    sigma_v_array = f_sigma_v(np.array([10, 20]), np.array([0, 20 / 180]), np.array([0, 45 / 180]))
    sigma_d_array = f_sigma_d(np.array([10, 20]), np.array([0, 20 / 180]), np.array([0, 45 / 180]))

    print(sigma_u_array)
    print(sigma_v_array)
    print(sigma_d_array)
