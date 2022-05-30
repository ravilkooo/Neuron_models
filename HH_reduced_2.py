import math
import random
import pygame
import os
import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt

pygame.init()
WIDTH = 1000
HEIGHT = 600
MARGIN_X = 20
MARGIN_Y = 20
WORK_WIDTH = WIDTH - MARGIN_X * 2
WORK_HEIGHT = HEIGHT - MARGIN_Y * 2
MAX_X = 120
MIN_X = -20
MAX_Y = 1
MIN_Y = 0
STEP_X = 10
STEP_Y = 0.1
SCALE_X = WORK_WIDTH / (MAX_X - MIN_X)
SCALE_Y = WORK_HEIGHT / (MAX_Y - MIN_Y)
# CENTER_XY = np.array([MAX_X+MIN_X, MAX_Y+MIN_Y]) / 2
CENTER_XY = np.array([0, 0])
SCALE_T = 0.5
PYGAME_START_TIME = 0


def init_model_update_timer():
    global PYGAME_START_TIME
    PYGAME_START_TIME = pygame.time.get_ticks()


def get_time():
    return pygame.time.get_ticks() - PYGAME_START_TIME


sc = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("FHN")
font = pygame.font.SysFont('arial', 10)

FPS = 60
clock = pygame.time.Clock()
WHITE = (255, 255, 255)
GREY = (200, 200, 200)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)


def draw_text(text, pos, col=BLACK):
    text_surface = font.render(text, True, col)
    sc.blit(text_surface, pos)


def real_to_pygame(r_cord):
    return np.array([(r_cord[0] - MIN_X) * SCALE_X + MARGIN_X, (MAX_Y - r_cord[1]) * SCALE_Y + MARGIN_Y])


def draw_net():
    pg_center = real_to_pygame(CENTER_XY)

    vert = np.append(np.arange(pg_center[0], 0, -SCALE_X * STEP_X), np.arange(pg_center[0], WIDTH, SCALE_X * STEP_X))
    horiz = np.append(np.arange(pg_center[1], 0, -SCALE_Y * STEP_Y), np.arange(pg_center[1], HEIGHT, SCALE_Y * STEP_Y))
    for v_line in vert:
        pygame.draw.line(sc, GREY, (v_line, 0), (v_line, HEIGHT))
    for h_line in horiz:
        pygame.draw.line(sc, GREY, (0, h_line), (WIDTH, h_line))

    pygame.draw.line(sc, BLACK, (pg_center[0], 0), (pg_center[0], HEIGHT))
    pygame.draw.line(sc, BLACK, (0, pg_center[1]), (WIDTH, pg_center[1]))

    for i in range(1, int((CENTER_XY[0] - MIN_X) / STEP_X)):
        draw_text(str(round(CENTER_XY[0] - i * STEP_X, 2)), (pg_center[0] - i * STEP_X * SCALE_X, pg_center[1]))
    for i in range(1, int((MAX_X - CENTER_XY[0]) / STEP_X)):
        draw_text(str(round(CENTER_XY[0] + i * STEP_X, 2)), (pg_center[0] + i * STEP_X * SCALE_X, pg_center[1]))
    for i in range(1, int((CENTER_XY[1] - MIN_Y) / STEP_Y)):
        draw_text(str(round(CENTER_XY[1] - i * STEP_Y, 2)), (pg_center[0], pg_center[1] + i * STEP_Y * SCALE_Y))
    for i in range(1, int((MAX_Y - CENTER_XY[1]) / STEP_Y)):
        draw_text(str(round(CENTER_XY[1] + i * STEP_Y, 2)), (pg_center[0], pg_center[1] - i * STEP_Y * SCALE_Y))


# -------------------------

v_0 = 0
n_0 = 0.318
x = np.array([v_0, n_0])

curve_dot_cnt = 2000
VN_curve = [real_to_pygame(x) for i in range(curve_dot_cnt)]

I_const = 0.0
I_impulse_flag = False
I_per = 23
I_last_imp = 0.0
I_incr = True
I_impulse_val = 0.15


def get_I_app(t):
    global I_last_imp
    if not I_impulse_flag:
        return I_const
    if t - I_last_imp > I_per:
        I_last_imp = t
    if t - I_last_imp < I_per/2:
        return I_const + I_impulse_val
    else:
        return I_const


def a_n(v):
    return 0.01 * (10 - v) / (np.exp((10 - v) / 10) - 1)


def b_n(v):
    return 0.125 * np.exp(-v / 80)


def a_m(v):
    return 0.1 * (25 - v) / (np.exp((25 - v) / 10) - 1)


def b_m(v):
    return 4 * np.exp(-v / 18)


def n_inf(v):
    return a_n(v)/(a_n(v) + b_n(v))


def m_inf(v):
    return a_m(v)/(a_m(v) + b_m(v))


def tau_n(v):
    return 1/(a_n(v) + b_n(v))


C = 1

E_K = -12
E_Na = 115
E_L = 10.613

g_K = 36
g_Na = 120
g_L = 0.3


def HH_reduced(v, n, t):
    return np.array([(get_I_app(t) - g_K * (n ** 4) * (v - E_K)
                     - g_Na * (m_inf(v) ** 3) * (0.89 - 1.1*n) * (v - E_Na) - g_L * (v - E_L))/C,
                     a_n(v)*(1-n)-b_n(v)*n])


# def get_equilibrium_points(t):
#     poly = [-g_K*()]
#     eq_points_w = np.roots(poly)
#     res = []
#     for eq_w in eq_points_w:
#         if np.isreal(eq_w) and MAX_X >= eq_w >= MIN_X:
#             eq_point = (eq_w*gamma, eq_w)
#             res.extend([eq_point])
#     return res


# -------------------------


def RK4_step(y, dt, t):
    v = y[0]
    n = y[1]
    k1, q1 = HH_reduced(v, n, t)
    k2, q2 = HH_reduced(v + 0.5 * k1 * dt, n + 0.5 * q1 * dt, t)
    k3, q3 = HH_reduced(v + 0.5 * k2 * dt, n + 0.5 * q2 * dt, t)
    k4, q4 = HH_reduced(v + k3 * dt, n + q3 * dt, t)
    return dt * np.array([k1 + 2 * k2 + 2 * k3 + k4, q1 + 2 * q2 + 2 * q3 + q4])


max_time = 40
delta_t = 0.001
time_measure = np.array([0])
# time-stepping solution
V = np.array([v_0])
N = np.array([n_0])
I_out = np.array([get_I_app(0)])

last_upd = 0
time_from_last_update = 0

init_model_update_timer()
model_time = 0
escape = False

while 1:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()
        elif event.type == pygame.KEYDOWN:
            event_keys = pygame.key.get_pressed()
            if event.key == pygame.K_UP:
                if event_keys[pygame.K_LSHIFT]:
                    I_impulse_val += 0.075
                    print(f"I_impulse_val = {I_impulse_val}")
                elif event_keys[pygame.K_LCTRL]:
                    I_per += 0.5
                    print(f"I_per = {I_per}")
                else:
                    I_const += 10
                    print(f"I_const = {I_const}")
            elif event.key == pygame.K_DOWN:
                if event_keys[pygame.K_LSHIFT]:
                    I_impulse_val -= 0.075
                    print(f"I_impulse_val = {I_impulse_val}")
                elif event_keys[pygame.K_LCTRL]:
                    I_per -= 0.5
                    print(f"I_per = {I_per}")
                else:
                    I_const -= 10
                    print(f"I_const = {I_const}")
            elif event.key == pygame.K_SPACE:
                I_impulse_flag = not I_impulse_flag
            elif event.key == pygame.K_ESCAPE:
                pygame.event.post(pygame.event.Event(pygame.QUIT))
            elif event.key == pygame.K_LEFTBRACKET:
                SCALE_T -= 0.1
                print(f"TIME SCALE = {SCALE_T}")
            elif event.key == pygame.K_RIGHTBRACKET:
                SCALE_T += 0.1
                print(f"TIME SCALE = {SCALE_T}")
    if get_time() < 1000*delta_t:
        continue

    init_model_update_timer()
    time_from_last_update += SCALE_T * delta_t
    model_time = time_measure[-1] + SCALE_T * delta_t

    x = x + RK4_step(x, SCALE_T * delta_t, model_time)

    V = np.append(V, x[0])
    N = np.append(N, x[1])
    I_out = np.append(I_out, get_I_app(model_time))
    time_measure = np.append(time_measure, model_time)

    if time_from_last_update - last_upd >= 1 / 60:
        sc.fill(WHITE)
        last_upd = time_from_last_update

        # nullclines (v, w)
        # v_nullcl = []
        # n_nullcl = []
        # for v in np.linspace(MIN_X, MAX_X, 100):
        #     poly_n = [-g_K*(v-E_K), 0, 0, 1.1*g_Na*(m_inf(v)**3)*(v-E_Na),
        #             get_I_app(model_time)-0.89*g_Na*(m_inf(v)**3)*(v-E_Na)-g_L*(v-E_K)]
        #     n_sols = np.roots(poly_n)
        #     n_real_sols = []
        #     for n_sol in n_sols:
        #         if np.isreal(n_sol):
        #             print(n_sol)
        #             v_nullcl.extend([real_to_pygame([v, n_sol])])
        #     n_nullcl.extend([real_to_pygame([v, 1/(1+np.exp((-53-v)/15))])])
        # print(v_nullcl)
        # # pygame.draw.aalines(sc, (255, 0, 255), False, v_nullcl)
        # pygame.draw.aalines(sc, (0, 255, 255), False, n_nullcl)

        # eq points
        # eq_points = get_equilibrium_points(model_time)
        # for eq_p in eq_points:
        #     f_v = np.poly1d([-gamma ** 3, (a + 1) * gamma ** 2, -a * gamma, (get_I_app(model_time) - eq_p[1])]).deriv()(
        #         eq_p[0])
        #     f_w = np.poly1d([-gamma ** 3, (a + 1) * (gamma ** 2), -(a * gamma + 1), get_I_app(model_time)]).deriv()(
        #         eq_p[1])
        #     g_v = np.poly1d([eps, -eps * gamma * eq_p[1]]).deriv()(eq_p[0])
        #     g_w = np.poly1d([-eps * gamma, eps * eq_p[0]]).deriv()(eq_p[1])
        #
        #     eig_val, eig_vec = np.linalg.eig([[f_v, f_w], [g_v, g_w]])
        #     # print(eig_val)
        #     is_stable = np.real(eig_val[0]) < 0 and np.real(eig_val[1]) < 0
        #
        #     pygame.draw.circle(sc, BLACK, real_to_pygame(eq_p), 6)
        #     if is_stable:
        #         pygame.draw.circle(sc, BLACK, real_to_pygame(eq_p), 5)
        #     else:
        #         pygame.draw.circle(sc, WHITE, real_to_pygame(eq_p), 5)

        # draw point
        point = real_to_pygame((x[0], x[1]))

        pygame.draw.circle(sc, RED, point, 5)

        # trajectory
        VN_curve.append([point[0], point[1]])
        pygame.draw.aalines(sc, BLUE, False, VN_curve[-curve_dot_cnt:])

        # net
        draw_net()

        pygame.display.update()

max_time = time_measure[-1]

exit()