import math
import random
import pygame
import os
import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt

pygame.init()
WIDTH = 600
HEIGHT = 600
MARGIN_X = 20
MARGIN_Y = 20
WORK_WIDTH = WIDTH - MARGIN_X * 2
WORK_HEIGHT = HEIGHT - MARGIN_Y * 2
MAX_X = 1.2
MIN_X = -0.4
MAX_Y = 1.0
MIN_Y = -0.1
STEP_X = 0.1
STEP_Y = 0.1
SCALE_X = WORK_WIDTH / (MAX_X - MIN_X)
SCALE_Y = WORK_HEIGHT / (MAX_Y - MIN_Y)
# CENTER_XY = np.array([MAX_X+MIN_X, MAX_Y+MIN_Y]) / 2
CENTER_XY = np.array([0, 0])
SCALE_T = 10
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

a = 0.25
eps = 0.005
I_const = 0.0
v_0 = 0
w_0 = 0
gamma = 1
# gamma = float(input())
x = np.array([v_0, w_0])

curve_dot_cnt = 2000
VW_curve = [real_to_pygame(x) for i in range(curve_dot_cnt)]

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


def FHN(v, w, t):
    return np.array([v * (1 - v) * (v - a) - w + get_I_app(t), eps * (v - gamma * w)])


def get_equilibrium_points(t):
    poly = [-gamma**3, (a+1)*(gamma**2), -(a*gamma + 1), get_I_app(t)]
    eq_points_w = np.roots(poly)
    res = []
    for eq_w in eq_points_w:
        if np.isreal(eq_w) and MAX_X >= eq_w >= MIN_X:
            eq_point = (eq_w*gamma, eq_w)
            res.extend([eq_point])
    return res


# -------------------------


def RK4_step(y, dt, t):
    v = y[0]
    w = y[1]
    [k1, q1] = FHN(v, w, t)
    [k2, q2] = FHN(v + 0.5 * k1 * dt, w + 0.5 * q1 * dt, t)
    [k3, q3] = FHN(v + 0.5 * k2 * dt, w + 0.5 * q2 * dt, t)
    [k4, q4] = FHN(v + k3 * dt, w + q3 * dt, t)
    return dt * np.array([k1 + 2 * k2 + 2 * k3 + k4, q1 + 2 * q2 + 2 * q3 + q4])


max_time = 40
delta_t = 0.001
time_measure = np.array([0])
# time-stepping solution
V = np.array([v_0])
W = np.array([w_0])

last_upd = 0
time_from_last_update = 0

init_model_update_timer()
model_time = 0
escape = False
print(len(VW_curve))

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
                    I_const += 0.075
                    print(f"I_const = {I_const}")
            elif event.key == pygame.K_DOWN:
                if event_keys[pygame.K_LSHIFT]:
                    I_impulse_val -= 0.075
                    print(f"I_impulse_val = {I_impulse_val}")
                elif event_keys[pygame.K_LCTRL]:
                    I_per -= 0.5
                    print(f"I_per = {I_per}")
                else:
                    I_const -= 0.075
                    print(f"I_const = {I_const}")
            elif event.key == pygame.K_SPACE:
                if not event_keys[pygame.K_LSHIFT]:
                    I_impulse_flag = not I_impulse_flag
                else:  # if event_keys[pygame.K_LSHIFT]:
                    x[0] += a * 1.3
            elif event.key == pygame.K_ESCAPE:
                pygame.event.post(pygame.event.Event(pygame.QUIT))
            elif event.key == pygame.K_LEFTBRACKET:
                SCALE_T -= 0.5
                print(f"TIME SCALE = {SCALE_T}")
            elif event.key == pygame.K_RIGHTBRACKET:
                SCALE_T += 0.5
                print(f"TIME SCALE = {SCALE_T}")
    if get_time() < delta_t * 1000:
        continue

    init_model_update_timer()
    time_from_last_update += SCALE_T * delta_t
    model_time = time_measure[-1] + SCALE_T * delta_t

    x = x + RK4_step(x, SCALE_T * delta_t, model_time)

    V = np.append(V, x[0])
    W = np.append(W, x[1])
    time_measure = np.append(time_measure, model_time)

    if time_from_last_update - last_upd >= 1 / 60:
        sc.fill(WHITE)

        last_upd = time_from_last_update

        # nullclines (v, w)
        v_nullcl = []
        w_nullcl = []
        for v in np.linspace(MIN_X, MAX_X, 100):
            v_nullcl.extend([real_to_pygame([v, v*(1-v)*(v-a) + get_I_app(model_time)])])
            w_nullcl.extend([real_to_pygame([v, v/gamma])])
        pygame.draw.aalines(sc, (255, 0, 255), False, v_nullcl)
        pygame.draw.aalines(sc, (0, 255, 255), False, w_nullcl)

        # eq points
        eq_points = get_equilibrium_points(model_time)
        for eq_p in eq_points:
            f_v = np.poly1d([-gamma**3, (a+1)*gamma**2, -a*gamma, (get_I_app(model_time) - eq_p[1])]).deriv()(eq_p[0])
            f_w = np.poly1d([-gamma**3, (a+1)*(gamma**2), -(a*gamma + 1), get_I_app(model_time)]).deriv()(eq_p[1])
            g_v = np.poly1d([eps, -eps*gamma*eq_p[1]]).deriv()(eq_p[0])
            g_w = np.poly1d([-eps*gamma, eps*eq_p[0]]).deriv()(eq_p[1])

            eig_val, eig_vec = np.linalg.eig([[f_v, f_w], [g_v, g_w]])
            # print(eig_val)
            is_stable = np.real(eig_val[0]) < 0 and np.real(eig_val[1]) < 0

            pygame.draw.circle(sc, BLACK, real_to_pygame(eq_p), 6)
            if is_stable:
                pygame.draw.circle(sc, BLACK, real_to_pygame(eq_p), 5)
            else:
                pygame.draw.circle(sc, WHITE, real_to_pygame(eq_p), 5)

        # draw point
        point = real_to_pygame(x)

        pygame.draw.circle(sc, RED, point, 5)

        # trajectory
        VW_curve.pop(0)
        VW_curve.append([point[0], point[1]])
        # print(len(VW_curve))
        pygame.draw.aalines(sc, BLUE, False, VW_curve[-curve_dot_cnt:])

        # net
        draw_net()

        pygame.display.update()


max_time = time_measure[-1]

# plot the result
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(time_measure, V)
ax.plot(time_measure, W)
ax.set(xlim=[0, max_time + 1], ylim=[min(MIN_X, MIN_Y), max(MAX_X, MAX_Y)], xlabel='time')
plt.show()

# plt.plot(time,V)
# plt.plot(time,W)
# plt.grid(True)
# plt.axis()
# plt.legend(['voltage', 'activation variable'], loc='lower right')
# plt.show()
