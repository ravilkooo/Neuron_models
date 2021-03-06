import pygame
import numpy as np
from matplotlib import pyplot as plt

pygame.init()
WIDTH = 1000
HEIGHT = 600
MARGIN_X = 20
MARGIN_Y = 20
WORK_WIDTH = WIDTH - MARGIN_X * 2
WORK_HEIGHT = HEIGHT - MARGIN_Y * 2
MAX_X = 20
MIN_X = -2
MAX_Y = 120
MIN_Y = -20
STEP_X = 2
STEP_Y = 10
SCALE_X = WORK_WIDTH / (MAX_X - MIN_X)
SCALE_Y = WORK_HEIGHT / (MAX_Y - MIN_Y)
# CENTER_XY = np.array([MAX_X+MIN_X, MAX_Y+MIN_Y]) / 2
CENTER_XY = np.array([0, 0])
SCALE_T = 4
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
VT_curve = [real_to_pygame([0, x[0]])]

def get_I_app(t):
    if 2 <= t <= 3:
        return 2
    if 10 <= t <= 11:
        return 2.3
    return 0

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
# -------------------------

def RK4_step(y, dt, t):
    v = y[0]
    n = y[1]
    k1, q1 = HH_reduced(v, n, t)
    k2, q2 = HH_reduced(v + 0.5 * k1 * dt, n + 0.5 * q1 * dt, t)
    k3, q3 = HH_reduced(v + 0.5 * k2 * dt, n + 0.5 * q2 * dt, t)
    k4, q4 = HH_reduced(v + k3 * dt, n + q3 * dt, t)
    return dt * np.array([k1 + 2 * k2 + 2 * k3 + k4, q1 + 2 * q2 + 2 * q3 + q4])

max_time = 20
delta_t = 0.001
time_measure = np.array([0])
V = np.array([v_0])
N = np.array([n_0])
I_out = np.array([get_I_app(0)])
last_upd = 0
time_from_last_update = 0
init_model_update_timer()
model_time = 0
escape = False
measure_cnt = 1
prespike_moment = 0
afterspike_moment = 0

while 1:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()
        elif event.type == pygame.KEYDOWN:
            event_keys = pygame.key.get_pressed()
            if event.key == pygame.K_ESCAPE:
                escape = True
            elif event.key == pygame.K_LEFTBRACKET:
                SCALE_T -= 0.5
                print(f"TIME SCALE = {SCALE_T}")
            elif event.key == pygame.K_RIGHTBRACKET:
                SCALE_T += 0.5
                print(f"TIME SCALE = {SCALE_T}")
    if escape or model_time >= max_time:
        break
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
    measure_cnt += 1
    if prespike_moment == 0 and model_time >= 9.9:
        prespike_moment = measure_cnt
    if afterspike_moment == 0 and model_time >= 15.:
        afterspike_moment = measure_cnt
    if model_time % 1 < 2*delta_t:
        print(model_time)
        print(x)
        print(f'I={get_I_app(model_time)}')
    if time_from_last_update - last_upd >= 1 / 60:
        sc.fill(WHITE)
        last_upd = time_from_last_update
        # E_k,na,l
        pygame.draw.line(sc, BLACK, real_to_pygame([0, E_K]), real_to_pygame([max_time, E_K]), 3)
        pygame.draw.line(sc, BLACK, real_to_pygame([0, E_Na]), real_to_pygame([max_time, E_Na]), 3)
        pygame.draw.line(sc, BLACK, real_to_pygame([0, E_L]), real_to_pygame([max_time, E_L]), 3)
        # draw point
        point = real_to_pygame((model_time, x[0]))
        pygame.draw.circle(sc, RED, point, 3)
        # trajectory
        VT_curve.append([point[0], point[1]])
        pygame.draw.aalines(sc, BLUE, False, VT_curve[:measure_cnt])
        # net
        draw_net()
        pygame.display.update()
max_time = time_measure[-1]
# plot the result
plt.plot(time_measure, V)
plt.plot(time_measure, np.ones_like(time_measure)*E_K, '--', label='E_K')
plt.plot(time_measure, np.ones_like(time_measure)*E_Na, ':', label='E_Na')
plt.plot(time_measure, np.ones_like(time_measure)*E_L, '-.', label='E_L')
plt.xlabel('Time, ms')
plt.ylabel('Voltage, mV', labelpad=0)
plt.legend()
plt.grid()
plt.show()

# repeat plots for [prespike_moment:]
plt.plot(time_measure[prespike_moment:afterspike_moment],
         V[prespike_moment:afterspike_moment])
plt.plot(time_measure[prespike_moment:afterspike_moment],
         np.ones_like(time_measure[prespike_moment:afterspike_moment])*E_K, '--', label='E_K')
plt.plot(time_measure[prespike_moment:afterspike_moment],
         np.ones_like(time_measure[prespike_moment:afterspike_moment])*E_Na, ':', label='E_Na')
plt.plot(time_measure[prespike_moment:afterspike_moment],
         np.ones_like(time_measure[prespike_moment:afterspike_moment])*E_L, '-.', label='E_L')
plt.xlabel('Time, ms')
plt.ylabel('Voltage, mV', labelpad=0)
plt.legend()
plt.grid()
plt.show()

exit()