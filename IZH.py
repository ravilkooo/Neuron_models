import pygame
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
MAX_X = 40
MIN_X = -100
MAX_Y = 60
MIN_Y = -20
STEP_X = 10
STEP_Y = 4
SCALE_X = WORK_WIDTH / (MAX_X - MIN_X)
SCALE_Y = WORK_HEIGHT / (MAX_Y - MIN_Y)
# CENTER_XY = np.array([MAX_X+MIN_X, MAX_Y+MIN_Y]) / 2
CENTER_XY = np.array([0, 0])
SCALE_T = 10
PYGAME_START_TIME = 0

def init_model_update_timer():
    global PYGAME_START_TIME
    PYGAME_START_TIME = pygame.time.get_ticks()

def model_update_timer():
    return pygame.time.get_ticks() - PYGAME_START_TIME

sc = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("IZH")
font = pygame.font.SysFont('arial', 14)

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
curr_type = 0
# Excitatory: [RS, IB, CH]
# Inhibitory: [FS, LTS]
# Thalamo- : [TC]
# Rezonator : [RZ]
a = [0.02, 0.02, 0.02, 0.1, 0.02, 0.02, 0.1]
b = [0.2, 0.2, 0.2, 0.2, 0.25, 0.25, 0.25]
c = [-65., -55., -50., -65., -65., -65., -65.]
d = [8., 4., 2., 2., 2., 0.05, 2.]
I_app = 10
v_0 = -70
w_0 = -14
v_thresh = 30.0
x = np.array([v_0, w_0])
curve_dot_cnt = 2000
VW_curve = [real_to_pygame(x) for i in range(curve_dot_cnt)]

def IZH(v, w):
    if v >= v_thresh:
        x[0] = c[curr_type]
        x[1] = x[1] + d[curr_type]
    return np.array([0.04 * v**2 + 5 * v + 140 - w + I_app, a[curr_type] * (b[curr_type] * v - w)])

def change_model_type(new_type):
    global curr_type
    curr_type = new_type

def get_equilibrium_points():
    poly = [0.04, (5-b[curr_type]), (140+I_app)]
    eq_points_v = np.roots(poly)
    res = []
    for eq_v in eq_points_v:
        if np.isreal(eq_v) and MAX_X >= eq_v >= MIN_X:
            eq_point = (eq_v, b[curr_type]*eq_v)
            res.extend([eq_point])
    return res
# -------------------------

def RK4_step(y, dt):
    v = y[0]
    w = y[1]
    [k1, q1] = IZH(v, w)
    [k2, q2] = IZH(v + 0.5 * k1 * dt, w + 0.5 * q1 * dt)
    [k3, q3] = IZH(v + 0.5 * k2 * dt, w + 0.5 * q2 * dt)
    [k4, q4] = IZH(v + k3 * dt, w + q3 * dt)
    return dt * np.array([k1 + 2 * k2 + 2 * k3 + k4, q1 + 2 * q2 + 2 * q3 + q4])

max_time = 40
delta_t = 0.001
time_measure = np.array([0])
V = np.array([v_0])
W = np.array([w_0])
last_upd = 0
time_from_last_update = 0
init_model_update_timer()
model_time = 0
escape = False

while 1:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            escape = True
        elif event.type == pygame.KEYDOWN:
            event_keys = pygame.key.get_pressed()
            if event.key == pygame.K_UP:
                I_app += 1
                if event_keys[pygame.K_LSHIFT]:
                    I_app += 4
                print(f"I_app = {I_app}")
            elif event.key == pygame.K_DOWN:
                I_app -= 1
                if event_keys[pygame.K_LSHIFT]:
                    I_app -= 4
                print(f"I_app = {I_app}")
            elif event.key == pygame.K_ESCAPE:
                pygame.event.post(pygame.event.Event(pygame.QUIT))
            elif event.key == pygame.K_LEFTBRACKET:
                SCALE_T -= 2
                print(f"TIME SCALE = {SCALE_T}")
            elif event.key == pygame.K_RIGHTBRACKET:
                SCALE_T += 2
                print(f"TIME SCALE = {SCALE_T}")
            elif event.key == pygame.K_1:
                change_model_type(0)
                I_app = 10
                print("Excitatory cortical neuron model.\n"
                      "Regular spiking.")
            elif event.key == pygame.K_2:
                change_model_type(1)
                I_app = 10
                print("Excitatory cortical neuron model.\n"
                      "Instrinsically bursting.")
            elif event.key == pygame.K_3:
                change_model_type(2)
                I_app = 10
                print("Excitatory cortical neuron model.\n"
                      "Chattering.")
            elif event.key == pygame.K_4:
                change_model_type(3)
                I_app = 30
                print("Inhibitotary cortical neuron model.\n"
                      "Fast spiking.")
            elif event.key == pygame.K_5:
                change_model_type(4)
                print("Inhibitotary cortical neuron model.\n"
                      "Low-threshold spiking.")
            elif event.key == pygame.K_6:
                change_model_type(5)
                print("Thalamo-cortical neuron model.\n")
            elif event.key == pygame.K_7:
                change_model_type(6)
                print("Resonator neuron model.\n")
    if escape:
        break
    if model_update_timer() < delta_t * 1000:
        continue
    init_model_update_timer()
    time_from_last_update += SCALE_T * delta_t
    model_time = time_measure[-1] + SCALE_T * delta_t
    x = x + RK4_step(x, SCALE_T * delta_t)
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
            v_nullcl.extend([real_to_pygame([v, 0.04*v**2 + 5*v + 140 + I_app])])
            w_nullcl.extend([real_to_pygame([v, b[curr_type]*v])])
        pygame.draw.aalines(sc, (255, 0, 255), False, v_nullcl)
        pygame.draw.aalines(sc, (0, 255, 255), False, w_nullcl)
        # eq points
        eq_points = get_equilibrium_points()
        for eq_p in eq_points:
            f_v = np.poly1d([0.04, 5, (140 + I_app - eq_p[1])]).deriv()(eq_p[0])
            f_w = np.poly1d([-1, 0.04*(eq_p[0]**2) + 5*eq_p[0] + 140 + I_app]).deriv()(eq_p[1])
            g_v = np.poly1d([a[curr_type]*b[curr_type], -a[curr_type]*eq_p[1]]).deriv()(eq_p[0])
            g_w = np.poly1d([-a[curr_type], a[curr_type]*b[curr_type]*eq_p[0]]).deriv()(eq_p[1])
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
        pygame.draw.circle(sc, RED, point, 4)
        # trajectory
        VW_curve.pop(0)
        VW_curve.append([point[0], point[1]])
        # print(len(VW_curve))
        pygame.draw.aalines(sc, BLUE, False, VW_curve[-curve_dot_cnt:])
        # net
        draw_net()
        pygame.display.update()

max_time = time_measure[-1]
plt.plot(time_measure, V)
plt.grid(True)
plt.axis()
plt.xlabel('Time, ms')
plt.ylabel('Voltage, mV')
plt.show()
exit()