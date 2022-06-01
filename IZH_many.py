
import pygame
import numpy as np
from numpy.linalg import inv

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
SCALE_T = 1
PYGAME_START_TIME = 0

def init_model_update_timer():
    global PYGAME_START_TIME
    PYGAME_START_TIME = pygame.time.get_ticks()

def model_update_timer():
    return pygame.time.get_ticks() - PYGAME_START_TIME

sc = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("IZH")
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
dot_cnt = 2000
curr_type = 0

# Excitatory: [RS, IB, CH]
# Inhibitory: [FS, LTS]
# Thalamo- : [TC]
# Rezonator : [RZ]
a = [0.02, 0.02, 0.02, 0.1, 0.02, 0.02, 0.1]
b = [0.2, 0.2, 0.2, 0.2, 0.25, 0.25, 0.25]
c = [-65., -55., -50., -65., -65., -65., -65.]
d = [8., 4., 2., 2., 2., 0.05, 2.]

I_app = 10.0
v_0 = -70
w_0 = -14
v_thresh = 30.0
x = np.zeros((dot_cnt, 2))
rads = np.ones(dot_cnt)*3
curves_dot_cnt = 5
VW_curves = np.array([[real_to_pygame(xx) for i in range(curves_dot_cnt)] for xx in x])

def IZH(v, w):
    for i in range(dot_cnt):
        if v[i] >= v_thresh:
            x[i][0] = c[curr_type]
            x[i][1] = x[i][1] + d[curr_type]
            VW_curves[i] = [real_to_pygame(x[i]) for j in range(curves_dot_cnt)]
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

def respawn_dots(i, dots_gens):
    low_b = int(i/dots_gens * dot_cnt)
    up_b = int(min((i+1)/dots_gens * dot_cnt, dot_cnt))
    x[low_b:up_b, 0] = np.random.uniform(MIN_X, MAX_X, up_b-low_b)
    x[low_b:up_b, 1] = np.random.uniform(MIN_Y, MAX_Y, up_b-low_b)
    rads[low_b:up_b] = np.ones(up_b-low_b)*(i+1)*5/dots_gens + 1
    VW_curves[low_b:up_b] = np.array([[real_to_pygame(xx) for i in range(curves_dot_cnt)] for xx in x[low_b:up_b]])
# -------------------------

def RK4_step(y, dt):
    v = y[:, 0]
    w = y[:, 1]
    k1, q1 = IZH(v, w)
    [k2, q2] = IZH(v + 0.5 * k1 * dt, w + 0.5 * q1 * dt)
    [k3, q3] = IZH(v + 0.5 * k2 * dt, w + 0.5 * q2 * dt)
    [k4, q4] = IZH(v + k3 * dt, w + q3 * dt)
    return dt * np.array([k1 + 2 * k2 + 2 * k3 + k4, q1 + 2 * q2 + 2 * q3 + q4]).T

max_time = 40
delta_t = 0.001
time_measure = np.array([0])
last_upd = 0
time_from_last_update = 0
init_model_update_timer()
model_time = 0
dots_lifetime = 2000
dots_generations = 10
for i in range(dots_generations):
    respawn_dots(i, dots_generations)
last_respawn = [pygame.time.get_ticks() - dots_lifetime*i/dots_generations for i in range(dots_generations)]

while 1:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                print(f"I_app = {I_app}")
                I_app += 0.4
            elif event.key == pygame.K_DOWN:
                print(f"I_app = {I_app}")
                I_app -= 0.4
            elif event.key == pygame.K_SPACE:
                print("dV")
                x[:, 0] += 40
                event_keys = pygame.key.get_pressed()
                if not event_keys[pygame.K_LSHIFT]:
                    x[:, 0] += 40
            elif event.key == pygame.K_ESCAPE:
                pygame.event.post(pygame.event.Event(pygame.QUIT))
            elif event.key == pygame.K_LEFTBRACKET:
                SCALE_T -= 0.5
                print(f"TIME SCALE = {SCALE_T}")
            elif event.key == pygame.K_RIGHTBRACKET:
                SCALE_T += 0.5
                print(f"TIME SCALE = {SCALE_T}")
            elif event.key == pygame.K_1:
                change_model_type(0)
                print("Excitatory cortical neuron model."
                      "Regular spiking.")
                I_app = 10
            elif event.key == pygame.K_2:
                change_model_type(1)
                print("Excitatory cortical neuron model."
                      "Instrinsically bursting.")
                I_app = 10
            elif event.key == pygame.K_3:
                change_model_type(2)
                print("Excitatory cortical neuron model."
                      "Chattering.")
                I_app = 10
            elif event.key == pygame.K_4:
                change_model_type(3)
                print("Inhibitotary cortical neuron model."
                      "Fast spiking.")
                I_app = 30
            elif event.key == pygame.K_5:
                change_model_type(4)
                print("Inhibitotary cortical neuron model."
                      "Low-threshold spiking.")
            elif event.key == pygame.K_6:
                change_model_type(5)
                print("Thalamo-cortical neuron model.")
            elif event.key == pygame.K_7:
                change_model_type(6)
                print("Resonator neuron model.")
    if model_update_timer() < delta_t * 1000:
        continue
    init_model_update_timer()
    time_from_last_update += SCALE_T * delta_t
    model_time = time_measure[-1] + SCALE_T * delta_t
    x = x + RK4_step(x, SCALE_T * delta_t)
    time_measure = np.append(time_measure, model_time)
    for i in range(dots_generations):
        if pygame.time.get_ticks()-last_respawn[i] >= dots_lifetime:
            last_respawn[i] = pygame.time.get_ticks()
            respawn_dots(i, dots_generations)
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
        for i in range(dots_generations):
            low_b = int(i / dots_generations * dot_cnt)
            up_b = int(min((i + 1) / dots_generations * dot_cnt, dot_cnt))
            dot_rad = np.sin(np.pi * (pygame.time.get_ticks() - last_respawn[i]) / dots_lifetime) * rads[low_b]
            dot_col = min(
                max(255 - np.sin(np.pi * (pygame.time.get_ticks() - last_respawn[i]) / dots_lifetime) * 255, 0), 255)
            for j in range(low_b, up_b):
                point = real_to_pygame(x[j])
                pygame.draw.circle(sc, (dot_col, dot_col, dot_col), point, dot_rad)
        # trajectory
        for i in range(dot_cnt):
            VW_curves[i][:-1] = VW_curves[i][1:]
            VW_curves[i][-1] = real_to_pygame((x[i][0], x[i][1]))
            pygame.draw.aalines(sc, BLUE, False, VW_curves[i])
        # net
        draw_net()
        pygame.display.update()