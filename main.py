"""
Minimal N‑body orbital demo (Leapfrog integrator) with GLFW + PyOpenGL.

Requirements
--------------------
pip install numpy glfw PyOpenGL

Controls
--------------------
W/A/S/D + mouse   : free-flight
Left-click        : select under crosshair → orbit mode
Mouse drag        : orbit body
Mouse wheel       : zoom
Any WASD in orbit : return to free-flight
Space             : pause/resume
T                 : trails
+ / -             : timescale (capped at 64×)
L                 : labels
H                 : toggle help (already open)
Esc               : quit
"""

import math
import sys
import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GLU import gluPerspective, gluLookAt, gluProject, gluOrtho2D
from collections import deque

# === GLUT labels (optional) ===
try:
    from OpenGL.GLUT import glutInit, glutBitmapCharacter, GLUT_BITMAP_9_BY_15
    GLUT_AVAILABLE = True
except (ImportError, AttributeError):
    GLUT_AVAILABLE = False
    print("WARNING: GLUT not available – name labels disabled.")


# -------------------------------------------------
# Constants
# -------------------------------------------------
G = 6.67430e-11


# -------------------------------------------------
# Barnes–Hut Octree (3D)
# -------------------------------------------------
class OctreeNode:
    def __init__(self, center, size):
        self.center = np.array(center, dtype=np.float64)
        self.size = size
        self.mass = 0.0
        self.com = np.zeros(3, dtype=np.float64)
        self.children = [None] * 8
        self.is_leaf = True
        self.body = None

    def _get_octant(self, pos):
        idx = 0
        if pos[0] >= self.center[0]:
            idx |= 1
        if pos[1] >= self.center[1]:
            idx |= 2
        if pos[2] >= self.center[2]:
            idx |= 4
        return idx

    def _subdivide(self):
        quarter = self.size / 4.0
        half_size = self.size / 2.0
        for i in range(8):
            dx = quarter if (i & 1) else -quarter
            dy = quarter if (i & 2) else -quarter
            dz = quarter if (i & 4) else -quarter
            child_center = self.center + np.array([dx, dy, dz], dtype=np.float64)
            self.children[i] = OctreeNode(child_center, half_size)

    def _insert_to_child(self, body):
        idx = self._get_octant(body.pos)
        self.children[idx].insert(body)

    def insert(self, body):
        if self.mass == 0.0:  # empty leaf
            self.body = body
            self.mass = body.mass
            self.com = body.pos.copy()
            return

        if self.is_leaf and self.body is not None:
            self._subdivide()
            old_body = self.body
            self.body = None
            self.is_leaf = False
            self._insert_to_child(old_body)

        if not self.is_leaf:
            self._insert_to_child(body)
        else:
            # should never reach here after subdivide
            pass

        # update node mass & COM
        old_mass = self.mass
        self.mass += body.mass
        self.com = (self.com * old_mass + body.pos * body.mass) / self.mass

    def compute_force(self, body, theta, G):
        if self.mass == 0.0:
            return np.zeros(3, dtype=np.float64)
        if self.is_leaf and self.body is body:
            return np.zeros(3, dtype=np.float64)

        dist_vec = self.com - body.pos
        dist_sq = np.dot(dist_vec, dist_vec)
        if dist_sq < 1e6:  # soft collision avoidance
            return np.zeros(3, dtype=np.float64)
        dist = np.sqrt(dist_sq)

        # Barnes–Hut criterion or leaf → approximate (or exact for leaf)
        if self.is_leaf or (self.size / dist < theta):
            inv_dist3 = 1.0 / (dist_sq * dist)
            acc = G * self.mass * dist_vec * inv_dist3
            return acc
        else:
            # recurse into children
            acc = np.zeros(3, dtype=np.float64)
            for child in self.children:
                if child is not None:
                    acc += child.compute_force(body, theta, G)
            return acc


# -------------------------------------------------
# Body
# -------------------------------------------------
class Body:
    def __init__(self, name, mass, position, velocity, radius, color):
        self.name = name
        self.mass = mass
        self.pos = np.array(position, dtype=np.float64)
        self.vel = np.array(velocity, dtype=np.float64)
        self.radius = radius
        self.color = color
        self.trail = None


# -------------------------------------------------
# Vectorized gravity replaced by Barnes–Hut
# -------------------------------------------------
def compute_accelerations(bodies):
    n = len(bodies)
    if n < 2:
        return [np.zeros(3, dtype=np.float64) for _ in bodies]

    pos_arr = np.stack([b.pos for b in bodies])
    minp = np.min(pos_arr, axis=0)
    maxp = np.max(pos_arr, axis=0)
    center = (minp + maxp) / 2
    extent = np.max(maxp - minp)
    size = max(extent * 1.2, 1e9)

    root = OctreeNode(center, size)
    for b in bodies:
        root.insert(b)

    acc_list = []
    for b in bodies:
        acc = root.compute_force(b, theta=0.5, G=G)
        acc_list.append(acc)
    return acc_list


def step_leapfrog(bodies, dt):
    accs = compute_accelerations(bodies)
    for b, a in zip(bodies, accs):
        b.vel += 0.5 * a * dt
    for b in bodies:
        b.pos += b.vel * dt
    accs = compute_accelerations(bodies)
    for b, a in zip(bodies, accs):
        b.vel += 0.5 * a * dt


# -------------------------------------------------
# Renderer
# -------------------------------------------------
class Renderer:
    def __init__(self, bodies, width=1000, height=700):
        if not glfw.init():
            raise RuntimeError("GLFW init failed")
        self.width, self.height = width, height
        self.window = glfw.create_window(width, height, "Minimal N‑Body", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Window creation failed")

        glfw.make_context_current(self.window)
        glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_DISABLED)

        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_cursor_pos_callback(self.window, self._on_mouse)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_scroll_callback(self.window, self._on_scroll)

        glEnable(GL_DEPTH_TEST)
        if GLUT_AVAILABLE:
            glutInit(sys.argv)

        self.bodies = bodies
        self.max_trail_len = 3000

        for b in self.bodies:
            b.trail = deque(maxlen=self.max_trail_len)

        # Camera & state
        max_dist = max(np.linalg.norm(b.pos) for b in bodies)
        self.cam_pos = np.array([0.0, 0.0, max_dist * 2.0], dtype=np.float64)
        self.yaw = -90.0
        self.pitch = 0.0
        self.cam_front = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        self.cam_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        self.last_x = width / 2
        self.last_y = height / 2
        self.first_mouse = True
        self.keys = {}

        self.mode = 'free'
        self.focused_body = None
        self.orbit_yaw = self.yaw
        self.orbit_pitch = self.pitch
        self.orbit_distance = 0.0

        self.trails = True
        self.timescale = 1.0
        self.show_labels = GLUT_AVAILABLE
        self.paused = False
        self.show_help = True

        self.current_dt = 3600.0

        # FPS
        self.fps = 60.0
        self.frame_count = 0
        self.last_fps_time = glfw.get_time()

        self.fov_factor = height / (2.0 * math.tan(math.radians(22.5)))

    # -------------------------------------------------
    # Input
    # -------------------------------------------------
    def _on_key(self, win, key, scancode, action, mods):
        pressed = action != glfw.RELEASE
        self.keys[key] = pressed
        if action == glfw.PRESS:
            if key == glfw.KEY_T:
                self.trails = not self.trails
            if key == glfw.KEY_L and GLUT_AVAILABLE:
                self.show_labels = not self.show_labels
            if key == glfw.KEY_H:
                self.show_help = not self.show_help
            if key == glfw.KEY_SPACE:
                self.paused = not self.paused
            if key in (glfw.KEY_KP_ADD, glfw.KEY_EQUAL):
                self.timescale = min(64.0, self.timescale * 2.0)
            if key in (glfw.KEY_KP_SUBTRACT, glfw.KEY_MINUS):
                self.timescale = max(0.01, self.timescale * 0.5)
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(self.window, True)

    def _on_mouse(self, win, xpos, ypos):
        if self.first_mouse:
            self.last_x = xpos
            self.last_y = ypos
            self.first_mouse = False
        xoff = xpos - self.last_x
        yoff = self.last_y - ypos
        self.last_x = xpos
        self.last_y = ypos
        sens = 0.1
        if self.mode == 'free':
            self.yaw += xoff * sens
            self.pitch += yoff * sens
            self.pitch = max(-89.0, min(89.0, self.pitch))
            self._update_cam_front()
        else:
            self.orbit_yaw += xoff * sens
            self.orbit_pitch += yoff * sens
            self.orbit_pitch = max(-89.0, min(89.0, self.orbit_pitch))

    def _update_cam_front(self):
        self.cam_front = np.array([
            math.cos(math.radians(self.yaw)) * math.cos(math.radians(self.pitch)),
            math.sin(math.radians(self.pitch)),
            math.sin(math.radians(self.yaw)) * math.cos(math.radians(self.pitch))
        ], dtype=np.float64)
        self.cam_front /= np.linalg.norm(self.cam_front)

    def _on_mouse_button(self, win, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS and self.mode == 'free':
            self._select_body_at_center()

    def _on_scroll(self, win, xoffset, yoffset):
        if self.mode == 'orbit' and self.focused_body:
            factor = 0.9 if yoffset > 0 else 1.11
            self.orbit_distance *= factor
            self.orbit_distance = max(5 * self.focused_body.radius, self.orbit_distance)

    def _select_body_at_center(self):
        model = glGetDoublev(GL_MODELVIEW_MATRIX)
        proj = glGetDoublev(GL_PROJECTION_MATRIX)
        viewport = glGetIntegerv(GL_VIEWPORT)
        cx, cy = self.width / 2, self.height / 2
        closest = None
        min_dist = float('inf')
        for b in self.bodies:
            wx, wy, wz = gluProject(b.pos[0], b.pos[1], b.pos[2], model, proj, viewport)
            if not (0 < wz < 1):
                continue
            d = math.hypot(wx - cx, wy - cy)
            if d < min_dist:
                min_dist = d
                closest = b
        if closest and min_dist < 80:
            self.focused_body = closest
            self.mode = 'orbit'
            cur = np.linalg.norm(self.cam_pos - closest.pos)
            self.orbit_distance = max(5 * closest.radius, cur * 0.6)
            self.orbit_yaw = self.yaw
            self.orbit_pitch = self.pitch

    # -------------------------------------------------
    # Adaptive timestep (Hill-sphere style)
    # -------------------------------------------------
    def _compute_adaptive_dt(self):
        n = len(self.bodies)
        if n < 2:
            return 3600.0
        pos = np.stack([b.pos for b in self.bodies])
        mass = np.array([b.mass for b in self.bodies], dtype=np.float64)
        dx = pos[None, :, :] - pos[:, None, :]
        dist_sq = np.sum(dx ** 2, axis=-1)
        dist = np.sqrt(dist_sq)
        i, j = np.triu_indices(n, k=1)
        d_ij = dist[i, j]
        msum = mass[i] + mass[j]
        tscales = np.sqrt(d_ij ** 3 / (G * msum + 1e-30))
        min_t = np.min(tscales)
        eta = 0.02
        return eta * min_t

    # -------------------------------------------------
    # Movement
    # -------------------------------------------------
    def _process_keyboard(self, dt):
        if self.mode != 'free':
            return
        speed = 3e11 * dt * (5.0 if self.keys.get(glfw.KEY_R, False) else 1.0)
        if self.keys.get(glfw.KEY_W, False):
            self.cam_pos += speed * self.cam_front
        if self.keys.get(glfw.KEY_S, False):
            self.cam_pos -= speed * self.cam_front
        right = np.cross(self.cam_front, self.cam_up)
        right /= np.linalg.norm(right)
        if self.keys.get(glfw.KEY_A, False):
            self.cam_pos -= speed * right
        if self.keys.get(glfw.KEY_D, False):
            self.cam_pos += speed * right

    def _update_trails(self):
        for b in self.bodies:
            b.trail.append(b.pos.copy())

    def _update_window_title(self):
        step_sec = self.current_dt
        if step_sec < 60:
            step_str = f"{step_sec:.1f} s"
        elif step_sec < 3600:
            step_str = f"{step_sec/60:.1f} min"
        elif step_sec < 86400:
            step_str = f"{step_sec/3600:.1f} h"
        else:
            step_str = f"{step_sec/86400:.1f} d"
        title = f"Minimal N‑Body – {step_str} (×{self.timescale:.2g}) | {self.fps:.0f} FPS"
        if self.paused:
            title += " [PAUSED]"
        if self.mode == 'orbit' and self.focused_body:
            title += f" | Orbiting {self.focused_body.name}"
        glfw.set_window_title(self.window, title)

    # -------------------------------------------------
    # Rendering
    # -------------------------------------------------
    def _draw_3d(self):
        if self.trails:
            glLineWidth(1.5)
            for b in self.bodies:
                n = len(b.trail)
                if n < 2:
                    continue
                trail_array = np.asarray(b.trail, dtype=np.float32)
                glEnableClientState(GL_VERTEX_ARRAY)
                glColor3f(*b.color)
                glVertexPointer(3, GL_FLOAT, 0, trail_array)
                glDrawArrays(GL_LINE_STRIP, 0, n)
                glDisableClientState(GL_VERTEX_ARRAY)

        for b in self.bodies:
            dist = max(1e6, np.linalg.norm(b.pos - self.cam_pos))
            size = max(3.0, min(40.0, (b.radius / dist) * self.fov_factor * 250.0))
            glPointSize(size)
            glBegin(GL_POINTS)
            glColor3f(*b.color)
            glVertex3f(*b.pos)
            glEnd()

    def _draw_hud(self, model, proj, viewport):
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, self.width, 0, self.height)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)
        glColor3f(1.0, 1.0, 1.0)

        # Crosshair
        if self.mode == 'free':
            cx, cy = self.width / 2, self.height / 2
            glLineWidth(1.5)
            glBegin(GL_LINES)
            glVertex2f(cx - 15, cy)
            glVertex2f(cx + 15, cy)
            glVertex2f(cx, cy - 15)
            glVertex2f(cx, cy + 15)
            glEnd()

        # Labels – small bodies hidden unless zoomed in
        if self.show_labels:
            for b in self.bodies:
                if b.radius < 1e6:
                    cam_dist = np.linalg.norm(b.pos - self.cam_pos)
                    if cam_dist > 5e9:
                        continue
                wx, wy, wz = gluProject(b.pos[0], b.pos[1], b.pos[2], model, proj, viewport)
                if not (0 < wz < 1):
                    continue
                glRasterPos2f(wx + 8, wy + 15)
                for char in b.name:
                    glutBitmapCharacter(GLUT_BITMAP_9_BY_15, ord(char))

        # Help overlay (open by default)
        if self.show_help:
            lines = ["=== CONTROLS ===", "Left-click/crosshair : select + orbit", "Mouse drag           : orbit selected",
                     "Mouse wheel          : zoom", "WASD (in orbit)      : back to free flight",
                     "Space                : pause/resume", "T                    : trails",
                     "+ / -                : timescale", "L                    : labels",
                     "H                    : toggle help", "Esc                  : quit"]
            y = self.height - 40
            for line in lines:
                glRasterPos2f(30, y)
                for char in line:
                    glutBitmapCharacter(GLUT_BITMAP_9_BY_15, ord(char))
                y -= 18

        glEnable(GL_DEPTH_TEST)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    def _draw(self):
        glViewport(0, 0, self.width, self.height)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, self.width / max(1.0, self.height), 1e8, 1e14)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        target = self.cam_pos + self.cam_front
        gluLookAt(*self.cam_pos, *target, *self.cam_up)

        model = glGetDoublev(GL_MODELVIEW_MATRIX)
        proj = glGetDoublev(GL_PROJECTION_MATRIX)
        viewport = glGetIntegerv(GL_VIEWPORT)

        self._draw_3d()
        self._draw_hud(model, proj, viewport)
        self._update_window_title()

    # -------------------------------------------------
    # Main loop
    # -------------------------------------------------
    def run(self):
        prev_time = glfw.get_time()
        while not glfw.window_should_close(self.window):
            now = glfw.get_time()
            frame_dt = now - prev_time
            prev_time = now

            glfw.poll_events()

            if self.mode == 'free':
                self._process_keyboard(frame_dt)
            elif self.mode == 'orbit' and self.focused_body:
                if any(self.keys.get(k, False) for k in (glfw.KEY_W, glfw.KEY_A, glfw.KEY_S, glfw.KEY_D)):
                    self.mode = 'free'
                    self.focused_body = None
                    self.yaw = self.orbit_yaw
                    self.pitch = self.orbit_pitch
                    self._update_cam_front()

            if self.mode == 'orbit' and self.focused_body:
                rx = self.orbit_distance * math.cos(math.radians(self.orbit_pitch)) * math.cos(math.radians(self.orbit_yaw))
                ry = self.orbit_distance * math.sin(math.radians(self.orbit_pitch))
                rz = self.orbit_distance * math.cos(math.radians(self.orbit_pitch)) * math.sin(math.radians(self.orbit_yaw))
                offset = np.array([rx, ry, rz], dtype=np.float64)
                self.cam_pos = self.focused_body.pos + offset
                self.cam_front = -offset / self.orbit_distance

            # Hill-sphere adaptive timestep (always computed for title)
            adaptive_base = self._compute_adaptive_dt()
            dt = adaptive_base * self.timescale
            if not self.paused:
                step_leapfrog(self.bodies, dt)
                self._update_trails()
            self.current_dt = dt

            self._draw()
            glfw.swap_buffers(self.window)

            self.frame_count += 1
            if now - self.last_fps_time > 1.0:
                self.fps = self.frame_count / (now - self.last_fps_time)
                self.frame_count = 0
                self.last_fps_time = now

        glfw.terminate()


# -------------------------------------------------
# Bodies (unchanged)
# -------------------------------------------------
bodies = [
    Body("Sun",     1.98847e30, [0.0, 0.0, 0.0],          [0.0, 0.0, 0.0],          6.9634e8, (1.0, 0.85, 0.0)),
    Body("Mercury", 3.3011e23,  [5.79e10, 0.0, 0.0],      [0.0, 4.79e4, 0.0],       2.44e6,   (0.7, 0.7, 0.7)),
    Body("Venus",   4.8675e24,  [1.082e11, 0.0, 0.0],     [0.0, 3.5e4, 0.0],        6.05e6,   (1.0, 0.9, 0.6)),
    Body("Earth",   5.97237e24,[1.496e11, 0.0, 0.0],     [0.0, 2.978e4, 0.0],      6.371e6,  (0.0, 0.5, 1.0)),
    Body("Moon",    7.342e22,   [1.496e11 + 3.844e8, 0.0, 0.0], [0.0, 2.978e4 + 1.022e3, 0.0], 1.737e6,  (0.8, 0.8, 0.8)),
    Body("Mars",    6.4171e23,  [2.279e11, 0.0, 0.0],    [0.0, 2.41e4, 0.0],       3.389e6,  (1.0, 0.4, 0.3)),
    Body("Jupiter", 1.8982e27,  [7.785e11, 0.0, 0.0],    [0.0, 1.307e4, 0.0],      6.9911e7, (1.0, 0.75, 0.5)),
    Body("Saturn",  5.6834e26,  [1.4335e12, 0.0, 0.0],   [0.0, 9.68e3, 0.0],       5.8232e7, (0.95, 0.9, 0.7)),
    Body("Uranus",  8.6810e25,  [2.8725e12, 0.0, 0.0],   [0.0, 6.8e3, 0.0],        2.5362e7, (0.6, 0.8, 1.0)),
    Body("Neptune", 1.02413e26, [4.4951e12, 0.0, 0.0],   [0.0, 5.4e3, 0.0],        2.4622e7, (0.4, 0.6, 1.0)),
]

jup = bodies[6]
bodies.extend([
    Body("Io",       8.9319e22, jup.pos + np.array([4.2170e8, 0.0, 0.0]), jup.vel + np.array([0.0, 1.7334e4, 0.0]), 1.8216e6, (1.0, 0.8, 0.6)),
    Body("Europa",   4.7998e22, jup.pos + np.array([6.7090e8, 0.0, 0.0]), jup.vel + np.array([0.0, 1.3740e4, 0.0]), 1.5608e6, (0.8, 0.9, 1.0)),
    Body("Ganymede", 1.4819e23, jup.pos + np.array([1.0704e9, 0.0, 0.0]), jup.vel + np.array([0.0, 1.0880e4, 0.0]), 2.631e6,   (0.7, 0.7, 0.7)),
    Body("Callisto", 1.0759e23, jup.pos + np.array([1.8827e9, 0.0, 0.0]), jup.vel + np.array([0.0, 8.204e3,  0.0]), 2.410e6,   (0.6, 0.6, 0.6)),
])
sat = bodies[7]
bodies.append(Body("Titan", 1.3452e23, sat.pos + np.array([1.2219e9, 0.0, 0.0]), sat.vel + np.array([0.0, 5.57e3, 0.0]), 2.575e6, (0.8, 0.8, 0.7)))
mars = bodies[5]
bodies.extend([
    Body("Phobos", 1.0659e16, mars.pos + np.array([9.377e6, 0.0, 0.0]), mars.vel + np.array([0.0, 2.138e3, 0.0]), 1.13e4, (0.5, 0.5, 0.5)),
    Body("Deimos", 1.4762e15, mars.pos + np.array([2.346e7, 0.0, 0.0]), mars.vel + np.array([0.0, 1.351e3, 0.0]), 6.2e3,  (0.6, 0.6, 0.6)),
])
bodies.append(Body("Ceres", 9.3835e20, [4.14e11, 0.0, 0.0], [0.0, 1.79e4, 0.0], 4.73e5, (0.6, 0.4, 0.3)))

total_mass = sum(b.mass for b in bodies)
cm = sum(b.mass * b.pos for b in bodies) / total_mass
for b in bodies:
    b.pos -= cm
total_mom = sum(b.mass * b.vel for b in bodies)
bodies[0].vel = -total_mom / bodies[0].mass


if __name__ == "__main__":
    renderer = Renderer(bodies)
    renderer.run()