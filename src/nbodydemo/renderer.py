import math
import sys
import numpy as np

import glfw
from OpenGL.GL import *
from OpenGL.GLU import gluPerspective, gluLookAt, gluProject, gluOrtho2D
from collections import deque

from .simulator import Simulator
from .body import Body

# GLUT for labels
try:
    from OpenGL.GLUT import glutInit, glutBitmapCharacter, GLUT_BITMAP_9_BY_15
    GLUT_AVAILABLE = True
except (ImportError, AttributeError):
    GLUT_AVAILABLE = False
    print("WARNING: GLUT not available – name labels disabled.")

from .integrator import step_leapfrog, G
from .body import Body

class Renderer:
    def __init__(self, simulator: Simulator, width=1000, height=700):
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

        self.simulator = simulator
        self.bodies = simulator.bodies

        # Camera & state
        max_distance = max(np.linalg.norm(b.pos) for b in self.bodies)
        self.cam_pos = np.array([0.0, 0.0, max_distance * 2.0], dtype=np.float64)
        self.yaw = -90.0
        self.pitch = 0.0
        self.cam_front = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        self.cam_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        self.previous_mouse_x = width / 2
        self.previous_mouse_y = height / 2
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

        self.current_timestep = 3600.0

        # FPS
        self.fps = 60.0
        self.frame_count = 0
        self.last_fps_update_time = glfw.get_time()

        self.fov_factor = height / (2.0 * math.tan(math.radians(22.5)))

    # -------------------------------------------------
    # Input
    # -------------------------------------------------
    def _on_key(self, window, key, scancode, action, mods):
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

    def _on_mouse(self, window, mouse_x, mouse_y):
        if self.first_mouse:
            self.previous_mouse_x = mouse_x
            self.previous_mouse_y = mouse_y
            self.first_mouse = False
        mouse_delta_x = mouse_x - self.previous_mouse_x
        mouse_delta_y = self.previous_mouse_y - mouse_y
        self.previous_mouse_x = mouse_x
        self.previous_mouse_y = mouse_y
        mouse_sensitivity = 0.1
        if self.mode == 'free':
            self.yaw += mouse_delta_x * mouse_sensitivity
            self.pitch += mouse_delta_y * mouse_sensitivity
            self.pitch = max(-89.0, min(89.0, self.pitch))
            self._update_cam_front()
        else:
            self.orbit_yaw += mouse_delta_x * mouse_sensitivity
            self.orbit_pitch += mouse_delta_y * mouse_sensitivity
            self.orbit_pitch = max(-89.0, min(89.0, self.orbit_pitch))

    def _update_cam_front(self):
        self.cam_front = np.array([
            math.cos(math.radians(self.yaw)) * math.cos(math.radians(self.pitch)),
            math.sin(math.radians(self.pitch)),
            math.sin(math.radians(self.yaw)) * math.cos(math.radians(self.pitch))
        ], dtype=np.float64)
        self.cam_front /= np.linalg.norm(self.cam_front)

    def _on_mouse_button(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS and self.mode == 'free':
            self._select_body_at_center()

    def _on_scroll(self, window, scroll_x, scroll_y):
        if self.mode == 'orbit' and self.focused_body:
            zoom_factor = 0.9 if scroll_y > 0 else 1.11
            self.orbit_distance *= zoom_factor
            self.orbit_distance = max(5 * self.focused_body.radius, self.orbit_distance)

    def _select_body_at_center(self):
        model = glGetDoublev(GL_MODELVIEW_MATRIX)
        proj = glGetDoublev(GL_PROJECTION_MATRIX)
        viewport = glGetIntegerv(GL_VIEWPORT)
        screen_center_x, screen_center_y = self.width / 2, self.height / 2
        closest_body = None
        min_screen_distance = float('inf')
        for body in self.bodies:
            projected_x, projected_y, projected_z = gluProject(body.pos[0], body.pos[1], body.pos[2], model, proj, viewport)
            if not (0 < projected_z < 1):
                continue
            screen_distance = math.hypot(projected_x - screen_center_x, projected_y - screen_center_y)
            if screen_distance < min_screen_distance:
                min_screen_distance = screen_distance
                closest_body = body
        if closest_body and min_screen_distance < 80:
            self.focused_body = closest_body
            self.mode = 'orbit'
            current_distance = np.linalg.norm(self.cam_pos - closest_body.pos)
            self.orbit_distance = max(5 * closest_body.radius, current_distance * 0.6)
            self.orbit_yaw = self.yaw
            self.orbit_pitch = self.pitch

    # -------------------------------------------------
    # Adaptive timestep (Hill-sphere style)
    # -------------------------------------------------
    def _compute_adaptive_dt(self):
        num_bodies = len(self.bodies)
        if num_bodies < 2:
            return 3600.0
        body_positions = np.stack([b.pos for b in self.bodies])
        body_masses = np.array([b.mass for b in self.bodies], dtype=np.float64)
        relative_positions = body_positions[None, :, :] - body_positions[:, None, :]
        pairwise_distance_squared = np.sum(relative_positions ** 2, axis=-1)
        pairwise_distances = np.sqrt(pairwise_distance_squared)
        pair_index_1, pair_index_2 = np.triu_indices(num_bodies, k=1)
        pair_distances = pairwise_distances[pair_index_1, pair_index_2]
        pairwise_mass_sums = body_masses[pair_index_1] + body_masses[pair_index_2]
        orbital_timescales = np.sqrt(pair_distances ** 3 / (G * pairwise_mass_sums + 1e-30))
        min_orbital_timescale = np.min(orbital_timescales)
        safety_factor = 0.02
        return safety_factor * min_orbital_timescale

    # -------------------------------------------------
    # Movement
    # -------------------------------------------------
    def _process_keyboard(self, frame_delta_time):
        if self.mode != 'free':
            return
        camera_speed = 3e11 * frame_delta_time * (5.0 if self.keys.get(glfw.KEY_R, False) else 1.0)
        if self.keys.get(glfw.KEY_W, False):
            self.cam_pos += camera_speed * self.cam_front
        if self.keys.get(glfw.KEY_S, False):
            self.cam_pos -= camera_speed * self.cam_front
        right = np.cross(self.cam_front, self.cam_up)
        right /= np.linalg.norm(right)
        if self.keys.get(glfw.KEY_A, False):
            self.cam_pos -= camera_speed * right
        if self.keys.get(glfw.KEY_D, False):
            self.cam_pos += camera_speed * right

    def _update_trails(self):
        for body in self.bodies:
            body.trail.append(body.pos.copy())

    def _update_window_title(self):
        timestep_seconds = self.current_timestep
        if timestep_seconds < 60:
            timestep_string = f"{timestep_seconds:.1f} s"
        elif timestep_seconds < 3600:
            timestep_string = f"{timestep_seconds/60:.1f} min"
        elif timestep_seconds < 86400:
            timestep_string = f"{timestep_seconds/3600:.1f} h"
        else:
            timestep_string = f"{timestep_seconds/86400:.1f} d"
        title = f"Minimal N‑Body – {timestep_string} (×{self.timescale:.2g}) | {self.fps:.0f} FPS"
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
            if self.trails:
                glLineWidth(1.5)
                for body in self.bodies:
                    if len(body.trail) < 2:
                        continue
                    trail_array = np.asarray(body.trail, dtype=np.float32)
                    glEnableClientState(GL_VERTEX_ARRAY)
                    glColor3f(*body.color)
                    glVertexPointer(3, GL_FLOAT, 0, trail_array)
                    glDrawArrays(GL_LINE_STRIP, 0, len(body.trail))
                    glDisableClientState(GL_VERTEX_ARRAY)
        for body in self.bodies:
            distance = max(1e6, np.linalg.norm(body.pos - self.cam_pos))
            size = max(3.0, min(40.0, (body.radius / distance) * self.fov_factor * 250.0))
            glPointSize(size)
            glBegin(GL_POINTS)
            glColor3f(*body.color)
            glVertex3f(*body.pos)
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
            screen_center_x, screen_center_y = self.width / 2, self.height / 2
            glLineWidth(1.5)
            glBegin(GL_LINES)
            glVertex2f(screen_center_x - 15, screen_center_y)
            glVertex2f(screen_center_x + 15, screen_center_y)
            glVertex2f(screen_center_x, screen_center_y - 15)
            glVertex2f(screen_center_x, screen_center_y + 15)
            glEnd()

        # Labels – small bodies hidden unless zoomed in
        if self.show_labels:
            for body in self.bodies:
                cam_dist = np.linalg.norm(body.pos - self.cam_pos)
                # Planets (radius >= 5e6) always shown. Moons hide when far.
                if body.radius < 5e6 and cam_dist > 5e9:
                    continue
                projected_x, projected_y, projected_z = gluProject(body.pos[0], body.pos[1], body.pos[2], model, proj, viewport)
                if not (0 < projected_z < 1):
                    continue
                glRasterPos2f(projected_x + 8, projected_y + 15)
                for char in body.name:
                    glutBitmapCharacter(GLUT_BITMAP_9_BY_15, ord(char))

        # Help overlay (open by default)
        if self.show_help:
            help_text_lines = ["=== CONTROLS ===", "Left-click/crosshair : select + orbit", "Mouse drag           : orbit selected",
                               "Mouse wheel          : zoom", "WASD (in orbit)      : back to free flight",
                               "Space                : pause/resume", "T                    : trails",
                               "+ / -                : timescale", "L                    : labels",
                               "H                    : toggle help", "Esc                  : quit"]
            current_text_y = self.height - 40
            for help_line in help_text_lines:
                glRasterPos2f(30, current_text_y)
                for char in help_line:
                    glutBitmapCharacter(GLUT_BITMAP_9_BY_15, ord(char))
                current_text_y -= 18

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
        previous_frame_time = glfw.get_time()
        while not glfw.window_should_close(self.window):
            current_frame_time = glfw.get_time()
            frame_delta_time = current_frame_time - previous_frame_time
            previous_frame_time = current_frame_time

            glfw.poll_events()

            if self.mode == 'free':
                self._process_keyboard(frame_delta_time)
            elif self.mode == 'orbit' and self.focused_body:
                if any(self.keys.get(k, False) for k in (glfw.KEY_W, glfw.KEY_A, glfw.KEY_S, glfw.KEY_D)):
                    self.mode = 'free'
                    self.focused_body = None
                    self.yaw = self.orbit_yaw
                    self.pitch = self.orbit_pitch
                    self._update_cam_front()

            if self.mode == 'orbit' and self.focused_body:
                x_offset = self.orbit_distance * math.cos(math.radians(self.orbit_pitch)) * math.cos(math.radians(self.orbit_yaw))
                y_offset = self.orbit_distance * math.sin(math.radians(self.orbit_pitch))
                z_offset = self.orbit_distance * math.cos(math.radians(self.orbit_pitch)) * math.sin(math.radians(self.orbit_yaw))
                offset = np.array([x_offset, y_offset, z_offset], dtype=np.float64)
                self.cam_pos = self.focused_body.pos + offset
                self.cam_front = -offset / self.orbit_distance

            # Hill-sphere adaptive timestep (always computed for title)
            base_adaptive_timestep = self.simulator.compute_adaptive_dt()
            simulation_timestep = base_adaptive_timestep * self.timescale
            if not self.paused:
                self.simulator.step(simulation_timestep)
            self.current_timestep = simulation_timestep

            self._draw()
            glfw.swap_buffers(self.window)

            self.frame_count += 1
            if current_frame_time - self.last_fps_update_time > 1.0:
                self.fps = self.frame_count / (current_frame_time - self.last_fps_update_time)
                self.frame_count = 0
                self.last_fps_update_time = current_frame_time

        glfw.terminate()