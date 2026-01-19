"""
HELOSFORGE is a real-time 3D Model Manipulation System with Gesture Control

Gestures:
    PINCH+MOVE  - Rotate model
    TWO_PINCH   - Scale model
    GRAB        - Pan/translate
    POINT       - Reset view
    SPREAD      - Toggle wireframe
    SWIPE_LEFT  - Undo (requires confirmation)
    SWIPE_RIGHT - Redo (requires confirmation)

Keys:
    1-5   - Load primitives
    S     - Save STL
    G     - AI generate model
    E     - AI edit loaded STL
    C     - Toggle camera view
    W     - Toggle wireframe
    R     - Reset view
    Y     - Confirm pending action
    N     - Cancel pending action
    ESC   - Quit
"""

import re
import sys
import time
import threading
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

import cv2
import numpy as np
import trimesh
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

from TACTUS_Gesture_Recognizer import (
    Gesture_Recognizer, GestureType, GestureState,
    draw_landmarks, draw_gesture_status
)

OPENAI_API_KEY = ""  # Set your OpenAI API key here or via environment variable

@dataclass
class ModelTransform:
    rotation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    scale: float = 1.0
    
    def copy(self) -> 'ModelTransform':
        return ModelTransform(self.rotation.copy(), self.translation.copy(), self.scale)


class AIModelGenerator:
    """
    Generates 3D models via OpenAI -> OpenSCAD -> STL pipeline with validation + auto-repair.
    """

    SYSTEM_PROMPT = """
    You are an OpenSCAD code generator.

    OUTPUT REQUIREMENTS:
    - Output ONLY valid OpenSCAD code. No markdown. No explanations.
    - First line: $fn=64;
    - Define: module main() { ... }
    - Last line: main();

    GEOMETRY:
    - Watertight, printable solid
    - Centered at origin
    - Size ~30-80mm

    ALLOWED:
    - Primitives: cube, sphere, cylinder, polyhedron, polygon, circle, square, text
    - Transforms: translate, rotate, scale, mirror, linear_extrude, rotate_extrude
    - Ops: union, difference, intersection, hull, minkowski
    - Math: sin, cos, tan, sqrt, pow, abs, min, max, floor, ceil

    FORBIDDEN:
    - use <...>; include <...>; import(); surface();
    - External libraries (MCAD/BOSL/etc)
    - File operations

    Output clean, working OpenSCAD. Nothing else.
    """

    SYSTEM_PROMPT_EDIT = """
    You are an OpenSCAD STL editor.

    You will be given a base() module that imports an existing STL.
    DO NOT redefine base(). DO NOT use import() yourself.

    OUTPUT REQUIREMENTS:
    - Output ONLY valid OpenSCAD code. No markdown. No explanations.
    - First line: $fn=64;
    - Define: module main() { ... }
    - Last line: main();

    EDITING RULES:
    - You must build from the existing model by calling base();
    - Prefer simple, printable CSG edits:
      * difference() for holes/cavities/channels
      * union() to add features (tabs, bosses, handles, feet)
      * intersection() to trim
      * minkowski() with a small sphere for gentle rounding/thickening (use sparingly)
    - Keep it centered-ish and printable.
    - Avoid extremely heavy operations (huge minkowski radii, super high $fn beyond 64).
    """
    
    PRIMITIVES = {
        "cube": lambda: trimesh.creation.box(extents=[1, 1, 1]),
        "sphere": lambda: trimesh.creation.icosphere(radius=0.5, subdivisions=3),
        "cylinder": lambda: trimesh.creation.cylinder(radius=0.5, height=1.0, sections=64),
        "cone": lambda: trimesh.creation.cone(radius=0.5, height=1.0, sections=64),
        "torus": lambda: trimesh.creation.torus(major_radius=0.5, minor_radius=0.15),
    }

    FORBIDDEN_PATTERNS = [r"\buse\s*<", r"\binclude\s*<", r"\bimport\s*\(", r"\bsurface\s*\("]

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5.2"):
        self.client = None
        self.model = model
        if api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key)
            except ImportError:
                print("OpenAI package not installed: pip install openai")

    def generate_primitive(self, shape: str = "cube") -> trimesh.Trimesh:
        return self.PRIMITIVES.get(shape, self.PRIMITIVES["cube"])()

    def _sanitize_scad(self, raw: str) -> str:
        if not raw:
            return ""

        code = raw.strip()

        #Strip markdown fences (both ```openscad and ```scad variants)
        if "```" in code:
            code = re.sub(r'```(?:openscad|scad)?\s*', '', code)
            code = code.replace('```', '').strip()

        #Remove preamble junk
        lines = code.splitlines()
        bad_starts = ('here', 'below', 'the following', 'sure', 'certainly', 'of course', 'openscad')
        while lines and any(lines[0].lower().strip().startswith(b) for b in bad_starts):
            lines.pop(0)
        while lines and not lines[0].strip():
            lines.pop(0)
        code = '\n'.join(lines).strip()

        #Ensure $fn=64; at top
        code = re.sub(r'^\s*\$fn\s*=\s*\d+\s*;?\s*\n?', '', code, flags=re.MULTILINE)
        code = '$fn=64;\n' + code

        #Wrap in main() if missing
        if 'module main' not in code:
            body = '\n'.join(code.splitlines()[1:])  #Skip $fn line
            code = '$fn=64;\nmodule main() {\n' + body + '\n}\nmain();'

        #Ensure main(); call at end
        if not re.search(r'main\s*\(\s*\)\s*;\s*$', code.strip()):
            code = code.rstrip() + '\nmain();'

        return code.strip() + '\n'

    def _validate_scad(self, code: str, *, forbid_import: bool = True, require_geom: bool = True) -> Optional[str]:
        if len(code.strip()) < 30:
            return "Code too short"

        forbidden = [r"\buse\s*<", r"\binclude\s*<", r"\bsurface\s*\("]
        if forbid_import:
            forbidden.append(r"\bimport\s*\(")

        for pat in forbidden:
            if re.search(pat, code, re.IGNORECASE):
                return f"Forbidden pattern: {pat}"

        if 'module main' not in code:
            return "Missing module main()"

        for a, b, name in [('{', '}', 'braces'), ('(', ')', 'parens'), ('[', ']', 'brackets')]:
            if code.count(a) != code.count(b):
                return f"Unbalanced {name}"

        if require_geom:
            geom = ('cube(', 'sphere(', 'cylinder(', 'polyhedron(', 'linear_extrude(',
                    'rotate_extrude(', 'union(', 'difference(', 'intersection(')
            if not any(g in code for g in geom):
                return "No geometry primitives found"

        return None

    def _find_openscad(self) -> Optional[str]:
        for exe in ('openscad', 'OpenSCAD'):
            try:
                if subprocess.run([exe, '--version'], capture_output=True, timeout=3).returncode == 0:
                    return exe
            except:
                pass
        mac_path = '/Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD'
        return mac_path if Path(mac_path).exists() else None

    def _compile_stl(self, scad_path: Path, stl_path: Path) -> tuple[bool, str]:
        exe = self._find_openscad()
        if not exe:
            return False, "OpenSCAD not found. Install from https://openscad.org"

        try:
            result = subprocess.run(
                [exe, '-o', str(stl_path), str(scad_path)],
                capture_output=True, text=True, timeout=120
            )
            log = f"{result.stdout}\n{result.stderr}".strip()
            
            if result.returncode != 0:
                return False, log
            if not stl_path.exists() or stl_path.stat().st_size < 500:
                return False, "Output STL empty or too small"
            return True, log
        except subprocess.TimeoutExpired:
            return False, "OpenSCAD timed out"
        except Exception as e:
            return False, str(e)

    def generate_openscad(self, prompt: str, repair_hint: Optional[str] = None) -> Optional[str]:
        if not self.client:
            print("ERROR: No OpenAI client")
            return None

        msg = f"Create 3D model: {prompt}"
        if repair_hint:
            msg += f"\n\nPREVIOUS ATTEMPT FAILED:\n{repair_hint}\n\nFix all issues. Output complete corrected file."

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                max_completion_tokens=2000,
                temperature=0.15,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": msg}
                ]
            )
            raw = resp.choices[0].message.content or ""
            code = self._sanitize_scad(raw)
            
            err = self._validate_scad(code, forbid_import=True, require_geom=True)
            if err:
                print(f"Validation failed: {err}")
                return None
            return code
        except Exception as e:
            print(f"API error: {e}")
            return None

    def generate_stl(self, prompt: str, output_path: str, max_attempts: int = 3) -> bool:
        stl_path = Path(output_path)
        scad_path = stl_path.with_suffix('.scad')
        repair_hint = None

        for attempt in range(1, max_attempts + 1):
            print(f"[Attempt {attempt}/{max_attempts}] Generating...")
            
            code = self.generate_openscad(prompt, repair_hint)
            if not code:
                repair_hint = "Output failed validation. Ensure: $fn=64; first, module main(){...}, main(); last. No external deps."
                continue

            scad_path.write_text(code)
            print(f"Saved: {scad_path}")

            ok, log = self._compile_stl(scad_path, stl_path)
            if ok:
                print(f"Success: {stl_path}")
                return True

            print(f"Compile failed:\n{log[:500]}")
            repair_hint = f"Compiler error:\n{log[:800]}\n\nFix the code. Self-contained, no imports."

        print(f"Failed after {max_attempts} attempts. Check {scad_path}")
        return False

    def _validate_scad_edit(self, code: str) -> Optional[str]:
        #Reuse basic validation checks from _validate_scad
        err = self._validate_scad(code, forbid_import=True, require_geom=False)
        if err:
            return err

        #Prevent AI from doing its own imports/includes
        if re.search(r"\bimport\s*\(", code, re.IGNORECASE):
            return "Edit mode forbids import(). Use base() only."
        if re.search(r"\buse\s*<|\binclude\s*<", code, re.IGNORECASE):
            return "Forbidden include/use."

        #Encourage they actually reference base()
        if "base()" not in code:
            return "Edit does not reference base(). Must call base()."

        return None

    def generate_openscad_edit(self, prompt: str, repair_hint: Optional[str] = None) -> Optional[str]:
        if not self.client:
            print("ERROR: No OpenAI client")
            return None

        msg = (
            "You are editing an existing STL.\n"
            "You MUST use base() as the starting geometry.\n\n"
            f"Edit request: {prompt}\n"
        )
        if repair_hint:
            msg += f"\nPREVIOUS ATTEMPT FAILED:\n{repair_hint}\nFix all issues. Output complete corrected file."

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                max_completion_tokens=2000,
                temperature=0.15,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT_EDIT},
                    {"role": "user", "content": msg}
                ]
            )
            raw = resp.choices[0].message.content or ""
            code = self._sanitize_scad(raw)

            err = self._validate_scad_edit(code)
            if err:
                print(f"Edit validation failed: {err}")
                return None
            return code
        except Exception as e:
            print(f"API error: {e}")
            return None

    def edit_stl(self, base_stl_path: str, edit_prompt: str, output_path: str, max_attempts: int = 3) -> bool:
        """
        base_stl_path: existing STL to modify
        edit_prompt: natural language edit request
        output_path: new STL output
        """
        base_stl = Path(base_stl_path)
        out_stl = Path(output_path)
        out_scad = out_stl.with_suffix(".scad")

        if not base_stl.exists():
            print(f"Base STL not found: {base_stl}")
            return False

        repair_hint = None

        for attempt in range(1, max_attempts + 1):
            print(f"[Edit Attempt {attempt}/{max_attempts}] Editing...")
            core = self.generate_openscad_edit(edit_prompt, repair_hint)
            if not core:
                repair_hint = (
                    "Output failed validation. Requirements: $fn=64 first, module main(){...}, main(); last. "
                    "Must call base(). Must not use import()/include/use."
                )
                continue

            #Inject a safe base() module ourselves (AI never gets to choose file paths).
            #Strip any accidental base() definition the model might have produced for extra safety.
            core_no_base = re.sub(r"module\s+base\s*\([^)]*\)\s*\{.*?\}", "", core, flags=re.DOTALL | re.IGNORECASE)

            injected = (
                "$fn=64;\n"
                f'module base() {{ import("{base_stl.as_posix()}"); }}\n\n'
                + "\n".join(core_no_base.splitlines()[1:])  #drop the model's $fn line if present
            ).strip() + "\n"

            #Final validation (must not contain import() except our injected one at base())
            if re.search(r"\bimport\s*\(", injected, re.IGNORECASE):
                #allow exactly one import from our injected base()
                #quick check: if more than one import(), reject
                if len(re.findall(r"\bimport\s*\(", injected, re.IGNORECASE)) > 1:
                    repair_hint = "Multiple import() detected. Only base() may import."
                    continue

            err = self._validate_scad(injected, forbid_import=False, require_geom=False)
            if err:
                repair_hint = f"SCAD invalid: {err}"
                continue

            #Save scad + compile
            out_scad.write_text(injected)
            ok, log = self._compile_stl(out_scad, out_stl)
            if ok:
                print(f"Edit success: {out_stl}")
                return True

            print(f"Edit compile failed:\n{log[:800]}")
            repair_hint = f"Compiler error:\n{log[:800]}\nFix code; use base(); no extra imports."

        print(f"Edit failed after {max_attempts} attempts.")
        return False

class ModelViewer:
    def __init__(self, width: int = 800, height: int = 600):
        self.width, self.height = width, height
        self.mesh: Optional[trimesh.Trimesh] = None
        self.transform = ModelTransform()
        self.wireframe = False
        self.camera_distance = 5.0
        self.vertices = self.normals = None
        self.undo_stack: deque = deque(maxlen=50)
        self.redo_stack: deque = deque(maxlen=50)
        self.last_gesture = GestureType.NONE
        self.pending_action = None
        self.pending_action_time = 0
        self.pending_timeout = 3.0

        pygame.init()
        pygame.display.set_caption("HELOSFORGE")
        self.screen = pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL | RESIZABLE)
        self.font = pygame.font.Font(None, 24)
        self._init_gl()

    def _init_gl(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_NORMALIZE)
        glLightfv(GL_LIGHT0, GL_POSITION, [1, 1, 1, 0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1])
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glClearColor(0.15, 0.15, 0.2, 1.0)
        self._setup_projection()

    def _setup_projection(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, self.width / self.height, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def load_mesh(self, path: str) -> bool:
        try:
            mesh = trimesh.load(path)
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            self.set_mesh(mesh)
            print(f"Loaded: {path} ({len(self.mesh.vertices)} verts, {len(self.mesh.faces)} faces)")
            return True
        except Exception as e:
            print(f"Load error: {e}")
            return False

    def set_mesh(self, mesh: trimesh.Trimesh):
        self.mesh = mesh
        self.mesh.vertices -= self.mesh.centroid
        self.mesh.vertices *= 2.0 / max(self.mesh.extents)
        self.vertices = self.mesh.vertices[self.mesh.faces].reshape(-1, 3).astype(np.float32)
        self.normals = np.repeat(self.mesh.face_normals, 3, axis=0).astype(np.float32) \
            if self.mesh.face_normals is not None else np.zeros_like(self.vertices)
        self.transform = ModelTransform()
        self.undo_stack.clear()
        self.redo_stack.clear()

    def save_mesh(self, path: str) -> bool:
        if not self.mesh:
            return False
        try:
            t = self.mesh.copy()
            t.vertices *= self.transform.scale
            rot = self.transform.rotation * np.pi / 180
            R = trimesh.transformations.euler_matrix(rot[0], rot[1], rot[2])[:3, :3]
            t.vertices = t.vertices @ R.T + self.transform.translation
            t.export(path)
            print(f"Saved: {path}")
            return True
        except Exception as e:
            print(f"Save error: {e}")
            return False

    def save_snapshot(self):
        self.undo_stack.append(self.transform.copy())
        self.redo_stack.clear()

    def undo(self):
        if self.undo_stack:
            self.redo_stack.append(self.transform.copy())
            self.transform = self.undo_stack.pop()
        self.pending_action = None

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(self.transform.copy())
            self.transform = self.redo_stack.pop()
        self.pending_action = None

    def reset_view(self):
        self.save_snapshot()
        self.transform = ModelTransform()

    def toggle_wireframe(self):
        self.wireframe = not self.wireframe

    def request_action(self, action: str):
        self.pending_action = action
        self.pending_action_time = time.time()

    def confirm_action(self):
        if self.pending_action == 'undo': self.undo()
        elif self.pending_action == 'redo': self.redo()

    def cancel_action(self):
        self.pending_action = None

    def check_timeout(self):
        if self.pending_action and time.time() - self.pending_action_time > self.pending_timeout:
            self.pending_action = None

    def apply_gesture(self, state: GestureState):
        g = state.gesture
        if g != self.last_gesture:
            if g in (GestureType.ROTATE, GestureType.GRAB, GestureType.TWO_PINCH):
                self.save_snapshot()
            self.last_gesture = g

        if g == GestureType.SWIPE_LEFT and self.pending_action != 'undo':
            self.request_action('undo')
        elif g == GestureType.SWIPE_RIGHT and self.pending_action != 'redo':
            self.request_action('redo')
        elif g == GestureType.POINT:
            self.reset_view()
        elif g == GestureType.SPREAD:
            self.toggle_wireframe()
        elif g == GestureType.ROTATE:
            self.transform.rotation[0] += state.rotation_delta[0] * 50.0
            self.transform.rotation[1] += state.rotation_delta[1] * 50.0
        elif g == GestureType.GRAB:
            self.transform.translation[0] += state.velocity[0] * 0.032
            self.transform.translation[1] -= state.velocity[1] * 0.032
        elif g == GestureType.TWO_PINCH:
            self.transform.scale = np.clip(self.transform.scale * state.scale_factor, 0.1, 10.0)

    def render(self, generating: bool = False):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        gluLookAt(0, 0, self.camera_distance, 0, 0, 0, 0, 1, 0)

        if self.mesh is not None and self.vertices is not None:
            glPushMatrix()
            glTranslatef(*self.transform.translation)
            glRotatef(self.transform.rotation[0], 1, 0, 0)
            glRotatef(self.transform.rotation[1], 0, 1, 0)
            glRotatef(self.transform.rotation[2], 0, 0, 1)
            glScalef(self.transform.scale, self.transform.scale, self.transform.scale)

            if self.wireframe:
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                glDisable(GL_LIGHTING)
                glColor3f(0.0, 1.0, 0.5)
            else:
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
                glEnable(GL_LIGHTING)
                glColor3f(0.3, 0.6, 0.9)

            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_NORMAL_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, self.vertices)
            glNormalPointer(GL_FLOAT, 0, self.normals)
            glDrawArrays(GL_TRIANGLES, 0, len(self.vertices))
            glDisableClientState(GL_VERTEX_ARRAY)
            glDisableClientState(GL_NORMAL_ARRAY)
            glPopMatrix()

        self._draw_grid()
        self._draw_hud(generating)
        pygame.display.flip()

    def _draw_grid(self):
        glDisable(GL_LIGHTING)
        glColor4f(0.4, 0.4, 0.4, 0.5)
        glBegin(GL_LINES)
        for i in range(-5, 6):
            glVertex3f(i, -2, -5); glVertex3f(i, -2, 5)
            glVertex3f(-5, -2, i); glVertex3f(5, -2, i)
        glEnd()
        glEnable(GL_LIGHTING)

    def _draw_hud(self, generating: bool):
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        #Status
        y = 15
        self._text(overlay, 15, y, f"Gesture: {self.last_gesture.name}", (0, 255, 128)); y += 22
        self._text(overlay, 15, y, f"Rotation: {self.transform.rotation.astype(int)}", (0, 255, 128)); y += 22
        self._text(overlay, 15, y, f"Scale: {self.transform.scale:.2f}", (0, 255, 128))
        if generating:
            self._text(overlay, 15, y + 28, "Generating...", (255, 255, 0))

        #Controls/Gesture reference
        rx = self.width - 200
        self._text(overlay, rx, 15, "GESTURES", (255, 180, 0))
        for i, t in enumerate(["PINCH+MOVE: Rotate", "TWO_PINCH: Scale", "GRAB: Pan",
                               "POINT: Reset", "SPREAD: Wireframe", "SWIPE: Undo/Redo"]):
            self._text(overlay, rx, 40 + i * 18, t, (160, 160, 160))

        self._text(overlay, rx, self.height - 110, "KEYS", (255, 180, 0))
        for i, t in enumerate(["1-5: Primitives", "S: Save", "G: Generate", "E: Edit STL", "C: Camera", "ESC: Quit"]):
            self._text(overlay, rx, self.height - 88 + i * 18, t, (160, 160, 160))

        if self.pending_action:
            w, h, cx, cy = 280, 50, self.width // 2, self.height - 40
            pygame.draw.rect(overlay, (20, 20, 25, 220), (cx - w//2, cy - h//2, w, h), border_radius=6)
            pygame.draw.rect(overlay, (255, 140, 0), (cx - w//2, cy - h//2, w, h), 2, border_radius=6)
            remaining = self.pending_timeout - (time.time() - self.pending_action_time)
            txt = f"{self.pending_action.upper()}?  Y/N  ({remaining:.0f}s)"
            r = self.font.render(txt, True, (255, 255, 255))
            overlay.blit(r, (cx - r.get_width() // 2, cy - 8))

        self._blit_overlay(overlay)

    def _text(self, surf, x, y, txt, color):
        surf.blit(self.font.render(txt, True, color), (x, y))

    def _blit_overlay(self, surface):
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        data = pygame.image.tostring(surface, "RGBA", True)
        glRasterPos2f(0, self.height)
        glDrawPixels(self.width, self.height, GL_RGBA, GL_UNSIGNED_BYTE, data)
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def handle_resize(self, w, h):
        self.width, self.height = w, h
        glViewport(0, 0, w, h)
        self._setup_projection()


class ModelManipulatorApp:
    def __init__(self, model_path: Optional[str] = None):
        self.viewer = ModelViewer()
        self.recognizer = Gesture_Recognizer()
        self.ai = AIModelGenerator(OPENAI_API_KEY)

        if model_path and Path(model_path).exists():
            self.viewer.load_mesh(model_path)
        else:
            self.viewer.set_mesh(self.ai.generate_primitive("cube"))

        self.running = True
        self.show_camera = False
        self.generating = False
        self.cap = None
        self.gesture_state = GestureState()
        self.frame = None
        self.lock = threading.Lock()
        self._camera_window_open = False
        
        self.base_dir = Path(__file__).resolve().parent / "outputs"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.export_dir = self.base_dir / "exports"
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
        self.gen_dir = self.base_dir / "generations"
        self.gen_dir.mkdir(parents=True, exist_ok=True)        
        
    def run(self):
        self._start_capture()
        clock = pygame.time.Clock()

        while self.running:
            self.viewer.check_timeout()

            for event in pygame.event.get():
                if event.type == QUIT:
                    self.running = False
                elif event.type == VIDEORESIZE:
                    self.viewer.handle_resize(event.w, event.h)
                elif event.type == KEYDOWN:
                    self._handle_key(event.key)
                elif event.type == DROPFILE:
                    self.viewer.load_mesh(event.file)

            with self.lock:
                state, frame = self.gesture_state, self.frame

            if self.show_camera and frame is not None:
                cv2.imshow("Camera", frame)
                cv2.waitKey(1)
                self._camera_window_open = True
            else:
                if self._camera_window_open:
                    cv2.destroyAllWindows()
                    self._camera_window_open = False

            self.viewer.apply_gesture(state)
            self.viewer.render(self.generating)
            clock.tick(60)

        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

    def _start_capture(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Warning: No camera")
            return
        threading.Thread(target=self._capture_loop, daemon=True).start()

    def _capture_loop(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            state, results = self.recognizer.process_frame(frame, time.monotonic())
            frame = draw_landmarks(frame, results)
            frame = draw_gesture_status(frame, state)
            with self.lock:
                self.gesture_state = state
                self.frame = frame
            time.sleep(0.01)

    def _handle_key(self, key):
        if key == K_y: self.viewer.confirm_action()
        elif key == K_n: self.viewer.cancel_action()
        elif key == K_ESCAPE: self.running = False
        elif key == K_e: self._prompt_edit()
        elif key == K_r: self.viewer.reset_view()
        elif key == K_w: self.viewer.toggle_wireframe()
        elif key == K_c: self.show_camera = not self.show_camera
        elif key == K_s:
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_path = self.export_dir / f"export_{ts}.stl"
            self.viewer.save_mesh(str(out_path))
        elif key in (K_1, K_2, K_3, K_4, K_5):
            shapes = {K_1: "cube", K_2: "sphere", K_3: "cylinder", K_4: "cone", K_5: "torus"}
            self.viewer.set_mesh(self.ai.generate_primitive(shapes[key]))
        elif key == K_g: self._prompt_generate()

    def _prompt_generate(self):
        if not self.ai.client or self.generating:
            return
        self.generating = True
        threading.Thread(target=self._generate_thread, daemon=True).start()

    def _generate_thread(self):
        try:
            prompt = input("Describe the stl model you want to generate: ").strip()
            if not prompt:
                return

            ts = time.strftime("%Y%m%d_%H%M%S")
            out_stl = self.gen_dir / f"gen_{ts}.stl"

            if self.ai.generate_stl(prompt, str(out_stl)):
                self.viewer.load_mesh(str(out_stl))
        finally:
            self.generating = False
        
    def _prompt_edit(self):
        if not self.ai.client or self.generating or self.viewer.mesh is None:
            return
        self.generating = True
        threading.Thread(target=self._edit_thread, daemon=True).start()

    def _edit_thread(self):
        try:
            edit_prompt = input("Edit current model (e.g., 'add a 6mm hole through the center'): ").strip()
            if not edit_prompt:
                return

            #Save the currently displayed geometry (with transforms applied) as the base STL for editing
            workdir = self.base_dir / "edits"
            workdir.mkdir(parents=True, exist_ok=True)

            ts = time.strftime("%Y%m%d_%H%M%S")
            base_path = workdir / f"base_{ts}.stl"
            out_path = workdir / f"edit_{ts}.stl"

            #Export current mesh with transform applied
            if not self.viewer.save_mesh(str(base_path)):
                print("Could not export current model for editing.")
                return

            if self.ai.edit_stl(str(base_path), edit_prompt, str(out_path)):
                self.viewer.load_mesh(str(out_path))
        finally:
            self.generating = False
            
def main():
    ModelManipulatorApp(sys.argv[1] if len(sys.argv) > 1 else None).run()

if __name__ == "__main__":
    main()