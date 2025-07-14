import os
import struct
import math
from collections import defaultdict
from typing import Dict, List, Tuple

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
except ImportError:
    raise SystemExit("PyOpenGL required: pip install PyOpenGL PyOpenGL_accelerate")

try:
    from PySide6.QtWidgets import (
        QApplication,
        QMainWindow,
        QFileDialog,
        QAction,
    )
    from PySide6.QtCore import QTimer
    from PySide6.QtOpenGLWidgets import QOpenGLWidget
except ImportError:
    raise SystemExit(
        "PySide6 required for the viewer: pip install PySide6"
    )

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Pillow required: pip install Pillow")

# -----------------------------------------------------------------------------
# Basic binary reading helpers

def read_bytes(data: bytes, offset: int, count: int) -> Tuple[bytes, int]:
    return data[offset:offset + count], offset + count

def read_str(data: bytes, offset: int, count: int) -> Tuple[str, int]:
    b, offset = read_bytes(data, offset, count)
    return b.decode('ascii', errors='ignore'), offset

def read_u32(data: bytes, offset: int) -> Tuple[int, int]:
    val = struct.unpack('<I', data[offset:offset + 4])[0]
    return val, offset + 4

def read_u16(data: bytes, offset: int) -> Tuple[int, int]:
    val = struct.unpack('<H', data[offset:offset + 2])[0]
    return val, offset + 2

def read_u8(data: bytes, offset: int) -> Tuple[int, int]:
    val = data[offset]
    return val, offset + 1

def read_float(data: bytes, offset: int) -> Tuple[float, int]:
    val = struct.unpack('<f', data[offset:offset + 4])[0]
    if math.isnan(val):
        val = 0.0
    return val, offset + 4

def read_vec3(data: bytes, offset: int) -> Tuple[Tuple[float, float, float], int]:
    x, offset = read_float(data, offset)
    y, offset = read_float(data, offset)
    z, offset = read_float(data, offset)
    return (x, y, z), offset

# -----------------------------------------------------------------------------
# .msh loader based on shadowman_mesh.load

def load_msh(filepath: str) -> Dict:
    with open(filepath, 'rb') as f:
        data = f.read()

    offset = 0
    s_file, offset = read_str(data, offset, 4)
    if s_file != 'EMsh':
        raise ValueError('Not a valid .msh file')

    s_ver, offset = read_str(data, offset, 4)
    if s_ver != 'V001':
        raise ValueError('Unsupported msh version')

    face_count, offset = read_u32(data, offset)
    vert_count, offset = read_u32(data, offset)

    faces = []
    loop_count = 0
    for _ in range(face_count):
        num_verts, offset = read_u8(data, offset)
        fill_mode, offset = read_u8(data, offset)
        unk1, offset = read_u8(data, offset)
        unk2, offset = read_u8(data, offset)
        tex_index, offset = read_u16(data, offset)
        unk3, offset = read_u16(data, offset)
        attributes, offset = read_u16(data, offset)
        c_plane = []
        for i in range(4):
            v, offset = read_float(data, offset)
            v = -v
            c_plane.append(v)
        c_plane[0] = -c_plane[0]
        indices = []
        uvs = []
        colors = []
        for i in range(num_verts):
            vi, offset = read_u16(data, offset)
            u, offset = read_float(data, offset)
            v, offset = read_float(data, offset)
            uv = (u, -v)
            r, offset = read_u8(data, offset)
            g, offset = read_u8(data, offset)
            b, offset = read_u8(data, offset)
            a, offset = read_u8(data, offset)
            indices.append(vi)
            uvs.append(uv)
            colors.append((r, g, b, a))
            loop_count += 1
        faces.append({
            'numVerts': num_verts,
            'texIndex': tex_index,
            'indices': indices,
            'uvs': uvs,
            'colors': colors,
        })

    verts = []
    normals = []
    for i in range(vert_count):
        loc, offset = read_vec3(data, offset)
        nx, offset = read_float(data, offset)
        ny, offset = read_float(data, offset)
        nz, offset = read_float(data, offset)
        loc = (-loc[0], loc[1], loc[2])
        normal = (nx, -ny, -nz)
        verts.append(loc)
        normals.append(normal)

    loop_normals = []
    remaining = len(data) - offset
    expected_bytes = loop_count * 12
    if remaining < expected_bytes:
        for face in faces:
            for idx in face['indices']:
                loop_normals.append(normals[idx])
    else:
        for _ in range(loop_count):
            nx, offset = read_float(data, offset)
            ny, offset = read_float(data, offset)
            nz, offset = read_float(data, offset)
            loop_normals.append((nx, -ny, -nz))

    return {
        'faces': faces,
        'verts': verts,
        'normals': normals,
        'loop_normals': loop_normals,
    }

# -----------------------------------------------------------------------------
# texture loading helper

VALID_TEXTURE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.tga']

def load_textures(directory: str) -> Dict[int, int]:
    textures = {}
    for filename in os.listdir(directory):
        base, ext = os.path.splitext(filename)
        if ext.lower() not in VALID_TEXTURE_EXTENSIONS:
            continue
        try:
            index = int(base.split('_')[0]) if '_' in base else int(base)
        except ValueError:
            continue
        path = os.path.join(directory, filename)
        img = Image.open(path).transpose(Image.FLIP_TOP_BOTTOM)
        img_data = img.convert('RGBA').tobytes()
        width, height = img.size
        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
        textures[index] = tex_id
    return textures

# -----------------------------------------------------------------------------
# Simple OpenGL viewer

class GLViewer(QOpenGLWidget):
    """OpenGL widget that displays a .msh model."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = None
        self.textures = {}
        self.center = (0.0, 0.0, 0.0)
        self.scale = 1.0
        self.angle = 0.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_angle)
        self.timer.start(16)

    # ------------------------------------------------------------------
    def _update_angle(self):
        self.angle = (self.angle + 0.5) % 360
        self.update()

    def load_model(self, path: str):
        self.model = load_msh(path)
        self.textures = load_textures(os.path.dirname(path))
        self.center, self.scale = self._compute_bounds()
        self.update()

    def _compute_bounds(self):
        verts = self.model['verts'] if self.model else []
        if not verts:
            return (0.0, 0.0, 0.0), 1.0
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        zs = [v[2] for v in verts]
        min_v = (min(xs), min(ys), min(zs))
        max_v = (max(xs), max(ys), max(zs))
        center = (
            (min_v[0] + max_v[0]) / 2.0,
            (min_v[1] + max_v[1]) / 2.0,
            (min_v[2] + max_v[2]) / 2.0,
        )
        size = max(max_v[0] - min_v[0], max_v[1] - min_v[1], max_v[2] - min_v[2])
        scale = 2.0 / size if size != 0 else 1.0
        return center, scale

    # ------------------------------------------------------------------
    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        glClearColor(0.1, 0.1, 0.1, 1.0)

    def _group_faces_by_texture(self):
        groups = defaultdict(list)
        if not self.model:
            return groups
        for face in self.model['faces']:
            groups[face['texIndex']].append(face)
        return groups

    def _draw_model(self):
        if not self.model:
            return
        verts = self.model['verts']
        for tex_index, faces in self._group_faces_by_texture().items():
            tex_id = self.textures.get(tex_index)
            glBindTexture(GL_TEXTURE_2D, tex_id or 0)
            glBegin(GL_TRIANGLES)
            for face in faces:
                for vi, uv in zip(face['indices'], face['uvs']):
                    glTexCoord2f(uv[0], uv[1])
                    x, y, z = verts[vi]
                    x = (x - self.center[0]) * self.scale
                    y = (y - self.center[1]) * self.scale
                    z = (z - self.center[2]) * self.scale
                    glVertex3f(x, y, z)
            glEnd()

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -3.0)
        glRotatef(self.angle, 0.0, 1.0, 0.0)
        self._draw_model()

    def resizeGL(self, w: int, h: int):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, float(w) / float(h or 1), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)


class MainWindow(QMainWindow):
    """Main application window with a menu to load .msh files."""

    def __init__(self, mesh_path: str | None = None):
        super().__init__()
        self.setWindowTitle("Shadow Man .msh Viewer")
        self.viewer = GLViewer(self)
        self.setCentralWidget(self.viewer)
        self._create_actions()
        if mesh_path:
            self.viewer.load_model(mesh_path)

    def _create_actions(self):
        open_act = QAction("Open .msh", self)
        open_act.triggered.connect(self._open_file)
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(open_act)

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open msh",
            "",
            "Mesh Files (*.msh)",
        )
        if path:
            self.viewer.load_model(path)

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    mesh_path = sys.argv[1] if len(sys.argv) > 1 else None
    app = QApplication(sys.argv)
    window = MainWindow(mesh_path)
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec())
