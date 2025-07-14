#!/usr/bin/env python3
"""
Shadow Man SKN & Animation Visualizer - GUI Version
Interface gráfica para carregar e visualizar arquivos .skn e .anims do Shadow Man

CARACTERÍSTICAS:
    - Carregamento e análise de arquivos .skn (skinning)
    - Carregamento e análise de arquivos .anims (animações)
    - Visualização organizada dos dados de skinning e animação
    - Exportação completa para JSON
    - Interface moderna com tema escuro

INSTALAÇÃO:
    pip install PySide6

USO:
    python skn.py                 # Inicia interface gráfica
    python skn.py --test          # Testa dependências

ATALHOS:
    Ctrl+O: Abrir arquivo SKN
    Ctrl+Shift+O: Abrir arquivo ANIMS
    Ctrl+E: Exportar dados para JSON
    Ctrl+L: Limpar console
    Ctrl+Q: Sair

REQUISITOS:
    - Python 3.7+
    - PySide6
"""

import sys
import os
import struct
import math
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
except ImportError:
    print("❌ PyOpenGL não está instalado. Execute: pip install PyOpenGL PyOpenGL_accelerate")
    sys.exit(1)

# Verificar se PySide6 está instalado
try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QTextEdit, QFileDialog, QLabel,
        QGroupBox, QComboBox, QCheckBox
    )
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QFont, QPalette, QColor, QAction
    from PySide6.QtOpenGLWidgets import QOpenGLWidget
except ImportError as e:
    print("❌ Erro: PySide6 não está instalado ou há problema na instalação.")
    print("Por favor, execute: pip install PySide6")
    print(f"Erro específico: {e}")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("❌ Pillow não está instalado. Execute: pip install Pillow")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Basic binary reading helpers para arquivos .msh

def read_bytes(data: bytes, offset: int, count: int) -> Tuple[bytes, int]:
    return data[offset:offset + count], offset + count

def read_str(data: bytes, offset: int, count: int) -> Tuple[str, int]:
    b, offset = read_bytes(data, offset, count)
    return b.decode("ascii", errors="ignore"), offset

def read_u32(data: bytes, offset: int) -> Tuple[int, int]:
    val = struct.unpack("<I", data[offset:offset + 4])[0]
    return val, offset + 4

def read_u16(data: bytes, offset: int) -> Tuple[int, int]:
    val = struct.unpack("<H", data[offset:offset + 2])[0]
    return val, offset + 2

def read_u8(data: bytes, offset: int) -> Tuple[int, int]:
    val = data[offset]
    return val, offset + 1

def read_float(data: bytes, offset: int) -> Tuple[float, int]:
    val = struct.unpack("<f", data[offset:offset + 4])[0]
    if math.isnan(val):
        val = 0.0
    return val, offset + 4

def read_vec3(data: bytes, offset: int) -> Tuple[Tuple[float, float, float], int]:
    x, offset = read_float(data, offset)
    y, offset = read_float(data, offset)
    z, offset = read_float(data, offset)
    return (x, y, z), offset

# -----------------------------------------------------------------------------

# Carregamento simples de arquivos .msh baseado em shadowman_mesh.load

def load_msh(filepath: str) -> Dict:
    with open(filepath, "rb") as f:
        data = f.read()

    offset = 0
    s_file, offset = read_str(data, offset, 4)
    if s_file != "EMsh":
        raise ValueError("Not a valid .msh file")

    s_ver, offset = read_str(data, offset, 4)
    if s_ver != "V001":
        raise ValueError("Unsupported msh version")

    face_count, offset = read_u32(data, offset)
    vert_count, offset = read_u32(data, offset)

    faces = []
    loop_count = 0
    for _ in range(face_count):
        num_verts, offset = read_u8(data, offset)
        _fill, offset = read_u8(data, offset)
        _u1, offset = read_u8(data, offset)
        _u2, offset = read_u8(data, offset)
        tex_index, offset = read_u16(data, offset)
        _u3, offset = read_u16(data, offset)
        _attr, offset = read_u16(data, offset)
        for _ in range(4):
            v, offset = read_float(data, offset)
        indices = []
        uvs = []
        colors = []
        for _ in range(num_verts):
            vi, offset = read_u16(data, offset)
            u, offset = read_float(data, offset)
            v, offset = read_float(data, offset)
            r, offset = read_u8(data, offset)
            g, offset = read_u8(data, offset)
            b, offset = read_u8(data, offset)
            a, offset = read_u8(data, offset)
            indices.append(vi)
            uvs.append((u, -v))
            colors.append((r, g, b, a))
            loop_count += 1
        faces.append({
            "numVerts": num_verts,
            "texIndex": tex_index,
            "indices": indices,
            "uvs": uvs,
            "colors": colors,
        })

    verts = []
    normals = []
    for _ in range(vert_count):
        loc, offset = read_vec3(data, offset)
        nx, offset = read_float(data, offset)
        ny, offset = read_float(data, offset)
        nz, offset = read_float(data, offset)
        verts.append((-loc[0], loc[1], loc[2]))
        normals.append((nx, -ny, -nz))

    loop_normals = []
    if offset >= len(data):
        for face in faces:
            for idx in face["indices"]:
                loop_normals.append(normals[idx])
    else:
        for _ in range(loop_count):
            nx, offset = read_float(data, offset)
            ny, offset = read_float(data, offset)
            nz, offset = read_float(data, offset)
            loop_normals.append((nx, -ny, -nz))

    return {
        "faces": faces,
        "verts": verts,
        "normals": normals,
        "loop_normals": loop_normals,
    }

# -----------------------------------------------------------------------------
# Carregamento de texturas por índice

VALID_TEXTURE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tga"]

def load_textures(directory: str) -> Dict[int, int]:
    textures = {}
    for filename in os.listdir(directory):
        base, ext = os.path.splitext(filename)
        if ext.lower() not in VALID_TEXTURE_EXTENSIONS:
            continue
        try:
            index = int(base.split("_")[0]) if "_" in base else int(base)
        except ValueError:
            continue
        path = os.path.join(directory, filename)
        img = Image.open(path).transpose(Image.FLIP_TOP_BOTTOM)
        img_data = img.convert("RGBA").tobytes()
        width, height = img.size
        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
        textures[index] = tex_id
    return textures

# -----------------------------------------------------------------------------

class ANIMSParser:
    """Classe para parsing de arquivos ANIMS."""
    
    def __init__(self):
        self.anims_data = {}
        
    def read_string(self, data: bytes, offset: int, length: int = None) -> Tuple[str, int]:
        """Lê uma string de tamanho fixo ou terminada em null."""
        if length is not None:
            string_bytes = data[offset:offset + length]
            string_value = string_bytes.decode('ascii', errors='ignore').rstrip('\x00')
            return string_value, offset + length
        else:
            # String terminada em null
            end_offset = offset
            while end_offset < len(data) and data[end_offset] != 0:
                end_offset += 1
            string_bytes = data[offset:end_offset]
            string_value = string_bytes.decode('ascii', errors='ignore')
            return string_value, end_offset + 1  # +1 para pular o null terminator
    
    def read_uint32(self, data: bytes, offset: int, signed: bool = False) -> Tuple[int, int]:
        """Lê um inteiro de 32 bits."""
        format_char = '<i' if signed else '<I'
        value = struct.unpack(format_char, data[offset:offset + 4])[0]
        return value, offset + 4
    
    def read_float(self, data: bytes, offset: int) -> Tuple[float, int]:
        """Lê um float de 32 bits."""
        value = struct.unpack('<f', data[offset:offset + 4])[0]
        if math.isnan(value):
            value = 0.0
        return value, offset + 4
    
    def read_vector3(self, data: bytes, offset: int) -> Tuple[Tuple[float, float, float], int]:
        """Lê um vetor 3D (3 floats)."""
        x, offset = self.read_float(data, offset)
        y, offset = self.read_float(data, offset)
        z, offset = self.read_float(data, offset)
        return (x, y, z), offset
    
    def read_vector4(self, data: bytes, offset: int) -> Tuple[Tuple[float, float, float, float], int]:
        """Lê um vetor 4D (4 floats)."""
        x, offset = self.read_float(data, offset)
        y, offset = self.read_float(data, offset)
        z, offset = self.read_float(data, offset)
        w, offset = self.read_float(data, offset)
        return (x, y, z, w), offset
    
    def load_anims_file(self, filepath: str) -> Tuple[bool, str]:
        """Carrega e interpreta um arquivo .anims. Retorna (sucesso, mensagem)."""
        try:
            with open(filepath, 'rb') as f:
                data = f.read()
            
            offset = 0
            
            # Verificar cabeçalho
            file_type, offset = self.read_string(data, offset, 4)
            if file_type != 'mnAE':
                return False, f"Arquivo não é um ANIMS válido (cabeçalho: {file_type})"
            
            # Contagem de animações
            anim_count, offset = self.read_uint32(data, offset, True)
            self.anims_data['anim_count'] = anim_count
            
            # Ler offsets das animações
            anim_offsets = []
            for i in range(anim_count):
                anim_offset, offset = self.read_uint32(data, offset, False)
                anim_offsets.append(anim_offset)
            
            # Ler dados de cada animação
            animations = []
            for i in range(anim_count):
                anim = {
                    'index': i,
                    'name': '',
                    'num_bones': 0,
                    'num_frames': 0,
                    'bones': []
                }
                
                # Número de ossos
                num_bones, offset = self.read_uint32(data, offset, True)
                anim['num_bones'] = num_bones
                
                # Nome da animação
                anim_name, offset = self.read_string(data, offset)
                anim['name'] = anim_name
                
                # Número de frames
                num_frames, offset = self.read_uint32(data, offset, True)
                anim['num_frames'] = num_frames
                
                # Dados de cada osso
                for bone_idx in range(num_bones):
                    bone_data = {
                        'bone_index': bone_idx,
                        'unknown1': 0.0,
                        'trans_offset': (0.0, 0.0, 0.0),
                        'translation_keyframes': [],
                        'unknown2': 0.0,
                        'unknown3': 0.0,
                        'unused_pivot': (0.0, 0.0, 0.0),
                        'rotation_keyframes': []
                    }
                    
                    # Unknown1
                    unknown1, offset = self.read_float(data, offset)
                    bone_data['unknown1'] = unknown1
                    
                    # Translation offset
                    trans_offset, offset = self.read_vector3(data, offset)
                    bone_data['trans_offset'] = trans_offset
                    
                    # Translation keyframes
                    trans_key_count, offset = self.read_uint32(data, offset, True)
                    for k in range(trans_key_count):
                        frame, offset = self.read_uint32(data, offset, True)
                        location, offset = self.read_vector3(data, offset)
                        bone_data['translation_keyframes'].append({
                            'frame': frame - 1,  # Converter para 0-indexed
                            'location': location
                        })
                    
                    # Unknown2 e Unknown3
                    unknown2, offset = self.read_float(data, offset)
                    unknown3, offset = self.read_float(data, offset)
                    bone_data['unknown2'] = unknown2
                    bone_data['unknown3'] = unknown3
                    
                    # Unused pivot
                    unused_pivot, offset = self.read_vector3(data, offset)
                    bone_data['unused_pivot'] = unused_pivot
                    
                    # Rotation keyframes
                    rot_key_count, offset = self.read_uint32(data, offset, True)
                    for k in range(rot_key_count):
                        frame, offset = self.read_uint32(data, offset, True)
                        rotation, offset = self.read_vector4(data, offset)
                        bone_data['rotation_keyframes'].append({
                            'frame': frame - 1,  # Converter para 0-indexed
                            'rotation': rotation  # Quaternion (x, y, z, w)
                        })
                    
                    anim['bones'].append(bone_data)
                
                animations.append(anim)
            
            self.anims_data['animations'] = animations
            self.anims_data['filepath'] = filepath
            
            return True, f"Arquivo ANIMS carregado com sucesso!"
            
        except Exception as e:
            return False, f"Erro ao carregar arquivo: {str(e)}"

class SKNParser:
    """Classe para parsing de arquivos SKN."""
    
    def __init__(self):
        self.skin_data = {}
        
    def read_string(self, data: bytes, offset: int, length: int) -> Tuple[str, int]:
        """Lê uma string de tamanho fixo do buffer de dados."""
        string_bytes = data[offset:offset + length]
        string_value = string_bytes.decode('ascii', errors='ignore').rstrip('\x00')
        return string_value, offset + length
    
    def read_uint32(self, data: bytes, offset: int, signed: bool = False) -> Tuple[int, int]:
        """Lê um inteiro de 32 bits."""
        format_char = '<i' if signed else '<I'
        value = struct.unpack(format_char, data[offset:offset + 4])[0]
        return value, offset + 4
    
    def read_uint16(self, data: bytes, offset: int, signed: bool = False) -> Tuple[int, int]:
        """Lê um inteiro de 16 bits."""
        format_char = '<h' if signed else '<H'
        value = struct.unpack(format_char, data[offset:offset + 2])[0]
        return value, offset + 2
    
    def read_float(self, data: bytes, offset: int) -> Tuple[float, int]:
        """Lê um float de 32 bits."""
        value = struct.unpack('<f', data[offset:offset + 4])[0]
        if math.isnan(value):
            value = 0.0
        return value, offset + 4
    
    def read_vector3(self, data: bytes, offset: int) -> Tuple[Tuple[float, float, float], int]:
        """Lê um vetor 3D (3 floats)."""
        x, offset = self.read_float(data, offset)
        y, offset = self.read_float(data, offset)
        z, offset = self.read_float(data, offset)
        return (x, y, z), offset
    
    def load_skn_file(self, filepath: str) -> Tuple[bool, str]:
        """Carrega e interpreta um arquivo .skn. Retorna (sucesso, mensagem)."""
        try:
            with open(filepath, 'rb') as f:
                data = f.read()
            
            offset = 0
            
            # Verificar cabeçalho
            file_type, offset = self.read_string(data, offset, 4)
            if file_type != 'BSKN':
                return False, f"Arquivo não é um SKN válido (cabeçalho: {file_type})"
            
            # Contagem de ossos
            bone_count, offset = self.read_uint32(data, offset, False)
            self.skin_data['bone_count'] = bone_count
            
            # Seção HRCY (Hierarquia)
            hrcy_header, offset = self.read_string(data, offset, 4)
            if hrcy_header != 'HRCY':
                return False, f"Cabeçalho HRCY esperado, encontrado: {hrcy_header}"
            
            # Ler hierarquia dos ossos (parents)
            bones = []
            for i in range(bone_count):
                parent, offset = self.read_uint32(data, offset, True)
                bones.append({
                    'index': i,
                    'parent': parent,
                    'n_hards': 0,
                    'n_soft_types': 0,
                    'hard_i': 0,
                    'soft_type_i': 0
                })
            
            # Ler informações detalhadas dos ossos
            for i in range(bone_count):
                bone_header, offset = self.read_string(data, offset, 4)
                if bone_header != 'BONE':
                    return False, f"Cabeçalho BONE esperado, encontrado: {bone_header}"
                
                n_hards, offset = self.read_uint16(data, offset, False)
                n_soft_types, offset = self.read_uint16(data, offset, False)
                hard_i, offset = self.read_uint16(data, offset, False)
                soft_type_i, offset = self.read_uint16(data, offset, False)
                
                bones[i].update({
                    'n_hards': n_hards,
                    'n_soft_types': n_soft_types,
                    'hard_i': hard_i,
                    'soft_type_i': soft_type_i
                })
            
            self.skin_data['bones'] = bones
            
            # Seção SOFT (Soft Bones)
            soft_header, offset = self.read_string(data, offset, 4)
            if soft_header != 'SOFT':
                return False, f"Cabeçalho SOFT esperado, encontrado: {soft_header}"
            
            soft_bone_count, offset = self.read_uint32(data, offset, False)
            self.skin_data['soft_bone_count'] = soft_bone_count
            
            soft_bones = []
            for i in range(soft_bone_count):
                styp_header, offset = self.read_string(data, offset, 4)
                if styp_header != 'STYP':
                    return False, f"Cabeçalho STYP esperado, encontrado: {styp_header}"
                
                weight_raw, offset = self.read_uint16(data, offset, False)
                weight = float(weight_raw / 65535.0)
                n_softs, offset = self.read_uint16(data, offset, False)
                soft_i, offset = self.read_uint16(data, offset, False)
                
                # Ler matriz 4x3 (12 floats)
                matrix = []
                for row in range(4):
                    matrix_row = []
                    for col in range(3):
                        value, offset = self.read_float(data, offset)
                        matrix_row.append(value)
                    matrix.append(matrix_row)
                
                soft_bones.append({
                    'index': i,
                    'weight': weight,
                    'n_softs': n_softs,
                    'soft_i': soft_i,
                    'matrix': matrix
                })
            
            self.skin_data['soft_bones'] = soft_bones
            
            # Seção BIND (Índices)
            bind_header, offset = self.read_string(data, offset, 4)
            if bind_header != 'BIND':
                return False, f"Cabeçalho BIND esperado, encontrado: {bind_header}"
            
            indice_count, offset = self.read_uint16(data, offset, False)
            self.skin_data['indice_count'] = indice_count
            
            indices = []
            for i in range(indice_count):
                index, offset = self.read_uint16(data, offset, False)
                indices.append(index)
            
            self.skin_data['indices'] = indices
            
            # Seção VERT (Vértices)
            vert_header, offset = self.read_string(data, offset, 4)
            if vert_header != 'VERT':
                return False, f"Cabeçalho VERT esperado, encontrado: {vert_header}"
            
            vert_count, offset = self.read_uint32(data, offset, False)
            self.skin_data['vert_count'] = vert_count
            
            vertices = []
            for i in range(vert_count):
                vertex, offset = self.read_vector3(data, offset)
                vertices.append(vertex)
            
            self.skin_data['vertices'] = vertices
            self.skin_data['filepath'] = filepath
            
            return True, f"Arquivo SKN carregado com sucesso!"
            
        except Exception as e:
            return False, f"Erro ao carregar arquivo: {str(e)}"


class GLViewer(QOpenGLWidget):
    """Widget OpenGL para visualizar modelos .msh."""

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

class SKNVisualizerWidget(QWidget):
    """Widget principal do visualizador SKN."""
    
    def __init__(self):
        super().__init__()
        self.parser = SKNParser()
        self.anims_parser = ANIMSParser()
        self.current_file = None
        self.current_anims_file = None
        self.current_mesh_file = None
        self.setup_ui()
        
    def setup_ui(self):
        """Configura a interface do usuário."""
        layout = QVBoxLayout(self)
        
        # Cabeçalho
        header_layout = QHBoxLayout()
        
        # Botão para carregar arquivo
        self.load_button = QPushButton("📁 Carregar Arquivo SKN")
        self.load_button.clicked.connect(self.load_file)
        self.load_button.setMinimumHeight(40)
        header_layout.addWidget(self.load_button)
        
        # Botão para carregar animações
        self.load_anims_button = QPushButton("🎬 Carregar Animações")
        self.load_anims_button.clicked.connect(self.load_anims_file)
        self.load_anims_button.setMinimumHeight(40)
        header_layout.addWidget(self.load_anims_button)

        # Botão para carregar mesh
        self.load_mesh_button = QPushButton("📦 Carregar Mesh")
        self.load_mesh_button.clicked.connect(self.load_mesh_file)
        self.load_mesh_button.setMinimumHeight(40)
        header_layout.addWidget(self.load_mesh_button)
        
        # Botão para exportar JSON
        self.export_button = QPushButton("💾 Exportar JSON")
        self.export_button.clicked.connect(self.export_json)
        self.export_button.setMinimumHeight(40)
        self.export_button.setEnabled(False)  # Desabilitado até carregar um arquivo
        header_layout.addWidget(self.export_button)
        
        # Label do arquivo atual
        self.file_label = QLabel("Nenhum arquivo carregado")
        self.file_label.setStyleSheet("color: #666; font-style: italic;")
        header_layout.addWidget(self.file_label)
        
        # Label do arquivo de animações
        self.anims_label = QLabel("Nenhuma animação carregada")
        self.anims_label.setStyleSheet("color: #666; font-style: italic;")
        header_layout.addWidget(self.anims_label)

        # Label do mesh
        self.mesh_label = QLabel("Nenhum mesh carregado")
        self.mesh_label.setStyleSheet("color: #666; font-style: italic;")
        header_layout.addWidget(self.mesh_label)

        header_layout.addStretch()
        layout.addLayout(header_layout)

        # Visualizador 3D
        self.viewer = GLViewer(self)
        self.viewer.setMinimumHeight(300)
        layout.addWidget(self.viewer)
        
        # Controles de visualização
        controls_group = QGroupBox("Opções de Visualização")
        controls_layout = QHBoxLayout(controls_group)
        
        # Dropdown para tipo de visualização
        self.view_combo = QComboBox()
        self.view_combo.addItems([
            "Resumo Geral SKN",
            "Hierarquia de Ossos",
            "Soft Bones Detalhados",
            "Análise Bone-Vertex",
            "Mapeamento de Vértices",
            "Resumo dos Vértices",
            "--- ANIMAÇÕES ---",
            "Resumo das Animações",
            "Lista de Animações",
            "Detalhes dos Keyframes",
            "Análise de Movimento",
            "--- COMPLETO ---",
            "Visualização Completa SKN",
            "Visualização Completa ANIMS",
            "Visualização Completa TUDO"
        ])
        self.view_combo.currentTextChanged.connect(self.update_display)
        controls_layout.addWidget(QLabel("Visualizar:"))
        controls_layout.addWidget(self.view_combo)
        
        # Checkbox para rolagem automática
        self.auto_scroll_cb = QCheckBox("Rolagem Automática")
        self.auto_scroll_cb.setChecked(True)
        controls_layout.addWidget(self.auto_scroll_cb)
        
        # Botão para limpar console
        clear_button = QPushButton("🗑️ Limpar Console")
        clear_button.clicked.connect(self.clear_console)
        controls_layout.addWidget(clear_button)
        
        controls_layout.addStretch()
        layout.addWidget(controls_group)
        
        # Console de saída
        console_group = QGroupBox("Console de Saída")
        console_layout = QVBoxLayout(console_group)
        
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setFont(QFont("Consolas", 10))
        self.console.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #3e3e3e;
                border-radius: 5px;
            }
        """)
        console_layout.addWidget(self.console)
        
        # Status bar no console
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Pronto")
        self.status_label.setStyleSheet("color: #888;")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        console_layout.addLayout(status_layout)
        
        layout.addWidget(console_group)
        
        # Mensagem inicial
        self.print_to_console("🎮 Shadow Man SKN & Animation Visualizer - GUI")
        self.print_to_console("=" * 60)
        self.print_to_console("📁 Clique em 'Carregar Arquivo SKN' para dados de skinning")
        self.print_to_console("🎬 Clique em 'Carregar Animações' para dados de animação")
        self.print_to_console("💾 Após carregar, use 'Exportar JSON' para salvar os dados")
        self.print_to_console("🎯 Use o dropdown para escolher diferentes visualizações")
        
    def print_to_console(self, text: str, color: str = "#d4d4d4"):
        """Adiciona texto ao console com cor específica."""
        self.console.append(f'<span style="color: {color};">{text}</span>')
        if self.auto_scroll_cb.isChecked():
            scrollbar = self.console.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
    
    def export_json(self):
        """Exporta todos os dados do SKN e ANIMS para um arquivo JSON."""
        if not self.parser.skin_data and not self.anims_parser.anims_data:
            self.print_to_console("❌ Nenhum arquivo carregado!", "#F44336")
            return
        
        # Definir nome padrão do arquivo
        default_name = "shadowman_data.json"
        if self.current_file:
            default_name = f"{os.path.splitext(os.path.basename(self.current_file))[0]}_data.json"
        elif self.current_anims_file:
            default_name = f"{os.path.splitext(os.path.basename(self.current_anims_file))[0]}_data.json"
        
        # Dialog para salvar arquivo
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Salvar dados como JSON",
            default_name,
            "JSON Files (*.json);;All Files (*)"
        )
        
        if not file_path:
            return
            
        try:
            self.status_label.setText("Exportando dados para JSON...")
            
            # Preparar dados para exportação
            export_data = self.prepare_export_data()
            
            # Salvar arquivo JSON
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            # Mostrar resumo da exportação
            file_size = os.path.getsize(file_path)
            self.print_to_console(f"\n✅ Dados exportados com sucesso para: {file_path}", "#4CAF50")
            self.print_to_console(f"📁 Tamanho do arquivo JSON: {file_size:,} bytes", "#2196F3")
            
            # Mostrar resumo do conteúdo exportado
            self.print_to_console(f"📊 Dados exportados:", "#FFC107")
            
            if export_data.get("skn_data"):
                summary = export_data["skn_data"]["summary"]
                self.print_to_console(f"   🦴 Hard Bones: {summary['bone_count']}", "#4CAF50")
                self.print_to_console(f"   🔧 Soft Bones: {summary['soft_bone_count']}", "#FF9800")
                self.print_to_console(f"   📍 Vértices: {summary['vertex_count']}", "#9C27B0")
                self.print_to_console(f"   🔗 Índices: {summary['indice_count']}", "#E91E63")
            
            if export_data.get("anims_data"):
                anim_summary = export_data["anims_data"]["summary"]
                self.print_to_console(f"   🎬 Animações: {anim_summary['animation_count']}", "#E91E63")
                self.print_to_console(f"   📊 Total de Frames: {anim_summary['total_frames']}", "#F06292")
                self.print_to_console(f"   🔑 Total de Keyframes: {anim_summary['total_keyframes']}", "#BA68C8")
            
            self.status_label.setText("Exportação concluída com sucesso")
            
        except Exception as e:
            error_msg = f"Erro ao exportar JSON: {str(e)}"
            self.print_to_console(f"\n❌ {error_msg}", "#F44336")
            self.status_label.setText("Erro na exportação")
    
    def prepare_export_data(self) -> Dict[str, Any]:
        """Prepara os dados do SKN e ANIMS para exportação em JSON."""
        export_data = {
            "metadata": {
                "file_type": "Shadow Man Complete Data Export",
                "export_timestamp": datetime.now().isoformat(),
                "exporter": "Shadow Man SKN Visualizer",
                "version": "1.0",
                "files": {}
            }
        }
        
        # Adicionar dados SKN se disponível
        if self.parser.skin_data:
            skn_data = self.parser.skin_data.copy()
            export_data["metadata"]["files"]["skn"] = os.path.basename(self.current_file) if self.current_file else "unknown"
            export_data["skn_data"] = self.prepare_skn_export_data(skn_data)
        
        # Adicionar dados ANIMS se disponível
        if self.anims_parser.anims_data:
            anims_data = self.anims_parser.anims_data.copy()
            export_data["metadata"]["files"]["anims"] = os.path.basename(self.current_anims_file) if self.current_anims_file else "unknown"
            export_data["anims_data"] = self.prepare_anims_export_data(anims_data)
        
        return export_data
    
    def prepare_skn_export_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepara os dados SKN para exportação."""
        skn_export = {
            "summary": {
                "bone_count": data.get('bone_count', 0),
                "soft_bone_count": data.get('soft_bone_count', 0),
                "vertex_count": data.get('vert_count', 0),
                "indice_count": data.get('indice_count', 0)
            },
            "raw_data": {}
        }
        
        # Adicionar estatísticas calculadas
        if data.get('bones'):
            total_hard_verts = sum(bone['n_hards'] for bone in data['bones'])
            root_bones = [bone for bone in data['bones'] if bone['parent'] < 0]
            
            skn_export["summary"]["total_hard_vertices"] = total_hard_verts
            skn_export["summary"]["root_bone_count"] = len(root_bones)
        
        if data.get('soft_bones'):
            total_soft_verts = sum(soft['n_softs'] for soft in data['soft_bones'])
            skn_export["summary"]["total_soft_vertices"] = total_soft_verts
        
        if data.get('vertices'):
            vertices = data['vertices']
            min_x = min(v[0] for v in vertices) if vertices else 0
            max_x = max(v[0] for v in vertices) if vertices else 0
            min_y = min(v[1] for v in vertices) if vertices else 0
            max_y = max(v[1] for v in vertices) if vertices else 0
            min_z = min(v[2] for v in vertices) if vertices else 0
            max_z = max(v[2] for v in vertices) if vertices else 0
            
            skn_export["summary"]["bounding_box"] = {
                "min": [min_x, min_y, min_z],
                "max": [max_x, max_y, max_z],
                "size": [max_x - min_x, max_y - min_y, max_z - min_z]
            }
        
        if data.get('indices'):
            indices = data['indices']
            skn_export["summary"]["vertex_indices"] = {
                "min_index": min(indices),
                "max_index": max(indices),
                "unique_count": len(set(indices)),
                "duplicate_count": len(indices) - len(set(indices))
            }
        
        # Adicionar dados brutos organizados
        skn_export["raw_data"] = {
            "bones": data.get('bones', []),
            "soft_bones": data.get('soft_bones', []),
            "vertex_indices": data.get('indices', []),
            "vertices": data.get('vertices', [])
        }
        
        # Adicionar análise de hierarquia
        if data.get('bones'):
            skn_export["hierarchy_analysis"] = self.analyze_bone_hierarchy(data['bones'])
        
        # Adicionar análise de vertex binding
        if data.get('bones') and data.get('soft_bones') and data.get('indices'):
            skn_export["vertex_binding_analysis"] = self.analyze_vertex_binding(
                data['bones'], data['soft_bones'], data['indices']
            )
        
        return skn_export
    
    def prepare_anims_export_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepara os dados ANIMS para exportação."""
        animations = data.get('animations', [])
        
        anims_export = {
            "summary": {
                "animation_count": data.get('anim_count', 0),
                "total_frames": 0,
                "total_keyframes": 0,
                "average_frames_per_animation": 0,
                "average_bones_per_animation": 0
            },
            "raw_data": {
                "animations": animations
            },
            "analysis": {}
        }
        
        if animations:
            # Calcular estatísticas
            total_frames = sum(anim['num_frames'] for anim in animations)
            total_bones = sum(anim['num_bones'] for anim in animations)
            total_keyframes = 0
            
            for anim in animations:
                for bone in anim['bones']:
                    total_keyframes += len(bone['translation_keyframes'])
                    total_keyframes += len(bone['rotation_keyframes'])
            
            anims_export["summary"]["total_frames"] = total_frames
            anims_export["summary"]["total_keyframes"] = total_keyframes
            anims_export["summary"]["average_frames_per_animation"] = total_frames / len(animations)
            anims_export["summary"]["average_bones_per_animation"] = total_bones / len(animations)
            
            # Análise detalhada
            anims_export["analysis"] = {
                "frame_distribution": {},
                "keyframe_density": {},
                "movement_analysis": {}
            }
            
            # Distribuição de frames
            frame_counts = [anim['num_frames'] for anim in animations]
            anims_export["analysis"]["frame_distribution"] = {
                "min_frames": min(frame_counts),
                "max_frames": max(frame_counts),
                "median_frames": sorted(frame_counts)[len(frame_counts)//2]
            }
            
            # Densidade de keyframes por animação
            for anim in animations:
                anim_keyframes = 0
                for bone in anim['bones']:
                    anim_keyframes += len(bone['translation_keyframes'])
                    anim_keyframes += len(bone['rotation_keyframes'])
                
                anims_export["analysis"]["keyframe_density"][anim['name']] = {
                    "total_keyframes": anim_keyframes,
                    "keyframes_per_frame": anim_keyframes / max(anim['num_frames'], 1),
                    "keyframes_per_bone": anim_keyframes / max(anim['num_bones'], 1)
                }
        
        return anims_export
    
    def analyze_bone_hierarchy(self, bones: List[Dict]) -> Dict[str, Any]:
        """Analisa a hierarquia dos ossos."""
        analysis = {
            "root_bones": [],
            "bone_tree": {},
            "max_depth": 0,
            "bone_children": {}
        }
        
        # Encontrar ossos raiz
        for bone in bones:
            if bone['parent'] < 0:
                analysis["root_bones"].append(bone['index'])
        
        # Construir árvore de filhos
        for bone in bones:
            parent_idx = bone['parent']
            if parent_idx >= 0:
                if parent_idx not in analysis["bone_children"]:
                    analysis["bone_children"][parent_idx] = []
                analysis["bone_children"][parent_idx].append(bone['index'])
        
        # Calcular profundidade máxima
        def calculate_depth(bone_idx, current_depth=0):
            max_depth = current_depth
            if bone_idx in analysis["bone_children"]:
                for child in analysis["bone_children"][bone_idx]:
                    max_depth = max(max_depth, calculate_depth(child, current_depth + 1))
            return max_depth
        
        for root in analysis["root_bones"]:
            analysis["max_depth"] = max(analysis["max_depth"], calculate_depth(root))
        
        return analysis
    
    def load_anims_file(self):
        """Abre dialog para carregar arquivo ANIMS."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Selecionar Arquivo ANIMS",
            "",
            "Shadow Man Animation Files (*.anims);;All Files (*)"
        )
        
        if file_path:
            self.current_anims_file = file_path
            self.anims_label.setText(f"Animações: {os.path.basename(file_path)}")
            self.status_label.setText("Carregando animações...")
            
            success, message = self.anims_parser.load_anims_file(file_path)
            
            if success:
                self.print_to_console(f"\n🎬 {message}", "#4CAF50")
                self.anims_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
                self.status_label.setText("Animações carregadas com sucesso")
                self.update_export_button_state()
                self.update_display()
            else:
                self.print_to_console(f"\n❌ {message}", "#F44336")
                self.anims_label.setStyleSheet("color: #F44336;")
                self.status_label.setText("Erro ao carregar animações")

    def load_mesh_file(self):
        """Abre dialog para carregar arquivo MSH e exibi-lo."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Selecionar Arquivo MSH",
            "",
            "Shadow Man Mesh Files (*.msh);;All Files (*)",
        )

        if file_path:
            self.current_mesh_file = file_path
            self.mesh_label.setText(f"Mesh: {os.path.basename(file_path)}")
            self.status_label.setText("Carregando mesh...")
            try:
                self.viewer.load_model(file_path)
                self.mesh_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
                self.print_to_console(f"\n✅ Mesh carregado: {os.path.basename(file_path)}", "#4CAF50")
                self.status_label.setText("Mesh carregado com sucesso")
            except Exception as e:
                self.print_to_console(f"\n❌ Erro ao carregar mesh: {e}", "#F44336")
                self.mesh_label.setStyleSheet("color: #F44336;")
                self.status_label.setText("Erro ao carregar mesh")
    
    def update_export_button_state(self):
        """Atualiza o estado do botão de exportar baseado nos arquivos carregados."""
        has_data = bool(self.parser.skin_data or self.anims_parser.anims_data)
        self.export_button.setEnabled(has_data)
    
    def clear_console(self):
        """Limpa o console."""
        self.console.clear()
        self.print_to_console("Console limpo.", "#888")
        """Limpa o console."""
        self.console.clear()
        self.print_to_console("Console limpo.", "#888")
    
    def analyze_vertex_binding(self, bones: List[Dict], soft_bones: List[Dict], indices: List[int]) -> Dict[str, Any]:
        """Analisa como os vértices estão ligados aos ossos."""
        analysis = {
            "hard_bone_vertices": {},
            "soft_bone_vertices": {},
            "vertex_assignments": {}
        }
        
        # Analisar vértices de hard bones
        for bone in bones:
            if bone['n_hards'] > 0:
                start_idx = bone['hard_i']
                end_idx = start_idx + bone['n_hards']
                vertex_indices = indices[start_idx:end_idx]
                
                analysis["hard_bone_vertices"][bone['index']] = {
                    "vertex_count": bone['n_hards'],
                    "vertex_indices": vertex_indices,
                    "index_range": [start_idx, end_idx - 1]
                }
                
                # Mapear cada vértice para seu osso
                for v_idx in vertex_indices:
                    if v_idx not in analysis["vertex_assignments"]:
                        analysis["vertex_assignments"][v_idx] = {"hard_bone": None, "soft_bones": []}
                    analysis["vertex_assignments"][v_idx]["hard_bone"] = bone['index']
        
        # Analisar vértices de soft bones
        for soft_idx, soft in enumerate(soft_bones):
            if soft['n_softs'] > 0:
                start_idx = soft['soft_i']
                end_idx = start_idx + soft['n_softs']
                vertex_indices = indices[start_idx:end_idx]
                
                analysis["soft_bone_vertices"][soft_idx] = {
                    "vertex_count": soft['n_softs'],
                    "vertex_indices": vertex_indices,
                    "weight": soft['weight'],
                    "index_range": [start_idx, end_idx - 1]
                }
                
                # Mapear cada vértice para seus soft bones
                for v_idx in vertex_indices:
                    if v_idx not in analysis["vertex_assignments"]:
                        analysis["vertex_assignments"][v_idx] = {"hard_bone": None, "soft_bones": []}
                    analysis["vertex_assignments"][v_idx]["soft_bones"].append({
                        "soft_bone_index": soft_idx,
                        "weight": soft['weight']
                    })
        
        return analysis
    
    def load_file(self):
        """Abre dialog para carregar arquivo SKN."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Selecionar Arquivo SKN",
            "",
            "Shadow Man Skin Files (*.skn);;All Files (*)"
        )
        
        if file_path:
            self.current_file = file_path
            self.file_label.setText(f"Arquivo: {os.path.basename(file_path)}")
            self.status_label.setText("Carregando arquivo...")
            
            success, message = self.parser.load_skn_file(file_path)
            
            if success:
                self.print_to_console(f"\n✅ {message}", "#4CAF50")
                self.file_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
                self.status_label.setText("Arquivo carregado com sucesso")
                self.update_export_button_state()
                self.update_display()
            else:
                self.print_to_console(f"\n❌ {message}", "#F44336")
                self.file_label.setStyleSheet("color: #F44336;")
                self.status_label.setText("Erro ao carregar arquivo")
                self.update_export_button_state()
    
    def update_display(self):
        """Atualiza a visualização baseada na seleção atual."""
        view_type = self.view_combo.currentText()
        
        # Ignora separadores
        if view_type.startswith("---"):
            return
            
        self.status_label.setText(f"Exibindo: {view_type}")
        
        # Visualizações SKN
        if view_type == "Resumo Geral SKN":
            if self.parser.skin_data:
                self.display_overview()
            else:
                self.print_to_console("❌ Nenhum arquivo SKN carregado!", "#F44336")
        elif view_type == "Hierarquia de Ossos":
            if self.parser.skin_data:
                self.display_bone_hierarchy()
            else:
                self.print_to_console("❌ Nenhum arquivo SKN carregado!", "#F44336")
        elif view_type == "Soft Bones Detalhados":
            if self.parser.skin_data:
                self.display_soft_bones()
            else:
                self.print_to_console("❌ Nenhum arquivo SKN carregado!", "#F44336")
        elif view_type == "Análise Bone-Vertex":
            if self.parser.skin_data:
                self.display_bone_vertex_analysis()
            else:
                self.print_to_console("❌ Nenhum arquivo SKN carregado!", "#F44336")
        elif view_type == "Mapeamento de Vértices":
            if self.parser.skin_data:
                self.display_vertex_binding()
            else:
                self.print_to_console("❌ Nenhum arquivo SKN carregado!", "#F44336")
        elif view_type == "Resumo dos Vértices":
            if self.parser.skin_data:
                self.display_vertices_summary()
            else:
                self.print_to_console("❌ Nenhum arquivo SKN carregado!", "#F44336")
        
        # Visualizações de Animações
        elif view_type == "Resumo das Animações":
            if self.anims_parser.anims_data:
                self.display_animations_overview()
            else:
                self.print_to_console("❌ Nenhum arquivo ANIMS carregado!", "#F44336")
        elif view_type == "Lista de Animações":
            if self.anims_parser.anims_data:
                self.display_animations_list()
            else:
                self.print_to_console("❌ Nenhum arquivo ANIMS carregado!", "#F44336")
        elif view_type == "Detalhes dos Keyframes":
            if self.anims_parser.anims_data:
                self.display_keyframes_details()
            else:
                self.print_to_console("❌ Nenhum arquivo ANIMS carregado!", "#F44336")
        elif view_type == "Análise de Movimento":
            if self.anims_parser.anims_data:
                self.display_movement_analysis()
            else:
                self.print_to_console("❌ Nenhum arquivo ANIMS carregado!", "#F44336")
        
        # Visualizações Completas
        elif view_type == "Visualização Completa SKN":
            if self.parser.skin_data:
                self.display_all_skn()
            else:
                self.print_to_console("❌ Nenhum arquivo SKN carregado!", "#F44336")
        elif view_type == "Visualização Completa ANIMS":
            if self.anims_parser.anims_data:
                self.display_all_anims()
            else:
                self.print_to_console("❌ Nenhum arquivo ANIMS carregado!", "#F44336")
        elif view_type == "Visualização Completa TUDO":
            if self.parser.skin_data or self.anims_parser.anims_data:
                self.display_everything()
            else:
                self.print_to_console("❌ Nenhum arquivo carregado!", "#F44336")
    
    def display_animations_overview(self):
        """Exibe um resumo geral das animações."""
        self.print_separator("RESUMO GERAL DAS ANIMAÇÕES", "=", "#E91E63")
        
        data = self.anims_parser.anims_data
        animations = data.get('animations', [])
        
        self.print_to_console(f"🎬 Total de Animações: {data.get('anim_count', 0)}", "#E91E63")
        
        if animations:
            # Estatísticas gerais
            total_frames = sum(anim['num_frames'] for anim in animations)
            avg_frames = total_frames / len(animations) if animations else 0
            max_frames = max(anim['num_frames'] for anim in animations)
            min_frames = min(anim['num_frames'] for anim in animations)
            
            total_bones = sum(anim['num_bones'] for anim in animations)
            avg_bones = total_bones / len(animations) if animations else 0
            
            self.print_to_console(f"📊 Total de Frames: {total_frames}", "#F06292")
            self.print_to_console(f"📈 Frames por Animação: Média: {avg_frames:.1f}, Min: {min_frames}, Max: {max_frames}", "#F06292")
            self.print_to_console(f"🦴 Média de Ossos por Animação: {avg_bones:.1f}", "#F06292")
            
            # Informações do arquivo
            if 'filepath' in data:
                file_size = os.path.getsize(data['filepath'])
                self.print_to_console(f"📁 Tamanho do arquivo: {file_size:,} bytes", "#666")
    
    def display_animations_list(self):
        """Exibe lista detalhada das animações."""
        self.print_separator("LISTA DE ANIMAÇÕES", "=", "#9C27B0")
        
        animations = self.anims_parser.anims_data.get('animations', [])
        
        for anim in animations:
            self.print_to_console(f"\n🎬 Animação {anim['index']}: \"{anim['name']}\"", "#9C27B0")
            self.print_to_console(f"   📊 Frames: {anim['num_frames']}", "#BA68C8")
            self.print_to_console(f"   🦴 Ossos: {anim['num_bones']}", "#BA68C8")
            
            # Contar keyframes totais
            total_trans_keys = sum(len(bone['translation_keyframes']) for bone in anim['bones'])
            total_rot_keys = sum(len(bone['rotation_keyframes']) for bone in anim['bones'])
            
            self.print_to_console(f"   🔑 Keyframes de Translação: {total_trans_keys}", "#CE93D8")
            self.print_to_console(f"   🔄 Keyframes de Rotação: {total_rot_keys}", "#CE93D8")
            self.print_to_console(f"   📈 Total de Keyframes: {total_trans_keys + total_rot_keys}", "#E1BEE7")
    
    def display_keyframes_details(self):
        """Exibe detalhes dos keyframes."""
        self.print_separator("DETALHES DOS KEYFRAMES", "=", "#FF5722")
        
        animations = self.anims_parser.anims_data.get('animations', [])
        
        for anim in animations:
            self.print_to_console(f"\n🎬 Animação {anim['index']}: \"{anim['name']}\"", "#FF5722")
            
            # Mostrar detalhes dos primeiros ossos (limite para não sobrecarregar)
            bones_to_show = min(3, len(anim['bones']))
            for bone_idx in range(bones_to_show):
                bone = anim['bones'][bone_idx]
                self.print_to_console(f"\n   🦴 Osso {bone['bone_index']}:", "#FF7043")
                self.print_to_console(f"      📍 Offset de Translação: {bone['trans_offset']}", "#FF8A65")
                
                # Keyframes de translação
                if bone['translation_keyframes']:
                    self.print_to_console(f"      🔑 Keyframes de Translação ({len(bone['translation_keyframes'])}):", "#FFAB91")
                    for i, kf in enumerate(bone['translation_keyframes'][:5]):  # Mostrar apenas os primeiros 5
                        loc = kf['location']
                        self.print_to_console(f"         Frame {kf['frame']:3d}: ({loc[0]:7.3f}, {loc[1]:7.3f}, {loc[2]:7.3f})", "#FFCCBC")
                    if len(bone['translation_keyframes']) > 5:
                        self.print_to_console(f"         ... e mais {len(bone['translation_keyframes']) - 5} keyframes", "#FFCCBC")
                
                # Keyframes de rotação
                if bone['rotation_keyframes']:
                    self.print_to_console(f"      🔄 Keyframes de Rotação ({len(bone['rotation_keyframes'])}):", "#FFAB91")
                    for i, kf in enumerate(bone['rotation_keyframes'][:5]):  # Mostrar apenas os primeiros 5
                        rot = kf['rotation']
                        self.print_to_console(f"         Frame {kf['frame']:3d}: ({rot[0]:6.3f}, {rot[1]:6.3f}, {rot[2]:6.3f}, {rot[3]:6.3f})", "#FFCCBC")
                    if len(bone['rotation_keyframes']) > 5:
                        self.print_to_console(f"         ... e mais {len(bone['rotation_keyframes']) - 5} keyframes", "#FFCCBC")
            
            if len(anim['bones']) > bones_to_show:
                self.print_to_console(f"   ... e mais {len(anim['bones']) - bones_to_show} ossos", "#FFAB91")
    
    def display_movement_analysis(self):
        """Exibe análise de movimento das animações."""
        self.print_separator("ANÁLISE DE MOVIMENTO", "=", "#607D8B")
        
        animations = self.anims_parser.anims_data.get('animations', [])
        
        for anim in animations:
            self.print_to_console(f"\n🎬 Animação {anim['index']}: \"{anim['name']}\"", "#607D8B")
            
            # Analisar movimento por osso
            for bone in anim['bones'][:5]:  # Limitar para primeiros 5 ossos
                bone_idx = bone['bone_index']
                
                # Análise de translação
                trans_kf = bone['translation_keyframes']
                if len(trans_kf) > 1:
                    # Calcular distância total percorrida
                    total_distance = 0
                    for i in range(1, len(trans_kf)):
                        prev_loc = trans_kf[i-1]['location']
                        curr_loc = trans_kf[i]['location']
                        distance = math.sqrt(
                            (curr_loc[0] - prev_loc[0])**2 +
                            (curr_loc[1] - prev_loc[1])**2 +
                            (curr_loc[2] - prev_loc[2])**2
                        )
                        total_distance += distance
                    
                    self.print_to_console(f"   🦴 Osso {bone_idx}: Distância total percorrida: {total_distance:.3f}", "#78909C")
                
                # Análise de rotação
                rot_kf = bone['rotation_keyframes']
                if len(rot_kf) > 0:
                    self.print_to_console(f"   🔄 Osso {bone_idx}: {len(rot_kf)} keyframes de rotação", "#90A4AE")
    
    def display_all_skn(self):
        """Exibe todas as informações do arquivo SKN."""
        if not self.parser.skin_data:
            self.print_to_console("❌ Nenhum arquivo SKN carregado!", "#F44336")
            return
        
        self.display_overview()
        self.display_bone_hierarchy()
        self.display_soft_bones()
        self.display_bone_vertex_analysis()
        self.display_vertex_binding()
        self.display_vertices_summary()
        
        self.print_separator("ANÁLISE SKN COMPLETA", "=", "#4CAF50")
        self.print_to_console("✅ Visualização completa do arquivo SKN concluída!", "#4CAF50")
    
    def display_all_anims(self):
        """Exibe todas as informações do arquivo ANIMS."""
        if not self.anims_parser.anims_data:
            self.print_to_console("❌ Nenhum arquivo ANIMS carregado!", "#F44336")
            return
        
        self.display_animations_overview()
        self.display_animations_list()
        self.display_keyframes_details()
        self.display_movement_analysis()
        
        self.print_separator("ANÁLISE ANIMS COMPLETA", "=", "#E91E63")
        self.print_to_console("✅ Visualização completa do arquivo ANIMS concluída!", "#E91E63")
    
    def display_everything(self):
        """Exibe todas as informações de SKN e ANIMS."""
        if self.parser.skin_data:
            self.display_all_skn()
        
        if self.anims_parser.anims_data:
            self.display_all_anims()
        
        self.print_separator("ANÁLISE COMPLETA FINALIZADA", "=", "#673AB7")
        self.print_to_console("🎯 Análise completa de todos os arquivos carregados!", "#673AB7")
    
    def print_separator(self, title: str, char: str = "=", color: str = "#FFC107"):
        """Imprime um separador com título."""
        width = 60
        title_len = len(title)
        padding = (width - title_len - 2) // 2
        separator = f"{char * padding} {title} {char * padding}"
        self.print_to_console(f"\n{separator}", color)
    
    def display_overview(self):
        """Exibe um resumo geral do arquivo SKN."""
        self.print_separator("RESUMO GERAL DO ARQUIVO SKN", "=", "#2196F3")
        
        data = self.parser.skin_data
        self.print_to_console(f"📊 Contagem de Ossos (Hard Bones): {data['bone_count']}", "#4CAF50")
        self.print_to_console(f"🔧 Contagem de Soft Bones: {data['soft_bone_count']}", "#FF9800")
        self.print_to_console(f"🔗 Total de Índices de Vértices: {data['indice_count']}", "#9C27B0")
        self.print_to_console(f"📍 Total de Vértices: {data['vert_count']}", "#E91E63")
        
        # Estatísticas adicionais
        total_hard_verts = sum(bone['n_hards'] for bone in data['bones'])
        total_soft_verts = sum(soft['n_softs'] for soft in data['soft_bones'])
        
        self.print_to_console(f"🎯 Vértices ligados a Hard Bones: {total_hard_verts}", "#607D8B")
        self.print_to_console(f"🎯 Vértices ligados a Soft Bones: {total_soft_verts}", "#795548")
        
        # Informações do arquivo
        if 'filepath' in data:
            file_size = os.path.getsize(data['filepath'])
            self.print_to_console(f"📁 Tamanho do arquivo: {file_size:,} bytes", "#666")
    
    def display_bone_hierarchy(self):
        """Exibe a hierarquia dos ossos."""
        self.print_separator("HIERARQUIA DOS OSSOS (HARD BONES)", "=", "#4CAF50")
        
        for bone in self.parser.skin_data['bones']:
            parent_info = f"Pai: {bone['parent']}" if bone['parent'] >= 0 else "Pai: [ROOT]"
            bone_info = (f"🦴 Osso {bone['index']:3d} | {parent_info:12s} | "
                        f"Hard Verts: {bone['n_hards']:3d} | Soft Types: {bone['n_soft_types']:2d} | "
                        f"Hard Index: {bone['hard_i']:4d} | Soft Type Index: {bone['soft_type_i']:3d}")
            
            color = "#4CAF50" if bone['parent'] < 0 else "#81C784"
            self.print_to_console(bone_info, color)
    
    def display_soft_bones(self):
        """Exibe informações dos soft bones."""
        self.print_separator("SOFT BONES DETALHADOS", "=", "#FF9800")
        
        for soft in self.parser.skin_data['soft_bones']:
            self.print_to_console(f"\n🔧 Soft Bone {soft['index']}", "#FF9800")
            self.print_to_console(f"   ⚖️  Peso: {soft['weight']:.6f}", "#FFC107")
            self.print_to_console(f"   📍 Vértices Soft: {soft['n_softs']}", "#FFB74D")
            self.print_to_console(f"   🔗 Índice Inicial: {soft['soft_i']}", "#FFCC02")
            self.print_to_console(f"   🔢 Matriz de Transformação:", "#FF8A65")
            
            for i, row in enumerate(soft['matrix']):
                matrix_row = f"      Row {i}: [{row[0]:8.4f}, {row[1]:8.4f}, {row[2]:8.4f}]"
                self.print_to_console(matrix_row, "#BCAAA4")
    
    def display_bone_vertex_analysis(self):
        """Exibe análise detalhada de como vértices são atribuídos aos ossos."""
        self.print_separator("ANÁLISE BONE-VERTEX", "=", "#9C27B0")
        
        data = self.parser.skin_data
        indices = data['indices']
        bones = data['bones']
        soft_bones = data['soft_bones']
        
        self.print_to_console("🦴 Distribuição de vértices por Hard Bone:", "#4CAF50")
        
        for bone in bones:
            if bone['n_hards'] > 0:
                start_idx = bone['hard_i']
                end_idx = start_idx + bone['n_hards']
                vertex_indices = indices[start_idx:end_idx]
                
                vertex_preview = str(vertex_indices[:5])
                if len(vertex_indices) > 5:
                    vertex_preview = vertex_preview[:-1] + ", ...]"
                
                bone_info = (f"   Bone {bone['index']:2d}: {bone['n_hards']:3d} vértices "
                           f"(índices {start_idx}-{end_idx-1}) -> vértices {vertex_preview}")
                self.print_to_console(bone_info, "#81C784")
        
        self.print_to_console("\n🔧 Distribuição de vértices por Soft Bone:", "#FF9800")
        for soft in soft_bones:
            if soft['n_softs'] > 0:
                start_idx = soft['soft_i']
                end_idx = start_idx + soft['n_softs']
                vertex_indices = indices[start_idx:end_idx]
                
                vertex_preview = str(vertex_indices[:5])
                if len(vertex_indices) > 5:
                    vertex_preview = vertex_preview[:-1] + ", ...]"
                
                soft_info = (f"   Soft {soft['index']:2d}: {soft['n_softs']:3d} vértices "
                           f"(peso: {soft['weight']:.3f}) "
                           f"(índices {start_idx}-{end_idx-1}) -> vértices {vertex_preview}")
                self.print_to_console(soft_info, "#FFB74D")
    
    def display_vertex_binding(self):
        """Exibe informações de binding dos vértices."""
        self.print_separator("MAPEAMENTO DE VÉRTICES", "=", "#E91E63")
        
        indices = self.parser.skin_data['indices']
        self.print_to_console(f"📋 Lista de Índices de Vértices ({len(indices)} total):", "#E91E63")
        
        # Exibir índices em grupos de 16 para melhor legibilidade
        for i in range(0, len(indices), 16):
            chunk = indices[i:i+16]
            indices_str = ', '.join(f"{idx:3d}" for idx in chunk)
            range_info = f"   {i:4d}-{min(i+15, len(indices)-1):4d}: [{indices_str}]"
            self.print_to_console(range_info, "#F48FB1")
        
        # Análise dos mapeamentos
        self.print_to_console(f"\n📊 Análise dos Mapeamentos:", "#AD1457")
        self.print_to_console(f"   🔢 Menor índice de vértice: {min(indices)}", "#C2185B")
        self.print_to_console(f"   🔢 Maior índice de vértice: {max(indices)}", "#C2185B")
        self.print_to_console(f"   🔢 Índices únicos: {len(set(indices))}", "#C2185B")
        self.print_to_console(f"   🔢 Duplicatas: {len(indices) - len(set(indices))}", "#C2185B")
    
    def display_vertices_summary(self):
        """Exibe um resumo dos vértices."""
        self.print_separator("RESUMO DOS VÉRTICES DE SKIN", "=", "#673AB7")
        
        vertices = self.parser.skin_data['vertices']
        self.print_to_console(f"📍 Total de vértices de skin: {len(vertices)}", "#9C27B0")
        
        if vertices:
            # Calcular bounding box
            min_x = min(v[0] for v in vertices)
            max_x = max(v[0] for v in vertices)
            min_y = min(v[1] for v in vertices)
            max_y = max(v[1] for v in vertices)
            min_z = min(v[2] for v in vertices)
            max_z = max(v[2] for v in vertices)
            
            self.print_to_console(f"\n📦 Bounding Box:", "#7B1FA2")
            self.print_to_console(f"   X: {min_x:8.4f} até {max_x:8.4f} (tamanho: {max_x - min_x:8.4f})", "#8E24AA")
            self.print_to_console(f"   Y: {min_y:8.4f} até {max_y:8.4f} (tamanho: {max_y - min_y:8.4f})", "#8E24AA")
            self.print_to_console(f"   Z: {min_z:8.4f} até {max_z:8.4f} (tamanho: {max_z - min_z:8.4f})", "#8E24AA")
            
            # Exibir alguns vértices de exemplo
            self.print_to_console(f"\n📋 Primeiros 10 vértices:", "#9C27B0")
            for i in range(min(10, len(vertices))):
                v = vertices[i]
                vertex_info = f"   Vértice {i:3d}: ({v[0]:8.4f}, {v[1]:8.4f}, {v[2]:8.4f})"
                self.print_to_console(vertex_info, "#BA68C8")
            
            if len(vertices) > 10:
                self.print_to_console(f"   ... e mais {len(vertices) - 10} vértices", "#CE93D8")
    
    def display_all(self):
        """Exibe todas as informações do arquivo SKN."""
        self.display_overview()
        self.display_bone_hierarchy()
        self.display_soft_bones()
        self.display_bone_vertex_analysis()
        self.display_vertex_binding()
        self.display_vertices_summary()
        
        self.print_separator("ANÁLISE COMPLETA FINALIZADA", "=", "#4CAF50")
        self.print_to_console("✅ Visualização completa do arquivo SKN concluída!", "#4CAF50")

class MainWindow(QMainWindow):
    """Janela principal da aplicação."""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        """Configura a interface da janela principal."""
        self.setWindowTitle("Shadow Man SKN & Animation Visualizer")
        self.setGeometry(100, 100, 1200, 800)
        
        # Widget central
        self.visualizer = SKNVisualizerWidget()
        self.setCentralWidget(self.visualizer)
        
        # Menu bar
        self.create_menu_bar()
        
        # Status bar
        self.statusBar().showMessage("Pronto")
        
    def create_menu_bar(self):
        """Cria a barra de menu."""
        menubar = self.menuBar()
        
        # Menu Arquivo
        file_menu = menubar.addMenu("Arquivo")
        
        open_action = QAction("Abrir SKN...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.visualizer.load_file)
        file_menu.addAction(open_action)
        
        open_anims_action = QAction("Abrir Animações...", self)
        open_anims_action.setShortcut("Ctrl+Shift+O")
        open_anims_action.triggered.connect(self.visualizer.load_anims_file)
        file_menu.addAction(open_anims_action)

        open_mesh_action = QAction("Abrir Mesh...", self)
        open_mesh_action.setShortcut("Ctrl+M")
        open_mesh_action.triggered.connect(self.visualizer.load_mesh_file)
        file_menu.addAction(open_mesh_action)
        
        export_action = QAction("Exportar JSON...", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self.visualizer.export_json)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Sair", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Menu Visualização
        view_menu = menubar.addMenu("Visualização")
        
        clear_action = QAction("Limpar Console", self)
        clear_action.setShortcut("Ctrl+L")
        clear_action.triggered.connect(self.visualizer.clear_console)
        view_menu.addAction(clear_action)
        
        # Menu Ajuda
        help_menu = menubar.addMenu("Ajuda")
        
        about_action = QAction("Sobre", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def show_about(self):
        """Mostra diálogo sobre a aplicação."""
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.about(
            self,
            "Sobre",
            "Shadow Man SKN & Animation Visualizer\n\n"
            "Visualizador para arquivos .skn e .anims do Shadow Man\n"
            "• Interpreta dados de skinning e hierarquia de ossos\n"
            "• Analisa animações e keyframes\n"
            "• Exporta dados completos para JSON\n\n"
            "Desenvolvido com PySide6"
        )

def main():
    """Função principal da aplicação."""
    # Verificar argumentos da linha de comando
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("🧪 Modo de teste - verificando dependências...")
        print("✅ PySide6 importado com sucesso!")
        print("✅ Todas as dependências estão funcionais!")
        return
    
    app = QApplication(sys.argv)
    
    # Configurar estilo escuro
    app.setStyle('Fusion')
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, QColor(0, 0, 0))
    palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
    palette.setColor(QPalette.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)
    
    try:
        window = MainWindow()
        window.show()
        
        print("🎮 Shadow Man SKN & Animation Visualizer iniciado com sucesso!")
        print("Interface gráfica aberta.")
        print("💡 Dicas:")
        print("  - Ctrl+O: Carregar arquivo SKN")
        print("  - Ctrl+Shift+O: Carregar arquivo ANIMS")
        print("  - Ctrl+E: Exportar dados para JSON")
        print("  - Ctrl+L: Limpar console")
        
        sys.exit(app.exec())
    
    except Exception as e:
        print(f"❌ Erro ao iniciar a aplicação: {e}")
        print("Tente executar: python skn.py --test")
        sys.exit(1)

if __name__ == "__main__":
    main()