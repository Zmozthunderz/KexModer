#!/usr/bin/env python3
"""
KEX GUI - Shadow Man Model Viewer Interface
Interface gráfica moderna para visualização de modelos do Shadow Man

CARACTERÍSTICAS:
    - Interface moderna com PySide6
    - Visualizador OpenGL com suporte completo a animação/skinning
    - Sistema híbrido adaptativo (completo/básico/estático)
    - Controles intuitivos de animação
    - Debug detalhado do sistema
    - Suporte a texturas e UVs corretos

DEPENDÊNCIAS:
    - PySide6 (Interface)
    - OpenGL (Renderização)
    - PIL (Texturas)
    - numpy (Matemática)
    - kex_core (Interpretadores)

USO:
    python kexgui.py
"""

import sys
import os
import math
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

# Verificar dependências críticas
try:
    import numpy as np
    from PIL import Image
except ImportError as e:
    print(f"❌ Dependência faltando: {e}")
    print("Execute: pip install numpy pillow")
    sys.exit(1)

# Verificar PySide6
try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QTextEdit, QFileDialog, QLabel, QGroupBox,
        QComboBox, QCheckBox, QSlider, QSpinBox, QListWidget,
        QSplitter, QTabWidget, QSizePolicy
    )
    from PySide6.QtOpenGLWidgets import QOpenGLWidget
    from PySide6.QtCore import Qt, QTimer, QElapsedTimer
    from PySide6.QtGui import QFont, QPalette, QColor, QAction
except ImportError as e:
    print(f"❌ PySide6 não encontrado: {e}")
    print("Execute: pip install PySide6")
    sys.exit(1)

# Verificar OpenGL
try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
except ImportError as e:
    print(f"❌ OpenGL não encontrado: {e}")
    print("Execute: pip install PyOpenGL")
    sys.exit(1)

# Importar núcleo KEX
try:
    from kex_core import KEXCore, Model3DInterpreter, MeshInterpreter, SkinInterpreter, AnimationInterpreter, AnimationSystem
except ImportError as e:
    print(f"❌ KEX Core não encontrado: {e}")
    print("Certifique-se que kex_core.py está no mesmo diretório")
    sys.exit(1)


class ModelViewer(QOpenGLWidget):
    """
    🎮 VISUALIZADOR MODERNO OPENGL
    Widget OpenGL para renderização de modelos 3D com sistema completo de animação
    """
    
    def __init__(self):
        super().__init__()
        
        # Núcleo KEX
        self.kex_core = KEXCore()
        
        # Estados de renderização
        self.model_data = None
        self.skin_data = None
        self.animations = None
        self.textures = {}
        self.texture_gl_ids = {}
        
        # Controles de visualização
        self.rot_x = 0
        self.rot_y = 0
        self.zoom = -5
        self.show_wireframe = False
        self.show_texcoords = False
        self.show_textures = True
        self.show_bones = False
        
        # ✅ ANIMAÇÃO MODERNA
        self.current_animation = 0
        self.current_frame = 0.0
        self.animation_playing = False
        self.animation_speed = 1.0
        
        # Dados de vértices
        self.animated_vertices = None
        self.original_vertices = None
        
        # ✅ TIMER DE ANIMAÇÃO
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)
        self.elapsed = QElapsedTimer()
        
        # Mouse
        self.last_mouse_pos = None
        self.mouse_pressed = False
        
        # Status
        self.system_mode = "ESTÁTICO"
        
    def load_shadowman_model(self, mesh_path):
        """✅ CARREGAMENTO COMPLETO VIA KEX CORE"""
        print(f"🎮 Carregando modelo via KEX Core: {mesh_path}")
        
        success = self.kex_core.load_complete_model(mesh_path)
        if not success:
            print("❌ Falha no carregamento via KEX Core")
            return False
        
        # Extrair dados do KEX Core
        self.model_data = self.kex_core.current_mesh
        self.skin_data = self.kex_core.current_skin
        self.animations = self.kex_core.current_animations
        self.textures = self.kex_core.textures
        
        # ✅ PREPARAR VÉRTICES
        if self.model_data:
            self.original_vertices = self.model_data["verts"]["loc"].copy()
            self.animated_vertices = self.original_vertices.copy()
            print(f"✅ Vértices preparados: {len(self.original_vertices)}")
        
        # ✅ DETERMINAR MODO DE OPERAÇÃO
        status = self.kex_core.get_system_status()
        if status['has_mesh'] and status['has_skin'] and status['has_animations']:
            self.system_mode = "COMPLETO"
        elif status['has_mesh'] and status['has_animations']:
            self.system_mode = "BÁSICO"
        else:
            self.system_mode = "ESTÁTICO"
        
        print(f"🎯 Modo determinado: {self.system_mode}")
        
        # Carregar texturas no OpenGL
        self.load_opengl_textures()
        
        # Centralizar modelo
        self.center_model()
        
        self.update()
        return True
    
    def load_opengl_textures(self):
        """Carregar texturas no contexto OpenGL"""
        self.texture_gl_ids = {}
        
        for tex_index, tex_path in self.textures.items():
            try:
                img = Image.open(tex_path)
                img = img.convert('RGBA')
                img_data = np.array(img)
                
                tex_id = glGenTextures(1)
                glBindTexture(GL_TEXTURE_2D, tex_id)
                
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width, img.height, 
                           0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
                
                self.texture_gl_ids[tex_index] = tex_id
                print(f"✅ Textura GL carregada: {tex_index} -> {os.path.basename(tex_path)}")
                
            except Exception as e:
                print(f"❌ Erro ao carregar textura GL {tex_index}: {e}")
    
    def center_model(self):
        """Centralizar modelo na tela"""
        if not self.original_vertices:
            return
        
        try:
            vertices_array = np.array(self.original_vertices)
            min_coords = np.min(vertices_array, axis=0)
            max_coords = np.max(vertices_array, axis=0)
            center = (min_coords + max_coords) / 2
            
            # Centralizar vértices
            for i, vertex in enumerate(self.original_vertices):
                self.original_vertices[i] = tuple(np.array(vertex) - center)
            
            self.animated_vertices = self.original_vertices.copy()
            
            # Ajustar zoom
            size = np.max(max_coords - min_coords)
            self.zoom = -size * 2
            
            print(f"✅ Modelo centralizado, zoom: {self.zoom}")
            
        except Exception as e:
            print(f"❌ Erro ao centralizar: {e}")
    
    def update_animation(self):
        """Atualizar frame de animação"""
        if not self.animation_playing or not self.animations:
            return
        
        if self.current_animation >= len(self.animations):
            return
        
        anim = self.animations[self.current_animation]
        max_frames = anim["numFrames"]
        
        if max_frames <= 0:
            return
        
        # Incrementar frame usando tempo real decorrido
        delta = self.elapsed.elapsed() / 1000.0
        self.elapsed.restart()
        self.current_frame += self.animation_speed * delta * 60.0
        if self.current_frame >= max_frames:
            self.current_frame = 0.0
        
        # Aplicar animação
        try:
            self.apply_animation()
            self.update()  # Redesenhar
        except Exception as e:
            print(f"❌ Erro na animação: {e}")
            self.timer.stop()
    
    def apply_animation(self):
        """Aplicar animação baseada no modo do sistema"""
        if not self.animations or self.current_animation >= len(self.animations):
            return
        
        animation = self.animations[self.current_animation]
        
        if self.system_mode == "COMPLETO" and self.skin_data:
            # Modo completo com skinning
            self.kex_core.anim_system.update_bone_matrices(animation, self.current_frame)
            self.animated_vertices = self.kex_core.anim_system.apply_skinning(
                self.original_vertices, self.skin_data
            )
        elif self.system_mode == "BÁSICO":
            # Modo básico - animação simples
            self.apply_basic_animation(animation)
        else:
            # Modo estático
            self.animated_vertices = self.original_vertices.copy()
    
    def apply_basic_animation(self, animation):
        """Aplicar animação básica sem skinning completo"""
        if not animation.get('bones') or not self.original_vertices:
            return
        
        # Usar primeiro bone para transformação global simples
        anim_bone = animation['bones'][0]
        
        # Interpolação simples
        current_frame = int(self.current_frame)
        next_frame = current_frame + 1
        t = self.current_frame - current_frame
        
        # Obter posição interpolada
        trans_data = anim_bone.get('trans', {})
        offset = anim_bone.get('transOffset', (0, 0, 0))
        
        pos_current = trans_data.get(current_frame, offset)
        pos_next = trans_data.get(next_frame, pos_current)
        
        if isinstance(pos_current, (list, tuple)) and isinstance(pos_next, (list, tuple)) and len(pos_current) >= 3 and len(pos_next) >= 3:
            pos_interp = tuple(
                pos_current[i] + (pos_next[i] - pos_current[i]) * t 
                for i in range(3)
            )
        else:
            pos_interp = offset if isinstance(offset, (list, tuple)) else (0, 0, 0)
        
        # Aplicar transformação global simples
        self.animated_vertices = []
        for vertex in self.original_vertices:
            new_vertex = (
                vertex[0] + pos_interp[0] * 0.02,  # Fator pequeno
                vertex[1] + pos_interp[1] * 0.02,
                vertex[2] + pos_interp[2] * 0.02
            )
            self.animated_vertices.append(new_vertex)
    
    # =========================================================================
    # CONTROLES DE ANIMAÇÃO
    # =========================================================================
    
    def get_animation_names(self):
        """Lista de nomes das animações"""
        if not self.animations:
            return []
        return [f"{i:03d} - {anim['name']}" for i, anim in enumerate(self.animations)]
    
    def set_current_animation(self, index):
        """Definir animação atual"""
        if not self.animations or not (0 <= index < len(self.animations)):
            return False
        
        self.current_animation = index
        self.current_frame = 0.0
        
        anim = self.animations[index]
        print(f"✅ Animação selecionada: '{anim['name']}'")
        
        # Aplicar frame inicial
        if anim['numFrames'] > 0:
            try:
                self.apply_animation()
                self.update()
            except Exception as e:
                print(f"❌ Erro ao aplicar frame inicial: {e}")
        
        return True
    
    def play_animation(self, play=True):
        """Iniciar/parar animação"""
        if not self.animations:
            return False
        
        self.animation_playing = play
        
        if play:
            anim = self.animations[self.current_animation]
            print(f"▶️ Iniciando animação '{anim['name']}'")
            self.elapsed.start()
            # Intervalo ~60 FPS para fluidez
            self.timer.start(16)
        else:
            print("⏹️ Parando animação")
            self.timer.stop()
        
        return True
    
    def set_animation_frame(self, frame):
        """Definir frame atual"""
        if self.animations and self.current_animation < len(self.animations):
            max_frames = self.animations[self.current_animation]["numFrames"]
            self.current_frame = max(0.0, min(float(frame), max_frames - 1))
            self.apply_animation()
            self.update()
    
    def get_debug_info(self):
        """Informações de debug do visualizador"""
        info = []
        info.append("🎮 KEXGUI MODEL VIEWER DEBUG:")
        info.append(f"Modo: {self.system_mode}")
        info.append(f"Animação: {self.current_animation}")
        info.append(f"Frame: {self.current_frame:.2f}")
        info.append(f"Playing: {self.animation_playing}")
        info.append(f"Timer ativo: {self.timer.isActive()}")
        
        if self.model_data:
            info.append(f"Vértices mesh: {len(self.model_data['verts']['loc'])}")
            info.append(f"Faces: {len(self.model_data['faces'])}")
        
        info.append(f"Vértices originais: {len(self.original_vertices) if self.original_vertices else 0}")
        info.append(f"Vértices animados: {len(self.animated_vertices) if self.animated_vertices else 0}")
        info.append(f"Texturas GL: {len(self.texture_gl_ids)}")
        
        # Debug do KEX Core
        info.append("\n" + self.kex_core.get_debug_info())
        
        return "\n".join(info)
    
    # =========================================================================
    # OPENGL RENDERING
    # =========================================================================
    
    def initializeGL(self):
        """Inicializar OpenGL"""
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.2, 0.2, 0.3, 1.0)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        
        # Iluminação
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        glLightfv(GL_LIGHT0, GL_POSITION, [1.0, 1.0, 1.0, 0.0])
        
        glMaterialfv(GL_FRONT, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
        glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        
        glEnable(GL_TEXTURE_2D)
    
    def resizeGL(self, w, h):
        """Redimensionar viewport"""
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, w / h if h != 0 else 1, 0.1, 1000)
        glMatrixMode(GL_MODELVIEW)
    
    def paintGL(self):
        """Renderizar cena"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        glTranslatef(0, 0, self.zoom)
        glRotatef(self.rot_x, 1, 0, 0)
        glRotatef(self.rot_y, 0, 1, 0)
        
        if self.model_data:
            self.render_model()
        else:
            self.render_placeholder()
    
    def render_model(self):
        """Renderizar modelo 3D"""
        # Escolher vértices corretos
        if self.animated_vertices and self.system_mode in ["COMPLETO", "BÁSICO"]:
            vertices = self.animated_vertices
        else:
            vertices = self.model_data["verts"]["loc"]
        
        normals = self.model_data["verts"]["normals"]
        faces = self.model_data["faces"]
        
        # Configurar renderização
        if self.show_wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glDisable(GL_LIGHTING)
            glDisable(GL_TEXTURE_2D)
            glColor3f(1.0, 1.0, 1.0)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glEnable(GL_LIGHTING)
            if self.show_textures:
                glEnable(GL_TEXTURE_2D)
            else:
                glDisable(GL_TEXTURE_2D)
        
        # Renderizar por textura
        current_texture = -1
        
        for face in faces:
            # Texturas
            if face["texIndex"] != current_texture and self.show_textures and not self.show_wireframe:
                current_texture = face["texIndex"]
                if current_texture in self.texture_gl_ids:
                    glBindTexture(GL_TEXTURE_2D, self.texture_gl_ids[current_texture])
                else:
                    glBindTexture(GL_TEXTURE_2D, 0)
            
            glBegin(GL_TRIANGLES)
            
            for i, vert_idx in enumerate(face["indices"]):
                if vert_idx < len(vertices):
                    vertex = vertices[vert_idx]
                    normal = normals[vert_idx] if vert_idx < len(normals) else (0, 1, 0)
                    
                    # Cores
                    if i < len(face["loopColors"]) and not self.show_wireframe:
                        color = face["loopColors"][i]
                        glColor4f(color[0], color[1], color[2], color[3])
                    elif self.show_texcoords and i < len(face["loopUV"]):
                        uv = face["loopUV"][i]
                        glColor3f(uv[0], uv[1], 0.5)
                    elif not self.show_wireframe:
                        glColor3f(0.8, 0.8, 0.8)
                    
                    # UVs
                    if i < len(face["loopUV"]) and self.show_textures and not self.show_wireframe:
                        uv = face["loopUV"][i]
                        glTexCoord2f(uv[0], uv[1])
                    
                    glNormal3f(normal[0], normal[1], normal[2])
                    glVertex3f(vertex[0], vertex[1], vertex[2])
            
            glEnd()
        
        # Renderizar skeleton se habilitado
        if self.show_bones and self.skin_data and self.system_mode == "COMPLETO":
            self.render_skeleton()
        
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glBindTexture(GL_TEXTURE_2D, 0)
    
    def render_skeleton(self):
        """Renderizar skeleton dos bones"""
        if not hasattr(self.kex_core.anim_system, 'final_bone_matrices'):
            return
        
        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        glColor3f(1.0, 0.0, 0.0)  # Vermelho para bones
        
        glBegin(GL_LINES)
        
        # Renderizar hierarquia de bones
        if hasattr(self.kex_core.anim_system, 'bones') and self.kex_core.anim_system.bones:
            for bone_idx, bone in enumerate(self.kex_core.anim_system.bones):
                parent_idx = bone.get('parent', -1)
                
                if (parent_idx >= 0 and 
                    parent_idx < len(self.kex_core.anim_system.final_bone_matrices) and
                    bone_idx < len(self.kex_core.anim_system.final_bone_matrices)):
                    
                    # Posições dos bones
                    bone_matrix = self.kex_core.anim_system.final_bone_matrices[bone_idx]
                    parent_matrix = self.kex_core.anim_system.final_bone_matrices[parent_idx]
                    
                    # Verificar se as matrizes são numpy arrays ou listas
                    if hasattr(bone_matrix, 'shape'):  # numpy array
                        bone_pos = (bone_matrix[0, 3], bone_matrix[1, 3], bone_matrix[2, 3])
                        parent_pos = (parent_matrix[0, 3], parent_matrix[1, 3], parent_matrix[2, 3])
                    else:  # lista de listas
                        bone_pos = (bone_matrix[0][3], bone_matrix[1][3], bone_matrix[2][3])
                        parent_pos = (parent_matrix[0][3], parent_matrix[1][3], parent_matrix[2][3])
                    
                    # Linha do pai para o filho
                    glVertex3f(parent_pos[0], parent_pos[1], parent_pos[2])
                    glVertex3f(bone_pos[0], bone_pos[1], bone_pos[2])
        
        glEnd()
        
        # Soft bones em azul
        glColor3f(0.0, 0.0, 1.0)
        glPointSize(5.0)
        glBegin(GL_POINTS)
        
        if hasattr(self.kex_core.anim_system, 'soft_bones') and self.kex_core.anim_system.soft_bones:
            soft_start = len(self.kex_core.anim_system.bones) if self.kex_core.anim_system.bones else 0
            for i in range(len(self.kex_core.anim_system.soft_bones)):
                soft_idx = soft_start + i
                if soft_idx < len(self.kex_core.anim_system.final_bone_matrices):
                    soft_matrix = self.kex_core.anim_system.final_bone_matrices[soft_idx]
                    
                    # Verificar se a matriz é numpy array ou lista
                    if hasattr(soft_matrix, 'shape'):  # numpy array
                        soft_pos = (soft_matrix[0, 3], soft_matrix[1, 3], soft_matrix[2, 3])
                    else:  # lista de listas
                        try:
                            soft_pos = (soft_matrix[0][3], soft_matrix[1][3], soft_matrix[2][3])
                        except (IndexError, TypeError):
                            # Se a matriz não tem a estrutura esperada, usar posição padrão
                            soft_pos = (0, i * 0.1, 0)
                    
                    glVertex3f(soft_pos[0], soft_pos[1], soft_pos[2])
        
        glEnd()
        glPointSize(1.0)
    
    def render_placeholder(self):
        """Placeholder quando não há modelo"""
        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        glColor3f(0.5, 0.5, 0.5)
        
        # Cubo simples
        glBegin(GL_QUADS)
        # Frente
        glVertex3f(-1, -1,  1)
        glVertex3f( 1, -1,  1)
        glVertex3f( 1,  1,  1)
        glVertex3f(-1,  1,  1)
        # Trás
        glVertex3f(-1, -1, -1)
        glVertex3f(-1,  1, -1)
        glVertex3f( 1,  1, -1)
        glVertex3f( 1, -1, -1)
        glEnd()
    
    # =========================================================================
    # CONTROLES DE MOUSE E TECLADO
    # =========================================================================
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.mouse_pressed = True
            self.last_mouse_pos = event.position().toPoint()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.mouse_pressed = False
    
    def mouseMoveEvent(self, event):
        if self.mouse_pressed and self.last_mouse_pos:
            current_pos = event.position().toPoint()
            dx = current_pos.x() - self.last_mouse_pos.x()
            dy = current_pos.y() - self.last_mouse_pos.y()
            
            self.rot_y += dx * 0.5
            self.rot_x += dy * 0.5
            self.rot_x = max(-90, min(90, self.rot_x))
            
            self.last_mouse_pos = current_pos
            self.update()
    
    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        zoom_factor = 0.1
        if delta > 0:
            self.zoom += zoom_factor
        else:
            self.zoom -= zoom_factor
        self.zoom = max(-50, min(-1, self.zoom))
        self.update()
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            if self.animation_playing:
                self.play_animation(False)
            else:
                self.play_animation(True)
        elif event.key() == Qt.Key_R:
            self.rot_x = 0
            self.rot_y = 0
            self.zoom = -5
            self.update()
        elif event.key() == Qt.Key_B:
            self.show_bones = not self.show_bones
            self.update()


class SKNAnalyzerWidget(QWidget):
    """
    🔍 WIDGET DE ANÁLISE SKN/ANIMS
    Widget para análise detalhada de arquivos SKN e ANIMS
    (Baseado no código do skn_v2.py)
    """
    
    def __init__(self):
        super().__init__()
        self.kex_core = KEXCore()
        self.setup_ui()
    
    def setup_ui(self):
        """Configurar interface do analisador"""
        layout = QVBoxLayout(self)
        
        # Controles de carregamento
        load_layout = QHBoxLayout()
        
        self.load_skn_btn = QPushButton("📁 Carregar SKN")
        self.load_skn_btn.clicked.connect(self.load_skn_file)
        load_layout.addWidget(self.load_skn_btn)
        
        self.load_anims_btn = QPushButton("🎬 Carregar ANIMS")
        self.load_anims_btn.clicked.connect(self.load_anims_file)
        load_layout.addWidget(self.load_anims_btn)
        
        self.export_btn = QPushButton("💾 Exportar JSON")
        self.export_btn.clicked.connect(self.export_json)
        self.export_btn.setEnabled(False)
        load_layout.addWidget(self.export_btn)
        
        layout.addLayout(load_layout)
        
        # Status de arquivos
        status_layout = QHBoxLayout()
        self.skn_status = QLabel("SKN: Não carregado")
        self.anims_status = QLabel("ANIMS: Não carregado")
        status_layout.addWidget(self.skn_status)
        status_layout.addWidget(self.anims_status)
        layout.addLayout(status_layout)
        
        # Dropdown de visualização
        view_layout = QHBoxLayout()
        view_layout.addWidget(QLabel("Visualizar:"))
        
        self.view_combo = QComboBox()
        self.view_combo.addItems([
            "Resumo Geral SKN",
            "Hierarquia de Bones",
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
        view_layout.addWidget(self.view_combo)
        
        clear_btn = QPushButton("🗑️ Limpar")
        clear_btn.clicked.connect(self.clear_console)
        view_layout.addWidget(clear_btn)
        
        layout.addLayout(view_layout)
        
        # Console de saída
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
        layout.addWidget(self.console)
        
        # Mensagem inicial
        self.print_to_console("🔍 KEX SKN/ANIMS Analyzer")
        self.print_to_console("Carregue arquivos SKN e/ou ANIMS para análise detalhada")
    
    def print_to_console(self, text, color="#d4d4d4"):
        """Adicionar texto ao console"""
        self.console.append(f'<span style="color: {color};">{text}</span>')
    
    def load_skn_file(self):
        """Carregar arquivo SKN"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Selecionar Arquivo SKN", "", "Shadow Man Skin Files (*.skn);;All Files (*)"
        )
        
        if file_path:
            success = self.kex_core.skin.read_skn_file(file_path)
            if success:
                self.skn_status.setText(f"SKN: {os.path.basename(file_path)}")
                self.skn_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
                self.print_to_console(f"✅ SKN carregado: {file_path}", "#4CAF50")
                self.export_btn.setEnabled(True)
                self.update_display()
            else:
                self.print_to_console(f"❌ Erro ao carregar SKN: {file_path}", "#F44336")
    
    def load_anims_file(self):
        """Carregar arquivo ANIMS"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Selecionar Arquivo ANIMS", "", "Shadow Man Animation Files (*.anims);;All Files (*)"
        )
        
        if file_path:
            anims = self.kex_core.animation.load_animations(file_path)
            if anims:
                self.anims_status.setText(f"ANIMS: {os.path.basename(file_path)}")
                self.anims_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
                self.print_to_console(f"✅ ANIMS carregado: {file_path}", "#4CAF50")
                self.export_btn.setEnabled(True)
                self.update_display()
            else:
                self.print_to_console(f"❌ Erro ao carregar ANIMS: {file_path}", "#F44336")
    
    def export_json(self):
        """Exportar dados para JSON"""
        if not self.kex_core.current_skin and not self.kex_core.current_animations:
            self.print_to_console("❌ Nenhum arquivo carregado!", "#F44336")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Salvar dados como JSON", "shadowman_data.json", "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                # Preparar dados para exportação
                export_data = {
                    "metadata": {
                        "export_timestamp": datetime.now().isoformat(),
                        "exporter": "KEX GUI Analyzer",
                        "version": "1.0"
                    }
                }
                
                if self.kex_core.current_skin:
                    export_data["skn_data"] = self.kex_core.current_skin
                
                if self.kex_core.current_animations:
                    export_data["anims_data"] = self.kex_core.current_animations
                
                # Salvar JSON
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                
                file_size = os.path.getsize(file_path)
                self.print_to_console(f"✅ Dados exportados para: {file_path}", "#4CAF50")
                self.print_to_console(f"📁 Tamanho: {file_size:,} bytes", "#2196F3")
                
            except Exception as e:
                self.print_to_console(f"❌ Erro ao exportar: {e}", "#F44336")
    
    def update_display(self):
        """Atualizar visualização baseada na seleção"""
        view_type = self.view_combo.currentText()
        
        if view_type.startswith("---"):
            return
        
        # Implementar visualizações específicas baseadas no skn_v2.py
        if view_type == "Resumo Geral SKN":
            self.display_skn_overview()
        elif view_type == "Hierarquia de Bones":
            self.display_bone_hierarchy()
        elif view_type == "Resumo das Animações":
            self.display_animations_overview()
        elif view_type == "Visualização Completa TUDO":
            self.display_everything()
        # Adicionar mais visualizações conforme necessário
    
    def display_skn_overview(self):
        """Exibir resumo do SKN"""
        if not self.kex_core.current_skin:
            self.print_to_console("❌ Nenhum arquivo SKN carregado!", "#F44336")
            return
        
        self.print_separator("RESUMO GERAL DO ARQUIVO SKN")
        skin = self.kex_core.current_skin
        
        self.print_to_console(f"🦴 Bones: {skin.get('boneCount', 0)}", "#4CAF50")
        self.print_to_console(f"🔧 Soft Bones: {skin.get('softBoneCount', 0)}", "#FF9800")
        self.print_to_console(f"📍 Vértices: {skin.get('vertCount', 0)}", "#E91E63")
        self.print_to_console(f"🔗 Índices: {skin.get('indiceCount', 0)}", "#9C27B0")
    
    def display_bone_hierarchy(self):
        """Exibir hierarquia de bones"""
        if not self.kex_core.current_skin:
            self.print_to_console("❌ Nenhum arquivo SKN carregado!", "#F44336")
            return
        
        self.print_separator("HIERARQUIA DOS BONES")
        
        bones = self.kex_core.current_skin.get('bones', [])
        for bone in bones:
            parent_info = f"Pai: {bone['parent']}" if bone['parent'] >= 0 else "Pai: [ROOT]"
            bone_info = (f"🦴 Bone {bone.get('index', 0):3d} | {parent_info:12s} | "
                        f"Hard Verts: {bone.get('nHards', 0):3d}")
            
            color = "#4CAF50" if bone['parent'] < 0 else "#81C784"
            self.print_to_console(bone_info, color)
    
    def display_animations_overview(self):
        """Exibir resumo das animações"""
        if not self.kex_core.current_animations:
            self.print_to_console("❌ Nenhum arquivo ANIMS carregado!", "#F44336")
            return
        
        self.print_separator("RESUMO DAS ANIMAÇÕES")
        
        anims = self.kex_core.current_animations
        self.print_to_console(f"🎬 Total de Animações: {len(anims)}", "#E91E63")
        
        total_frames = sum(anim.get('numFrames', 0) for anim in anims)
        avg_frames = total_frames / len(anims) if anims else 0
        
        self.print_to_console(f"📊 Total de Frames: {total_frames}", "#F06292")
        self.print_to_console(f"📈 Frames por Animação (média): {avg_frames:.1f}", "#F06292")
    
    def display_everything(self):
        """Exibir tudo"""
        if self.kex_core.current_skin:
            self.display_skn_overview()
            self.display_bone_hierarchy()
        
        if self.kex_core.current_animations:
            self.display_animations_overview()
        
        self.print_separator("ANÁLISE COMPLETA FINALIZADA", "#4CAF50")
        self.print_to_console("✅ Visualização completa concluída!", "#4CAF50")
    
    def print_separator(self, title, color="#FFC107"):
        """Imprimir separador"""
        separator = f"\n{'='*60}\n{title}\n{'='*60}"
        self.print_to_console(separator, color)
    
    def clear_console(self):
        """Limpar console"""
        self.console.clear()
        self.print_to_console("Console limpo.", "#888")


class MainWindow(QMainWindow):
    """
    🖥️ JANELA PRINCIPAL KEX GUI
    Interface principal moderna para visualização Shadow Man
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🚀 KEX GUI - Shadow Man Advanced Viewer")
        self.setMinimumSize(1600, 1000)
        self.setup_modern_ui()
        self.create_menu_bar()
    
    def setup_modern_ui(self):
        """Configurar interface moderna"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal com tabs
        main_layout = QVBoxLayout(central_widget)
        
        # Header
        header = QLabel("🚀 KEX GUI - Shadow Man Advanced Model Viewer")
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: #2196F3; padding: 10px;")
        main_layout.addWidget(header)
        
        # Tabs principais
        self.tab_widget = QTabWidget()
        
        # Tab 1: Visualizador 3D
        self.viewer_tab = self.create_viewer_tab()
        self.tab_widget.addTab(self.viewer_tab, "🎮 Visualizador 3D")
        
        # Tab 2: Analisador SKN/ANIMS
        self.analyzer_tab = SKNAnalyzerWidget()
        self.tab_widget.addTab(self.analyzer_tab, "🔍 Analisador SKN/ANIMS")
        
        main_layout.addWidget(self.tab_widget)
        
        # Status bar
        self.statusBar().showMessage("KEX GUI pronto - Carregue um modelo para começar")
    
    def create_viewer_tab(self):
        """Criar tab do visualizador 3D"""
        tab_widget = QWidget()
        layout = QHBoxLayout(tab_widget)
        
        # Splitter principal
        splitter = QSplitter(Qt.Horizontal)
        
        # Visualizador OpenGL
        self.viewer = ModelViewer()
        splitter.addWidget(self.viewer)
        
        # Painel de controles
        controls_widget = QWidget()
        controls_widget.setMaximumWidth(400)
        controls_widget.setMinimumWidth(300)
        controls_layout = QVBoxLayout(controls_widget)
        
        # Botão de carregamento
        self.load_button = QPushButton("📁 Carregar Modelo (.msh)")
        self.load_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 12px;")
        self.load_button.clicked.connect(self.load_model)
        controls_layout.addWidget(self.load_button)
        
        # Status do sistema
        status_group = QGroupBox("📊 Status do Sistema")
        status_layout = QVBoxLayout(status_group)
        
        self.info_label = QLabel("Nenhum modelo carregado")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("font-family: monospace; font-size: 10px;")
        status_layout.addWidget(self.info_label)
        
        self.debug_button = QPushButton("🔍 Debug Completo")
        self.debug_button.setStyleSheet("background-color: #FF9800; color: white;")
        self.debug_button.clicked.connect(self.show_debug_info)
        status_layout.addWidget(self.debug_button)
        
        controls_layout.addWidget(status_group)
        
        # Controles de visualização
        view_group = QGroupBox("👁️ Visualização")
        view_layout = QVBoxLayout(view_group)
        
        self.wireframe_check = QCheckBox("🔗 Wireframe")
        self.wireframe_check.toggled.connect(self.toggle_wireframe)
        view_layout.addWidget(self.wireframe_check)
        
        self.textures_check = QCheckBox("🎨 Texturas")
        self.textures_check.setChecked(True)
        self.textures_check.toggled.connect(self.toggle_textures)
        view_layout.addWidget(self.textures_check)
        
        self.texcoords_check = QCheckBox("📐 UVs como cores")
        self.texcoords_check.toggled.connect(self.toggle_texcoords)
        view_layout.addWidget(self.texcoords_check)
        
        self.bones_check = QCheckBox("🦴 Mostrar Skeleton")
        self.bones_check.toggled.connect(self.toggle_bones)
        view_layout.addWidget(self.bones_check)
        
        controls_layout.addWidget(view_group)
        
        # Controles de animação
        anim_group = QGroupBox("🎬 Sistema de Animação")
        anim_layout = QVBoxLayout(anim_group)
        
        # Lista de animações
        anim_layout.addWidget(QLabel("📽️ Animações:"))
        self.anim_list = QListWidget()
        self.anim_list.setMinimumHeight(150)
        self.anim_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.anim_list.currentRowChanged.connect(self.select_animation)
        anim_layout.addWidget(self.anim_list)
        
        # Controles de reprodução
        play_layout = QHBoxLayout()
        self.play_button = QPushButton("▶️ Play")
        self.play_button.setStyleSheet("background-color: #4CAF50; color: white;")
        self.play_button.clicked.connect(self.start_animation)
        self.play_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.stop_button = QPushButton("⏹️ Stop")
        self.stop_button.setStyleSheet("background-color: #f44336; color: white;")
        self.stop_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.stop_button.clicked.connect(self.stop_animation)
        play_layout.addWidget(self.play_button)
        play_layout.addWidget(self.stop_button)
        anim_layout.addLayout(play_layout)
        
        # Controle de frame
        anim_layout.addWidget(QLabel("🎞️ Frame:"))
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.valueChanged.connect(self.set_frame)
        anim_layout.addWidget(self.frame_slider)
        
        self.frame_spin = QSpinBox()
        self.frame_spin.valueChanged.connect(self.set_frame)
        anim_layout.addWidget(self.frame_spin)
        
        # Velocidade
        anim_layout.addWidget(QLabel("⚡ Velocidade:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 50)
        self.speed_slider.setValue(10)
        self.speed_slider.valueChanged.connect(self.set_speed)
        anim_layout.addWidget(self.speed_slider)
        
        controls_layout.addWidget(anim_group)
        
        # Lista de texturas
        tex_group = QGroupBox("🎨 Texturas")
        tex_layout = QVBoxLayout(tex_group)
        
        self.texture_list = QListWidget()
        self.texture_list.setMaximumHeight(100)
        tex_layout.addWidget(self.texture_list)
        
        controls_layout.addWidget(tex_group)
        
        # Instruções
        instructions = QLabel("""
🎮 CONTROLES KEX:
• 🖱️ Mouse: Rotacionar modelo
• 🖱️ Scroll: Zoom in/out
• ⌨️ Espaço: Play/Pause animação
• ⌨️ R: Reset câmera
• ⌨️ B: Toggle skeleton

🚀 MODOS SUPORTADOS:
• COMPLETO: .msh + .skn + .anims
• BÁSICO: .msh + .anims
• ESTÁTICO: apenas .msh
        """)
        instructions.setStyleSheet("font-size: 10px; color: #666; background: #f5f5f5; padding: 10px; border-radius: 5px;")
        controls_layout.addWidget(instructions)
        
        controls_layout.addStretch()
        splitter.addWidget(controls_widget)
        
        # Proporções
        splitter.setSizes([1200, 400])
        
        layout.addWidget(splitter)
        return tab_widget
    
    def create_menu_bar(self):
        """Criar menu bar"""
        menubar = self.menuBar()
        
        # Menu Arquivo
        file_menu = menubar.addMenu("Arquivo")
        
        open_action = QAction("Abrir Modelo...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.load_model)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Sair", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Menu Visualização
        view_menu = menubar.addMenu("Visualização")
        
        debug_action = QAction("Debug Completo", self)
        debug_action.setShortcut("F12")
        debug_action.triggered.connect(self.show_debug_info)
        view_menu.addAction(debug_action)
        
        # Menu Ajuda
        help_menu = menubar.addMenu("Ajuda")
        
        about_action = QAction("Sobre KEX GUI", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def load_model(self):
        """Carregar modelo"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Abrir Modelo Shadow Man", "", 
            "Shadow Man Mesh (*.msh);;Todos os arquivos (*.*)"
        )
        
        if filename:
            success = self.viewer.load_shadowman_model(filename)
            
            if success:
                self.update_ui()
                self.statusBar().showMessage(f"Modelo carregado: {os.path.basename(filename)}")
            else:
                self.statusBar().showMessage("Erro ao carregar modelo")
    
    def update_ui(self):
        """Atualizar interface após carregamento"""
        # Atualizar informações
        status = self.viewer.kex_core.get_system_status()
        
        mode_info = f"""🚀 MODELO CARREGADO:
🎯 Modo: {self.viewer.system_mode}
📐 Vértices: {status['mesh_vertices']}
🦴 Bones: {status['skin_bones']}
🎬 Animações: {status['animation_count']}
🎨 Texturas: {status['texture_count']}

✅ Sistema KEX ativo!"""
        
        self.info_label.setText(mode_info)
        
        # Atualizar lista de animações
        self.anim_list.clear()
        if self.viewer.animations:
            anim_names = self.viewer.get_animation_names()
            self.anim_list.addItems(anim_names)
            
            # Habilitar controles
            self.play_button.setEnabled(True)
            self.stop_button.setEnabled(True)
            self.frame_slider.setEnabled(True)
            self.frame_spin.setEnabled(True)
            self.speed_slider.setEnabled(True)
            
            # Configurar controles de frame
            max_frames = max(anim["numFrames"] for anim in self.viewer.animations)
            self.frame_slider.setMaximum(max_frames - 1)
            self.frame_spin.setMaximum(max_frames - 1)
        else:
            self.play_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            self.frame_slider.setEnabled(False)
            self.frame_spin.setEnabled(False)
            self.speed_slider.setEnabled(False)
        
        # Atualizar lista de texturas
        self.texture_list.clear()
        for tex_index, tex_path in self.viewer.textures.items():
            item_text = f"🎨 {tex_index:03d} - {os.path.basename(tex_path)}"
            self.texture_list.addItem(item_text)
    
    def show_debug_info(self):
        """Mostrar informações de debug"""
        debug_info = self.viewer.get_debug_info()
        
        # Mostrar em dialog ou console
        print("=" * 80)
        print("🔍 KEX GUI DEBUG COMPLETO")
        print("=" * 80)
        print(debug_info)
        print("=" * 80)
        
        # Atualizar label com resumo
        lines = debug_info.split('\n')
        summary_lines = [line for line in lines[:15] if line.strip()]
        summary = '\n'.join(summary_lines) + f"\n\n📊 Total de {len(lines)} linhas de debug"
        self.info_label.setText(summary)
    
    def show_about(self):
        """Mostrar sobre"""
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.about(
            self,
            "Sobre KEX GUI",
            "🚀 KEX GUI - Shadow Man Advanced Model Viewer\n\n"
            "Sistema moderno de visualização para modelos do Shadow Man\n\n"
            "CARACTERÍSTICAS:\n"
            "• Sistema híbrido adaptativo (COMPLETO/BÁSICO/ESTÁTICO)\n"
            "• Interpretação completa de .msh, .skn e .anims\n"
            "• Animação e skinning modernos\n"
            "• Suporte a texturas e UVs corretos\n"
            "• Debug detalhado do sistema\n"
            "• Análise avançada de arquivos SKN/ANIMS\n\n"
            "NÚCLEO:\n"
            "• KEX Core - Sistema de interpretação\n"
            "• OpenGL - Renderização 3D\n"
            "• PySide6 - Interface moderna\n\n"
            "Desenvolvido com base no código oficial do Blender add-on"
        )
    
    # Métodos de controle (conectados aos widgets)
    def toggle_wireframe(self, checked):
        self.viewer.show_wireframe = checked
        self.viewer.update()
    
    def toggle_textures(self, checked):
        self.viewer.show_textures = checked
        self.viewer.update()
    
    def toggle_texcoords(self, checked):
        self.viewer.show_texcoords = checked
        self.viewer.update()
    
    def toggle_bones(self, checked):
        self.viewer.show_bones = checked
        self.viewer.update()
    
    def select_animation(self, index):
        if index >= 0:
            success = self.viewer.set_current_animation(index)
            if success and self.viewer.animations:
                anim = self.viewer.animations[index]
                max_frames = anim["numFrames"] - 1
                self.frame_slider.setMaximum(max_frames)
                self.frame_spin.setMaximum(max_frames)
    
    def start_animation(self):
        success = self.viewer.play_animation(True)
        if success:
            self.play_button.setStyleSheet("background-color: #FF9800; color: white;")
    
    def stop_animation(self):
        self.viewer.play_animation(False)
        self.play_button.setStyleSheet("background-color: #4CAF50; color: white;")
    
    def set_frame(self, frame):
        self.viewer.set_animation_frame(frame)
        self.frame_slider.setValue(frame)
        self.frame_spin.setValue(frame)
    
    def set_speed(self, speed):
        self.viewer.animation_speed = speed / 10.0


def setup_dark_theme(app):
    """Configurar tema escuro moderno"""
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


def main():
    """Função principal do KEX GUI"""
    app = QApplication(sys.argv)
    
    # Configurar tema escuro
    setup_dark_theme(app)
    
    try:
        window = MainWindow()
        window.show()
        
        print("🚀" + "=" * 80)
        print("🚀 KEX GUI - Shadow Man Advanced Model Viewer")
        print("🚀" + "=" * 80)
        print("✨ SISTEMA HÍBRIDO ADAPTATIVO:")
        print("   🦴 MODO COMPLETO: .msh + .skn + .anims (skinning completo)")
        print("   🎬 MODO BÁSICO: .msh + .anims (animação simples)")
        print("   📐 MODO ESTÁTICO: apenas .msh (visualização)")
        print("")
        print("🎯 CARACTERÍSTICAS:")
        print("   🔍 Análise detalhada de SKN/ANIMS")
        print("   🎨 Suporte completo a texturas")
        print("   🦴 Visualização de skeleton")
        print("   📊 Debug completo do sistema")
        print("   💾 Exportação para JSON")
        print("")
        print("🎮 COMO USAR:")
        print("   📁 Tab 'Visualizador 3D': Carregue um modelo .msh")
        print("   🔍 Tab 'Analisador': Analise arquivos SKN/ANIMS individualmente")
        print("   ⌨️ Controles: Espaço (Play/Pause), R (Reset), B (Bones)")
        print("   🖱️ Mouse: Arrastar (Rotar), Scroll (Zoom)")
        print("🚀" + "=" * 80)
        
        sys.exit(app.exec())
    
    except Exception as e:
        print(f"❌ Erro ao iniciar KEX GUI: {e}")
        print("Verifique se todas as dependências estão instaladas:")
        print("  pip install PySide6 PyOpenGL numpy pillow")
        sys.exit(1)


if __name__ == "__main__":
    main()
