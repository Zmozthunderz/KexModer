import struct
import os
import math
import sys
import time
import numpy as np
from collections import defaultdict

# --- OpenGL/Qt imports (GUI) ---
try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QFileDialog, QPushButton, QLabel,
        QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem, QSplitter, QSlider,
        QFormLayout, QCheckBox, QLineEdit, QMessageBox
    )
    from PySide6.QtCore import Qt, QTimer, QSize
    from PySide6.QtOpenGLWidgets import QOpenGLWidget
    from OpenGL.GL import (
        glClearColor, glClear, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
        glEnable, GL_DEPTH_TEST, glBegin, glEnd, GL_LINES, glVertex3f, glColor3f,
        glMatrixMode, GL_PROJECTION, GL_MODELVIEW, glLoadIdentity, glViewport,
        glRotatef, glTranslatef
    )
    from OpenGL.GLU import gluPerspective
    PYSIDE_OK = True
except Exception as e:
    PYSIDE_OK = False
    _GUI_IMPORT_ERR = e

class ShadowManAdvancedAnalyzer:
    """
    🔍 ANALISADOR AVANÇADO SHADOW MAN
    Usando conhecimento ASF/AMC como base para engenharia reversa
    """
    
    def __init__(self):
        self.debug_mode = True
        
        # Conhecimento ASF/AMC aplicado
        self.asf_patterns = {
            'bone_fields': ['id', 'name', 'direction', 'length', 'axis', 'dof', 'limits'],
            'typical_bone_count': (20, 50),  # Range típico
            'coordinate_system': 'Y_UP',
            'hierarchy_depth': (3, 8),  # Profundidade típica
        }
        
        self.amc_patterns = {
            'frame_structure': ['frame_number', 'root_motion', 'bone_rotations'],
            'typical_frame_count': (30, 10000),
            'rotation_components': (3, 6),  # 3 euler ou 6 (pos+rot)
        }
    
    def analyze_all_formats(self, base_path):
        """Análise completa de todos os formatos Shadow Man"""
        results = {}
        
        print("🎮 ANÁLISE COMPLETA SHADOW MAN")
        print("="*60)
        print("Usando conhecimento ASF/AMC como referência...")
        print()
        
        # Buscar arquivos Shadow Man
        files = {
            'msh': self._find_files(base_path, '.msh'),
            'skn': self._find_files(base_path, '.skn'), 
            'anims': self._find_files(base_path, '.anims'),
            'anm': self._find_files(base_path, '.anm')
        }
        
        for format_type, file_list in files.items():
            if file_list:
                print(f"\n📁 Analisando arquivos {format_type.upper()}:")
                results[format_type] = []
                
                for filepath in file_list[:50]:  # Limite maior p/ UI
                    print(f"\n🔍 {os.path.basename(filepath)}")
                    
                    if format_type == 'msh':
                        analysis = self.analyze_msh_with_asf_knowledge(filepath)
                    elif format_type == 'skn':
                        analysis = self.analyze_skn_with_asf_knowledge(filepath)
                    elif format_type in ['anims', 'anm']:
                        analysis = self.analyze_animation_with_amc_knowledge(filepath)
                    
                    results[format_type].append(analysis)
        
        # Análise cruzada
        self._cross_format_analysis(results)
        
        return results
    
    def analyze_skn_with_asf_knowledge(self, filepath):
        """Analisar SKN aplicando conhecimento de ASF"""
        
        with open(filepath, 'rb') as f:
            data = f.read()
        
        print(f"📊 Tamanho: {len(data)} bytes")
        
        analysis = {
            'filepath': filepath,
            'size': len(data),
            'header': self._analyze_skn_header(data),
            'bone_structure': self._analyze_bone_structure(data),
            'hierarchy_hints': self._find_hierarchy_patterns(data),
            'string_data': self._find_bone_names(data),
            'confidence': 0
        }
        
        # Calcular confiança baseada nos padrões ASF
        analysis['confidence'] = self._calculate_skn_confidence(analysis)
        
        self._print_skn_analysis(analysis)
        return analysis
    
    def _analyze_skn_header(self, data):
        """Analisar cabeçalho do SKN baseado no padrão ASF"""
        header = {}
        
        # Buscar assinatura (como :version em ASF)
        possible_signatures = []
        for i in range(min(20, len(data)-4)):
            chunk = data[i:i+4]
            possible_signatures.append((i, chunk, chunk.hex()))
        
        header['signatures'] = possible_signatures
        
        # Buscar contador de bones (equivalente ao número de entradas :bonedata)
        bone_count_candidates = []
        for i in range(0, min(100, len(data)-4), 4):
            try:
                value = struct.unpack('<I', data[i:i+4])[0]
                if self.asf_patterns['typical_bone_count'][0] <= value <= self.asf_patterns['typical_bone_count'][1]:
                    bone_count_candidates.append((i, value))
            except:
                continue
        
        header['bone_count_candidates'] = bone_count_candidates
        
        return header
    
    def _analyze_bone_structure(self, data):
        """Analisar estrutura de bones baseada no padrão ASF"""
        
        bone_structures = []
        
        # Em ASF cada bone tem: direction(3f), length(1f), axis(3f), etc
        # Buscar padrões similares no binário
        
        for i in range(0, len(data)-48, 4):  # 48 bytes = estimativa mínima por bone
            chunk = data[i:i+48]
            
            try:
                # Tentar interpretar como dados de bone
                floats = struct.unpack('<12f', chunk)
                
                # Analisar se parece dados de bone válidos
                direction = floats[0:3]
                length_candidate = floats[3]
                axis_candidate = floats[4:7]
                
                bone_analysis = self._validate_bone_data(direction, length_candidate, axis_candidate)
                
                if bone_analysis['valid']:
                    bone_structures.append({
                        'offset': i,
                        'direction': direction,
                        'length': length_candidate,
                        'axis': axis_candidate,
                        'confidence': bone_analysis['confidence']
                    })
                    
            except:
                continue
        
        return bone_structures[:200]  # Mais p/ UI
    
    def _validate_bone_data(self, direction, length, axis):
        """Validar se dados parecem bone válido (baseado em ASF)"""
        
        confidence = 0
        reasons = []
        
        # Validar direction (deve ser vetor com magnitude razoável)
        dir_magnitude = math.sqrt(sum(d*d for d in direction))
        if 0.1 < dir_magnitude < 5.0:
            confidence += 30
            reasons.append("direction_magnitude_ok")
        
        # Validar length (deve ser positivo e razoável) 
        if 0.01 < length < 50.0:
            confidence += 25
            reasons.append("length_reasonable")
            
        # Validar axis (ângulos em graus ou radianos)
        axis_magnitude = math.sqrt(sum(a*a for a in axis))
        if axis_magnitude < 10.0:  # Provavelmente radianos ou pequenos ângulos
            confidence += 20
            reasons.append("axis_reasonable")
        elif axis_magnitude < 360.0:  # Provavelmente graus
            confidence += 15
            reasons.append("axis_degrees_possible")
        
        # Verificar se direction é aproximadamente unitário
        if 0.8 < dir_magnitude < 1.2:
            confidence += 25
            reasons.append("direction_unit_vector")
        
        return {
            'valid': confidence > 50,
            'confidence': confidence,
            'reasons': reasons
        }
    
    def analyze_animation_with_amc_knowledge(self, filepath):
        """Analisar ANIMS/ANM aplicando conhecimento de AMC"""
        
        with open(filepath, 'rb') as f:
            data = f.read()
        
        print(f"📊 Tamanho: {len(data)} bytes")
        
        analysis = {
            'filepath': filepath,
            'size': len(data),
            'header': self._analyze_animation_header(data),
            'frame_structure': self._analyze_frame_structure(data),
            'rotation_data': self._analyze_rotation_data(data),
            'temporal_patterns': self._find_temporal_patterns(data)
        }
        
        self._print_animation_analysis(analysis)
        return analysis
    
    def _analyze_animation_header(self, data):
        """Analisar cabeçalho de animação baseado no AMC"""
        header = {}
        
        # Buscar identificador de formato (como :FULLY-SPECIFIED em AMC)
        header['format_signatures'] = []
        for i in range(min(20, len(data)-4)):
            chunk = data[i:i+4]
            header['format_signatures'].append((i, chunk.hex()))
        
        # Buscar frame count (primeira informação estrutural em AMC)
        frame_count_candidates = []
        for i in range(4, min(200, len(data)-4), 4):
            try:
                value = struct.unpack('<I', data[i:i+4])[0]
                if self.amc_patterns['typical_frame_count'][0] <= value <= self.amc_patterns['typical_frame_count'][1]:
                    frame_count_candidates.append((i, value))
            except:
                continue
        
        header['frame_count_candidates'] = frame_count_candidates
        
        return header
    
    def _analyze_frame_structure(self, data):
        """Analisar estrutura de frames baseada no AMC"""
        
        frame_patterns = []
        
        # Em AMC: frame_number seguido de bone_name valor1 valor2 valor3...
        # Em ANIMS: deve ter padrão similar mas binário
        
        # Buscar sequências que podem ser dados de frame
        for i in range(0, len(data)-100, 4):
            
            # Tentar ler possível número de frame
            try:
                frame_num = struct.unpack('<I', data[i:i+4])[0]
                
                # Se frame_num parece razoável, analisar dados seguintes
                if 0 <= frame_num <= 10000:
                    
                    # Buscar dados de rotação/transformação após frame number
                    rotation_data = self._extract_frame_rotations(data[i+4:i+100])
                    
                    if rotation_data['valid']:
                        frame_patterns.append({
                            'offset': i,
                            'frame_number': frame_num,
                            'rotation_data': rotation_data,
                            'confidence': rotation_data['confidence']
                        })
                        
            except:
                continue
        
        return frame_patterns[:200]
    
    def _extract_frame_rotations(self, chunk):
        """Extrair dados de rotação de um chunk (baseado no AMC)"""
        
        rotations = []
        confidence = 0
        
        try:
            # Tentar interpretar como sequência de floats (rotações)
            float_count = len(chunk) // 4
            floats = struct.unpack(f'<{float_count}f', chunk[:float_count*4])
            
            # Analisar padrões de rotação
            for i in range(0, len(floats)-2, 3):
                triplet = floats[i:i+3]
                
                # Verificar se parece rotação Euler
                if self._looks_like_euler_rotation(triplet):
                    rotations.append(('euler', triplet))
                    confidence += 10
            
            # Verificar se há quaternions (grupos de 4)
            for i in range(0, len(floats)-3, 4):
                quartet = floats[i:i+4]
                
                if self._looks_like_quaternion(quartet):
                    rotations.append(('quaternion', quartet))
                    confidence += 15
            
        except:
            pass
        
        return {
            'valid': len(rotations) > 0,
            'rotations': rotations,
            'confidence': min(confidence, 100)
        }
    
    def _looks_like_euler_rotation(self, triplet):
        """Verificar se triplet parece rotação Euler"""
        x, y, z = triplet
        
        # Ângulos em graus: -360 a +360
        degrees_range = all(-360 <= angle <= 360 for angle in triplet)
        
        # Ângulos em radianos: -2π a +2π  
        radians_range = all(-7 <= angle <= 7 for angle in triplet)
        
        return degrees_range or radians_range
    
    def _looks_like_quaternion(self, quartet):
        """Verificar se quartet parece quaternion"""
        w, x, y, z = quartet
        
        # Quaternion normalizado tem magnitude ~1
        magnitude = w*w + x*x + y*y + z*z
        return 0.8 < magnitude < 1.2
    
    def _find_hierarchy_patterns(self, data):
        """Buscar padrões de hierarquia (baseado no :hierarchy do ASF)"""
        
        patterns = []
        
        # Buscar sequências de IDs que podem representar hierarquia pai-filho
        for i in range(0, len(data)-20, 4):
            try:
                # Ler 5 valores consecutivos
                values = struct.unpack('<5i', data[i:i+20])
                
                # Analisar se parece hierarquia
                if self._looks_like_hierarchy(values):
                    patterns.append({
                        'offset': i,
                        'values': values,
                        'analysis': self._analyze_hierarchy_pattern(values)
                    })
                    
            except:
                continue
        
        return patterns[:20]
    
    def _looks_like_hierarchy(self, values):
        """Verificar se valores parecem hierarquia de bones"""
        
        # Hierarquia típica: alguns -1 (sem pai), valores sequenciais
        has_root_markers = any(v == -1 or v == 0 for v in values)
        reasonable_range = all(-1 <= v < 100 for v in values)
        has_variation = len(set(values)) > 1
        
        return has_root_markers and reasonable_range and has_variation
    
    def _analyze_hierarchy_pattern(self, values):
        """Analisar padrão de hierarquia em detalhes"""
        
        analysis = {
            'root_bones': [i for i, v in enumerate(values) if v == -1],
            'max_id': max(v for v in values if v >= 0) if any(v >= 0 for v in values) else 0,
            'unique_parents': len(set(v for v in values if v >= 0))
        }
        
        return analysis
    
    def _find_bone_names(self, data):
        """Buscar possíveis nomes de bones (strings)"""
        
        strings = []
        current_string = ""
        
        for byte in data:
            if 32 <= byte <= 126:  # ASCII imprimível
                current_string += chr(byte)
            else:
                if len(current_string) >= 3:  # Strings de 3+ caracteres
                    strings.append(current_string)
                current_string = ""
        
        # Filtrar strings que parecem nomes de bones
        bone_like_strings = []
        bone_keywords = ['bone', 'joint', 'hip', 'knee', 'ankle', 'shoulder', 'elbow', 'wrist', 
                        'head', 'neck', 'spine', 'root', 'left', 'right', 'upper', 'lower']
        
        for s in strings:
            s_lower = s.lower()
            if any(keyword in s_lower for keyword in bone_keywords):
                bone_like_strings.append(s)
        
        return {
            'all_strings': strings[:50],
            'bone_like': bone_like_strings,
            'total_found': len(strings)
        }
    
    def _find_temporal_patterns(self, data):
        """Buscar padrões temporais (sequências de frames)"""
        
        patterns = []
        
        # Buscar sequências crescentes que podem ser números de frame
        for i in range(0, len(data)-20, 4):
            try:
                # Ler 5 valores consecutivos
                values = struct.unpack('<5I', data[i:i+20])
                
                # Verificar se é sequência crescente (frame numbers)
                if self._looks_like_frame_sequence(values):
                    patterns.append({
                        'offset': i,
                        'sequence': values,
                        'step': values[1] - values[0] if len(values) > 1 else 0
                    })
                    
            except:
                continue
        
        return patterns[:20]
    
    def _looks_like_frame_sequence(self, values):
        """Verificar se valores parecem sequência de frames"""
        
        # Verificar se é sequência crescente
        is_increasing = all(values[i] <= values[i+1] for i in range(len(values)-1))
        
        # Verificar se os passos são consistentes
        if len(values) > 2:
            steps = [values[i+1] - values[i] for i in range(len(values)-1)]
            consistent_step = len(set(steps)) <= 2  # Permitir variação pequena
        else:
            consistent_step = True
        
        # Valores devem estar em range razoável para frames
        reasonable_range = all(0 <= v <= 100000 for v in values)
        
        return is_increasing and consistent_step and reasonable_range
    
    def _analyze_rotation_data(self, data):
        """Análise detalhada de dados de rotação"""
        
        rotation_analysis = {
            'euler_candidates': [],
            'quaternion_candidates': [],
            'matrix_candidates': [],
            'statistics': {}
        }
        
        # Buscar diferentes tipos de dados de rotação
        for i in range(0, len(data)-16, 4):
            try:
                # Testar como Euler (3 floats)
                euler = struct.unpack('<3f', data[i:i+12])
                if all(-360 <= e <= 360 for e in euler):
                    rotation_analysis['euler_candidates'].append((i, euler))
                
                # Testar como Quaternion (4 floats)
                quat = struct.unpack('<4f', data[i:i+16])
                magnitude = sum(q*q for q in quat) ** 0.5
                if 0.9 < magnitude < 1.1:
                    rotation_analysis['quaternion_candidates'].append((i, quat))
                
                # Testar como matriz 3x3 (9 floats)
                if i + 36 <= len(data):
                    matrix = struct.unpack('<9f', data[i:i+36])
                    if self._looks_like_rotation_matrix(matrix):
                        rotation_analysis['matrix_candidates'].append((i, matrix))
                        
            except:
                continue
        
        # Estatísticas
        rotation_analysis['statistics'] = {
            'euler_count': len(rotation_analysis['euler_candidates']),
            'quaternion_count': len(rotation_analysis['quaternion_candidates']),
            'matrix_count': len(rotation_analysis['matrix_candidates'])
        }
        
        return rotation_analysis
    
    def _looks_like_rotation_matrix(self, matrix):
        """Verificar se 9 floats parecem matriz de rotação 3x3"""
        try:
            # Reorganizar como matriz 3x3
            m = np.array(matrix).reshape(3, 3)
            
            # Verificar se é ortogonal (det ≈ 1)
            det = np.linalg.det(m)
            if not (0.8 < abs(det) < 1.2):
                return False
            
            # Verificar se linhas são aproximadamente unitárias
            for row in m:
                magnitude = np.linalg.norm(row)
                if not (0.8 < magnitude < 1.2):
                    return False
            
            return True
            
        except:
            return False
    
    def _cross_format_analysis(self, results):
        """Análise cruzada entre formatos para validar descobertas"""
        
        print("\n🔄 ANÁLISE CRUZADA DE FORMATOS")
        print("="*50)
        
        # Verificar consistência entre SKN e ANIMS
        if 'skn' in results and 'anims' in results:
            self._validate_skeleton_animation_consistency(results['skn'], results['anims'])
        
        # Analisar relação MSH com outros formatos
        if 'msh' in results:
            self._analyze_mesh_relationships(results)
    
    def _validate_skeleton_animation_consistency(self, skn_results, anims_results):
        """Validar consistência entre skeleton e animação"""
        
        print("\n🦴➡️🎬 Validação SKN ↔ ANIMS:")
        
        for skn_analysis in skn_results:
            bone_count_estimates = [est[1] for est in skn_analysis['header']['bone_count_candidates'][:3]]
            
            for anims_analysis in anims_results:
                frame_counts = [est[1] for est in anims_analysis['header']['frame_count_candidates'][:3]]
                
                print(f"   SKN bones estimados: {bone_count_estimates}")
                print(f"   ANIMS frames estimados: {frame_counts}")
                
                # Verificar se proporções fazem sentido
                if bone_count_estimates and frame_counts:
                    bone_est = bone_count_estimates[0]
                    frame_est = frame_counts[0]
                    
                    # Calcular tamanho esperado de dados
                    expected_size_per_frame = bone_est * 12  # 3 floats por bone (rotação)
                    expected_total_size = frame_est * expected_size_per_frame
                    actual_size = anims_analysis['size']
                    
                    ratio = actual_size / expected_total_size if expected_total_size > 0 else 0
                    
                    print(f"   Tamanho esperado: {expected_total_size:,} bytes")
                    print(f"   Tamanho real: {actual_size:,} bytes")
                    print(f"   Proporção: {ratio:.2f}")
                    
                    if 0.5 < ratio < 2.0:
                        print("   ✅ Proporção consistente!")
                    else:
                        print("   ⚠️ Proporção inconsistente")
    
    def _calculate_skn_confidence(self, analysis):
        """Calcular confiança na análise do SKN"""
        
        confidence = 0
        
        # Pontuação por candidatos a bone count válidos
        bone_candidates = len(analysis['header']['bone_count_candidates'])
        confidence += min(bone_candidates * 10, 30)
        
        # Pontuação por estruturas de bone válidas
        valid_bones = sum(1 for bone in analysis['bone_structure'] if bone['confidence'] > 50)
        confidence += min(valid_bones * 5, 25)
        
        # Pontuação por padrões de hierarquia
        hierarchy_patterns = len(analysis['hierarchy_hints'])
        confidence += min(hierarchy_patterns * 10, 20)
        
        # Pontuação por strings relacionadas a bones
        bone_strings = len(analysis['string_data']['bone_like'])
        confidence += min(bone_strings * 5, 15)
        
        return min(confidence, 100)
    
    def _print_skn_analysis(self, analysis):
        """Imprimir análise do SKN de forma organizada"""
        
        print(f"   📊 Confiança: {analysis['confidence']}%")
        
        print(f"\n   🏷️ Possíveis contadores de bones:")
        for offset, count in analysis['header']['bone_count_candidates'][:5]:
            print(f"      {offset:04x}: {count} bones")
        
        print(f"\n   🦴 Estruturas de bone válidas: {len(analysis['bone_structure'])}")
        for bone in analysis['bone_structure'][:3]:
            print(f"      {bone['offset']:04x}: len={bone['length']:.2f}, conf={bone['confidence']}%")
        
        print(f"\n   🌳 Padrões de hierarquia: {len(analysis['hierarchy_hints'])}")
        for pattern in analysis['hierarchy_hints'][:2]:
            print(f"      {pattern['offset']:04x}: {pattern['values']}")
        
        print(f"\n   📛 Strings relacionadas a bones: {len(analysis['string_data']['bone_like'])}")
        for s in analysis['string_data']['bone_like'][:5]:
            print(f"      '{s}'")
    
    def _print_animation_analysis(self, analysis):
        """Imprimir análise de animação de forma organizada"""
        
        print(f"\n   🎞️ Possíveis contadores de frames:")
        for offset, count in analysis['header']['frame_count_candidates'][:5]:
            print(f"      {offset:04x}: {count} frames")
        
        print(f"\n   📐 Estruturas de frame: {len(analysis['frame_structure'])}")
        for frame in analysis['frame_structure'][:3]:
            print(f"      {frame['offset']:04x}: frame #{frame['frame_number']}, conf={frame['confidence']}%")
        
        stats = analysis['rotation_data']['statistics']
        print(f"\n   🔄 Dados de rotação encontrados:")
        print(f"      Euler: {stats['euler_count']}")
        print(f"      Quaternion: {stats['quaternion_count']}")
        print(f"      Matrix: {stats['matrix_count']}")
        
        print(f"\n   ⏱️ Padrões temporais: {len(analysis['temporal_patterns'])}")
        for pattern in analysis['temporal_patterns'][:2]:
            print(f"      {pattern['offset']:04x}: {pattern['sequence'][:3]}... (step={pattern['step']})")
    
    def _find_files(self, base_path, extension):
        """Buscar arquivos com extensão específica"""
        files = []
        
        for root, dirs, filenames in os.walk(base_path):
            for filename in filenames:
                if filename.lower().endswith(extension):
                    files.append(os.path.join(root, filename))
        
        return files
    
    def analyze_msh_with_asf_knowledge(self, filepath):
        """Analisar MSH (mesh) com insights de formatos de skeleton"""
        
        with open(filepath, 'rb') as f:
            data = f.read()
        
        print(f"📊 Tamanho: {len(data)} bytes")
        
        # MSH não tem equivalente direto em ASF/AMC, mas pode ter referências de skeleton
        analysis = {
            'filepath': filepath,
            'size': len(data),
            'geometry_data': self._analyze_geometry_structure(data),
            'skeleton_references': self._find_skeleton_references(data),
            'vertex_bone_weights': self._find_bone_weight_patterns(data)
        }
        
        self._print_msh_analysis(analysis)
        return analysis
    
    def _analyze_geometry_structure(self, data):
        """Analisar estrutura geométrica do MSH"""
        
        # Buscar padrões típicos de mesh: vértices, faces, UVs
        geometry = {
            'vertex_candidates': [],
            'face_candidates': [],
            'uv_candidates': []
        }
        
        # Buscar grupos de 3 floats (vértices/normais)
        for i in range(0, len(data)-12, 4):
            try:
                triplet = struct.unpack('<3f', data[i:i+12])
                if all(-1000 < v < 1000 for v in triplet):  # Range razoável para coordenadas
                    geometry['vertex_candidates'].append((i, triplet))
            except:
                continue
        
        return geometry
    
    def _find_skeleton_references(self, data):
        """Buscar referências de skeleton no MSH"""
        
        # Buscar IDs de bones ou índices que podem referenciar skeleton
        bone_refs = []
        
        for i in range(0, len(data)-4, 4):
            try:
                value = struct.unpack('<I', data[i:i+4])[0]
                if 0 <= value < 100:  # Range típico de bone IDs
                    bone_refs.append((i, value))
            except:
                continue
        
        return bone_refs[:200]
    
    def _find_bone_weight_patterns(self, data):
        """Buscar padrões de bone weights (skinning)"""
        
        # Buscar grupos de floats que podem ser weights (soma ≈ 1.0)
        weight_patterns = []
        
        for i in range(0, len(data)-16, 4):
            try:
                weights = struct.unpack('<4f', data[i:i+16])
                total = sum(weights)
                if 0.9 < total < 1.1 and all(0 <= w <= 1 for w in weights):
                    weight_patterns.append((i, weights))
            except:
                continue
        
        return weight_patterns[:200]
    
    def _print_msh_analysis(self, analysis):
        """Imprimir análise do MSH"""
        
        print(f"\n   📐 Candidatos a vértices: {len(analysis['geometry_data']['vertex_candidates'])}")
        print(f"   🔗 Referências de skeleton: {len(analysis['skeleton_references'])}")
        print(f"   ⚖️ Padrões de bone weights: {len(analysis['vertex_bone_weights'])}")
    
    def _analyze_mesh_relationships(self, results):
        """Analisar relação entre mesh e outros formatos"""
        
        print("\n🎯 Relação MSH com outros formatos:")
        
        if 'msh' in results and 'skn' in results:
            msh_analysis = results['msh'][0] if results['msh'] else None
            skn_analysis = results['skn'][0] if results['skn'] else None
            
            if msh_analysis and skn_analysis:
                bone_refs = len(msh_analysis['skeleton_references'])
                estimated_bones = skn_analysis['header']['bone_count_candidates'][0][1] if skn_analysis['header']['bone_count_candidates'] else 0
                
                print(f"   MSH referências de bones: {bone_refs}")
                print(f"   SKN bones estimados: {estimated_bones}")
                
                if bone_refs > 0 and estimated_bones > 0:
                    if bone_refs <= estimated_bones * 4:  # Considerando múltiplas referências por bone
                        print("   ✅ Referências consistentes com skeleton!")
                    else:
                        print("   ⚠️ Muitas referências para poucos bones")

# ------------------------
# Funções utilitárias GUI
# ------------------------

class ShadowManViewer(QOpenGLWidget if PYSIDE_OK else object):
    """Visualizador 3D simples (wireframe + eixo) para pré-visualização.
    Nota: Sem parsing real de MSH/SKN ainda; usa placeholders ou pontos detectados.
    """

    def __init__(self, parent=None):
        if not PYSIDE_OK:  # evitar chamada ao super errado
            return
        super().__init__(parent)
        self.setMinimumSize(QSize(640, 480))
        self._angle = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._playing = False

        # Dados simulados/placeholder (serão alimentados pela UI)
        self.point_cloud = []   # Lista de tuplas (x,y,z)
        self.frame_count = 0
        self.current_frame = 0

    def initializeGL(self):
        glClearColor(0.08, 0.08, 0.1, 1.0)
        glEnable(GL_DEPTH_TEST)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h if h>0 else 1)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = w / float(h if h>0 else 1)
        gluPerspective(45.0, aspect, 0.1, 1000.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Câmera simples
        glTranslatef(0.0, 0.0, -6.0)
        glRotatef(self._angle, 0.0, 1.0, 0.0)

        # Eixos
        self._draw_axes()

        # Placeholder: desenhar um cubo wireframe se não houver pontos
        if not self.point_cloud:
            self._draw_wire_cube(size=1.5)
        else:
            self._draw_point_cloud(self.point_cloud)

    def _draw_axes(self):
        glBegin(GL_LINES)
        # X (vermelho)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(-2.0, 0.0, 0.0); glVertex3f(2.0, 0.0, 0.0)
        # Y (verde)
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0.0, -2.0, 0.0); glVertex3f(0.0, 2.0, 0.0)
        # Z (azul)
        glColor3f(0.0, 0.7, 1.0)
        glVertex3f(0.0, 0.0, -2.0); glVertex3f(0.0, 0.0, 2.0)
        glEnd()

    def _draw_wire_cube(self, size=1.0):
        s = size
        glColor3f(0.9, 0.9, 0.9)
        edges = [
            (-s,-s,-s,  s,-s,-s), (s,-s,-s,  s, s,-s), (s, s,-s, -s, s,-s), (-s, s,-s, -s,-s,-s),
            (-s,-s, s,  s,-s, s), (s,-s, s,  s, s, s), (s, s, s, -s, s, s), (-s, s, s, -s,-s, s),
            (-s,-s,-s, -s,-s, s), (s,-s,-s,  s,-s, s), (s, s,-s,  s, s, s), (-s, s,-s, -s, s, s)
        ]
        glBegin(GL_LINES)
        for x1,y1,z1,x2,y2,z2 in edges:
            glVertex3f(x1,y1,z1); glVertex3f(x2,y2,z2)
        glEnd()

    def _draw_point_cloud(self, points):
        glColor3f(0.9, 0.9, 0.9)
        glBegin(GL_LINES)
        # Desenhar pequenos "ticks" para cada ponto
        tick = 0.02
        for x,y,z in points:
            glVertex3f(x-tick, y, z); glVertex3f(x+tick, y, z)
            glVertex3f(x, y-tick, z); glVertex3f(x, y+tick, z)
            glVertex3f(x, y, z-tick); glVertex3f(x, y, z+tick)
        glEnd()

    def _tick(self):
        # Atualização por frame: gira cena e avança contador de frames
        self._angle = (self._angle + 0.4) % 360.0
        if self.frame_count > 0 and self._playing:
            self.current_frame = (self.current_frame + 1) % self.frame_count
        self.update()

    def set_points(self, pts):
        self.point_cloud = pts or []
        self.update()

    def set_frame_count(self, count):
        self.frame_count = max(0, int(count))
        self.current_frame = 0
        self.update()

    def play(self):
        self._playing = True
        if not self._timer.isActive():
            self._timer.start(16)  # ~60 FPS

    def pause(self):
        self._playing = False

class MainWindow(QMainWindow if PYSIDE_OK else object):
    def __init__(self):
        if not PYSIDE_OK:
            return
        super().__init__()
        self.setWindowTitle("Shadow Man - Analyzer & Viewer (ASF/AMC-aware)")
        self.resize(1100, 680)

        self.analyzer = ShadowManAdvancedAnalyzer()

        # Widgets
        self.viewer = ShadowManViewer(self)

        self.btn_pick = QPushButton("Escolher Pasta…")
        self.ed_path = QLineEdit()
        self.ed_path.setPlaceholderText("Selecione a pasta que contém .msh / .skn / .anims/.anm")
        self.btn_scan = JButton = QPushButton("Analisar")
        self.btn_play = QPushButton("▶ Play")
        self.btn_pause = QPushButton("⏸ Pausar")

        self.list_msh = QListWidget()
        self.list_skn = QListWidget()
        self.list_anims = QListWidget()

        self.lbl_status = QLabel("Pronto.")
        self.slider_speed = QSlider(Qt.Horizontal)
        self.slider_speed.setMinimum(1)
        self.slider_speed.setMaximum(200)
        self.slider_speed.setValue(60)

        # Layout esquerdo (listas)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.addWidget(QLabel("MSH (meshes)"))
        lv.addWidget(self.list_msh, 1)
        lv.addWidget(QLabel("SKN (skeletons)"))
        lv.addWidget(self.list_skn, 1)
        lv.addWidget(QLabel("ANIMS/ANM (animações)"))
        lv.addWidget(self.list_anims, 1)

        # Layout topo
        top = QWidget()
        ht = QHBoxLayout(top)
        ht.addWidget(self.btn_pick)
        ht.addWidget(self.ed_path, 1)
        ht.addWidget(self.btn_scan)
        ht.addWidget(self.btn_play)
        ht.addWidget(self.btn_pause)
        ht.addWidget(QLabel("Velocidade (FPS):"))
        ht.addWidget(self.slider_speed)

        # Splitter principal
        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(self.viewer)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # Container central
        central = QWidget()
        vc = QVBoxLayout(central)
        vc.addWidget(top)
        vc.addWidget(splitter, 1)
        vc.addWidget(self.lbl_status)

        self.setCentralWidget(central)

        # Conexões
        self.btn_pick.clicked.connect(self._choose_folder)
        self.btn_scan.clicked.connect(self._scan_and_analyze)
        self.list_msh.itemSelectionChanged.connect(self._on_select_msh)
        self.list_anims.itemSelectionChanged.connect(self._on_select_anim)
        self.slider_speed.valueChanged.connect(self._change_speed)
        self.btn_play.clicked.connect(self.viewer.play)
        self.btn_pause.clicked.connect(self.viewer.pause)

        # Estado
        self.results = {}
        self.msh_files = []
        self.skn_files = []
        self.anim_files = []

        # Timer de status
        self.status_timer = QTimer(self)
        self.status_timer.setInterval(3000)
        self.status_timer.timeout.connect(self._clear_status)

    def _set_status(self, text):
        self.lbl_status.setText(text)
        self.status_timer.start()

    def _clear_status(self):
        self.lbl_status.setText("Pronto.")
        self.status_timer.stop()

    def _choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Escolher Pasta")
        if folder:
            self.ed_path.setText(folder)

    def _scan_and_analyze(self):
        base = self.ed_path.text().strip()
        if not base or not os.path.isdir(base):
            QMessageBox.warning(self, "Atenção", "Pasta inválida.")
            return

        self.list_msh.clear(); self.list_skn.clear(); self.list_anims.clear()
        self.results = self.analyzer.analyze_all_formats(base)

        # Popular listas
        self.msh_files = [a['filepath'] for a in self.results.get('msh', [])]
        self.skn_files = [a['filepath'] for a in self.results.get('skn', [])]
        self.anim_files = [a['filepath'] for a in self.results.get('anims', [])] + \
                          [a['filepath'] for a in self.results.get('anm', [])]

        for p in self.msh_files:
            self.list_msh.addItem(QListWidgetItem(os.path.basename(p)))
        for p in self.skn_files:
            self.list_skn.addItem(QListWidgetItem(os.path.basename(p)))
        for p in self.anim_files:
            self.list_anims.addItem(QListWidgetItem(os.path.basename(p)))

        # Atualizar viewer com uma nuvem de pontos muito simples baseada nos primeiros "candidatos a vértice"
        if self.results.get('msh'):
            verts = self.results['msh'][0]['geometry_data']['vertex_candidates'][:2000]
            pts = []
            # Normalizar grossamente para caber na tela
            for _, (x,y,z) in verts[::50]:  # diminuir para não fritar
                pts.append((x*0.01, y*0.01, z*0.01))
            self.viewer.set_points(pts)

        self._set_status("Análise concluída. Selecione itens para pré-visualizar.")

        # Se achou animações, estimar framecount e setar no viewer para playback dummy
        framecount = 0
        if self.results.get('anims'):
            header = self.results['anims'][0]['header']
            if header['frame_count_candidates']:
                framecount = header['frame_count_candidates'][0][1]
        elif self.results.get('anm'):
            header = self.results['anm'][0]['header']
            if header['frame_count_candidates']:
                framecount = header['frame_count_candidates'][0][1]

        self.viewer.set_frame_count(framecount or 0)

    def _on_select_msh(self):
        idxs = self.list_msh.selectedIndexes()
        if not idxs:
            return
        i = idxs[0].row()
        if i < len(self.results.get('msh', [])):
            analysis = self.results['msh'][i]
            verts = analysis['geometry_data']['vertex_candidates'][:2000]
            pts = [(x*0.01, y*0.01, z*0.01) for _, (x,y,z) in verts[::50]]
            self.viewer.set_points(pts)
            self._set_status(f"Pré-visualizando MSH: {os.path.basename(analysis['filepath'])}")

    def _on_select_anim(self):
        idxs = self.list_anims.selectedIndexes()
        if not idxs:
            return
        i = idxs[0].row()
        # anim pode estar em 'anims' ou 'anm' — concatenamos acima
        combined = (self.results.get('anims', []) + self.results.get('anm', []))
        if i < len(combined):
            analysis = combined[i]
            fcc = analysis['header']['frame_count_candidates']
            fc = fcc[0][1] if fcc else 0
            self.viewer.set_frame_count(fc)
            self._set_status(f"Animação: {os.path.basename(analysis['filepath'])} | Frames estimados: {fc}")

    def _change_speed(self, value):
        # Ajusta intervalo do timer do viewer com base no "FPS"
        fps = max(1, int(value))
        interval_ms = int(1000.0 / fps)
        if self.viewer._timer.isActive():
            self.viewer._timer.start(interval_ms)
        else:
            self.viewer._timer.setInterval(interval_ms)
        self._set_status(f"Velocidade ~{fps} FPS")

def analyze_shadowman_files(base_path):
    """Função principal para analisar arquivos Shadow Man (CLI)"""
    
    analyzer = ShadowManAdvancedAnalyzer()
    results = analyzer.analyze_all_formats(base_path)
    
    print(f"\n🎯 RESUMO DA ANÁLISE")
    print("="*50)
    
    for format_type, analyses in results.items():
        if analyses:
            print(f"\n{format_type.upper()}: {len(analyses)} arquivo(s) analisado(s)")
            
            if format_type == 'skn':
                avg_confidence = sum(a['confidence'] for a in analyses) / len(analyses)
                print(f"   Confiança média: {avg_confidence:.1f}%")
                
                best_bone_candidates = []
                for analysis in analyses:
                    if analysis['header']['bone_count_candidates']:
                        best_bone_candidates.append(analysis['header']['bone_count_candidates'][0][1])
                
                if best_bone_candidates:
                    print(f"   Estimativas de bones: {set(best_bone_candidates)}")
    
    return results

def _run_gui():
    if not PYSIDE_OK:
        print("ERRO: A interface requer PySide6 e PyOpenGL instalados.")
        print("Detalhe:", _GUI_IMPORT_ERR)
        print("Instale com:")
        print("  pip install PySide6 PyOpenGL")
        sys.exit(2)
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

# Exemplo de uso
if __name__ == "__main__":
    # Modos:
    # 1) CLI: python anal.py <pasta>
    # 2) GUI: python anal.py --gui
    if len(sys.argv) > 1 and sys.argv[1] == "--gui":
        _run_gui()
    elif len(sys.argv) > 1:
        base_path = sys.argv[1]
        results = analyze_shadowman_files(base_path)
    else:
        print("Uso:")
        print("  GUI: python anal.py --gui")
        print("  CLI: python anal.py <caminho_para_arquivos_shadowman>")
        print("\nEste analisador usa conhecimento ASF/AMC para decifrar formatos Shadow Man:")
        print("📁 MSH  - Geometria de mesh")
        print("🦴 SKN  - Dados de skeleton") 
        print("🎬 ANIMS/ANM - Dados de animação")
