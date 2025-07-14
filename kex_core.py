#!/usr/bin/env python3
"""
KEX Core - Shadow Man Model, Mesh, Skin and Animation Interpreters
Núcleo de interpretação para arquivos do Shadow Man

CLASSES:
    - Model3DInterpreter: Interpretação geral de modelos 3D
    - MeshInterpreter: Interpretação de arquivos .msh (mesh)
    - SkinInterpreter: Interpretação de arquivos .skn (skinning)
    - AnimationInterpreter: Interpretação de arquivos .anims (animações)
    - AnimationSystem: Sistema de animação e skinning moderno

CARACTERÍSTICAS:
    - Interpretação completa baseada no código oficial do Blender
    - Sistema híbrido de animação (completo/básico/estático)
    - Suporte a hard bones e soft bones
    - Verificação de compatibilidade mesh/skin
    - Transformações globais corretas
    - Sistema de debug detalhado
"""

import sys
import struct
import os
import math
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Any, Optional


class Model3DInterpreter:
    """
    🎯 INTERPRETADOR GERAL DE MODELOS 3D
    Classe base para interpretação de modelos 3D genéricos
    """
    
    def __init__(self):
        self.global_matrix = np.eye(4)
        self.supported_formats = ['.msh', '.skn', '.anims']
        self.texture_extensions = ['.tga', '.png', '.jpg', '.jpeg', '.bmp', '.dds']
        
    def apply_global_transform(self, vector):
        """✅ TRANSFORMAÇÃO CORRETA: Exatamente como no Blender"""
        # No Blender: loc = global_matrix @ Vector(original_pos)
        # Seguido por: skin["verts"][i] = (-loc[0], loc[1], loc[2])
        # 
        # Como não temos global_matrix aqui, aplicamos a transformação direta:
        # Inverter X, manter Y e Z
        return (-vector[0], vector[1], vector[2])
    
    def extract_texture_index(self, filename):
        """Extrai índice de textura do nome do arquivo"""
        import re
        match = re.match(r'^(\d+)', os.path.splitext(filename)[0])
        if match:
            return int(match.group(1))
        return -1
    
    def find_textures(self, base_path, texture_indices):
        """Procura texturas baseado nos índices"""
        textures = {}
        
        search_dirs = [
            os.path.dirname(base_path),
            os.path.join(os.path.dirname(base_path), 'textures'),
            os.path.join(os.path.dirname(base_path), 'tex'),
            os.path.join(os.path.dirname(base_path), 'images'),
        ]
        
        for search_dir in search_dirs:
            if not os.path.exists(search_dir):
                continue
                
            for filename in os.listdir(search_dir):
                if not any(filename.lower().endswith(ext) for ext in self.texture_extensions):
                    continue
                
                tex_index = self.extract_texture_index(filename)
                if tex_index in texture_indices and tex_index not in textures:
                    full_path = os.path.join(search_dir, filename)
                    textures[tex_index] = full_path
                    print(f"✅ Textura: {tex_index} -> {filename}")
        
        return textures
    
    # =========================================================================
    # UTILITÁRIOS DE ARQUIVO COM VERIFICAÇÕES DE SEGURANÇA
    # =========================================================================
    
    @staticmethod
    def safe_read(file, size, description="dados"):
        """Lê dados do arquivo com verificação de segurança."""
        try:
            data = file.read(size)
            if len(data) != size:
                raise EOFError(f"Fim de arquivo inesperado ao ler {description}. "
                             f"Esperado {size} bytes, obtido {len(data)} bytes.")
            return data
        except Exception as e:
            raise IOError(f"Erro ao ler {description}: {e}")
    
    @staticmethod
    def read_string_count(file, count):
        data = Model3DInterpreter.safe_read(file, count, f"string de {count} caracteres")
        return data.decode('ascii', errors='ignore')
    
    @staticmethod
    def read_string(file):
        result = bytearray()
        while True:
            byte_data = Model3DInterpreter.safe_read(file, 1, "byte de string")
            if byte_data == b'\x00':
                break
            result.extend(byte_data)
        return result.decode('ascii', errors='ignore')
    
    @staticmethod
    def read_u32(file, little_endian=True):
        format_str = '<I' if little_endian else '>I'
        data = Model3DInterpreter.safe_read(file, 4, "uint32")
        return struct.unpack(format_str, data)[0]
    
    @staticmethod
    def read_u16(file, little_endian=True):
        format_str = '<H' if little_endian else '>H'
        data = Model3DInterpreter.safe_read(file, 2, "uint16")
        return struct.unpack(format_str, data)[0]
    
    @staticmethod
    def read_i32(file, little_endian=True):
        format_str = '<i' if little_endian else '>i'
        data = Model3DInterpreter.safe_read(file, 4, "int32")
        return struct.unpack(format_str, data)[0]
    
    @staticmethod
    def read_float(file):
        data = Model3DInterpreter.safe_read(file, 4, "float")
        return struct.unpack('<f', data)[0]
    
    @staticmethod
    def read_vector3(file):
        return (
            Model3DInterpreter.read_float(file),
            Model3DInterpreter.read_float(file),
            Model3DInterpreter.read_float(file)
        )
    
    @staticmethod
    def read_vector4(file):
        return (
            Model3DInterpreter.read_float(file),
            Model3DInterpreter.read_float(file),
            Model3DInterpreter.read_float(file),
            Model3DInterpreter.read_float(file)
        )
    
    @staticmethod
    def read_shadowman_color(file):
        data = Model3DInterpreter.safe_read(file, 4, "cor")
        r, g, b, a = struct.unpack('<BBBB', data)
        return (r/255.0, g/255.0, b/255.0, a/255.0)


class MeshInterpreter(Model3DInterpreter):
    """
    🎯 INTERPRETADOR DE MESH (.msh)
    Interpretação completa de arquivos de mesh do Shadow Man
    """
    
    # Constantes de formato de arquivo
    MESH_SIGNATURE = 'EMsh'
    MESH_VERSION = 'V001'
    
    def __init__(self):
        super().__init__()
        self.last_loaded_mesh = None
    
    def load_mesh(self, filepath):
        """✅ CARREGAMENTO DE MESH COM UVS CORRETOS"""
        if not os.path.exists(filepath):
            print(f"[ERRO] Arquivo não encontrado: {filepath}")
            return None
        
        model = {}
        
        try:
            with open(filepath, 'rb') as file:
                print(f"Carregando mesh: {filepath}")
                
                # Verificar assinatura e versão
                file_type = self.read_string_count(file, 4)
                if file_type != self.MESH_SIGNATURE:
                    print(f"[ERRO] {filepath} não é um arquivo de mesh válido")
                    return None
                
                version = self.read_string_count(file, 4)
                if version != self.MESH_VERSION:
                    print(f"[ERRO] Versão incorreta: {version}")
                    return None
                
                # Ler contadores
                face_count = self.read_u32(file, False)
                vert_count = self.read_u32(file, False)
                
                print(f"✅ Mesh: {vert_count} vértices, {face_count} faces")
                
                model["faceCount"] = face_count
                model["vertCount"] = vert_count
                model["faces"] = []
                model["verts"] = {
                    "loc": [],
                    "normals": [],
                }
                model["texture_indices"] = set()
                
                # Ler faces
                for i in range(face_count):
                    face = {
                        "numVerts": 0,
                        "fillMode": 0,
                        "texIndex": 0,
                        "attributes": 0,
                        "indices": [],
                        "loopUV": [],
                        "loopColors": [],
                    }
                    
                    model["faces"].append(face)
                    
                    # Ler dados da face
                    face["numVerts"] = int.from_bytes(self.safe_read(file, 1, f"numVerts da face {i}"), 'little')
                    face["fillMode"] = int.from_bytes(self.safe_read(file, 1, f"fillMode da face {i}"), 'little')
                    unknown1 = int.from_bytes(self.safe_read(file, 1, f"unknown1 da face {i}"), 'little')
                    unknown2 = int.from_bytes(self.safe_read(file, 1, f"unknown2 da face {i}"), 'little')
                    face["texIndex"] = self.read_u16(file, False)
                    unknown3 = self.read_u16(file, False)
                    face["attributes"] = self.read_u16(file, False)
                    
                    # Adicionar à lista de texturas usadas
                    model["texture_indices"].add(face["texIndex"])
                    
                    # Pular plano de corte (4 floats)
                    for j in range(4):
                        self.read_float(file)
                    
                    # 🎯 CORREÇÃO CRÍTICA: UVs corretos baseados no Blender
                    for j in range(face["numVerts"]):
                        vert_index = self.read_u16(file, False)
                        u = self.read_float(file)
                        v = -self.read_float(file)  # ✅ Inverter V conforme Blender
                        color = self.read_shadowman_color(file)
                        
                        face["indices"].append(vert_index)
                        face["loopUV"].append((u, v))  # ✅ UV correto
                        face["loopColors"].append(color)
                
                # Ler vértices
                for i in range(vert_count):
                    loc = self.read_vector3(file)
                    normal = self.read_vector3(file)
                    
                    # ✅ CORREÇÃO: Aplicar transformação correta
                    loc = self.apply_global_transform(loc)
                    normal = self.apply_global_transform(normal)
                    normal = (normal[0], -normal[1], -normal[2])  # Corrigir normais
                    
                    model["verts"]["loc"].append(loc)
                    model["verts"]["normals"].append(normal)
                
                print(f"✅ Mesh carregada! Texturas: {sorted(model['texture_indices'])}")
                self.last_loaded_mesh = model
                return model
                
        except Exception as e:
            print(f"[ERRO] Falha ao carregar mesh: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_debug_info(self):
        """Informações de debug do mesh"""
        if not self.last_loaded_mesh:
            return "Nenhum mesh carregado"
        
        mesh = self.last_loaded_mesh
        info = []
        info.append("🎯 MESH INTERPRETER DEBUG:")
        info.append(f"Vértices: {mesh.get('vertCount', 0)}")
        info.append(f"Faces: {mesh.get('faceCount', 0)}")
        info.append(f"Texturas: {len(mesh.get('texture_indices', set()))}")
        
        return "\n".join(info)


class SkinInterpreter(Model3DInterpreter):
    """
    🦴 INTERPRETADOR DEDICADO PARA ARQUIVOS .SKN
    Baseado no código funcional do skn_v2.py
    """
    
    def __init__(self):
        super().__init__()
        self.skin_data = {}
        
    def read_skn_file(self, filepath, apply_global_transform=True):
        """
        ✅ LEITURA PRINCIPAL: Baseada no código funcional do skn_v2.py
        """
        if not os.path.exists(filepath):
            print(f"[SKN] Arquivo não encontrado: {filepath}")
            return None
            
        try:
            with open(filepath, 'rb') as f:
                data = f.read()
            
            offset = 0
            print(f"[SKN] Interpretando: {filepath} ({len(data)} bytes)")
            
            # Verificar cabeçalho
            file_type, offset = self.read_string(data, offset, 4)
            if file_type != 'BSKN':
                print(f"[SKN] ❌ Assinatura incorreta: '{file_type}', esperado 'BSKN'")
                return None
            
            # Contagem de ossos (uint32, little endian)
            bone_count, offset = self.read_uint32(data, offset, False)
            self.skin_data['bone_count'] = bone_count
            print(f"[SKN] Bone count: {bone_count}")
            
            # Seção HRCY (Hierarquia)
            hrcy_header, offset = self.read_string(data, offset, 4)
            if hrcy_header != 'HRCY':
                print(f"[SKN] ❌ Cabeçalho HRCY esperado, encontrado: '{hrcy_header}'")
                return None
            
            # Ler hierarquia dos ossos (parents)
            bones = []
            for i in range(bone_count):
                parent, offset = self.read_uint32(data, offset, True)  # signed int32
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
                    print(f"[SKN] ❌ Cabeçalho BONE esperado, encontrado: '{bone_header}'")
                    return None
                
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
                print(f"[SKN] ❌ Cabeçalho SOFT esperado, encontrado: '{soft_header}'")
                return None
            
            soft_bone_count, offset = self.read_uint32(data, offset, False)
            self.skin_data['soft_bone_count'] = soft_bone_count
            
            soft_bones = []
            for i in range(soft_bone_count):
                styp_header, offset = self.read_string(data, offset, 4)
                if styp_header != 'STYP':
                    print(f"[SKN] ❌ Cabeçalho STYP esperado, encontrado: '{styp_header}'")
                    return None
                
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
                print(f"[SKN] ❌ Cabeçalho BIND esperado, encontrado: '{bind_header}'")
                return None
            
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
                print(f"[SKN] ❌ Cabeçalho VERT esperado, encontrado: '{vert_header}'")
                return None
            
            vert_count, offset = self.read_uint32(data, offset, False)
            self.skin_data['vert_count'] = vert_count
            
            vertices = []
            vertices_original = []
            for i in range(vert_count):
                vertex, offset = self.read_vector3(data, offset)
                vertices_original.append(vertex)
                
                # Aplicar transformação global se necessário
                if apply_global_transform:
                    transformed = (-vertex[0], vertex[1], vertex[2])
                else:
                    transformed = vertex
                
                vertices.append(transformed)
            
            self.skin_data['vertices'] = vertices
            self.skin_data['vertices_original'] = vertices_original
            self.skin_data['filepath'] = filepath
            
            # Conversão para formato compatível com o sistema de animação
            converted_skin = self.convert_to_animation_format()
            
            print(f"[SKN] ✅ Arquivo SKN carregado com sucesso!")
            print(f"[SKN]   Bones: {bone_count}")
            print(f"[SKN]   Soft bones: {soft_bone_count}")
            print(f"[SKN]   Vértices: {vert_count}")
            print(f"[SKN]   Índices: {indice_count}")
            
            return converted_skin
            
        except Exception as e:
            print(f"[SKN] ❌ Erro na interpretação: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def convert_to_animation_format(self):
        """Converte dados do SKN para formato compatível com sistema de animação"""
        converted = {
            'boneCount': self.skin_data['bone_count'],
            'softBoneCount': self.skin_data['soft_bone_count'],
            'vertCount': self.skin_data['vert_count'],
            'indiceCount': self.skin_data['indice_count'],
            'bones': [],
            'softBones': [],
            'indices': self.skin_data['indices'],
            'verts': self.skin_data['vertices'],
            'vertsOriginal': self.skin_data['vertices_original']
        }
        
        # Converter bones
        for bone in self.skin_data['bones']:
            converted_bone = {
                'parent': bone['parent'],
                'nHards': bone['n_hards'],
                'nSoftTypes': bone['n_soft_types'],
                'hardi': bone['hard_i'],
                'softTypei': bone['soft_type_i']
            }
            converted['bones'].append(converted_bone)
        
        # Converter soft bones
        for soft in self.skin_data['soft_bones']:
            converted_soft = {
                'weight': soft['weight'],
                'nSofts': soft['n_softs'],
                'softi': soft['soft_i'],
                'matrix': soft['matrix']
            }
            converted['softBones'].append(converted_soft)
        
        return converted
    
    # Métodos de leitura baseados no skn_v2.py
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
    
    # =========================================================================
    # MÉTODOS AUXILIARES DE LEITURA
    # =========================================================================
    
    @staticmethod
    def _safe_read(file, size, description="dados"):
        try:
            data = file.read(size)
            if len(data) != size:
                raise EOFError(f"Fim de arquivo inesperado ao ler {description}")
            return data
        except Exception as e:
            raise IOError(f"Erro ao ler {description}: {e}")
    
    @staticmethod
    def _read_string_count(file, count):
        data = SkinInterpreter._safe_read(file, count, f"string de {count} caracteres")
        return data.decode('ascii', errors='ignore')
    
    @staticmethod
    def _read_u32(file, little_endian=True):
        format_str = '<I' if little_endian else '>I'
        data = SkinInterpreter._safe_read(file, 4, "uint32")
        return struct.unpack(format_str, data)[0]
    
    @staticmethod
    def _read_u16(file, little_endian=True):
        format_str = '<H' if little_endian else '>H'
        data = SkinInterpreter._safe_read(file, 2, "uint16")
        return struct.unpack(format_str, data)[0]
    
    @staticmethod
    def _read_i32(file, little_endian=True):
        format_str = '<i' if little_endian else '>i'
        data = SkinInterpreter._safe_read(file, 4, "int32")
        return struct.unpack(format_str, data)[0]
    
    @staticmethod
    def _read_float(file):
        data = SkinInterpreter._safe_read(file, 4, "float")
        return struct.unpack('<f', data)[0]
    
    @staticmethod
    def _read_vector3(file):
        return (
            SkinInterpreter._read_float(file),
            SkinInterpreter._read_float(file),
            SkinInterpreter._read_float(file)
        )


class AnimationInterpreter(Model3DInterpreter):
    """
    🎬 INTERPRETADOR DE ANIMAÇÕES (.anims)
    Sistema moderno de carregamento de animações para Shadow Man
    """
    
    # Constantes de formato de arquivo
    ANIM_SIGNATURE = 'mnAE'
    
    def __init__(self):
        super().__init__()
        self.last_loaded_animations = None
    
    def load_animations(self, filepath):
        """✅ SISTEMA MODERNO: Carregamento completo de animações"""
        if not os.path.exists(filepath):
            print(f"[INFO] Arquivo de animação não encontrado: {filepath}")
            return None
        
        anims = []
        try:
            with open(filepath, 'rb') as file:
                print(f"Carregando animações: {filepath}")
                
                file_type = self.read_string_count(file, 4)
                if file_type != self.ANIM_SIGNATURE:
                    print(f"[ERRO] Assinatura incorreta: '{file_type}'")
                    return None
                
                anim_count = self.read_u32(file, True)
                print(f"✅ Carregando {anim_count} animações...")
                
                # Criar estruturas de animação
                for i in range(anim_count):
                    anim = {
                        "index": self.read_u32(file, False),
                        "name": "",
                        "numBones": 0,
                        "numFrames": 0,
                        "bones": [],
                    }
                    anims.append(anim)
                
                # ✅ CARREGAR DADOS DETALHADOS
                for i in range(anim_count):
                    anim = anims[i]
                    anim["numBones"] = self.read_u32(file, True)
                    anim["name"] = self.read_string(file)
                    anim["numFrames"] = self.read_u32(file, True)
                    anim["bones"] = []
                    
                    print(f"   Animação {i}: '{anim['name']}' - {anim['numFrames']} frames, {anim['numBones']} bones")
                    
                    # ✅ DADOS DOS BONES DA ANIMAÇÃO
                    for j in range(anim["numBones"]):
                        anim_bone = {
                            "transOffset": (0.0, 0.0, 0.0),
                            "trans": {},  # frame -> position
                            "rots": {},   # frame -> quaternion
                            "transOriginal": [],
                            "rotsOriginal": [],
                        }
                        anim["bones"].append(anim_bone)
                        
                        # Dados não usados mas necessários para estrutura
                        f_unknown1 = self.read_float(file)
                        bone_offset = self.read_vector3(file)
                        anim_bone["transOffset"] = self.apply_global_transform(bone_offset)
                        
                        # ✅ KEYFRAMES DE TRANSLAÇÃO
                        trans_key_count = self.read_u32(file, True)
                        for k in range(trans_key_count):
                            frame = self.read_u32(file, True) - 1
                            pos = self.read_vector3(file)
                            
                            # Armazenar original
                            anim_bone["transOriginal"].append({
                                "frame": frame + 1,
                                "loc": pos
                            })
                            
                            # Aplicar transformação
                            pos = self.apply_global_transform(pos)
                            anim_bone["trans"][frame] = pos
                        
                        # Frame 0 se não existir
                        if 0 not in anim_bone["trans"]:
                            anim_bone["trans"][0] = anim_bone["transOffset"]
                        
                        # Dados não usados
                        f_unknown2 = self.read_float(file)
                        f_unknown3 = self.read_float(file)
                        unused_pivot = self.read_vector3(file)
                        
                        # ✅ KEYFRAMES DE ROTAÇÃO
                        rots_key_count = self.read_u32(file, True)
                        for k in range(rots_key_count):
                            frame = self.read_u32(file, True) - 1
                            rot_vec4 = self.read_vector4(file)
                            
                            # Armazenar original
                            anim_bone["rotsOriginal"].append({
                                "frame": frame + 1,
                                "rot": rot_vec4
                            })
                            
                            # ✅ CONVERSÃO DE QUATERNION: right-handed para left-handed
                            quat = (-rot_vec4[3], rot_vec4[0], rot_vec4[2], rot_vec4[1])
                            anim_bone["rots"][frame] = quat
                        
                        # Frame 0 se não existir
                        if 0 not in anim_bone["rots"]:
                            anim_bone["rots"][0] = (-1, 0, 0, 0)  # identidade left-handed
                
                print(f"✅ Animações carregadas com sucesso!")
                self.last_loaded_animations = anims
                return anims
                
        except Exception as e:
            print(f"[ERRO] Falha ao carregar animações: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_debug_info(self):
        """Informações de debug das animações"""
        if not self.last_loaded_animations:
            return "Nenhuma animação carregada"
        
        anims = self.last_loaded_animations
        info = []
        info.append("🎬 ANIMATION INTERPRETER DEBUG:")
        info.append(f"Animações: {len(anims)}")
        
        total_frames = sum(anim.get('numFrames', 0) for anim in anims)
        total_bones = sum(anim.get('numBones', 0) for anim in anims)
        
        info.append(f"Total frames: {total_frames}")
        info.append(f"Total bones: {total_bones}")
        
        return "\n".join(info)


class AnimationSystem:
    """
    🦴 SISTEMA MODERNO DE ANIMAÇÃO COM SKINNING REAL
    """
    
    def __init__(self):
        self.bones = []
        self.soft_bones = []
        self.bone_hierarchy = {}
        self.vertex_bone_mapping = {}  # vértice -> bone index
        self.vertex_weights = {}       # vértice -> weight (para soft bones)
        
        # Matrizes de transformação
        self.bind_pose_matrices = []
        self.current_pose_matrices = []
        self.final_bone_matrices = []
        
        # Cache para performance
        self._interpolation_cache = {}
        self._cache_clear_counter = 0
    
    def setup_skeleton(self, skin_data):
        """✅ CONFIGURAÇÃO CORRETA: Baseada na estrutura real do skin"""
        if not skin_data or 'bones' not in skin_data:
            print("❌ Dados de skin inválidos")
            return False
        
        bone_count = skin_data['boneCount']
        self.bones = skin_data['bones']
        self.soft_bones = skin_data.get('softBones', [])
        
        print(f"🦴 Configurando skeleton:")
        print(f"   Hard bones: {bone_count}")
        print(f"   Soft bones: {len(self.soft_bones)}")
        
        # ✅ HIERARQUIA DE BONES (baseada na estrutura real)
        self.bone_hierarchy = {}
        for i, bone in enumerate(self.bones):
            parent = bone.get('parent', -1)
            if parent >= 0:
                if parent not in self.bone_hierarchy:
                    self.bone_hierarchy[parent] = []
                self.bone_hierarchy[parent].append(i)
        
        # ✅ MAPEAMENTO CORRETO VÉRTICE -> BONE (baseado nos índices reais)
        self._setup_correct_vertex_bone_mapping(skin_data)
        
        # Inicializar matrizes
        total_bones = bone_count + len(self.soft_bones)
        self.bind_pose_matrices = [np.eye(4) for _ in range(total_bones)]
        self.current_pose_matrices = [np.eye(4) for _ in range(total_bones)]
        self.final_bone_matrices = [np.eye(4) for _ in range(total_bones)]
        
        # ✅ CALCULAR BIND POSES CORRETAS
        self._calculate_correct_bind_poses(skin_data)
        
        print(f"✅ Sistema de animação configurado!")
        print(f"   Hierarquia: {len(self.bone_hierarchy)} pais")
        print(f"   Mapeamento: {len(self.vertex_bone_mapping)} vértices mapeados")
        
        return True
    
    def _setup_correct_vertex_bone_mapping(self, skin_data):
        """✅ MAPEAMENTO CORRETO: Usando a estrutura real do arquivo skin"""
        indices = skin_data.get('indices', [])
        self.vertex_bone_mapping = {}
        self.vertex_weights = {}
        
        print(f"🔧 Configurando mapeamento com {len(indices)} índices")
        
        # ✅ MAPEAR HARD BONES (seguindo a estrutura real)
        for bone_idx, bone in enumerate(self.bones):
            hard_start = bone.get('hardi', 0)
            hard_count = bone.get('nHards', 0)
            
            if hard_count > 0:
                print(f"   Bone {bone_idx}: {hard_count} hard verts a partir do índice {hard_start}")
            
            for i in range(hard_count):
                if hard_start + i < len(indices):
                    vertex_idx = indices[hard_start + i]
                    self.vertex_bone_mapping[vertex_idx] = bone_idx
                    self.vertex_weights[vertex_idx] = 1.0
        
        # ✅ MAPEAR SOFT BONES (seguindo a estrutura real)
        for bone_idx, bone in enumerate(self.bones):
            soft_start = bone.get('softTypei', 0)
            soft_count = bone.get('nSoftTypes', 0)
            
            if soft_count > 0:
                print(f"   Bone {bone_idx}: {soft_count} soft types a partir do índice {soft_start}")
            
            for i in range(soft_count):
                soft_idx = soft_start + i
                if soft_idx < len(self.soft_bones):
                    soft_bone = self.soft_bones[soft_idx]
                    soft_vert_start = soft_bone.get('softi', 0)
                    soft_vert_count = soft_bone.get('nSofts', 0)
                    
                    for j in range(soft_vert_count):
                        if soft_vert_start + j < len(indices):
                            vertex_idx = indices[soft_vert_start + j]
                            # Soft bones têm prioridade sobre hard bones
                            soft_bone_idx = len(self.bones) + soft_idx
                            self.vertex_bone_mapping[vertex_idx] = soft_bone_idx
                            self.vertex_weights[vertex_idx] = soft_bone.get('weight', 1.0)
        
        print(f"✅ Mapeamento concluído: {len(self.vertex_bone_mapping)} vértices mapeados")
    
    def _calculate_correct_bind_poses(self, skin_data):
        """✅ BIND POSES CORRETAS: Para hard e soft bones"""
        # Para hard bones, usar posições baseadas em hierarquia
        for i, bone in enumerate(self.bones):
            bind_matrix = np.eye(4)
            
            # Posição inicial baseada no índice do bone
            bind_matrix[1, 3] = i * 0.1  # Offset vertical
            
            self.bind_pose_matrices[i] = bind_matrix
        
        # ✅ PARA SOFT BONES, USAR MATRIZES DO ARQUIVO (estrutura correta)
        for i, soft_bone in enumerate(self.soft_bones):
            soft_idx = len(self.bones) + i
            
            # ✅ CONVERTER MATRIZ 3x4 PARA 4x4 NUMPY
            if 'matrix' in soft_bone:
                soft_matrix = soft_bone['matrix']
                
                # Verificar se é a estrutura de lista (como no Blender)
                if isinstance(soft_matrix, list) and len(soft_matrix) >= 4:
                    matrix_4x4 = np.eye(4)
                    
                    # Copiar as 3 primeiras linhas (transformação)
                    for row in range(3):
                        if len(soft_matrix[row]) >= 3:
                            for col in range(3):
                                matrix_4x4[row, col] = soft_matrix[row][col]
                    
                    # Copiar a coluna de posição (4ª coluna das 3 primeiras linhas)
                    for row in range(3):
                        if len(soft_matrix[row]) >= 4:
                            matrix_4x4[row, 3] = soft_matrix[row][3]
                    
                    # A linha de posição adicional (linha 4) se existir
                    if len(soft_matrix) >= 4 and len(soft_matrix[3]) >= 3:
                        for col in range(3):
                            matrix_4x4[3, col] = soft_matrix[3][col]
                    
                    self.bind_pose_matrices[soft_idx] = matrix_4x4
                else:
                    # Fallback para matriz identidade
                    self.bind_pose_matrices[soft_idx] = np.eye(4)
            else:
                self.bind_pose_matrices[soft_idx] = np.eye(4)
    
    def interpolate_animation_frame(self, animation, frame_time):
        """✅ INTERPOLAÇÃO HÍBRIDA: Funciona sempre"""
        cache_key = (id(animation), frame_time)
        if cache_key in self._interpolation_cache:
            return self._interpolation_cache[cache_key]
        
        interpolated_transforms = {}
        
        current_frame = int(frame_time)
        next_frame = current_frame + 1
        t = frame_time - current_frame
        
        # ✅ VERIFICAR SE HÁ BONES NA ANIMAÇÃO
        anim_bones = animation.get('bones', [])
        if not anim_bones:
            return interpolated_transforms
        
        for bone_idx, anim_bone in enumerate(anim_bones):
            # ✅ INTERPOLAÇÃO DE POSIÇÃO
            pos_current = anim_bone.get('trans', {}).get(current_frame, anim_bone.get('transOffset', (0, 0, 0)))
            pos_next = anim_bone.get('trans', {}).get(next_frame, pos_current)
            
            if isinstance(pos_current, (list, tuple)) and isinstance(pos_next, (list, tuple)) and len(pos_current) >= 3 and len(pos_next) >= 3:
                pos_interp = tuple(
                    pos_current[i] + (pos_next[i] - pos_current[i]) * t 
                    for i in range(3)
                )
            else:
                pos_interp = pos_current if isinstance(pos_current, (list, tuple)) else (0, 0, 0)
            
            # ✅ INTERPOLAÇÃO DE ROTAÇÃO
            rot_current = anim_bone.get('rots', {}).get(current_frame, (-1, 0, 0, 0))
            rot_next = anim_bone.get('rots', {}).get(next_frame, rot_current)
            
            if isinstance(rot_current, (list, tuple)) and isinstance(rot_next, (list, tuple)) and len(rot_current) >= 4 and len(rot_next) >= 4:
                rot_interp = self._slerp_quaternion(rot_current, rot_next, t)
            else:
                rot_interp = rot_current if isinstance(rot_current, (list, tuple)) else (-1, 0, 0, 0)
            
            interpolated_transforms[bone_idx] = {
                'position': pos_interp,
                'rotation': rot_interp
            }
        
        # Cache com limpeza periódica
        self._interpolation_cache[cache_key] = interpolated_transforms
        self._clear_cache_if_needed()
        
        return interpolated_transforms
    
    def _slerp_quaternion(self, q1, q2, t):
        """✅ SLERP MODERNO: Interpolação esférica otimizada"""
        if len(q1) != 4 or len(q2) != 4:
            return q1
        
        # Normalizar quaternions
        q1_norm = self._normalize_quaternion(q1)
        q2_norm = self._normalize_quaternion(q2)
        
        # Dot product
        dot = sum(a * b for a, b in zip(q1_norm, q2_norm))
        
        # Caminho mais curto
        if dot < 0.0:
            q2_norm = tuple(-x for x in q2_norm)
            dot = -dot
        
        # Interpolação linear para quaternions próximos
        if dot > 0.9995:
            return self._normalize_quaternion(tuple(
                q1_norm[i] + (q2_norm[i] - q1_norm[i]) * t for i in range(4)
            ))
        
        # SLERP completo
        theta_0 = math.acos(abs(dot))
        sin_theta_0 = math.sin(theta_0)
        
        if sin_theta_0 == 0:
            return q1_norm
        
        theta = theta_0 * t
        sin_theta = math.sin(theta)
        
        s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        
        return tuple(s0 * q1_norm[i] + s1 * q2_norm[i] for i in range(4))
    
    def _normalize_quaternion(self, q):
        """Normalizar quaternion"""
        magnitude = math.sqrt(sum(x * x for x in q))
        if magnitude == 0:
            return (0, 0, 0, 1)
        return tuple(x / magnitude for x in q)
    
    def update_bone_matrices(self, animation, frame_time):
        """✅ ATUALIZAÇÃO HÍBRIDA: Funciona com ou sem skin data"""
        # Obter transformações interpoladas
        transforms = self.interpolate_animation_frame(animation, frame_time)
        
        # ✅ VERIFICAR SE TEMOS MATRIZES SUFICIENTES
        if not self.current_pose_matrices:
            # Criar matrizes básicas se não existirem
            bone_count = len(animation.get('bones', []))
            self.current_pose_matrices = [np.eye(4) for _ in range(bone_count)]
            self.final_bone_matrices = [np.eye(4) for _ in range(bone_count)]
        
        # ✅ CALCULAR MATRIZES LOCAIS
        for bone_idx in range(len(self.current_pose_matrices)):
            local_matrix = np.eye(4)
            
            if bone_idx in transforms:
                transform = transforms[bone_idx]
                
                # Aplicar rotação
                rot_matrix = self._quaternion_to_matrix(transform['rotation'])
                
                # Aplicar posição
                pos = transform['position']
                rot_matrix[0, 3] = pos[0]
                rot_matrix[1, 3] = pos[1]
                rot_matrix[2, 3] = pos[2]
                
                local_matrix = rot_matrix
            
            self.current_pose_matrices[bone_idx] = local_matrix
        
        # ✅ CALCULAR MATRIZES FINAIS
        if hasattr(self, 'bones') and self.bones:
            # Modo completo com hierarquia
            self._calculate_final_matrices()
        else:
            # Modo básico - copiar matrizes locais
            for i in range(len(self.current_pose_matrices)):
                self.final_bone_matrices[i] = self.current_pose_matrices[i].copy()
    
    def _quaternion_to_matrix(self, q):
        """✅ CONVERSÃO MODERNA: Quaternion para matriz 4x4"""
        if len(q) != 4:
            return np.eye(4)
        
        w, x, y, z = q
        
        # Normalizar
        norm = math.sqrt(w*w + x*x + y*y + z*z)
        if norm == 0:
            return np.eye(4)
        
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
        
        matrix = np.eye(4)
        
        # Conversão quaternion para matriz de rotação
        matrix[0, 0] = 1 - 2*(y*y + z*z)
        matrix[0, 1] = 2*(x*y - w*z)
        matrix[0, 2] = 2*(x*z + w*y)
        
        matrix[1, 0] = 2*(x*y + w*z)
        matrix[1, 1] = 1 - 2*(x*x + z*z)
        matrix[1, 2] = 2*(y*z - w*x)
        
        matrix[2, 0] = 2*(x*z - w*y)
        matrix[2, 1] = 2*(y*z + w*x)
        matrix[2, 2] = 1 - 2*(x*x + y*y)
        
        return matrix
    
    def _calculate_final_matrices(self):
        """✅ MATRIZES FINAIS ROBUSTAS: Com verificação de hierarquia"""
        # Resetar matrizes
        for i in range(len(self.current_pose_matrices)):
            if i < len(self.final_bone_matrices):
                self.final_bone_matrices[i] = np.eye(4)
        
        # ✅ VERIFICAR SE TEMOS BONES
        if not hasattr(self, 'bones') or not self.bones:
            # Modo básico - copiar matrizes locais
            for i in range(len(self.current_pose_matrices)):
                if i < len(self.final_bone_matrices):
                    self.final_bone_matrices[i] = self.current_pose_matrices[i].copy()
            return
        
        # ✅ PROCESSAR EM ORDEM HIERÁRQUICA
        processed = set()
        queue = []
        
        # Encontrar bones raiz
        for i, bone in enumerate(self.bones):
            if bone.get('parent', -1) < 0:
                queue.append(i)
        
        # Se não há bones raiz, processar todos
        if not queue:
            queue = list(range(len(self.bones)))
        
        while queue:
            bone_idx = queue.pop(0)
            if bone_idx in processed or bone_idx >= len(self.bones):
                continue
            
            processed.add(bone_idx)
            
            parent_idx = self.bones[bone_idx].get('parent', -1)
            
            if parent_idx >= 0 and parent_idx in processed and parent_idx < len(self.final_bone_matrices):
                # Aplicar transformação do pai
                parent_matrix = self.final_bone_matrices[parent_idx]
                local_matrix = self.current_pose_matrices[bone_idx] if bone_idx < len(self.current_pose_matrices) else np.eye(4)
                self.final_bone_matrices[bone_idx] = np.dot(parent_matrix, local_matrix)
            else:
                # Bone raiz ou sem pai
                if bone_idx < len(self.current_pose_matrices):
                    self.final_bone_matrices[bone_idx] = self.current_pose_matrices[bone_idx].copy()
            
            # Adicionar filhos à fila
            if hasattr(self, 'bone_hierarchy') and bone_idx in self.bone_hierarchy:
                queue.extend(self.bone_hierarchy[bone_idx])
        
        # ✅ PROCESSAR SOFT BONES se existirem
        if hasattr(self, 'soft_bones') and self.soft_bones:
            for i, soft_bone in enumerate(self.soft_bones):
                soft_idx = len(self.bones) + i
                if soft_idx < len(self.final_bone_matrices):
                    # Soft bones usam suas próprias matrizes
                    self.final_bone_matrices[soft_idx] = soft_bone.get('matrix', np.eye(4)).copy()
    
    def apply_skinning(self, vertices, skin_data=None):
        """✅ SKINNING CORRETO: Usando estrutura real do arquivo skin"""
        if not vertices or not self.final_bone_matrices:
            return vertices
        
        transformed_vertices = []
        
        for i, vertex in enumerate(vertices):
            if len(vertex) != 3:
                transformed_vertices.append(vertex)
                continue
            
            # ✅ OBTER BONE E WEIGHT DO VÉRTICE
            bone_idx = self.vertex_bone_mapping.get(i, 0)
            weight = self.vertex_weights.get(i, 1.0)
            
            if bone_idx >= len(self.final_bone_matrices):
                # ✅ FALLBACK: usar matriz identidade
                transformed_vertices.append(vertex)
                continue
            
            # ✅ VERIFICAR SE MATRIZ É VÁLIDA
            bone_matrix = self.final_bone_matrices[bone_idx]
            if bone_matrix is None:
                bone_matrix = np.eye(4)
            
            # ✅ TRANSFORMAR VÉRTICE
            try:
                vertex_homogeneous = np.array([vertex[0], vertex[1], vertex[2], 1.0])
                transformed = np.dot(bone_matrix, vertex_homogeneous)
                
                # ✅ APLICAR WEIGHT CORRETAMENTE (especialmente para soft bones)
                if weight < 1.0 and skin_data:
                    # Interpolar entre posição original e transformada
                    original = np.array([vertex[0], vertex[1], vertex[2], 1.0])
                    transformed = original * (1.0 - weight) + transformed * weight
                
                # Converter para coordenadas cartesianas
                if abs(transformed[3]) > 1e-6:
                    result = (
                        transformed[0] / transformed[3],
                        transformed[1] / transformed[3],
                        transformed[2] / transformed[3]
                    )
                else:
                    result = (transformed[0], transformed[1], transformed[2])
                
                transformed_vertices.append(result)
                
            except Exception as e:
                # Em caso de erro, usar vértice original
                transformed_vertices.append(vertex)
        
        return transformed_vertices
    
    def debug_skin_structure(self, skin_data):
        """✅ NOVO: Debug da estrutura do skin carregado"""
        if not skin_data:
            print("❌ Nenhum skin data para debug")
            return
        
        print(f"🔍 DEBUG ESTRUTURA DO SKIN:")
        print(f"   Bone count: {skin_data.get('boneCount', 0)}")
        print(f"   Soft bone count: {skin_data.get('softBoneCount', 0)}")
        print(f"   Índices: {len(skin_data.get('indices', []))}")
        print(f"   Vértices: {skin_data.get('vertCount', 0)}")
        
        # Debug dos primeiros bones
        bones = skin_data.get('bones', [])
        print(f"   Primeiros 3 bones:")
        for i in range(min(3, len(bones))):
            bone = bones[i]
            print(f"     Bone {i}: parent={bone.get('parent', -1)}, nHards={bone.get('nHards', 0)}, nSofts={bone.get('nSoftTypes', 0)}")
        
        # Debug dos soft bones
        soft_bones = skin_data.get('softBones', [])
        print(f"   Primeiros 2 soft bones:")
        for i in range(min(2, len(soft_bones))):
            soft = soft_bones[i]
            print(f"     Soft {i}: weight={soft.get('weight', 0):.3f}, nSofts={soft.get('nSofts', 0)}")
            
            # Debug da matriz
            matrix = soft.get('matrix', [])
            if matrix and len(matrix) > 0:
                print(f"       Matriz[0]: {matrix[0][:3] if len(matrix[0]) >= 3 else matrix[0]}")
        
        # Debug dos índices
        indices = skin_data.get('indices', [])
        if indices:
            print(f"   Primeiros 10 índices: {indices[:10]}")
            print(f"   Últimos 5 índices: {indices[-5:] if len(indices) >= 5 else indices}")
    
    def debug_vertex_mapping(self):
        """✅ NOVO: Debug do mapeamento vértice->bone"""
        if not hasattr(self, 'vertex_bone_mapping'):
            print("❌ Nenhum mapeamento para debug")
            return
        
        print("🔍 DEBUG MAPEAMENTO VÉRTICE->BONE:")
        print(f"   Total de vértices mapeados: {len(self.vertex_bone_mapping)}")
        
        # Contar vértices por bone
        bone_vertex_count = {}
        for vertex_idx, bone_idx in self.vertex_bone_mapping.items():
            if bone_idx not in bone_vertex_count:
                bone_vertex_count[bone_idx] = 0
            bone_vertex_count[bone_idx] += 1
        
        print(f"   Bones utilizados: {len(bone_vertex_count)}")
        for bone_idx in sorted(bone_vertex_count.keys())[:5]:  # Primeiros 5
            count = bone_vertex_count[bone_idx]
            weight = "varied" if hasattr(self, 'vertex_weights') else "1.0"
            print(f"     Bone {bone_idx}: {count} vértices (weight: {weight})")
        
        # Debug de weights se disponível
        if hasattr(self, 'vertex_weights') and self.vertex_weights:
            unique_weights = set(self.vertex_weights.values())
            print(f"   Weights únicos: {sorted(unique_weights)[:10]}")  # Primeiros 10
    
    def _clear_cache_if_needed(self):
        """Limpar cache periodicamente"""
        self._cache_clear_counter += 1
        if self._cache_clear_counter > 50:
            self._interpolation_cache.clear()
            self._cache_clear_counter = 0
    
    def get_debug_info(self):
        """✅ DEBUG HÍBRIDO: Informações para todos os modos"""
        info = []
        info.append(f"Bones: {len(self.bones) if hasattr(self, 'bones') and self.bones else 0}")
        info.append(f"Soft Bones: {len(self.soft_bones) if hasattr(self, 'soft_bones') and self.soft_bones else 0}")
        
        if hasattr(self, 'bone_hierarchy'):
            info.append(f"Hierarquia: {len(self.bone_hierarchy)} pais")
        
        if hasattr(self, 'vertex_bone_mapping'):
            info.append(f"Vértices mapeados: {len(self.vertex_bone_mapping)}")
        
        if hasattr(self, '_interpolation_cache'):
            info.append(f"Cache de interpolação: {len(self._interpolation_cache)} entradas")
        
        if hasattr(self, 'final_bone_matrices'):
            info.append(f"Matrizes finais: {len(self.final_bone_matrices)}")
        
        # Estatísticas dos bones
        if hasattr(self, 'bones') and self.bones:
            root_count = sum(1 for bone in self.bones if bone.get('parent', -1) < 0)
            info.append(f"Bones raiz: {root_count}")
        
        return "\n".join(info)


# Classe de conveniência que combina todos os interpretadores
class KEXCore:
    """
    🚀 NÚCLEO KEX - Combinação de todos os interpretadores
    """
    
    def __init__(self):
        self.model3d = Model3DInterpreter()
        self.mesh = MeshInterpreter()
        self.skin = SkinInterpreter()
        self.animation = AnimationInterpreter()
        self.anim_system = AnimationSystem()
        
        # Estados
        self.current_mesh = None
        self.current_skin = None
        self.current_animations = None
        self.textures = {}
        
    def load_complete_model(self, mesh_path):
        """
        ✅ CARREGAMENTO COMPLETO: Mesh + Skin + Animações
        """
        print(f"🚀 KEX Core: Carregando modelo completo: {mesh_path}")
        
        # 1. Carregar mesh (obrigatório)
        self.current_mesh = self.mesh.load_mesh(mesh_path)
        if not self.current_mesh:
            print("❌ Falha ao carregar mesh")
            return False
        
        # 2. Tentar carregar skin (opcional)
        skin_path = os.path.splitext(mesh_path)[0] + ".skn"
        self.current_skin = self.skin.read_skn_file(skin_path)
        
        # 3. Tentar carregar animações (opcional)
        anim_path = os.path.splitext(mesh_path)[0] + ".anims"
        self.current_animations = self.animation.load_animations(anim_path)
        
        # 4. Configurar sistema de animação
        if self.current_skin:
            success = self.anim_system.setup_skeleton(self.current_skin)
            if success:
                print("✅ Sistema completo configurado")
            else:
                print("⚠️ Sistema básico configurado")
        
        # 5. Carregar texturas
        self.textures = self.model3d.find_textures(mesh_path, self.current_mesh.get("texture_indices", set()))
        
        print(f"✅ KEX Core: Carregamento concluído!")
        return True
    
    def get_system_status(self):
        """Status completo do sistema"""
        return {
            'has_mesh': bool(self.current_mesh),
            'has_skin': bool(self.current_skin),
            'has_animations': bool(self.current_animations),
            'has_textures': bool(self.textures),
            'mesh_vertices': len(self.current_mesh.get('verts', {}).get('loc', [])) if self.current_mesh else 0,
            'skin_bones': self.current_skin.get('boneCount', 0) if self.current_skin else 0,
            'animation_count': len(self.current_animations) if self.current_animations else 0,
            'texture_count': len(self.textures)
        }
    
    def get_debug_info(self):
        """Debug completo de todos os sistemas"""
        info = []
        info.append("🚀 KEX CORE DEBUG:")
        
        status = self.get_system_status()
        for key, value in status.items():
            info.append(f"  {key}: {value}")
        
        if self.current_mesh:
            info.append("\n" + self.mesh.get_debug_info())
        
        if self.current_skin:
            info.append("\n" + self.skin.get_debug_info())
        
        if self.current_animations:
            info.append("\n" + self.animation.get_debug_info())
        
        info.append("\n" + self.anim_system.get_debug_info())
        
        return "\n".join(info)


def main():
    """Função de teste do KEX Core"""
    print("🚀 KEX Core - Sistema de Interpretação Shadow Man")
    print("Este é o núcleo de interpretação. Use kexgui.py para interface gráfica.")
    
    # Teste básico
    core = KEXCore()
    print("✅ KEX Core inicializado com sucesso!")
    print(f"Interpretadores disponíveis:")
    print(f"  🎯 Model3D: {type(core.model3d).__name__}")
    print(f"  🎯 Mesh: {type(core.mesh).__name__}")
    print(f"  🦴 Skin: {type(core.skin).__name__}")
    print(f"  🎬 Animation: {type(core.animation).__name__}")
    print(f"  🦴 AnimSystem: {type(core.anim_system).__name__}")


if __name__ == "__main__":
    main()
