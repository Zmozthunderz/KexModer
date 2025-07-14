import struct
import math
import os
from typing import Dict, Tuple, Optional

try:
    from PIL import Image
except ImportError:
    Image = None  # allow py_compile without Pillow

try:
    from OpenGL.GL import (
        glGenTextures, glBindTexture, glTexParameteri, glTexImage2D,
        GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER,
        GL_LINEAR, GL_RGBA, GL_UNSIGNED_BYTE
    )
except Exception:
    glGenTextures = glBindTexture = glTexParameteri = glTexImage2D = None
    GL_TEXTURE_2D = GL_TEXTURE_MIN_FILTER = GL_TEXTURE_MAG_FILTER = None
    GL_LINEAR = GL_RGBA = GL_UNSIGNED_BYTE = None


# --- basic binary helpers --------------------------------------------------

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


# --- msh parsing -----------------------------------------------------------

def _parse_msh(data: bytes, read_cplane: bool, read_colors: bool) -> Dict:
    """Internal parser used by `load_msh` with optional sections."""
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
        _, offset = read_u8(data, offset)  # fill mode
        _, offset = read_u8(data, offset)
        _, offset = read_u8(data, offset)
        tex_index, offset = read_u16(data, offset)
        _, offset = read_u16(data, offset)
        _, offset = read_u16(data, offset)

        if read_cplane:
            for _ in range(4):
                _, offset = read_float(data, offset)

        indices = []
        uvs = []
        colors = []
        for _ in range(num_verts):
            vi, offset = read_u16(data, offset)
            u, offset = read_float(data, offset)
            v, offset = read_float(data, offset)
            if read_colors:
                r, offset = read_u8(data, offset)
                g, offset = read_u8(data, offset)
                b, offset = read_u8(data, offset)
                a, offset = read_u8(data, offset)
            else:
                r = g = b = a = 255
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
    remaining = len(data) - offset
    expected_bytes = loop_count * 12
    if remaining >= expected_bytes:
        for _ in range(loop_count):
            nx, offset = read_float(data, offset)
            ny, offset = read_float(data, offset)
            nz, offset = read_float(data, offset)
            loop_normals.append((nx, -ny, -nz))
    else:
        for face in faces:
            for idx in face["indices"]:
                loop_normals.append(normals[idx])

    return {
        "faces": faces,
        "verts": verts,
        "normals": normals,
        "loop_normals": loop_normals,
    }


def load_msh(filepath: str) -> Tuple[Dict, str]:
    """Try to load a .msh file using several strategies."""
    with open(filepath, "rb") as f:
        data = f.read()

    attempts = [
        ("padrao", True, True),
        ("sem_cores", True, False),
        ("sem_cplane", False, True),
        ("minimo", False, False),
    ]
    last_error: Optional[Exception] = None
    for name, cplane, colors in attempts:
        try:
            model = _parse_msh(data, read_cplane=cplane, read_colors=colors)
            return model, name
        except Exception as e:
            last_error = e
            continue
    raise last_error if last_error else RuntimeError("Falha ao carregar MSH")


# --- texture helpers -------------------------------------------------------

VALID_TEXTURE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tga"]


def load_textures(directory: str):
    """Load textures in a directory keyed by numeric prefix."""
    if Image is None or glGenTextures is None:
        return {}
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
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data
        )
        textures[index] = tex_id
    return textures


