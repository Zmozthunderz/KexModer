import bpy
import os
import bmesh
import math
from mathutils import (
    Vector,
    Quaternion,
    Matrix,
)
from bpy_extras.wm_utils.progress_report import (
    ProgressReport,
    ProgressReportSubstep,
)
from . import (
    utils,
    file_utils,
)

# -----------------------------------------------------------------------------
# Returns dictionary of mesh structure. Returns None if there was an error.
def load(filepath, global_matrix):
    model = {}
    with open(filepath, 'rb') as data:
        data.seek(0, 2)
        fileByteSize = data.tell()
        data.seek(0)
        sFileType = file_utils.readStringCount(data, 4) #'EMsh'
        if sFileType != 'EMsh':
            utils.show_error("[Import Mesh] %s is not a valid mesh file." % (filepath))
            return None
        #end if
        sVersion = file_utils.readStringCount(data, 4) #'V001'
        if sVersion != 'V001':
            utils.show_error("[Import Mesh] %s is has incorrect version of %s." % (filepath, sVersion))
            return None
        #end if
        
        model["faceCount"] = file_utils.readU32(data, False)
        model["vertCount"] = file_utils.readU32(data, False)
        model["faces"] = []
        model["verts"] = {
            "loc": [(0.0, 0.0, 0.0)] * model["vertCount"],
            "normals": [(0.0, 0.0, 0.0)] * model["vertCount"],
        }
        loopCount = 0
        for i in range(model["faceCount"]):
            face = {
                "numVerts": 0,
                "fillMode": 0,
                "texIndex": 0,
                "attributes": 0,
                "cPlane": [0.0, 0.0, 0.0, 0.0],
                "unknown1": 0,
                "unknown2": 0,
                "unknown3": 0,
                "indices": [],
                "loopUV": [],
                "loopColors": [],
            }
            model["faces"].append(face)
            face["numVerts"] = file_utils.read8(data, False) #is always a triangle (3)
            face["fillMode"] = file_utils.read8(data, False)
            face["unknown1"] = file_utils.read8(data, False)
            face["unknown2"] = file_utils.read8(data, False)
            face["texIndex"] = file_utils.readU16(data, False)
            face["unknown3"] = file_utils.readU16(data, False)
            face["attributes"] = file_utils.readU16(data, False)
            for i2 in range(len(face["cPlane"])):
                face["cPlane"][i2] = -file_utils.readFloat(data)
                if math.isnan(face["cPlane"][i2]):
                    face["cPlane"][i2] = 0.0
                #end if
            #end for
            face["cPlane"][0] = -face["cPlane"][0] #just need to flip normal x to make it correct with blender

            #Verts for the face
            face["indices"] = [0] * face["numVerts"]
            face["loopUV"] = [(0.0, 0.0)] * face["numVerts"]
            face["loopColors"] = [(0.0, 0.0, 0.0, 0.0)] * face["numVerts"] #color range (0..255)
            for i2 in range(face["numVerts"]):
                vertIndex = file_utils.readU16(data, False)
                uv = (file_utils.readFloat(data), -file_utils.readFloat(data))
                color = file_utils.readShadowManColor(data)
                face["indices"][i2] = vertIndex
                face["loopUV"][i2] = uv
                face["loopColors"][i2] = color
                loopCount += 1
            #end for
        #end for

        for i in range(model["vertCount"]):
            loc = global_matrix @ Vector(file_utils.readVector(data))
            loc.x = -loc.x
            normal = global_matrix @ Vector(file_utils.readVector(data))
            #normal = normal.normalized()
            normal = (normal[0], -normal[1], -normal[2])

            model["verts"]["loc"][i] = (loc[0], loc[1], loc[2])
            model["verts"]["normals"][i] = normal
        #end for

        model["loopNormals"] = [(0, 0, 0)] * loopCount
        loopIndex = 0
        #check if this is the end of the file
        if data.tell() >= fileByteSize:
            for face in model["faces"]:
                for i in range(face["numVerts"]):
                    model["loopNormals"][loopIndex] = model["verts"]["normals"][face["indices"][i]]
                    loopIndex += 1
                #end for
            #end for
        else:
            for face in model["faces"]:
                for i in range(face["numVerts"]):
                    normal = global_matrix @ Vector(file_utils.readVector(data))
                    # normal = normal.normalized()
                    normal = (normal[0], -normal[1], -normal[2])
                    model["loopNormals"][loopIndex] = normal
                    loopIndex += 1
                #end for
            #end for
        #end if
    #end with

    return model
#end def

# -----------------------------------------------------------------------------
def do_export(context, obj, filepath, global_matrix, useCustomNormals):
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    utils.create_BMLayers(bm)
    uv_layer = None
    vc_layer = None
    if mesh.has_custom_normals and mesh.use_auto_smooth and useCustomNormals:
        mesh.calc_normals_split() #required to get the loop.normal to work and for normals_split_custom_set to work
    #end if
    if mesh.uv_layers.active is not None:
        uv_layer = mesh.uv_layers.active.data
    #end if
    if mesh.vertex_colors.active is not None:
        vc_layer = mesh.vertex_colors.active.data
    #end if
    with ProgressReport(context.window_manager) as progress:
        stepCount = 1
        progress.enter_substeps(stepCount)
        print("Exporting Shadow Man Mesh %r ..." % filepath)

        fileDir, fileName = os.path.split(filepath)
        fileNameNoExt = os.path.splitext(fileName)[0]
        material_texture_indexs = {} #mat_index, tex_index

        scene = bpy.context.scene
        with open(filepath, 'wb') as data:
            data.seek(0)
            file_utils.writeString(data, "EMshV001", False)
            file_utils.writeU32(data, len(mesh.polygons), False)
            file_utils.writeU32(data, len(mesh.vertices), False)
            
            for face in bm.faces:
                file_utils.write8(data, len(face.loops), False)
                file_utils.write8(data, face[utils.FL_MESH_FILLMODE], False)
                file_utils.write8(data, face[utils.FL_MESH_UNKNOWN1], False)
                file_utils.write8(data, face[utils.FL_MESH_UNKNOWN2], False)
                if face.material_index in material_texture_indexs:
                    texIndex = material_texture_indexs[face.material_index]
                else:
                    texIndex = utils.get_material_texture_index(mesh.materials[face.material_index])
                #end if
                file_utils.writeU16(data, texIndex, False)
                file_utils.writeU16(data, face[utils.FL_MESH_UNKNOWN3], False)
                file_utils.writeU16(data, face[utils.FL_MESH_ATTRIBUTES], False)
                #cPlane
                plane_abc = global_matrix @ face.normal
                plane_d = -face.calc_center_median().dot(face.normal)
                file_utils.writeFloat(data, plane_abc[0]) #only need to flip the plane a value
                file_utils.writeFloat(data, -plane_abc[1])
                file_utils.writeFloat(data, -plane_abc[2])
                file_utils.writeFloat(data, -plane_d)
                
                loopIndices = [(loop.vert.index, loop.index) for loop in face.loops]
                for loopInfo in loopIndices:
                    vertIndex = loopInfo[0]
                    loopIndex = loopInfo[1]
                    file_utils.writeU16(data, vertIndex, False)
                    if uv_layer:
                        file_utils.writeFloat(data, uv_layer[loopIndex].uv[0])
                        file_utils.writeFloat(data, -uv_layer[loopIndex].uv[1])
                    else:
                        file_utils.writeFloat(data, 0.0)
                        file_utils.writeFloat(data, 0.0)
                    #end if
                    if vc_layer:
                        file_utils.writeShadowManColor(data, vc_layer[loopIndex].color)
                    else:
                        file_utils.writeShadowManColor(data, [1.0, 1.0, 1.0, 1.0])
                    #end if
                #end for
            #end for

            vertNormals = {}
            bOutputLoopNormals = False
            if mesh.has_custom_normals and mesh.use_auto_smooth and useCustomNormals:
                #if all loop vertices have the same normals then don't write loop normals to the end of the file
                for loop in mesh.loops:
                    if loop.vertex_index not in vertNormals:
                        vertNormals[loop.vertex_index] = loop.normal
                        continue
                    #end if
                    if loop.normal != vertNormals[loop.vertex_index]:
                        bOutputLoopNormals = True
                        vertNormals.clear()
                        for vert in mesh.vertices:
                            vertNormals[vert.index] = vert.normal
                        #end for
                        break
                    #end if
                #end for
            else:
                for vert in mesh.vertices:
                    vertNormals[vert.index] = vert.normal
                #end for
            #end if

            for vert in mesh.vertices:
                loc = global_matrix @ vert.co
                loc.x = -loc.x
                normal = global_matrix @ Vector(vertNormals[vert.index])
                normal = (normal[0], -normal[1], -normal[2])
                file_utils.writeVector3(data, loc)
                file_utils.writeVector3(data, normal)
            #end for

            #Export custom loop normals for ShadowManEX
            if bOutputLoopNormals:
                # vertNormals = {}
                # for loop in mesh.loops:
                #     if loop.vertex_index not in vertNormals:
                #         vertNormals[loop.vertex_index] = {}
                #         vertNormals[loop.vertex_index]["norms"] = []
                #     #end if
                #     vertNormals[loop.vertex_index]["norms"].append(loop.normal)
                # #end for

                for face in bm.faces:
                    loopIndices = [(loop.vert.index, loop.index) for loop in face.loops]
                    for loopInfo in loopIndices:
                        vertIndex = loopInfo[0]
                        loopIndex = loopInfo[1]
                        normal = global_matrix @ Vector(mesh.loops[loopIndex].normal)
                        normal = (normal[0], -normal[1], -normal[2])
                        file_utils.writeVector3(data, normal)
                    #end for
                #end for
            #end if
        #end with
        
        progress.step("Mesh Successfully Exported!")
    #end with
    bm.free()

    return {'FINISHED'}
#end def
