import bpy
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
def load(filepath, global_matrix):
    """Returns dictionary of skin structure. Returns None if there was an error."""

    skin = {}
    with open(filepath, 'rb') as data:
        data.seek(0)
        sFileType = file_utils.readStringCount(data, 4) #'BSKN'
        if sFileType != 'BSKN':
            utils.show_error("[Import Skin] %s is not a valid skin file." % (filepath))
            return None
        #end if
        skin["boneCount"] = file_utils.read32(data, False)
        file_utils.read32(data, False) #'HRCY'
        skin["bones"] = []
        for i in range(skin["boneCount"]):
            bone = {
                "nHards": 0,
                "nSoftTypes": 0,
                "hardi": 0,
                "softTypei": 0,
                "parent": -1,
            }
            skin["bones"].append(bone)
            bone["parent"] = file_utils.read32(data, True)
        #end for
        for i in range(skin["boneCount"]):
            bone = skin["bones"][i]
            file_utils.read32(data, False) #'BONE'
            bone["nHards"] = file_utils.read16(data, False)
            bone["nSoftTypes"] = file_utils.read16(data, False)
            bone["hardi"] = file_utils.read16(data, False)
            bone["softTypei"] = file_utils.read16(data, False)
        #end for
        file_utils.read32(data, False) #'SOFT'
        skin["softBoneCount"] = file_utils.read32(data, False)
        skin["softBones"] = []
        for i in range(skin["softBoneCount"]):
            softBone = {
                "weight": 0.0,
                "nSofts": 0,
                "softi": 0,
                "matrix": [[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]]
            }
            skin["softBones"].append(softBone)
            file_utils.read32(data, False) #'STYP'
            softBone["weight"] = float(file_utils.read16(data, False) / 65535.0)
            softBone["nSofts"] = file_utils.read16(data, False)
            softBone["softi"] = file_utils.read16(data, False)
            softBone["matrix"][0][0] = file_utils.readFloat(data)
            softBone["matrix"][0][1] = file_utils.readFloat(data)
            softBone["matrix"][0][2] = file_utils.readFloat(data)
            softBone["matrix"][1][0] = file_utils.readFloat(data)
            softBone["matrix"][1][1] = file_utils.readFloat(data)
            softBone["matrix"][1][2] = file_utils.readFloat(data)
            softBone["matrix"][2][0] = file_utils.readFloat(data)
            softBone["matrix"][2][1] = file_utils.readFloat(data)
            softBone["matrix"][2][2] = file_utils.readFloat(data)
            softBone["matrix"][3][0] = file_utils.readFloat(data)
            softBone["matrix"][3][1] = file_utils.readFloat(data)
            softBone["matrix"][3][2] = file_utils.readFloat(data)
            #get rid of nan values
            for i in range(len(softBone["matrix"])):
                for j in range(len(softBone["matrix"][i])):
                    if math.isnan(softBone["matrix"][i][j]):
                        softBone["matrix"][i][j] = 0.0
                    #end if
                #end for
            #end for
        #end for
        file_utils.read32(data, False) #'BIND'
        skin["indiceCount"] = file_utils.read16(data, False)
        skin["indices"] = [0] * skin["indiceCount"] #elements are the hard and soft index lookup, and value is the mesh vertex index
        for i in range(skin["indiceCount"]):
            skin["indices"][i] = file_utils.read16(data, False)
        #end for

        #these verts overwrite the position of the mesh verts and must have the same amount of vertices as the .mesh
        file_utils.read32(data, False) #'VERT'
        skin["vertCount"] = file_utils.read32(data, False)
        skin["verts"] = [(0.0, 0.0, 0.0)] * skin["vertCount"]
        skin["vertsOriginal"] = [(0.0, 0.0, 0.0)] * skin["vertCount"]
        for i in range(skin["vertCount"]):
            skin["vertsOriginal"][i] = file_utils.readVector(data)
            loc = global_matrix @ Vector(skin["vertsOriginal"][i])
            skin["verts"][i] = (-loc[0], loc[1], loc[2])
        #end for
    #end with

    return skin
#end def

# -----------------------------------------------------------------------------
def do_export(context, armObj, meshObj, filepath, global_matrix):
    armature = armObj.data
    hardBones, softBones = utils.get_bones(armObj)
    boneHardVertGroups = {vertGroup.index:bone for bone in hardBones for vertGroup in meshObj.vertex_groups if vertGroup.name == bone.name}
    bonesInfo = {} #bone index, {}
    usedSofts = []
    lastHardBoneIndex = 0
    for hardBoneIndex, hardBone in enumerate(hardBones):
        bonesInfo[hardBoneIndex] = { "hardi": 0, "verts": [], "nSoftTypes": 0, "softTypei": 0 }
        if hardBoneIndex > 0:
            bonesInfo[hardBoneIndex]["hardi"] = bonesInfo[hardBoneIndex-1]["hardi"] + len(bonesInfo[hardBoneIndex-1]["verts"])
            bonesInfo[hardBoneIndex]["softTypei"] = bonesInfo[hardBoneIndex-1]["softTypei"] + bonesInfo[hardBoneIndex-1]["nSoftTypes"]
        #end if
        for vertIndex, vert in enumerate(meshObj.data.vertices):
            for vg in vert.groups:
                if vg.group in boneHardVertGroups and boneHardVertGroups[vg.group].name == hardBone.name:
                    bonesInfo[hardBoneIndex]["verts"].append(vert.index)
                    lastHardBoneIndex += 1
                #end if
            #end for
        #end for
        #check how many softs this bone uses
        for softBone in softBones:
            if softBone.smex.attachedBone == hardBone.name:
                usedSofts.append(softBone)
                bonesInfo[hardBoneIndex]["nSoftTypes"] += 1
            #end if
        #end for
    #end for

    boneSoftVertGroups = {vertGroup.index:bone for bone in usedSofts for vertGroup in meshObj.vertex_groups if vertGroup.name == bone.name}
    softBonesInfo = {} #soft bone index, {}
    for softBoneIndex, softBone in enumerate(usedSofts):
        softBonesInfo[softBoneIndex] = { "softi": lastHardBoneIndex, "verts": [] }
        if softBoneIndex > 0:
            softBonesInfo[softBoneIndex]["softi"] = softBonesInfo[softBoneIndex-1]["softi"] + len(softBonesInfo[softBoneIndex-1]["verts"])
        #end if
        for vertIndex, vert in enumerate(meshObj.data.vertices):
            for vg in vert.groups:
                if vg.group in boneSoftVertGroups and boneSoftVertGroups[vg.group].name == softBone.name:
                    softBonesInfo[softBoneIndex]["verts"].append(vert.index)
                #end if
            #end for
        #end for
    #end for

    indices = []
    for i, b in bonesInfo.items():
        for vi in b["verts"]:
            indices.append(vi)
        #end for
    #end for
    for i, b in softBonesInfo.items():
        for vi in b["verts"]:
            indices.append(vi)
        #end for
    #end for

    #get shape key that has "skin" in the name. if is none then use mesh verts
    shapeKeys = [shapeKey for shapeKey in meshObj.data.shape_keys.key_blocks]
    skinShapeKey = None
    for shapeKey in shapeKeys:
        if "skin" in shapeKey.name.lower():
            skinShapeKey = shapeKey
            break
        #end if
    #end for

    with ProgressReport(context.window_manager) as progress:
        stepCount = 1
        progress.enter_substeps(stepCount)
        print("Exporting Shadow Man Skin %r ..." % filepath)

        with open(filepath, 'wb') as data:
            data.seek(0)
            file_utils.writeString(data, 'BSKN', False)
            file_utils.write32(data, len(hardBones), False)
            file_utils.writeString(data, 'HRCY', False)

            for i in range(len(hardBones)):
                parentIndex = -1
                if hardBones[i].parent:
                    parentIndex = utils.index_of(hardBones, hardBones[i].parent)
                #end if
                file_utils.write32(data, parentIndex, True)
            #end for
            for i, hardBone in enumerate(hardBones):
                file_utils.writeString(data, 'BONE', False)
                file_utils.write16(data, len(bonesInfo[i]["verts"]), False) #number of hard vertices
                file_utils.write16(data, bonesInfo[i]["nSoftTypes"], False) #number of softs connected to this bone
                file_utils.write16(data, bonesInfo[i]["hardi"], False) #start hard vertice count
                file_utils.write16(data, bonesInfo[i]["softTypei"], False) #starting soft bone index connected to this bone
            #end for
            file_utils.writeString(data, 'SOFT', False)
            file_utils.write32(data, len(usedSofts), False)
            for i, soft in enumerate(usedSofts):
                smSoft = soft.smex
                file_utils.writeString(data, 'STYP', False)
                file_utils.write16(data, int(utils.clamp(smSoft.softWeight, 0.0, 1.0) * 65535), False)
                file_utils.write16(data, len(softBonesInfo[i]["verts"]), False)
                file_utils.write16(data, softBonesInfo[i]["softi"], False)
                file_utils.writeFloat(data, smSoft.softMatrixRow1[0])
                file_utils.writeFloat(data, smSoft.softMatrixRow1[1])
                file_utils.writeFloat(data, smSoft.softMatrixRow1[2])
                file_utils.writeFloat(data, smSoft.softMatrixRow2[0])
                file_utils.writeFloat(data, smSoft.softMatrixRow2[1])
                file_utils.writeFloat(data, smSoft.softMatrixRow2[2])
                file_utils.writeFloat(data, smSoft.softMatrixRow3[0])
                file_utils.writeFloat(data, smSoft.softMatrixRow3[1])
                file_utils.writeFloat(data, smSoft.softMatrixRow3[2])
                file_utils.writeFloat(data, smSoft.softMatrixRow4[0])
                file_utils.writeFloat(data, smSoft.softMatrixRow4[1])
                file_utils.writeFloat(data, smSoft.softMatrixRow4[2])
            #end for
            file_utils.writeString(data, 'BIND', False)
            file_utils.write16(data, len(indices), False)
            for i in range(len(indices)):
                file_utils.write16(data, indices[i], False)
            #end for
            file_utils.writeString(data, 'VERT', False)
            file_utils.write32(data, len(meshObj.data.vertices), False)
            if skinShapeKey:
                for i in range(len(meshObj.data.vertices)):
                    loc = global_matrix @ Vector(skinShapeKey.data[i].co)
                    file_utils.writeVector3(data, (-loc[0], loc[1], loc[2]))
                #end for
            else:
                for vert in enumerate(meshObj.data.vertices):
                    loc = global_matrix @ Vector(vert.co)
                    file_utils.writeVector3(data, (-loc[0], loc[1], loc[2]))
                #end for
            #end if
        #end with
        progress.step("Skin Successfully Exported! (Hard Bones:%i Soft Bones:%i" % (len(hardBones), len(usedSofts)))
    #end with

    return {'FINISHED'}
#end def
