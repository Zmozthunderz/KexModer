import bpy
import os
import bmesh
import re
import sys
import math

from mathutils import (
    Vector,
    Quaternion,
    Matrix,
)
from shutil import copyfile
from bpy_extras.image_utils import load_image
from bpy_extras.wm_utils.progress_report import (
    ProgressReport,
    ProgressReportSubstep,
)
from bpy_extras.io_utils import (
        ImportHelper,
        ExportHelper,
        orientation_helper,
        axis_conversion,
        )
from . import (
    utils,
    file_utils,
    paths,
    defs,
    shadowman_mesh,
    shadowman_skin,
    shadowman_animation,
)

# -----------------------------------------------------------------------------
@orientation_helper(axis_forward='Z', axis_up='Y')
class SHADOWMAN_OT_export_model(bpy.types.Operator, ExportHelper):
    """Export Shadow Man Model files"""
    bl_idname = "shadowman_object.export_model"
    bl_label = 'Export Model'
    bl_options = {'PRESET', 'REGISTER', 'UNDO'}

    filename_ext = ""
    filter_glob: bpy.props.StringProperty(default="*.*", options={'HIDDEN'})

    exportMesh: bpy.props.BoolProperty(name="Export Mesh", default=True)
    exportCustomNormals: bpy.props.BoolProperty(name="Export Custom Normals", default=True, description="Appends loop normals to the mesh file (ShadowManEX only feature)")
    exportSkin: bpy.props.BoolProperty(name="Export Skin", default=True)
    exportAnims: bpy.props.BoolProperty(name="Export Anims", default=True)

    meshObj = None

    @classmethod
    def poll(cls, context):
        obj = context.object
        if obj is None:
            return False
        #end if

        armObj = utils.get_root_sm_object2(obj, "MODEL", 'ARMATURE')
        #obj is a mesh or armature or has root armature shadowman model object
        return obj.type == "MESH" or obj.type == "ARMATURE" or armObj != obj
    #end def
    
    def execute(self, context):
        utils.mode_set('OBJECT')
        global_matrix = utils.global_export_matrix()

        success = export_model(context, self.filepath, global_matrix, self.exportMesh, self.exportSkin, self.exportAnims, self.exportCustomNormals)

        return {'FINISHED'} if success else {'CANCELLED'}
    #end def
        
    def invoke(self, context, event):
        obj = context.object
        armObj = utils.get_root_sm_object2(obj, "MODEL", 'ARMATURE')
        self.filepath = armObj.name
        utils.mode_set('OBJECT')
        results = invoke_validation(context, obj, "Export Model")
        if len(results["errors"]) > 0:
            return {'CANCELLED'}
        #end if
        if armObj.type == "ARMATURE":
            self.meshObj = utils.find_armature_mesh(armObj)
        else:
            self.meshObj = armObj
        #end if
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    #end def

    def draw(self, context):
        layout = self.layout
        obj = utils.get_root_sm_object2(context.object, "MODEL", 'ARMATURE')
        hasCustomNormals = False
        if self.meshObj is not None:
            hasCustomNormals = self.meshObj.data.has_custom_normals and self.meshObj.data.use_auto_smooth
        #end if
        if obj.type == "ARMATURE":
            row = layout.row()
            row.prop(self, "exportMesh")
            if hasCustomNormals:
                row = layout.row()
                row.separator()
                row.prop(self, "exportCustomNormals")
            #end if
            row = layout.row()
            row.prop(self, "exportSkin")
            row = layout.row()
            row.prop(self, "exportAnims")
        else:
            if hasCustomNormals:
                row = layout.row()
                row.prop(self, "exportCustomNormals")
            #end if
        #end if
    #end def
#end class

# -----------------------------------------------------------------------------
# 
def do_import(context, filepath, texDirs, skinPath, global_matrix, forceObjType="MODEL", isSilent=False):
    """Returns the model object, skin(or None) if successful else None, None"""

    if not os.path.isfile(filepath):
        utils.show_error("[Import Mesh] mesh path %s not found." % (filepath))
        return None, None
    #end if
    if skinPath and not os.path.isfile(skinPath):
        utils.show_error("[Import Mesh] skin path %s not found." % (skinPath))
        return None, None
    #end if

    meshObj = None
    armObj = None
    skin = None
    if isSilent:
        progress = utils.dummy_context_manager()
    else:
        progress = ProgressReport(context.window_manager)
    #end if
    with progress:
        if not isSilent:
            progress.enter_substeps(1)
            print("Importing Shadow Man Mesh %r ..." % (filepath))
            if skinPath:
                print("Importing Shadow Man Skin %r ..." % (skinPath))
            #end if
        #end if

        fileName = os.path.split(filepath)[1]
        fileNameNoExt = os.path.splitext(fileName)[0]
        scene = bpy.context.scene
        
        model = shadowman_mesh.load(filepath, global_matrix)
        if model is None:
            return None, None
        #end if

        if skinPath:
            skin = shadowman_skin.load(skinPath, global_matrix)
            if skin is None:
                return None, None
            #end if

            #Make sure this skin belongs to this mesh
            if len(model["verts"]["loc"]) != len(skin["indices"]):
                utils.show_error("[Import Mesh] skin %s vert indice count of %d doesn't match mesh vert count of %d" % (skinPath, len(model["verts"]["loc"]), len(skin["indices"])))
                return None, None
            #end if
            if len(model["verts"]["loc"]) != len(skin["verts"]):
                utils.show_error("[Import Mesh] skin %s vert count of %d doesn't match mesh vert count of %d" % (skinPath, len(model["verts"]["loc"]), len(skin["verts"])))
                return None, None
            #end if
            for i in range(len(model["verts"]["loc"])):
                idx = utils.index_of(skin["indices"], i)
                if idx == -1:
                    utils.show_error("[Import Mesh] skin %s has no entry for vertex %d" % (skinPath, i))
                    return None, None
                #end if
            #end for
            
        #end if
        
        #get all the texture indexs that the model uses and only load those materials
        usedTextureIndexs = set()
        for face in model["faces"]:
            usedTextureIndexs.add(face["texIndex"])
        #end for

        materialsInfo = {} #textureIndex, {"mat":material, mat_index:0}
        #get all the textures in the texture directory (just filename.ext)
        for texDir in texDirs:
            texFileNames = file_utils.get_files_in_dir(texDir, utils.VALID_TEXTURE_EXTENSIONS, True)
            for filename in texFileNames:
                index = utils.get_filename_texture_index(filename)[0]
                #create blender materials for the new texture index
                if index in usedTextureIndexs and materialsInfo.get(index) is None:
                    filenameNoExt = os.path.splitext(filename)[0]
                    texPath = os.path.join(texDir, filename)
                    mat = utils.get_material(filenameNoExt, texPath)
                    materialsInfo[index] = { "mat": mat }
                    utils.setup_material(mat, texPath)
                #end if
            #end for
        #end for

        for texIndex in usedTextureIndexs:
            if materialsInfo.get(texIndex) is None:
                mat = utils.get_material("%03d Missing" % (texIndex))
                materialsInfo[texIndex] = { "mat": mat }
            #end if
        #end if

        mesh = bpy.data.meshes.new('mesh')
        for texIndex in sorted(materialsInfo):
            materialsInfo[texIndex]["mat_index"] = len(mesh.materials)
            mesh.materials.append(materialsInfo[texIndex]["mat"])
        #end for

        meshFaceIndices = [face["indices"] for face in model["faces"]]
        set_model_mesh_geo(mesh, model, materialsInfo, model["verts"]["loc"], meshFaceIndices)
        
        # create scene objects
        if skin:
            meshObj = utils.create_shadowman_object(fileNameNoExt + "_mesh", mesh, forceObjType)

            # create armature object
            armature = bpy.data.armatures.new(name='Armature')
            armObj = utils.create_shadowman_object(fileNameNoExt, armature, forceObjType)
            armObj.data.display_type = 'STICK'
            armObj.show_in_front = True

            meshObj.parent = armObj
            meshObj.parent_type = 'OBJECT'

            #create shape keys for new skin vertex positions
            sk_basis = meshObj.shape_key_add(name='Basis')
            sk_basis.interpolation = 'KEY_LINEAR'
            mesh.shape_keys.use_relative = True

            # Create shape key for skin verts
            sk = meshObj.shape_key_add(name='Skin')
            sk.interpolation = 'KEY_LINEAR'
            sk.value = 1.0
            sk.mute = False
            for i in range(len(mesh.vertices)):
                sk.data[i].co = skin["verts"][i]
            #end for

            # create armature bones
            bpy.context.view_layer.objects.active = armObj
            if len(skin["bones"]) > 0:
                utils.mode_set('EDIT')

                #Create edit bones
                boneNames = {} #boneindex, name
                for i in range(len(skin["bones"])):
                    editBone = utils.create_editbone(armature, utils.get_bonename(i))
                    boneNames[i] = editBone.name
                #end for

                #Create edit bones for softs
                softBoneNames = {} #softBoneindex, name
                for i in range(len(skin["softBones"])):
                    editBone = utils.create_editbone(armature, utils.get_softbonename(i))
                    softBoneNames[i] = editBone.name
                #end for

                #Parent edit bones
                for i in range(len(skin["bones"])):
                    parentBoneIndex = skin["bones"][i]["parent"]
                    if parentBoneIndex < 0: #don't parent root bones
                        continue
                    #end if
                    curBoneName = boneNames[i]
                    parentBoneName = boneNames[parentBoneIndex]
                    curBone = armature.edit_bones[curBoneName]
                    curBone.parent = armature.edit_bones[parentBoneName]
                #end for

                #Setup vertex groups
                vertGroups = {} #key is bone index, value is vert group (with it's name as the bone)
                vertToBonesLoopup = [-1] * len(skin["indices"]) #vertIndex (from skin indices), boneIndex
                for boneIndex, bone in enumerate(skin["bones"]):
                    for i in range(bone["nHards"]):
                        vertIndex = skin["indices"][bone["hardi"] + i] #vertex that belongs to this bone
                        vertToBonesLoopup[vertIndex] = boneIndex
                        if boneIndex not in vertGroups:
                            vertGroups[boneIndex] = {"group":meshObj.vertex_groups.new(name=boneNames[boneIndex]), "verts":[], "isSoft":False}
                        #end if
                        vertGroups[boneIndex]["verts"].append(vertIndex)
                    #end for
                    for i in range(bone["nSoftTypes"]):
                        softIndex = bone["softTypei"] + i
                        soft = skin["softBones"][softIndex]
                        for i2 in range(soft["nSofts"]):
                            vertIndex = skin["indices"][soft["softi"] + i2] #vertex that belongs to this softs bone
                            vertToBonesLoopup[vertIndex] = boneIndex

                            softBoneIndex = len(skin["bones"]) + softIndex
                            if softBoneIndex not in vertGroups:
                                vertGroups[softBoneIndex] = {"group":meshObj.vertex_groups.new(name=softBoneNames[softIndex]), "verts":[], "isSoft":True, "weight":soft["weight"]}
                            #end if
                            vertGroups[softBoneIndex]["verts"].append(vertIndex)
                            pass
                        #end for
                    #end for
                #end for

                #assign any verts that weren't assigned to a bone to bone 0
                for vertIndex, boneIndex in enumerate(vertToBonesLoopup):
                    if boneIndex == -1:
                        if 0 not in vertGroups:
                            vertGroups[0] = {"group":meshObj.vertex_groups.new(name=boneNames[0]), "verts":[], "isSoft":False}
                        #end if
                        vertGroups[0]["verts"].append(vertIndex)
                    #end if
                #end for

                #Add Vertices to vertex groups all at once
                for key in vertGroups:
                    if vertGroups[key]["isSoft"]:
                        vertGroups[key]["group"].add(vertGroups[key]["verts"], 1.0, 'ADD')
                    else:
                        vertGroups[key]["group"].add(vertGroups[key]["verts"], 1.0, 'ADD')
                    #end if
                #end for

                #Exit Armature editing so the edit bone is created internally at this point
                utils.mode_set('OBJECT')

                utils.mode_set('POSE')

                #Create soft bone constraints
                for i in range(len(skin["bones"])):
                    skinBone = skin["bones"][i]
                    poseBone = armObj.pose.bones[boneNames[i]]
                    for i2 in range(skinBone["nSoftTypes"]):
                        softBoneIndex = skinBone["softTypei"] + i2
                        skinSoftBone = skin["softBones"][softBoneIndex]
                        poseSoftBone = armObj.pose.bones[softBoneNames[softBoneIndex]]
                        armBone = armature.bones[softBoneNames[softBoneIndex]]
                        armBone.smex.attachedBone = boneNames[i]

                        utils.setup_soft_bone_constraints(poseSoftBone, poseBone, armObj, skinSoftBone["weight"])
                    #end for
                #end for

                utils.mode_set('OBJECT')
            #end if

            modArm = meshObj.modifiers.new("Armature", 'ARMATURE')
            modArm.show_in_editmode = False
            modArm.show_on_cage = False
            modArm.object = armObj

            #Setup Shadow Man Bone Properties
            for skinBoneIndex, skinBone in enumerate(skin["bones"]):
                bone = armature.bones[boneNames[skinBoneIndex]]
                bone.smex.boneIndex = skinBoneIndex
            #end for

            for softBoneIndex, softBone in enumerate(skin["softBones"]):
                bone = armature.bones[softBoneNames[softBoneIndex]]
                bone.smex.isSoft = True
                bone.smex.softWeight = softBone["weight"]
                bone.smex.softMatrixRow1 = softBone["matrix"][0]
                bone.smex.softMatrixRow2 = softBone["matrix"][1]
                bone.smex.softMatrixRow3 = softBone["matrix"][2]
                bone.smex.softMatrixRow4 = softBone["matrix"][3]
            #end for

            #Select only the armature object
            for ob in bpy.context.selected_objects:
                ob.select_set(False)
            #end for
            
            # if utils.prefs().smoothShadeImport:
            #     meshObj.select_set(True)
            #     bpy.ops.object.shade_smooth()
            #     meshObj.select_set(False)
            # #end if

            armObj.select_set(True)
        else: #it's only a mesh
            meshObj = utils.create_shadowman_object(fileNameNoExt, mesh, forceObjType)
            if not isSilent:
                utils.select_object(meshObj)
            #end if
            # if utils.prefs().smoothShadeImport:
            #     bpy.ops.object.shade_smooth()
            # #end if
        #end if
        if not isSilent:
            progress.leave_substeps("Finished! (verts:%i faces:%i materials:%i)" % (model["vertCount"], len(model["faces"]), len(materialsInfo)))
        #end if
    #end with
    
    return (armObj if armObj else meshObj), skin
#end def

# -----------------------------------------------------------------------------
# 
def set_model_mesh_geo(mesh, model, materialsInfo, vertexLocations, faceIndices, printWarnings = False):
    if len(model["faces"]) <= 0 or model["vertCount"] < 3:
        return
    #end if

    mesh.from_pydata(vertexLocations, [], faceIndices)
    # bInvalidGeoFixed = mesh.validate(verbose=False, clean_customdata=False)
    # if bInvalidGeoFixed:
    #     mesh.clear_geometry()
    #     mesh.from_pydata(vertexLocations, [], faceIndices)
    # else:
    #     mesh.use_auto_smooth = True #required to show custom normals and for for loop.normal to work
    # #end if

    mesh.use_auto_smooth = True #required to show custom normals and for for loop.normal to work
    mesh.calc_normals_split() #required to get the loop.normal to work and for normals_split_custom_set to work
    #mesh.normals_split_custom_set_from_vertices(model["verts"]["normals"]) #set custom normals here
    mesh.normals_split_custom_set(model["loopNormals"]) #set custom normals here
    mesh.update(calc_edges=True)

    blMesh = bmesh.new()
    blMesh.from_mesh(mesh)
    utils.create_BMLayers(blMesh)

    for faceIndex, bFace in enumerate(blMesh.faces):
        modelFace = model["faces"][faceIndex]
        bFace[utils.FL_MESH_FILLMODE] = modelFace["fillMode"]
        bFace[utils.FL_MESH_ATTRIBUTES] = modelFace["attributes"]
        bFace[utils.FL_MESH_UNKNOWN1] = modelFace["unknown1"]
        bFace[utils.FL_MESH_UNKNOWN2] = modelFace["unknown2"]
        bFace[utils.FL_MESH_UNKNOWN3] = modelFace["unknown3"]
        bFace.material_index = materialsInfo[modelFace["texIndex"]]["mat_index"]
    #end for
    
    blMesh.to_mesh(mesh)
    blMesh.free()

    # create uv layer and set uv values
    uv_layer = mesh.uv_layers.new()
    vc_layer = mesh.vertex_colors.new()

    for polyIndex, poly in enumerate(mesh.polygons):
        polyIndices = tuple([mesh.loops[loopIndex].vertex_index for loopIndex in range(poly.loop_start, poly.loop_start + poly.loop_total)])
        matchFace = model["faces"][poly.index]

        if matchFace is not None:
            modelLoopIndex = 0
            for loopIndex in range(poly.loop_start, poly.loop_start + poly.loop_total):
                uv_layer.data[loopIndex].uv = matchFace["loopUV"][modelLoopIndex]

                vc = matchFace["loopColors"][modelLoopIndex]
                vc_layer.data[loopIndex].color = (vc[0] / 255, vc[1] / 255, vc[2] / 255, vc[3] / 255)
                modelLoopIndex += 1
            #end for
        else:
            print("Poly %s indices %s was not found in model faces" % (poly.index, polyIndices))
        #end if
    #end for

#end def

# -----------------------------------------------------------------------------
# Returns True if succeeded
# filepath should have no extenstion
def export_model(context, filepath, global_matrix, exportMesh, exportSkin, exportAnims, useCustomNormals):
    filePathNoExt = os.path.splitext(filepath)[0]
    meshPath = filePathNoExt + ".msh"
    skinPath = filePathNoExt + ".skn"
    animsPath = filePathNoExt + ".anims"
    obj = utils.get_root_sm_object2(context.object, "MODEL", 'ARMATURE')

    if obj.type == "MESH":
        shadowman_mesh.do_export(context, obj, meshPath, global_matrix, useCustomNormals)
    elif obj.type == "ARMATURE":
        meshObj = utils.find_armature_mesh(obj)
        if meshObj is not None:
            if exportMesh:
                shadowman_mesh.do_export(context, meshObj, meshPath, global_matrix, useCustomNormals)
            #end if
            if exportSkin:
                shadowman_skin.do_export(context, obj, meshObj, skinPath, global_matrix)
            #end if
            if exportAnims:
                #export animations if has any animations
                actions = utils.get_actions(obj)
                if len(actions) > 0:
                    shadowman_animation.do_export(context, obj, animsPath, global_matrix)
                #end if
            #end if
        #end if
    #end if
#end def

# -----------------------------------------------------------------------------
def invoke_validation(context, obj, msgPrefix):
    """Invokes the validation process showing messages in console and in popup window
    Parameters
    ----------
    context : current blender context
    obj : the blender object
    string msgPrefix : the text to show before each message

    Returns
    -------
    results : dictionary containing keys: errors, warnings, info
    """

    validateResults = validate_model(context, obj)
    if len(validateResults["errors"]) > 0 and len(validateResults["warnings"]) > 0:
        utils.show_error("%s" % (validateResults["errors"][0]), "Errors and Warnings - See System Console for full details", toConsole=False)
    elif len(validateResults["errors"]) > 0:
        utils.show_error("%s" % (validateResults["errors"][0]), "Error - See System Console for full details", toConsole=False)
    elif len(validateResults["warnings"]) > 0:
        utils.show_error("%s" % (validateResults["warnings"][0]), "Warning - See System Console for full details", toConsole=False)
    else:
        utils.show_message("Everything's good!", msgPrefix, toConsole=False)
    #end if
    if len(validateResults["errors"]) > 0:
        for message in validateResults["errors"]:
            print("[%s Error] %s" % (msgPrefix, message))
        #end for
        return validateResults
    #end if
    if len(validateResults["warnings"]) > 0:
        for message in validateResults["warnings"]:
            print("[%s Warning] %s" % (msgPrefix, message))
        #end for
    #end if

    return validateResults
#end def

# -----------------------------------------------------------------------------
def validate_model(context, obj):
    """Validates the object model to make sure it's ok to be exported
    Parameters
    ----------
    context : current blender context
    obj : the blender object

    Returns
    -------
    results : dictionary containing keys: errors, warnings, info
    """
    result = { "errors": [], "warnings": [], "info": [] }
    
    #Error: Check if object is not None
    if obj is None:
        result["errors"].append("Object is None")
        return result
    #end if
    
    meshObj = None
    if obj.type == "MESH":
        meshObj = obj
    elif obj.type == "ARMATURE":
        validate_armature(context, obj, result)
        meshObj = utils.find_armature_mesh(obj)
        if meshObj is None:
            return result
        #end if
    else:
        result["errors"].append("Object is not a Mesh or Armature")
        return result
    #end if

    validate_mesh(context, meshObj, result)
    
    return result
#end def

# -----------------------------------------------------------------------------
def validate_mesh(context, obj, result):
    material_indexs = set() #material indexes used
    for poly in obj.data.polygons:
        material_indexs.add(poly.material_index)
    #end for

    #Error: Check if textures use an index at the start of there names
    for matIndex in material_indexs:
        mat = obj.data.materials[matIndex]
        if mat is None:
            result["errors"].append("Object \"%s\", No material is set in slot index %d" % (obj.name, matIndex))
            continue
        #end if
    #end for

    #Error: Make sure each mesh has only 3 vertices per poly.
    for poly in obj.data.polygons:
        if poly.loop_total != 3:
            result["errors"].append("Object \"%s\" has %d verts per poly. Each poly must only have 3 verts. Triangulate the mesh." % (obj.name, poly.loop_total))
            break
        #end if
    #end for

    if obj.matrix_world != Matrix():
        result["errors"].append("Object \"%s\" must have it's location and rotation set to 0 and must be a scale of 1. Make sure you applied all transforms to the mesh. Ctrl + A > All Transforms" % (obj.name))
    #end if
#end def

# -----------------------------------------------------------------------------
def validate_armature(context, obj, result):
    #make sure armature has a mesh
    meshObj = utils.find_armature_mesh(obj)
    if meshObj is None:
        result["errors"].append("Armature \"%s\" has no shadow man model mesh object parented to it" % (obj.name))
        return
    #end if

    if obj.matrix_world != Matrix():
        result["errors"].append("Armature \"%s\" must have it's location and rotation set to 0 and must be a scale of 1." % (obj.name))
    #end if

    #check bones are valid
    for poseBone in obj.pose.bones:
        bone = obj.data.bones[poseBone.name]
        if bone.smex.isSoft:
            softIsValid = utils.validate_soft_bone(obj, obj.pose.bones[bone.name])
            if not softIsValid:
                result["errors"].append("Object \"%s\", Bone \"%s\" is an invalid soft bone." % (obj.name, poseBone.name))
            #end if
        else:
            #make sure not parented to any soft bones
            if bone.parent and bone.parent.smex.isSoft:
                result["errors"].append("Object \"%s\", Bone \"%s\" can not be parented to soft bone \"%s\"." % (obj.name, bone.name, bone.parent.name))
            #end if
            if not utils.hard_bone_index_is_valid(obj, bone):
                result["errors"].append("Object \"%s\", Bone \"%s\" has a invalid bone index." % (obj.name, bone.name))
            #end if
        #end if
    #end for

    #Error: Make sure each vertice is in only 1 bone vertex group and it's weight is 1.0
    boneVertGroups = {vertGroup.index for bone in obj.data.bones for vertGroup in meshObj.vertex_groups if vertGroup.name == bone.name}
    for vertIndex, vert in enumerate(meshObj.data.vertices):
        vertGroupsIn = [vg for vg in vert.groups if vg.group in boneVertGroups]
        if len(vertGroupsIn) == 0:
            result["errors"].append("Mesh \"%s\", Vert %d, is not in any Bone Vertex Groups." % (meshObj.name, vert.index))
            break
        elif len(vertGroupsIn) > 1:
            result["errors"].append("Mesh \"%s\", Vert %d, is in more than 1 Bone Vertex Groups. Shadow Man Only Supports 1 Bone per vertice." % (meshObj.name, vert.index))
            break
        else:
            vgName = meshObj.vertex_groups[vertGroupsIn[0].group].name
            if vertGroupsIn[0].weight != 1.0:
                result["errors"].append("Mesh \"%s\", Vert %d, has a weight of %s in vertex group \"%s\". It must have a weight of 1.0." % (meshObj.name, vert.index, vertGroupsIn[0].weight, vgName))
                break
            #end if
        #end if
    #end for

    #check animations are valid
    actions = utils.get_actions(obj)
    if len(actions) == 0:
        result["errors"].append("Object \"%s\", has no animation actions. Make sure you named your actions correctly with the armature name then the animation index and then the animation name. eg. 'Fish 000 chubswim'" % (obj.name))
    #end if

    hardBones, softBones = utils.get_bones(obj)
    for action in actions:
        frameStart, frameEnd = action.frame_range
        frameStart = int(frameStart)
        frameEnd = int(frameEnd)
        frameCount = (frameEnd - frameStart) + 1
        actionHardBoneInfo = {bone.name:{"isLocSet":[False] * 3} for bone in hardBones}

        if not utils.anim_action_index_is_valid(obj, action):
            result["warnings"].append("Object \"%s\", Animation Action \"%s\" has an invalid animation index of %i" % (obj.name, action.name, utils.get_anim_index(obj, action)))
        #end if

        #make sure key frames don't include soft bones
        for f in action.fcurves:
            propObjPath, propDot, propType = f.data_path.rpartition('.')
            propObj = obj.path_resolve(propObjPath)
            if type(propObj) is bpy.types.PoseBone:
                fBone = obj.data.bones[propObj.name]
                if fBone.smex.isSoft:
                    result["errors"].append("Object \"%s\", Animation Action \"%s\" has an fcurve \"%s\" with a soft bone \"%s\"." % (obj.name, action.name, f.data_path, fBone.name))
                    break
                #end if
                if propType == "location" and fBone.name in actionHardBoneInfo:
                    for keyPoint in f.keyframe_points:
                        fFrame = keyPoint.co[0]
                        frame = int(fFrame) # - frameStart
                        if frame == 0:
                            actionHardBoneInfo[fBone.name]["isLocSet"][f.array_index] = True
                        #end if
                    #end for
                #end if
            #end if
        #end for

        #Check if location xyz components at frame 0 exist for all bones in this animation (which are required for the games anim bone translation)
        for boneName in actionHardBoneInfo:
            isLocSet = actionHardBoneInfo[boneName]["isLocSet"]
            if utils.index_of(isLocSet, False) != -1:
                result["errors"].append("Object \"%s\", Animation Action \"%s\" is missing location fcurves for the first frame, for bone \"%s\"." % (obj.name, action.name, boneName))
            #end if
        #end for
    #end for
    
#end def

# -----------------------------------------------------------------------------
def menu_func_export_shadowman_model(self, context):
    self.layout.operator(SHADOWMAN_OT_export_model.bl_idname, text="Shadow Man (.msh, .skn, .anims)")
#end def

# -----------------------------------------------------------------------------
def register():
    bpy.utils.register_class(SHADOWMAN_OT_export_model)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export_shadowman_model)
#end def

# -----------------------------------------------------------------------------
def unregister():
    bpy.utils.unregister_class(SHADOWMAN_OT_export_model)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export_shadowman_model)
#end def
