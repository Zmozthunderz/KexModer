import bpy
import math
import os
from mathutils import (
    Vector,
    Quaternion,
    Euler,
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
def load(filepath, global_matrix, skin):
    """Returns an array of a dictionary of an animation structure. Returns None if there was an error."""

    anims = []
    with open(filepath, 'rb') as data:
        data.seek(0)
        sFileType = file_utils.readStringCount(data, 4) #'mnAE' 'EAnm'
        if sFileType != 'mnAE':
            utils.show_error("[Import Anims] %s is not a valid animation package" % (filepath))
            return None
        #end if

        animCount = file_utils.read32(data, True)
        for i in range(animCount):
            anim = {
                "index": 0,
                "name": "",
                "numBones": 0,
                "numFrames": 0,
                "bones": [],
            }
            anims.append(anim)
            anim["index"] = file_utils.read32(data, False) #byte offset to each anim in this file. Not needed and also incorrect.
        #end for
        for i in range(animCount):
            # print("Anim %i: %i, %i, dif:%i" % (i, data.tell(), anims[i]["index"], anims[i]["index"] - data.tell()))

            anim = anims[i]
            anim["numBones"] = file_utils.read32(data, True)
            anim["name"] = file_utils.readString(data)
            anim["numFrames"] = file_utils.read32(data, True)
            anim["bones"] = []
            for i2 in range(anim["numBones"]):
                animBone = {
                    "transOffset": (0.0, 0.0, 0.0), #offset all the pose bone keyframes
                    "trans": {}, #key is frame index, value is position
                    "transOriginal": [],
                    "rots": {},
                    "rotsOriginal": [],
                }
                anim["bones"].append(animBone)
                animBone["fUnknown1"] = file_utils.readFloat(data) #unused
                boneOffsetX = file_utils.readFloat(data)
                boneOffsetY = file_utils.readFloat(data)
                boneOffsetZ = file_utils.readFloat(data)
                animBone["transOffset"] = (boneOffsetX, boneOffsetY, boneOffsetZ)
                transKeyCount = file_utils.read32(data, True) #number of used trans keys for this bone
                for i3 in range(transKeyCount):
                    frame = file_utils.read32(data, True) - 1
                    x = file_utils.readFloat(data)
                    y = file_utils.readFloat(data)
                    z = file_utils.readFloat(data)
                    
                    orgTrans = {}
                    animBone["transOriginal"].append(orgTrans)
                    orgTrans["frame"] = frame + 1
                    orgTrans["loc"] = (x, y, z)

                    loc = global_matrix @ Vector((x, y, z))
                    animBone["trans"][frame] = (-loc[0], loc[1], loc[2])
                #end for
                if 0 not in animBone["trans"]:
                    loc = global_matrix @ Vector((boneOffsetX, boneOffsetY, boneOffsetZ))
                    animBone["trans"][0] = (-loc[0], loc[1], loc[2])
                #end if
                animBone["fUnknown2"] = file_utils.readFloat(data) #unused
                animBone["fUnknown3"] = file_utils.readFloat(data) #unused
                animBone["unusedPivot"] = file_utils.readVector(data) #unused bone rot pivot
                rotsKeyCount = file_utils.read32(data, True) #number of used rotations keys for this bone
                for i3 in range(rotsKeyCount):
                    frame = file_utils.read32(data, True) - 1
                    vec4Rot = file_utils.readVector4(data)

                    orgRots = {}
                    animBone["rotsOriginal"].append(orgRots)
                    orgRots["frame"] = frame + 1
                    orgRots["rot"] = vec4Rot

                    #Convert right-handed to left-handed quaternion
                    qRot = Quaternion((-vec4Rot[3], vec4Rot[0], vec4Rot[2], vec4Rot[1]))
                    animBone["rots"][frame] = qRot
                #end for
                if 0 not in animBone["rots"]:
                    animBone["rots"][0] = Quaternion((-1, 0, 0, 0)) #left-handed identity quaternion is w = -1
                #end if

                #Fixes jumping around with rotations in blender
                if len(animBone["rots"]) > 1:
                    sortedRotKeys = sorted(animBone["rots"])
                    for i3 in range(1, len(sortedRotKeys), 2):
                        animBone["rots"][sortedRotKeys[i3]].make_compatible(animBone["rots"][sortedRotKeys[i3-1]])
                    #end for
                #end if
            #end for
        #end for
    #end with

    return anims
#end def

# -----------------------------------------------------------------------------
def do_import(context, armObj, filepath, global_matrix, skin):
    """Returns True if successful"""

    if armObj is None or armObj.type != "ARMATURE":
        utils.show_error("[Import Anim] No Armature Object Selected")
        return False
    #end if

    scene = bpy.context.scene
    with ProgressReport(context.window_manager) as progress:
        progress.enter_substeps(1)
        print("Importing Shadow Man Animations %r ..." % filepath)

        anims = load(filepath, global_matrix, skin)
        if not anims or anims is None:
            return False
        #end if

        armBoneCount = len(armObj.data.bones)
        for anim in anims:
            if anim["numBones"] > armBoneCount:
                utils.show_error("[Import Anim] Armature (%s), Anim %s, requires %d Bones but the Armature only has %d Bones" % (
                                    armObj.name, anim["name"], anim["numBones"], armBoneCount))
                return False
            #end if
        #end for

        #Create bone names
        boneNames = {} #boneindex, name
        for bone in armObj.data.bones:
            if not bone.smex.isSoft:
                if bone.smex.boneIndex in boneNames:
                    utils.show_error("[Import Anim] Bone \"%s\" has the same boneIndex of %i as Bone \"%s\". Make sure all bones have a unique bone index before importing animations." % (bone.name, bone.smex.boneIndex, boneNames[bone.smex.boneIndex]))
                    return False
                #end if
                boneNames[bone.smex.boneIndex] = bone.name
            #end if
        #end for

        fileName = os.path.split(filepath)[1]
        fileNameNoExt = os.path.splitext(fileName)[0]
        bpy.context.view_layer.objects.active = armObj

        utils.mode_set('POSE')
        animData = armObj.animation_data_create()
        firstAction = None #to set the current active animation to the first anim
        totalKeyFrameInserts = 0 #for stat tracking purposes only

        progress.enter_substeps(len(anims))
        for animIndex, anim in enumerate(anims):
            animName = "%s %03d %s" % (armObj.name, animIndex, os.path.splitext(anim["name"])[0])
            animAction = bpy.data.actions.new(animName)
            animData.action = animAction
            if animIndex == 0:
                firstAction = animAction
                scene.frame_start = 0
                scene.frame_set(0)
            #end if

            for boneIndex, animBone in enumerate(anim["bones"]):
                boneName = boneNames[boneIndex]
                boneTrans = animBone["trans"]
                boneRots = animBone["rots"]

                smAnimBone = None
                for armSMBone in animAction.smex.bones:
                    if armSMBone.boneIndex == boneIndex:
                        smAnimBone = armSMBone
                        break
                    #end if
                #end for
                if not smAnimBone:
                    smAnimBone = animAction.smex.bones.add()
                #end if
                smAnimBone.boneIndex = boneIndex
                smAnimBone.unknown1 = animBone["fUnknown1"]
                smAnimBone.unknown2 = animBone["fUnknown2"]
                smAnimBone.unknown3 = animBone["fUnknown3"]
                smAnimBone.unknownPivot = animBone["unusedPivot"]

                #set initial poseBone position and rotation to first frame
                poseBone = armObj.pose.bones[boneName]
                poseBone.location = boneTrans[0]
                poseBone.rotation_mode = 'QUATERNION'
                poseBone.rotation_quaternion = boneRots[0]

                #create key frame curve points for translations
                trans_curves = []
                for i in range(3):
                    trans_curves.append(animAction.fcurves.new(poseBone.path_from_id("location"), index=i))
                    trans_curves[i].keyframe_points.add(count=len(boneTrans))
                #end for
                for i in range(3):
                    totalKeyFrameInserts += len(boneTrans)
                    keyPointIndex = 0
                    for frame, loc in boneTrans.items():
                        trans_curves[i].keyframe_points[keyPointIndex].co = (frame, loc[i])
                        trans_curves[i].keyframe_points[keyPointIndex].interpolation = 'LINEAR'
                        keyPointIndex += 1
                    #end for
                #end for
                for transCurve in trans_curves:
                    transCurve.update()
                #end for

                #create key frame curve points for rotations
                rot_curves = []
                for i in range(4):
                    rot_curves.append(animAction.fcurves.new(poseBone.path_from_id("rotation_quaternion"), index=i))
                    rot_curves[i].keyframe_points.add(count=len(boneRots))
                #end for
                for i in range(4):
                    totalKeyFrameInserts += len(boneRots)
                    keyPointIndex = 0
                    for frame, rot in boneRots.items():
                        rot_curves[i].keyframe_points[keyPointIndex].co = (frame, rot[i])
                        rot_curves[i].keyframe_points[keyPointIndex].interpolation = 'LINEAR'
                        keyPointIndex += 1
                    #end for
                #end for
                for rotCurve in rot_curves:
                    rotCurve.update()
                #end for
            #end for

            progress.step()
        #end for

        #Exit Armature pose mode
        utils.mode_set('OBJECT')
        if firstAction is not None:
            animData.action = firstAction
        #end if
        #select only the armature
        for ob in bpy.context.selected_objects:
            ob.select_set(False)
        #end for
        armObj.select_set(True)
        
        progress.leave_substeps("Finished! (actions:%d bones:%d keyframes:%d)" % (len(anims), armBoneCount, totalKeyFrameInserts))
    #end with

    if not utils.prefs().fps60playback:
        scene.render.fps = 30
        scene.render.fps_base = 1.0
        scene.render.frame_map_old = 100
        scene.render.frame_map_new = 100
    #end if

    return True
#end def

# -----------------------------------------------------------------------------
def do_export(context, armObj, filepath, global_matrix):
    actions = utils.get_actions(armObj)
    actions.sort(key=lambda action: utils.get_anim_index(armObj, action))
    armature = armObj.data
    hardBones, softBones = utils.get_bones(armObj)
    boneCount = len(hardBones)

    #get location and rotation key frames for all actions and bones
    actionBoneLocRots = {}
    for i in range(len(actions)):
        actionBoneLocRots[i] = {}
        for i2 in range(boneCount):
            locs, rots = utils.get_action_bone_locrots(armObj, actions[i], hardBones[i2])
            actionBoneLocRots[i][i2] = {"locs": locs, "rots": rots}
        #end for
    #end for

    #Get Animation Byte Offsets
    actionOffset = [0] * len(actions)
    offset = 8 + len(actions) * 4
    for i in range(len(actions)):
        actionOffset[i] = offset
        offset += 8
        offset += len(utils.get_anim_name(armObj, actions[i], ".anm")) + 1
        for i2 in range(boneCount):
            offset += 44
            offset += len(actionBoneLocRots[i][i2]["locs"]) * 16
            offset += len(actionBoneLocRots[i][i2]["rots"]) * 20
        #end for
    #end for

    with ProgressReport(context.window_manager) as progress:
        stepCount = 1
        progress.enter_substeps(stepCount)
        print("Exporting Shadow Man Animations %r ..." % filepath)

        with open(filepath, 'wb') as data:
            data.seek(0)
            file_utils.writeString(data, 'mnAE', False)
            file_utils.write32(data, len(actions), True)
            for i in range(len(actions)):
                file_utils.write32(data, actionOffset[i], False)
            #end for
            for i in range(len(actions)):
                action = actions[i]
                frameStart, frameEnd = action.frame_range
                frameStart = int(frameStart)
                frameEnd = int(frameEnd)
                frameCount = (frameEnd - frameStart) #+ 1
                animName = utils.get_anim_name(armObj, action, ".anm")

                file_utils.write32(data, boneCount, True)
                file_utils.writeString(data, animName, True)
                file_utils.write32(data, frameCount, True)
                for i2 in range(boneCount):
                    bone = hardBones[i2]
                    unknown1 = 0
                    unknown2 = 0
                    unknown3 = 0
                    unusedPivot = (0.0, 1.0, 0.0)
                    for item in action.smex.bones:
                        if item.boneIndex == i2:
                            unknown1 = item.unknown1
                            unknown2 = item.unknown2
                            unknown3 = item.unknown3
                            unusedPivot = item.unknownPivot
                            break
                        #end if
                    #end for

                    locs = actionBoneLocRots[i][i2]["locs"]
                    rots = actionBoneLocRots[i][i2]["rots"]

                    for frame in locs:
                        locs[frame] = global_matrix @ locs[frame]
                        locs[frame].x = -locs[frame].x
                    #end for

                    for frame in rots:
                        #Convert left-handed to right-handed quaternion
                        q = rots[frame]
                        rots[frame] = Quaternion((q[1], q[3], q[2], -q[0]))
                    #end for

                    file_utils.writeFloat(data, unknown1)
                    transOffset = Vector((0.0, 0.0, 0.0))
                    if 0 in locs:
                        transOffset = locs[0]
                    #end if
                    file_utils.writeVector3(data, transOffset)
                    file_utils.write32(data, len(locs), True)
                    for frame in locs:
                        file_utils.write32(data, frame + 1, True)
                        file_utils.writeVector3(data, locs[frame])
                    #end for
                    file_utils.writeFloat(data, unknown2)
                    file_utils.writeFloat(data, unknown3)
                    file_utils.writeVector3(data, unusedPivot)
                    file_utils.write32(data, len(rots), True)
                    for frame in rots:
                        file_utils.write32(data, frame + 1, True)
                        file_utils.writeVector4(data, rots[frame])
                    #end for
                #end for
            #end for
        #end with

        progress.step("Animation Successfully Exported! (anims:%i bones:%i)" % (len(actions), boneCount))
    #end with

    return {'FINISHED'}
#end def
