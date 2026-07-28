using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEditor.Animations;
using UnityEngine;
using VRC.SDK3.Avatars.Components;
using VRC.SDKBase;

namespace VRChatCameraOsc.AvatarSetup
{
    /// <summary>
    /// Builds Animator Controller layers that drive an existing blend shape
    /// or the Humanoid head bone from one of the wizard's OSC float
    /// parameters (issue #16). Blend-shape layers go in the FX controller;
    /// <see cref="AddHeadPoseLayer"/> goes in the Gesture controller instead
    /// (see its doc comment for why). Each parameter gets its own layer,
    /// named <c>OSC_&lt;ParamName&gt;</c> (with <c>/</c> replaced by
    /// <c>_</c> — <c>v2/Head/Yaw</c> parameters keep their slash on the
    /// Animator *parameter*, but layer/state/clip asset names must not
    /// contain path separators), so re-running the wizard replaces rather
    /// than duplicates.
    ///
    /// Deliberately Animator-only: VRChat strips arbitrary MonoBehaviours from
    /// uploaded avatars, so nothing here can be a runtime script — see the
    /// package README.
    /// </summary>
    public static class OscAnimatorLayerBuilder
    {
        const float BlendShapeFullWeight = 100f;

        /// <summary><c>OSC_*</c> layer/asset name for a parameter, slash-safe.</summary>
        static string LayerNameFor(string paramName) => "OSC_" + paramName.Replace('/', '_');

        /// <summary>0..1 parameter driving a single blend shape from 0 to 100.</summary>
        public static void AddBlendShapeLayer(
            AnimatorController controller,
            Transform avatarRoot,
            string paramName,
            SkinnedMeshRenderer renderer,
            string blendShapeName)
        {
            var tree = NewTree(controller, paramName, BlendTreeType.Simple1D);
            tree.AddChild(BlendShapeClip(controller, avatarRoot, renderer, blendShapeName, 0f), 0f);
            tree.AddChild(BlendShapeClip(controller, avatarRoot, renderer, blendShapeName, BlendShapeFullWeight), 1f);
            AddLayer(controller, paramName, tree, AnimatorLayerBlendingMode.Override, null);
        }

        /// <summary>
        /// A VRCFT <c>v2/EyeLid*</c> parameter (0 = closed, ~0.75 = relaxed
        /// open, 1 = wide) driving a *blink/close* blend shape — the mapping
        /// is inverted: shape weight 100 at parameter 0, weight 0 at
        /// <see cref="OscParameterSpec.EyeLidNeutral"/>. Above the neutral
        /// threshold Simple1D clamps to the last child, so 0.75..1 (eye-wide
        /// territory) keeps the blink shape at 0 — avatars with an eye-wide
        /// shape can wire it manually later; most don't have one.
        /// </summary>
        public static void AddEyeLidLayer(
            AnimatorController controller,
            Transform avatarRoot,
            string paramName,
            SkinnedMeshRenderer renderer,
            string blendShapeName)
        {
            var tree = NewTree(controller, paramName, BlendTreeType.Simple1D);
            tree.AddChild(BlendShapeClip(controller, avatarRoot, renderer, blendShapeName, BlendShapeFullWeight), 0f);
            tree.AddChild(BlendShapeClip(controller, avatarRoot, renderer, blendShapeName, 0f), OscParameterSpec.EyeLidNeutral);
            AddLayer(controller, paramName, tree, AnimatorLayerBlendingMode.Override, null);
        }

        /// <summary>
        /// -1..1 parameter driving one Humanoid muscle via an additive layer,
        /// so it composes with whatever else animates the avatar instead of
        /// overriding it. <paramref name="muscleName"/> must be one of
        /// <see cref="HumanTrait.MuscleName"/> (e.g. "Head Nod Down-Up").
        ///
        /// <paramref name="controller"/> must be the avatar's **Gesture**
        /// playable-layer controller, not FX. Two documented VRChat facts
        /// force this (creators.vrchat.com/avatars/playable-layers/,
        /// /avatars/state-behaviors/): (1) at avatar init the FX layer's
        /// default mask "disables all humanoid muscles", so a muscle-curve
        /// layer placed in FX is silently inert in the VRChat client even
        /// though it rotates the head in the Unity editor; (2) the Head bone
        /// is IK-driven on Desktop and only an Animator layer whose state
        /// carries a <see cref="VRCAnimatorTrackingControl"/> with
        /// <c>trackingHead = Animation</c> makes the Animator's own values
        /// win over that IK — which is why this method also attaches one.
        /// The Gesture layer is VRChat's documented home for "animations
        /// that need to act on individual body parts while still playing the
        /// underlying animations for the rest of the body".
        /// </summary>
        public static void AddHeadPoseLayer(AnimatorController controller, string paramName, string muscleName)
        {
            var tree = NewTree(controller, paramName, BlendTreeType.Simple1D);
            tree.AddChild(MuscleClip(controller, paramName, muscleName, -1f), -1f);
            tree.AddChild(MuscleClip(controller, paramName, muscleName, 0f), 0f);
            tree.AddChild(MuscleClip(controller, paramName, muscleName, 1f), 1f);
            AddLayer(controller, paramName, tree, AnimatorLayerBlendingMode.Additive, GetOrCreateHeadOnlyMask(controller), AddHeadTrackingControlBehaviour);
        }

        /// <summary>
        /// Forces Head = Animation (so this layer's muscle curves win over
        /// Desktop head IK) and leaves every other tracked body part at
        /// NoChange, so this layer never overrides hands/eyes/mouth/etc. it
        /// knows nothing about. Field/enum names confirmed against the VRC
        /// SDK's compiled <c>VRCSDKBase.dll</c> /
        /// <c>VRCAnimatorTrackingControlEditor.cs</c> — the type is
        /// <c>VRC.SDK3.Avatars.Components.VRCAnimatorTrackingControl</c>,
        /// deriving its <c>TrackingType</c> enum (NoChange/Tracking/Animation)
        /// from <see cref="VRC_AnimatorTrackingControl"/>.
        /// </summary>
        static void AddHeadTrackingControlBehaviour(AnimatorState state)
        {
            var behaviour = state.AddStateMachineBehaviour<VRCAnimatorTrackingControl>();
            behaviour.trackingHead = VRC_AnimatorTrackingControl.TrackingType.Animation;
            behaviour.trackingLeftHand = VRC_AnimatorTrackingControl.TrackingType.NoChange;
            behaviour.trackingRightHand = VRC_AnimatorTrackingControl.TrackingType.NoChange;
            behaviour.trackingHip = VRC_AnimatorTrackingControl.TrackingType.NoChange;
            behaviour.trackingLeftFoot = VRC_AnimatorTrackingControl.TrackingType.NoChange;
            behaviour.trackingRightFoot = VRC_AnimatorTrackingControl.TrackingType.NoChange;
            behaviour.trackingLeftFingers = VRC_AnimatorTrackingControl.TrackingType.NoChange;
            behaviour.trackingRightFingers = VRC_AnimatorTrackingControl.TrackingType.NoChange;
            behaviour.trackingEyes = VRC_AnimatorTrackingControl.TrackingType.NoChange;
            behaviour.trackingMouth = VRC_AnimatorTrackingControl.TrackingType.NoChange;
        }

        /// <summary>
        /// A shared <see cref="AvatarMask"/>, restricted to only the Head
        /// humanoid body part, reused across all three head-pose layers.
        ///
        /// Without this, a real bug: Unity's Humanoid retargeting solver can
        /// leak tiny perturbations into Chest/Spine even when a clip only
        /// contains Head muscle curves, invisible in a single static
        /// parameter scrub but enough — at the continuous, jittery rate real
        /// OSC head-tracking updates at — to kick a chest-mounted VRCPhysBone
        /// (e.g. wings) into a runaway spin. The mask contains the additive
        /// layer's *written* effect to Head only, regardless of what the
        /// solver computes internally for other bones.
        /// </summary>
        static AvatarMask GetOrCreateHeadOnlyMask(AnimatorController controller)
        {
            const string maskName = "OSC_HeadOnlyMask";
            var existing = AssetDatabase.LoadAllAssetsAtPath(AssetDatabase.GetAssetPath(controller))
                .OfType<AvatarMask>()
                .FirstOrDefault(m => m.name == maskName);
            if (existing != null)
            {
                return existing;
            }

            var mask = new AvatarMask { name = maskName };
            for (var part = AvatarMaskBodyPart.Root; part < AvatarMaskBodyPart.LastBodyPart; part++)
            {
                mask.SetHumanoidBodyPartActive(part, part == AvatarMaskBodyPart.Head);
            }
            AssetDatabase.AddObjectToAsset(mask, controller);
            return mask;
        }

        /// <summary>Whether an <c>OSC_&lt;paramName&gt;</c> layer currently exists — the
        /// "ON" state the wizard's toggle button reads to decide Apply vs. Remove.</summary>
        public static bool HasLayer(AnimatorController controller, string paramName)
        {
            return controller != null && controller.layers.Any(l => l.name == LayerNameFor(paramName));
        }

        /// <summary>
        /// Removes the <c>OSC_&lt;paramName&gt;</c> layer, its parameter, and
        /// every sub-asset (BlendTree + AnimationClips) it owns — the "OFF"
        /// side of the wizard's apply/remove toggle. No-op (returns false) if
        /// the layer isn't present.
        /// </summary>
        public static bool RemoveLayer(AnimatorController controller, string paramName)
        {
            var layerName = LayerNameFor(paramName);
            var layers = controller.layers.ToList();
            var index = layers.FindIndex(l => l.name == layerName);
            if (index < 0)
            {
                return false;
            }

            var layer = layers[index];
            layers.RemoveAt(index);
            controller.layers = layers.ToArray();

            if (layer.stateMachine != null)
            {
                var visited = new HashSet<int>();
                foreach (var childState in layer.stateMachine.states)
                {
                    DestroyMotion(childState.state.motion, visited);

                    // StateMachineBehaviours (e.g. the head-pose layer's
                    // VRCAnimatorTrackingControl) are separate sub-assets of
                    // the controller, same as BlendTrees/AnimationClips —
                    // destroying the state/stateMachine does not take them
                    // with it, so without this an orphaned behaviour asset
                    // would linger in the controller file after removal.
                    foreach (var behaviour in childState.state.behaviours)
                    {
                        if (behaviour == null)
                        {
                            continue;
                        }
                        AssetDatabase.RemoveObjectFromAsset(behaviour);
                        Object.DestroyImmediate(behaviour, true);
                    }
                }
                AssetDatabase.RemoveObjectFromAsset(layer.stateMachine);
                Object.DestroyImmediate(layer.stateMachine, true);
            }

            // The head-only mask is shared across all 3 head-pose layers —
            // only destroy it once no remaining layer references it.
            if (layer.avatarMask != null && !layers.Any(l => l.avatarMask == layer.avatarMask))
            {
                AssetDatabase.RemoveObjectFromAsset(layer.avatarMask);
                Object.DestroyImmediate(layer.avatarMask, true);
            }

            controller.parameters = controller.parameters.Where(p => p.name != paramName).ToArray();
            return true;
        }

        static void DestroyMotion(Motion motion, HashSet<int> visited)
        {
            if (motion == null || !visited.Add(motion.GetInstanceID()))
            {
                return;
            }

            if (motion is BlendTree tree)
            {
                foreach (var child in tree.children)
                {
                    DestroyMotion(child.motion, visited);
                }
                AssetDatabase.RemoveObjectFromAsset(tree);
                Object.DestroyImmediate(tree, true);
            }
            else if (motion is AnimationClip clip)
            {
                AssetDatabase.RemoveObjectFromAsset(clip);
                Object.DestroyImmediate(clip, true);
            }
        }

        static BlendTree NewTree(AnimatorController controller, string paramName, BlendTreeType type)
        {
            EnsureFloatParameter(controller, paramName);
            var tree = new BlendTree
            {
                name = LayerNameFor(paramName),
                blendType = type,
                blendParameter = paramName,
            };
            AssetDatabase.AddObjectToAsset(tree, controller);
            return tree;
        }

        static void EnsureFloatParameter(AnimatorController controller, string paramName)
        {
            if (controller.parameters.Any(p => p.name == paramName))
            {
                return;
            }
            controller.AddParameter(paramName, AnimatorControllerParameterType.Float);
        }

        /// <summary>
        /// Replaces an existing <c>OSC_&lt;paramName&gt;</c> layer if present
        /// (idempotent re-runs), otherwise appends a new one.
        /// </summary>
        static void AddLayer(
            AnimatorController controller,
            string paramName,
            BlendTree tree,
            AnimatorLayerBlendingMode blendingMode,
            AvatarMask mask,
            System.Action<AnimatorState> configureState = null)
        {
            var layerName = LayerNameFor(paramName);
            var layers = controller.layers.ToList();
            var existingIndex = layers.FindIndex(l => l.name == layerName);
            if (existingIndex >= 0)
            {
                layers.RemoveAt(existingIndex);
            }

            var stateMachine = new AnimatorStateMachine { name = layerName, hideFlags = HideFlags.HideInHierarchy };
            AssetDatabase.AddObjectToAsset(stateMachine, controller);
            var state = stateMachine.AddState(layerName);
            state.motion = tree;
            state.writeDefaultValues = false;
            stateMachine.defaultState = state;
            configureState?.Invoke(state);

            layers.Add(new AnimatorControllerLayer
            {
                name = layerName,
                stateMachine = stateMachine,
                defaultWeight = 1f,
                blendingMode = blendingMode,
                avatarMask = mask,
            });
            controller.layers = layers.ToArray();
        }

        static AnimationClip BlendShapeClip(
            AnimatorController controller,
            Transform avatarRoot,
            SkinnedMeshRenderer renderer,
            string blendShapeName,
            float weight)
        {
            var clip = new AnimationClip { name = $"{renderer.name}_{blendShapeName}_{weight:0}".Replace('/', '_') };
            var path = AnimationUtility.CalculateTransformPath(renderer.transform, avatarRoot);
            var binding = EditorCurveBinding.FloatCurve(path, typeof(SkinnedMeshRenderer), "blendShape." + blendShapeName);
            AnimationUtility.SetEditorCurve(clip, binding, AnimationCurve.Constant(0f, 0f, weight));
            AssetDatabase.AddObjectToAsset(clip, controller);
            return clip;
        }

        // Additive layers subtract each clip's own value at time 0 as its
        // implicit reference pose before adding the rest on top. A flat
        // AnimationCurve.Constant clip has the same value at time 0 as
        // everywhere else, so that delta — and the entire additive layer's
        // contribution — is always zero, regardless of AvatarMask. Ramping
        // from 0 at t=0 up to the target value makes the reference pose 0
        // (matching the other two children of the blend tree, which also
        // start at 0) so the additive delta correctly reflects the blended
        // muscle value once the ramp finishes, one frame later.
        const float AdditiveReferenceRampSeconds = 1f / 60f;

        static AnimationClip MuscleClip(AnimatorController controller, string paramName, string muscleName, float value)
        {
            var clip = new AnimationClip { name = $"{LayerNameFor(paramName)}_{value:0.##}" };
            var binding = EditorCurveBinding.FloatCurve(string.Empty, typeof(Animator), muscleName);
            AnimationUtility.SetEditorCurve(clip, binding, AnimationCurve.Linear(0f, 0f, AdditiveReferenceRampSeconds, value));
            AssetDatabase.AddObjectToAsset(clip, controller);
            return clip;
        }
    }
}
