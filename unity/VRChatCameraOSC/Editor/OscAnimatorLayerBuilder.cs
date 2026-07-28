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

        /// <summary>0..1 parameter driving a single blend shape from 0 to 100.
        /// <paramref name="fullScale"/> is the parameter value at which the
        /// shape reaches weight 100 (Simple1D clamps above the last child, so
        /// higher values stay at 100) — below-1 values rescale avatar-side
        /// for channels whose tracked values never approach 1.0 (issue #23).</summary>
        public static void AddBlendShapeLayer(
            AnimatorController controller,
            Transform avatarRoot,
            string paramName,
            SkinnedMeshRenderer renderer,
            string blendShapeName,
            float fullScale = 1f)
        {
            var tree = NewTree(controller, paramName, BlendTreeType.Simple1D);
            tree.AddChild(BlendShapeClip(controller, avatarRoot, renderer, blendShapeName, 0f), 0f);
            tree.AddChild(BlendShapeClip(controller, avatarRoot, renderer, blendShapeName, BlendShapeFullWeight), fullScale);
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
            string blendShapeName,
            SkinnedMeshRenderer wideRenderer = null,
            string wideBlendShapeName = null)
        {
            var tree = NewTree(controller, paramName, BlendTreeType.Simple1D);
            var hasWide = wideRenderer != null && !string.IsNullOrEmpty(wideBlendShapeName);
            if (!hasWide)
            {
                tree.AddChild(BlendShapeClip(controller, avatarRoot, renderer, blendShapeName, BlendShapeFullWeight), 0f);
                tree.AddChild(BlendShapeClip(controller, avatarRoot, renderer, blendShapeName, 0f), OscParameterSpec.EyeLidNeutral);
            }
            else
            {
                // With an eye-wide shape wired (issue #24), the 0.75..1 range
                // stops clamping and drives it — zero extra parameter bits,
                // since EyeLid already encodes wide above neutral. Every
                // child animates BOTH curves so the blend never leaves one
                // shape at an unanimated default mid-segment.
                tree.AddChild(
                    TwoBlendShapeClip(controller, avatarRoot, renderer, blendShapeName, BlendShapeFullWeight,
                        wideRenderer, wideBlendShapeName, 0f),
                    0f);
                tree.AddChild(
                    TwoBlendShapeClip(controller, avatarRoot, renderer, blendShapeName, 0f,
                        wideRenderer, wideBlendShapeName, 0f),
                    OscParameterSpec.EyeLidNeutral);
                tree.AddChild(
                    TwoBlendShapeClip(controller, avatarRoot, renderer, blendShapeName, 0f,
                        wideRenderer, wideBlendShapeName, BlendShapeFullWeight),
                    1f);
            }
            AddLayer(controller, paramName, tree, AnimatorLayerBlendingMode.Override, null);
        }

        /// <summary>
        /// -1..1 parameter driving one Humanoid muscle via an additive layer,
        /// so it composes with whatever else animates the avatar instead of
        /// overriding it. <paramref name="muscleName"/> must be one of
        /// <see cref="HumanTrait.MuscleName"/> (e.g. "Head Nod Down-Up").
        ///
        /// <paramref name="controller"/> must be the avatar's **Additive**
        /// playable-layer controller (issue #25), not FX or Gesture.
        /// Documented VRChat facts force this
        /// (creators.vrchat.com/avatars/playable-layers/,
        /// /avatars/state-behaviors/): (1) at avatar init the FX layer's
        /// default mask "disables all humanoid muscles", so a muscle-curve
        /// layer placed in FX is silently inert in the VRChat client even
        /// though it rotates the head in the Unity editor; (2) on Gesture,
        /// the first layer's avatar mask is AND-ed over the entire playable
        /// layer, and stock hand-gesture setups carry vrc_HandsOnly there —
        /// same silent-inert failure (observed live, issue #25). The
        /// Additive playable layer is VRChat's documented home for additive
        /// humanoid muscle animation on top of Base ("things like
        /// breathing"), and it ignores the first layer's mask; (3) the Head
        /// bone is IK-driven on Desktop and only an Animator layer whose
        /// state carries a <see cref="VRCAnimatorTrackingControl"/> with
        /// <c>trackingHead = Animation</c> makes the Animator's own values
        /// win over that IK — which is why this method also attaches one.
        /// </summary>
        public static void AddHeadPoseLayer(AnimatorController controller, string paramName, string muscleName)
        {
            var tree = NewTree(controller, paramName, BlendTreeType.Simple1D);
            tree.AddChild(MuscleClip(controller, paramName, muscleName, -1f), -1f);
            tree.AddChild(MuscleClip(controller, paramName, muscleName, 0f), 0f);
            tree.AddChild(MuscleClip(controller, paramName, muscleName, 1f), 1f);
            // Override, not Additive (issue #25 third attempt): additive
            // animator layers derive each clip's reference pose from its t=0
            // sample, which we compensated with a 1/60s 0→value ramp — that
            // trick verifies in the editor but produced zero head motion in
            // the VRChat client on both the Gesture and Additive playable
            // layers, consistent with the client pinning state time at 0
            // (ramp@t0 = 0 ⇒ additive delta identically 0). Emotes prove the
            // client DOES honor plain muscle clips + TrackingControl on a
            // head that Animation-releases from IK, so use plain Override
            // blending with constant clips; the head-only AvatarMask keeps
            // this layer from touching anything else.
            //
            // bufferEntryState (issue #25 attempts 5-6): VRChat misses State
            // Behaviours on a layer's initial state at avatar load (confirmed
            // live — the head only moved once the behaviour fired via a
            // transition), and the client's own systems (jump/landing, emote
            // exit) later reset tracking to Tracking, permanently freezing a
            // fire-once design (also confirmed live). The state machine
            // therefore ping-pongs between two identical states so the
            // tracking control re-asserts Head = Animation continuously.
            AddLayer(controller, paramName, tree, AnimatorLayerBlendingMode.Override,
                GetOrCreateHeadOnlyMask(controller), AddHeadTrackingControlBehaviour, bufferEntryState: true);
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

                    // Transitions (the buffered-entry Init state has one) and
                    // the AnimatorState objects themselves are sub-assets
                    // too; destroying the state machine does not take them
                    // with it.
                    foreach (var transition in childState.state.transitions)
                    {
                        if (transition == null)
                        {
                            continue;
                        }
                        AssetDatabase.RemoveObjectFromAsset(transition);
                        Object.DestroyImmediate(transition, true);
                    }
                    AssetDatabase.RemoveObjectFromAsset(childState.state);
                    Object.DestroyImmediate(childState.state, true);
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
                // Default is true, and automatic mode redistributes child
                // thresholds across the tree's 0..1 default range on AddChild
                // — which silently moved the eyelid tree's 0.75 neutral
                // threshold to 1.0 (caught by the EditMode suite, issue #21).
                // Every tree here passes explicit thresholds; never let Unity
                // rewrite them.
                useAutomaticThresholds = false,
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
            System.Action<AnimatorState> configureState = null,
            bool bufferEntryState = false)
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
            configureState?.Invoke(state);
            if (bufferEntryState)
            {
                // Two identical states ping-ponging on exit time, each
                // carrying the motion AND the configured behaviours, so
                // OnStateEnter re-fires continuously. This both dodges the
                // missed-behaviour-on-initial-state load quirk (confirmed
                // live: the head only started moving once the behaviour fired
                // via a transition) and re-asserts the tracking control after
                // VRChat's own systems reset it — observed live: a jump (or
                // any client animation event that sets tracking back to
                // Tracking) permanently froze the head under a fire-once
                // design (issue #25).
                var loop = stateMachine.AddState(layerName + "_Loop");
                loop.motion = tree;
                loop.writeDefaultValues = false;
                configureState?.Invoke(loop);
                var forward = state.AddTransition(loop);
                forward.hasExitTime = true;
                forward.exitTime = 1f;
                forward.duration = 0f;
                var back = loop.AddTransition(state);
                back.hasExitTime = true;
                back.exitTime = 1f;
                back.duration = 0f;
            }
            stateMachine.defaultState = state;

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
            SetBlendShapeCurve(clip, avatarRoot, renderer, blendShapeName, weight);
            AssetDatabase.AddObjectToAsset(clip, controller);
            return clip;
        }

        /// <summary>One clip animating two blend shapes (the EyeLid blink +
        /// eye-wide pair) — Simple1D children must all animate the same
        /// property set or Unity leaves the missing curve at its previous
        /// value mid-blend.</summary>
        static AnimationClip TwoBlendShapeClip(
            AnimatorController controller,
            Transform avatarRoot,
            SkinnedMeshRenderer rendererA,
            string shapeA,
            float weightA,
            SkinnedMeshRenderer rendererB,
            string shapeB,
            float weightB)
        {
            var clip = new AnimationClip
            {
                name = $"{rendererA.name}_{shapeA}_{weightA:0}_{shapeB}_{weightB:0}".Replace('/', '_'),
            };
            SetBlendShapeCurve(clip, avatarRoot, rendererA, shapeA, weightA);
            SetBlendShapeCurve(clip, avatarRoot, rendererB, shapeB, weightB);
            AssetDatabase.AddObjectToAsset(clip, controller);
            return clip;
        }

        static void SetBlendShapeCurve(
            AnimationClip clip,
            Transform avatarRoot,
            SkinnedMeshRenderer renderer,
            string blendShapeName,
            float weight)
        {
            var path = AnimationUtility.CalculateTransformPath(renderer.transform, avatarRoot);
            var binding = EditorCurveBinding.FloatCurve(path, typeof(SkinnedMeshRenderer), "blendShape." + blendShapeName);
            AnimationUtility.SetEditorCurve(clip, binding, AnimationCurve.Constant(0f, 0f, weight));
        }

        /// <summary>Seconds of flat curve per muscle clip. The value is
        /// constant, but the clip must have REAL length: the head layer's
        /// ping-pong transitions fire on exit time, and a zero-length clip
        /// (single t=0 key) never advances normalized time — live-confirmed
        /// as "jump freezes the head and it never recovers" (the tracking
        /// re-assert transition simply never fired, issue #25). With 1s
        /// clips, Head = Animation is re-asserted every second.</summary>
        const float MuscleClipSeconds = 1f;

        static AnimationClip MuscleClip(AnimatorController controller, string paramName, string muscleName, float value)
        {
            // Flat (constant-value) curve with real duration; Override
            // blending (see AddHeadPoseLayer — the additive + t=0-ramp
            // variant is retired, it produced zero motion in the client).
            var clip = new AnimationClip { name = $"{LayerNameFor(paramName)}_{value:0.##}" };
            var binding = EditorCurveBinding.FloatCurve(string.Empty, typeof(Animator), muscleName);
            var curve = new AnimationCurve(
                new Keyframe(0f, value),
                new Keyframe(MuscleClipSeconds, value));
            AnimationUtility.SetEditorCurve(clip, binding, curve);
            AssetDatabase.AddObjectToAsset(clip, controller);
            return clip;
        }
    }
}
