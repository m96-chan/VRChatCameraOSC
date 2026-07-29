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

        /// <summary>The combined head layer's pseudo-parameter key: used only
        /// to derive the layer/asset name (<c>OSC_v2_Head</c>) for
        /// <see cref="HasLayer"/>/<see cref="RemoveLayer"/>. The actual
        /// Animator parameters are the three <c>v2/Head/*</c> floats.</summary>
        public const string CombinedHeadKey = "v2/Head";

        /// <summary>Humanoid muscle per head axis, in nesting order (yaw
        /// outermost); names verified against
        /// <see cref="HumanTrait.MuscleName"/> (MuscleNameValidityTests).
        /// <c>sign</c> maps the parameter's VRCFT convention onto
        /// the muscle's: <c>v2/Head/Yaw</c> is +1 = turn toward the
        /// subject's LEFT (HeadPose/VRCFT), but the "Head Turn Left-Right"
        /// muscle is +1 = right — live-confirmed inverted (いやいや mirrored,
        /// issue #27 follow-up). Pitch (+1 = up = muscle up) and roll (sign
        /// already flipped tracker-side) verified correct live.</summary>
        static readonly (string param, string muscle, float sign)[] HeadAxes =
        {
            ("v2/Head/Yaw", "Head Turn Left-Right", -1f),
            ("v2/Head/Pitch", "Head Nod Down-Up", 1f),
            ("v2/Head/Roll", "Head Tilt Left-Right", 1f),
        };

        /// <summary>
        /// ONE Override layer driving all three head axes through a nested
        /// 3-axis blend tree (Yaw → Pitch → Roll Simple1D trees, 3×3×3 = 27
        /// leaf clips each animating ALL three muscles).
        ///
        /// Why one layer (issue #27, observed live): Unity/VRChat compose
        /// humanoid Override layers per masked muscle GROUP, not per muscle —
        /// with one layer per axis, the last (Roll) layer overrode the whole
        /// Head group with {tilt: value, turn: 0, nod: 0}: roll moved, yaw
        /// and pitch were pinned straight ahead in-client (the editor blends
        /// per-muscle and hides this). All three muscles must therefore be
        /// written by the same layer, i.e. the same blend tree.
        ///
        /// <paramref name="controller"/> must be the avatar's **Gesture**
        /// playable-layer controller with a first-layer mask that allows
        /// Head (see AvatarSetupWindow.EnsureGestureMaskAllowsHead). The
        /// placement/construction history — FX (masked out), Additive
        /// playable layer (client-additive zeroes it), additive blending +
        /// t=0 ramp (reference-pose delta 0), fire-once tracking control
        /// (client resets on jump), zero-length clips (exit time never
        /// reached) — is recorded in issues #25/#27. A
        /// <see cref="VRCAnimatorTrackingControl"/> (Head = Animation) rides
        /// the ping-pong states so the Animator keeps winning over Desktop
        /// head IK.
        /// </summary>
        public static void AddCombinedHeadLayer(AnimatorController controller)
        {
            foreach (var (param, _, _) in HeadAxes)
            {
                EnsureFloatParameter(controller, param);
            }

            var top = new BlendTree
            {
                name = LayerNameFor(CombinedHeadKey),
                blendType = BlendTreeType.Simple1D,
                blendParameter = HeadAxes[0].param,
                useAutomaticThresholds = false,
            };
            AssetDatabase.AddObjectToAsset(top, controller);
            foreach (var yaw in new[] { -1f, 0f, 1f })
            {
                var mid = new BlendTree
                {
                    name = $"{LayerNameFor(CombinedHeadKey)}_y{yaw:0.#}",
                    blendType = BlendTreeType.Simple1D,
                    blendParameter = HeadAxes[1].param,
                    useAutomaticThresholds = false,
                };
                AssetDatabase.AddObjectToAsset(mid, controller);
                top.AddChild(mid, yaw);
                foreach (var pitch in new[] { -1f, 0f, 1f })
                {
                    var leaf = new BlendTree
                    {
                        name = $"{LayerNameFor(CombinedHeadKey)}_y{yaw:0.#}_p{pitch:0.#}",
                        blendType = BlendTreeType.Simple1D,
                        blendParameter = HeadAxes[2].param,
                        useAutomaticThresholds = false,
                    };
                    AssetDatabase.AddObjectToAsset(leaf, controller);
                    mid.AddChild(leaf, pitch);
                    foreach (var roll in new[] { -1f, 0f, 1f })
                    {
                        leaf.AddChild(HeadMuscleClip(controller, yaw, pitch, roll), roll);
                    }
                }
            }

            AddLayer(controller, CombinedHeadKey, top, AnimatorLayerBlendingMode.Override,
                GetOrCreateHeadOnlyMask(controller), AddHeadTrackingControlBehaviour, bufferEntryState: true);
        }

        /// <summary>One leaf clip writing ALL THREE head muscles (flat
        /// 1-second curves — real length keeps the ping-pong clock ticking,
        /// see <see cref="MuscleClipSeconds"/>). Writing every muscle in
        /// every leaf is the point: the whole-group override then always
        /// carries all three blended values instead of defaults.</summary>
        static AnimationClip HeadMuscleClip(AnimatorController controller, float yaw, float pitch, float roll)
        {
            var clip = new AnimationClip
            {
                name = $"{LayerNameFor(CombinedHeadKey)}_y{yaw:0.#}_p{pitch:0.#}_r{roll:0.#}",
            };
            var values = new[] { yaw, pitch, roll };
            for (var i = 0; i < HeadAxes.Length; i++)
            {
                var v = values[i] * HeadAxes[i].sign;
                var binding = EditorCurveBinding.FloatCurve(string.Empty, typeof(Animator), HeadAxes[i].muscle);
                var curve = new AnimationCurve(
                    new Keyframe(0f, v),
                    new Keyframe(MuscleClipSeconds, v));
                AnimationUtility.SetEditorCurve(clip, binding, curve);
            }
            AssetDatabase.AddObjectToAsset(clip, controller);
            return clip;
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

        /// <summary>Animator/expression parameter names for the hand-gesture
        /// Int layers (issue #8) — the tracker's custom transport for the
        /// standard 0-7 VRChat gesture scale (native <c>GestureLeft/Right</c>
        /// are read-only over OSC, vrchat-community/osc#42).</summary>
        public const string GestureLeftParam = "VCO_GestureLeft";
        public const string GestureRightParam = "VCO_GestureRight";

        /// <summary>Pose state names by gesture index − 1 (index 0 is the
        /// empty Neutral state): the standard VRChat gesture table
        /// (creators.vrchat.com/avatars/animator-parameters).</summary>
        internal static readonly string[] GesturePoseNames =
        {
            "Fist", "HandOpen", "FingerPoint", "Victory", "RockNRoll", "HandGun", "ThumbsUp",
        };

        /// <summary>Finger order used by the pose table below.</summary>
        static readonly string[] Fingers = { "Thumb", "Index", "Middle", "Ring", "Little" };

        /// <summary>
        /// Per-gesture finger-muscle targets, finger order Thumb / Index /
        /// Middle / Ring / Little. <c>stretch</c> feeds every "N Stretched"
        /// joint muscle of that finger (+1 = fully extended/straight, −1 =
        /// fully curled — thumb has joints 1..3 + Spread, same as the other
        /// fingers in Unity's Humanoid rig); <c>spread</c> feeds the finger's
        /// single "Spread" muscle (positive fans fingers apart, matching the
        /// grouped finger-spread muscle-settings slider). Spread values are
        /// cosmetic tuning (webcam gestures are recognized by curl pattern
        /// alone): HandOpen fans slightly, Victory separates the V,
        /// ThumbsUp/HandGun stick the thumb out.
        /// </summary>
        static readonly (string pose, float[] stretch, float[] spread)[] GesturePoseTable =
        {
            // pose            Thumb Index Middle Ring Little
            ("Fist", new[] { -1f, -1f, -1f, -1f, -1f }, new[] { 0f, 0f, 0f, 0f, 0f }),
            ("HandOpen", new[] { 1f, 1f, 1f, 1f, 1f }, new[] { 0.5f, 0.5f, 0.5f, 0.5f, 0.5f }),
            ("FingerPoint", new[] { -1f, 1f, -1f, -1f, -1f }, new[] { 0f, 0f, 0f, 0f, 0f }),
            ("Victory", new[] { -1f, 1f, 1f, -1f, -1f }, new[] { 0f, 0.5f, -0.5f, 0f, 0f }),
            ("RockNRoll", new[] { 1f, 1f, -1f, -1f, 1f }, new[] { 0.5f, 0f, 0f, 0f, 0f }),
            ("HandGun", new[] { 1f, 1f, -1f, -1f, -1f }, new[] { 1f, 0f, 0f, 0f, 0f }),
            ("ThumbsUp", new[] { 1f, -1f, -1f, -1f, -1f }, new[] { 1f, 0f, 0f, 0f, 0f }),
        };

        /// <summary>Seconds of cross-fade between hand poses — condition-based
        /// transitions, so this is pure smoothing (an Int snap would otherwise
        /// teleport the fingers).</summary>
        const float GestureTransitionSeconds = 0.1f;

        /// <summary>
        /// The animatable muscle-curve property names for one hand's finger
        /// muscles, 4 per finger (joints 1..3 Stretched + Spread), 20 total.
        /// NOTE the finger quirk: <see cref="HumanTrait.MuscleName"/> lists
        /// these as e.g. <c>"Left Thumb 1 Stretched"</c>, but the property
        /// name an <see cref="EditorCurveBinding"/> (and the Animation
        /// window) actually binds is <c>"LeftHand.Thumb.1 Stretched"</c> —
        /// unlike the head muscles, where the two names coincide. Verified
        /// empirically in MuscleNameValidityTests (a clip bound with these
        /// names reports <c>humanMotion == true</c>; the HumanTrait spelling
        /// does not).
        /// </summary>
        internal static IEnumerable<string> FingerMuscleProperties(bool leftHand)
        {
            var hand = leftHand ? "LeftHand" : "RightHand";
            foreach (var finger in Fingers)
            {
                yield return $"{hand}.{finger}.1 Stretched";
                yield return $"{hand}.{finger}.Spread";
                yield return $"{hand}.{finger}.2 Stretched";
                yield return $"{hand}.{finger}.3 Stretched";
            }
        }

        /// <summary>
        /// ONE Gesture-controller layer per hand (<c>OSC_VCO_GestureLeft</c> /
        /// <c>OSC_VCO_GestureRight</c>, issue #8) posing that hand's fingers
        /// from the tracker's <c>VCO_Gesture*</c> Int (standard 0-7 scale):
        ///
        /// - The default <b>Neutral</b> state is EMPTY (no motion): at 0 the
        ///   layer writes nothing, so VRChat's own keyboard/controller hand
        ///   gestures on the stock layers below keep working untouched.
        /// - 7 pose states (Fist … ThumbsUp), each a 1-second flat
        ///   finger-muscle clip (real length by convention — zero-length
        ///   clips broke exit-time machinery elsewhere in this controller,
        ///   issue #25 — though these transitions are condition-based).
        /// - Any-state transitions conditioned <c>Equals</c> the int value,
        ///   0.1 s fixed duration for smoothing, canTransitionToSelf off so a
        ///   held gesture doesn't retrigger the cross-fade.
        /// - A per-hand AvatarMask restricted to that hand's Fingers body
        ///   part only (mirrors GetOrCreateHeadOnlyMask — Humanoid
        ///   retargeting must not leak outside the hand).
        /// - Deliberately NO VRCAnimatorTrackingControl: fingers aren't
        ///   IK-held on Desktop — the stock gesture layers prove plain
        ///   muscle-driven hand poses just work. The Gesture playable layer's
        ///   first-layer mask (stock vrc_HandsOnly, or the wizard's
        ///   OSC_GestureMask) already allows both Fingers parts, so nothing
        ///   extra is needed there either.
        /// </summary>
        public static void AddHandGestureLayer(AnimatorController controller, bool leftHand)
        {
            var paramName = leftHand ? GestureLeftParam : GestureRightParam;

            // Full sub-asset cleanup before rebuilding (idempotent re-Apply
            // without orphaned clips/transitions accumulating in the asset).
            RemoveLayer(controller, paramName);

            if (!controller.parameters.Any(p => p.name == paramName))
            {
                controller.AddParameter(paramName, AnimatorControllerParameterType.Int);
            }

            var layerName = LayerNameFor(paramName);
            var stateMachine = new AnimatorStateMachine { name = layerName, hideFlags = HideFlags.HideInHierarchy };
            AssetDatabase.AddObjectToAsset(stateMachine, controller);

            var neutral = stateMachine.AddState("Neutral");
            neutral.motion = null;
            neutral.writeDefaultValues = false;
            stateMachine.defaultState = neutral;
            ConfigureGestureTransition(stateMachine.AddAnyStateTransition(neutral), paramName, 0);

            for (var i = 0; i < GesturePoseTable.Length; i++)
            {
                var (pose, stretch, spread) = GesturePoseTable[i];
                var state = stateMachine.AddState(pose);
                state.motion = HandPoseClip(controller, layerName, leftHand, pose, stretch, spread);
                state.writeDefaultValues = false;
                ConfigureGestureTransition(stateMachine.AddAnyStateTransition(state), paramName, i + 1);
            }

            var layers = controller.layers.ToList();
            layers.Add(new AnimatorControllerLayer
            {
                name = layerName,
                stateMachine = stateMachine,
                defaultWeight = 1f,
                blendingMode = AnimatorLayerBlendingMode.Override,
                avatarMask = GetOrCreateFingersOnlyMask(controller, leftHand),
            });
            controller.layers = layers.ToArray();
        }

        static void ConfigureGestureTransition(AnimatorStateTransition transition, string paramName, int value)
        {
            transition.hasExitTime = false;
            transition.hasFixedDuration = true;
            transition.duration = GestureTransitionSeconds;
            transition.canTransitionToSelf = false;
            transition.AddCondition(AnimatorConditionMode.Equals, value, paramName);
        }

        /// <summary>One pose clip writing ALL 20 of that hand's finger
        /// muscles as flat 1-second curves (<see cref="MuscleClipSeconds"/>)
        /// — every pose animates the full set so the whole-group override
        /// always carries every finger, never a stale default.</summary>
        static AnimationClip HandPoseClip(
            AnimatorController controller,
            string layerName,
            bool leftHand,
            string pose,
            float[] stretch,
            float[] spread)
        {
            var clip = new AnimationClip { name = $"{layerName}_{pose}" };
            var hand = leftHand ? "LeftHand" : "RightHand";
            for (var f = 0; f < Fingers.Length; f++)
            {
                foreach (var joint in new[] { "1 Stretched", "2 Stretched", "3 Stretched", "Spread" })
                {
                    var value = joint == "Spread" ? spread[f] : stretch[f];
                    var binding = EditorCurveBinding.FloatCurve(
                        string.Empty, typeof(Animator), $"{hand}.{Fingers[f]}.{joint}");
                    var curve = new AnimationCurve(
                        new Keyframe(0f, value),
                        new Keyframe(MuscleClipSeconds, value));
                    AnimationUtility.SetEditorCurve(clip, binding, curve);
                }
            }
            AssetDatabase.AddObjectToAsset(clip, controller);
            return clip;
        }

        /// <summary>Per-hand fingers-only mask (mirrors
        /// <see cref="GetOrCreateHeadOnlyMask"/>): contains the layer's
        /// written effect to that hand's Fingers body part so Humanoid
        /// retargeting can't perturb the arm/body — fingers only is the
        /// safest restriction, and it's all the pose clips animate.</summary>
        static AvatarMask GetOrCreateFingersOnlyMask(AnimatorController controller, bool leftHand)
        {
            var maskName = leftHand ? "OSC_LeftFingersMask" : "OSC_RightFingersMask";
            var allowed = leftHand ? AvatarMaskBodyPart.LeftFingers : AvatarMaskBodyPart.RightFingers;
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
                mask.SetHumanoidBodyPartActive(part, part == allowed);
            }
            AssetDatabase.AddObjectToAsset(mask, controller);
            return mask;
        }

        /// <summary>Pseudo-parameter keys for the per-hand finger-curl layers
        /// (issue #8 phase 3), mirroring <see cref="CombinedHeadKey"/>: they
        /// derive the layer/asset name (<c>OSC_VCO_L_FingerCurls</c>); the
        /// real Animator parameters are the five <c>VCO_*Curl</c> floats of
        /// that hand (<see cref="CurlParams"/>).</summary>
        public const string FingerCurlLeftKey = "VCO_L_FingerCurls";
        public const string FingerCurlRightKey = "VCO_R_FingerCurls";

        /// <summary>Layer keys for the per-arm layers (issue #28). Kept at
        /// the phase-1 UpDown parameter names so the layer asset name stays
        /// <c>OSC_VCO_L/R_ArmUpDown</c> — re-applying over a phase-1 avatar
        /// then replaces the old 1-D layer in place instead of orphaning it.
        /// The layer's real Animator parameters are the four
        /// <see cref="ArmParams"/> of that arm (phase 2: Bool gate + three
        /// direction/elbow floats).</summary>
        public const string ArmLeftParam = "VCO_L_ArmUpDown";
        public const string ArmRightParam = "VCO_R_ArmUpDown";

        /// <summary>The four Animator/expression parameters of one arm
        /// (issue #28 phase 2), gate first: <c>VCO_x_ArmTracked</c> (Bool,
        /// true while the arm is tracked — the state-machine gate),
        /// <c>VCO_x_ArmUpDown</c> (Float -1 hanging .. +1 overhead, 0 =
        /// horizontal OR pointing at the camera), <c>VCO_x_ArmAcross</c>
        /// (Float +1 across the chest .. -1 out away from the body) and
        /// <c>VCO_x_Elbow</c> (Float 0 straight .. 1 fully bent). The
        /// tracker sends ArmTracked=false and decays the floats to exactly
        /// 0.0 while the arm is untracked.</summary>
        public static string[] ArmParams(bool leftArm)
        {
            var side = leftArm ? "L" : "R";
            return new[]
            {
                $"VCO_{side}_ArmTracked",
                $"VCO_{side}_ArmUpDown",
                $"VCO_{side}_ArmAcross",
                $"VCO_{side}_Elbow",
            };
        }

        /// <summary>The five per-finger curl parameter names of one hand, in
        /// tree-nesting order (Thumb outermost — matches <see cref="Fingers"/>).
        /// 0 = straight, 1 = fully curled.</summary>
        public static string[] CurlParams(bool leftHand)
        {
            var side = leftHand ? "L" : "R";
            return Fingers.Select(f => $"VCO_{side}_{f}Curl").ToArray();
        }

        /// <summary>
        /// ONE Gesture-controller layer per hand (issue #8 phase 3,
        /// <c>OSC_VCO_L_FingerCurls</c> / <c>OSC_VCO_R_FingerCurls</c>)
        /// driving all 15 of that hand's "Stretched" joint muscles from the
        /// five <c>VCO_*Curl</c> floats through a nested Simple1D blend tree:
        /// depth 5 (Thumb → Index → Middle → Ring → Little), two children per
        /// level at thresholds 0 (straight) and 1 (curled), so 2^5 = 32 leaf
        /// clips. Every leaf writes ALL 20 finger muscles — the 15 stretch
        /// joints at +1 (straight) / -1 (curled) per that leaf's finger
        /// combination, plus the 5 Spread muscles pinned at 0 (neutral).
        /// Writing the spreads too is deliberate (issue #27 group lesson,
        /// same as the gesture pose clips): the whole-Fingers-group override
        /// must always carry every muscle of the group, never leave one at a
        /// stale default. A half-curled param blends linearly between the 0
        /// and 1 slots of its level, so each finger moves smoothly and
        /// independently.
        ///
        /// MUTUALLY EXCLUSIVE with <see cref="AddHandGestureLayer"/>: both
        /// write the same per-hand Fingers muscle group, and two Override
        /// layers masking one group fight (last-one-wins for the whole
        /// group, issue #27) — the wizard's hand mode
        /// (<see cref="ApplyHandMode"/>) applies one and removes the other.
        /// Like the gesture layers: Override blending, per-hand fingers-only
        /// mask, no VRCAnimatorTrackingControl (fingers aren't IK-held on
        /// Desktop).
        /// </summary>
        public static void AddFingerCurlLayer(AnimatorController controller, bool leftHand)
        {
            var key = leftHand ? FingerCurlLeftKey : FingerCurlRightKey;

            // Full sub-asset cleanup before rebuilding (idempotent re-Apply).
            RemoveLayer(controller, key);

            var curlParams = CurlParams(leftHand);
            foreach (var p in curlParams)
            {
                EnsureFloatParameter(controller, p);
            }

            var layerName = LayerNameFor(key);
            var root = (BlendTree)BuildCurlTree(controller, layerName, leftHand, curlParams, 0, new bool[Fingers.Length]);
            AddLayer(controller, key, root, AnimatorLayerBlendingMode.Override,
                GetOrCreateFingersOnlyMask(controller, leftHand));
        }

        /// <summary>Recursive nested-Simple1D construction for
        /// <see cref="AddFingerCurlLayer"/>: at <paramref name="depth"/> ==
        /// <c>Fingers.Length</c> the branch bottoms out in a leaf clip for
        /// the accumulated straight/curled combination.</summary>
        static Motion BuildCurlTree(
            AnimatorController controller,
            string layerName,
            bool leftHand,
            string[] curlParams,
            int depth,
            bool[] curled)
        {
            if (depth == Fingers.Length)
            {
                return CurlPoseClip(controller, layerName, leftHand, curled);
            }

            var tree = new BlendTree
            {
                name = $"{layerName}_{CurlSuffix(curled, depth)}{Fingers[depth]}",
                blendType = BlendTreeType.Simple1D,
                blendParameter = curlParams[depth],
                useAutomaticThresholds = false,
            };
            AssetDatabase.AddObjectToAsset(tree, controller);
            curled[depth] = false;
            tree.AddChild(BuildCurlTree(controller, layerName, leftHand, curlParams, depth + 1, curled), 0f);
            curled[depth] = true;
            tree.AddChild(BuildCurlTree(controller, layerName, leftHand, curlParams, depth + 1, curled), 1f);
            curled[depth] = false;
            return tree;
        }

        /// <summary>Asset-name suffix encoding a straight/curled combination
        /// prefix, e.g. <c>T1I0</c> = thumb curled, index straight.</summary>
        static string CurlSuffix(bool[] curled, int depth)
        {
            var s = new System.Text.StringBuilder();
            for (var i = 0; i < depth; i++)
            {
                s.Append(Fingers[i][0]).Append(curled[i] ? '1' : '0');
            }
            return s.ToString();
        }

        /// <summary>One curl leaf clip writing ALL 20 of that hand's finger
        /// muscles as flat 1-second curves: stretch joints -1 when that
        /// finger's slot is curled / +1 when straight, spreads 0.</summary>
        static AnimationClip CurlPoseClip(
            AnimatorController controller,
            string layerName,
            bool leftHand,
            bool[] curled)
        {
            var clip = new AnimationClip { name = $"{layerName}_{CurlSuffix(curled, curled.Length)}" };
            var hand = leftHand ? "LeftHand" : "RightHand";
            for (var f = 0; f < Fingers.Length; f++)
            {
                foreach (var joint in new[] { "1 Stretched", "2 Stretched", "3 Stretched", "Spread" })
                {
                    var value = joint == "Spread" ? 0f : (curled[f] ? -1f : 1f);
                    var binding = EditorCurveBinding.FloatCurve(
                        string.Empty, typeof(Animator), $"{hand}.{Fingers[f]}.{joint}");
                    var curve = new AnimationCurve(
                        new Keyframe(0f, value),
                        new Keyframe(MuscleClipSeconds, value));
                    AnimationUtility.SetEditorCurve(clip, binding, curve);
                }
            }
            AssetDatabase.AddObjectToAsset(clip, controller);
            return clip;
        }

        /// <summary>
        /// Applies the wizard's hand mode (issue #8, see
        /// <see cref="HandMode"/>) to the Gesture controller: adds the
        /// selected mode's per-hand layers and removes the other mode's —
        /// gesture poses and per-finger curls both write the same Fingers
        /// muscle groups, so they must never coexist (issue #27 per-group
        /// override lesson). <see cref="HandMode.Off"/> removes both.
        /// </summary>
        public static void ApplyHandMode(AnimatorController controller, HandMode mode)
        {
            if (mode == HandMode.Gestures)
            {
                AddHandGestureLayer(controller, leftHand: true);
                AddHandGestureLayer(controller, leftHand: false);
            }
            else
            {
                RemoveLayer(controller, GestureLeftParam);
                RemoveLayer(controller, GestureRightParam);
            }

            if (mode == HandMode.FingerCurls)
            {
                AddFingerCurlLayer(controller, leftHand: true);
                AddFingerCurlLayer(controller, leftHand: false);
            }
            else
            {
                RemoveLayer(controller, FingerCurlLeftKey);
                RemoveLayer(controller, FingerCurlRightKey);
            }
        }

        /// <summary>Seconds of cross-fade for the arm layer's Idle ⇄
        /// Active ⇄ Rest transitions — long enough that gaining/losing arm
        /// tracking raises/lowers the arm smoothly instead of snapping.</summary>
        const float ArmTransitionSeconds = 0.25f;

        /// <summary>The nine muscles of one arm's muscle group, in the order
        /// the <see cref="ArmAnchorTable"/> pose rows list their values.
        /// Sign conventions (verified live for the head muscles, issue #27):
        /// the name's SECOND word is +1 — "Down-Up" +1 = up, "Front-Back"
        /// +1 = back (so reaching forward is NEGATIVE), "In-Out" +1 = out.
        /// "Forearm Stretch" +1 = straight, -1 = fully bent (same as the
        /// finger "Stretched" muscles). Muscle names are per-side
        /// ("Left Arm Down-Up" / "Right Arm Down-Up") with identical,
        /// body-relative values — Unity mirrors them, so one table serves
        /// both arms.</summary>
        static readonly string[] ArmMuscles =
        {
            "Shoulder Down-Up",
            "Shoulder Front-Back",
            "Arm Down-Up",
            "Arm Front-Back",
            "Arm Twist In-Out",
            "Forearm Stretch",
            "Forearm Twist In-Out",
            "Hand Down-Up",
            "Hand In-Out",
        };

        /// <summary>
        /// The five 2D anchor poses of <see cref="AddArmLayer"/>'s Freeform
        /// Cartesian blend tree (issue #28 phase 2), each with an
        /// elbow-straight and an elbow-fully-bent muscle row (the nested
        /// Simple1D elbow dimension), 10 poses per arm total. Anchor
        /// positions are (Across, UpDown) in the tracker's parameter space:
        /// UpDown -1 = hanging / +1 = overhead / 0 = horizontal OR pointing
        /// at the camera; Across +1 = toward the opposite shoulder / -1 =
        /// out away from the body / 0 = vertical-forward plane.
        ///
        /// Values are PROVISIONAL live-tuning material, calibrated against
        /// the phase-1 table that WAS live-tuned (hang: Arm Down-Up -0.5,
        /// Forearm Stretch 0.85; overhead: Arm Down-Up 0.95, Forearm
        /// Stretch 0.7). Muscle order per row = <see cref="ArmMuscles"/>.
        /// Every row writes ALL nine muscles (issue #27 group lesson —
        /// VRChat composes humanoid Override layers per masked muscle
        /// group, partial writes break the unwritten muscles).
        /// </summary>
        static readonly (string name, float across, float upDown, float[] straight, float[] bent)[] ArmAnchorTable =
        {
            // Muscle order:  ShDU   ShFB   ArmDU  ArmFB  ATwIO  FaStr  FaTwIO HaDU  HaIO
            ("down", 0f, -1f,
                // Relaxed hang — the phase-1 live-tuned rest pose: upper arm
                // a soft -0.5 (not a rigid pole), forearm nearly straight.
                new[] { -0.1f,  0f,   -0.5f,  0f,    0f,    0.85f, 0f,    0f,   0f },
                // Bicep-curl hang: upper arm stays down, forearm folds up in
                // front (hand near the shoulder), slight palm-in twist.
                new[] { -0.1f,  0f,   -0.5f,  0f,    0f,   -0.8f, -0.3f,  0f,   0f }),
            ("up", 0f, 1f,
                // Straight overhead — the phase-1 live-tuned top pose:
                // shoulder shrugged up, arm almost vertical, forearm mostly
                // straight.
                new[] {  0.5f,  0f,    0.95f, 0f,    0f,    0.7f,  0f,    0f,   0f },
                // Overhead with bent elbow: hand drops behind/above the head
                // (stretch/scratch-the-neck pose).
                new[] {  0.5f,  0f,    0.95f, 0f,    0f,   -0.8f, -0.3f,  0f,   0f }),
            ("across", 1f, 0f,
                // Horizontal reach across the chest toward the opposite
                // shoulder: strong arm-forward (Front-Back negative) plus
                // shoulder protraction (Shoulder Front-Back negative) gives
                // the cross-body sweep; upper arm slightly under horizontal.
                new[] {  0.1f, -0.5f,  0.25f, -0.9f, 0f,    0.85f, 0f,    0f,   0f },
                // Hand lands on the opposite shoulder (arm hugs the chest).
                new[] {  0.1f, -0.5f,  0.25f, -0.9f, 0f,   -0.8f, -0.3f,  0f,   0f }),
            ("out", -1f, 0f,
                // Horizontal out to the side, T-pose-ish: Arm Down-Up in the
                // 0.4-ish horizontal range, no front-back component.
                new[] {  0.2f,  0f,    0.4f,  0f,    0f,    0.85f, 0f,    0f,   0f },
                // Bicep-flex from the T: upper arm stays out, forearm folds
                // up (the 💪 pose).
                new[] {  0.2f,  0f,    0.4f,  0f,    0f,   -0.8f, -0.3f,  0f,   0f }),
            ("forward", 0f, 0f,
                // Horizontal pointing forward at the camera: strong
                // arm-forward, slight shoulder protraction (matches the
                // phase-1 mid pose's direction values).
                new[] {  0.1f, -0.2f,  0.3f, -0.6f,  0f,    0.85f, 0f,    0f,   0f },
                // Elbow bent with the upper arm forward: hand comes back
                // toward the face — the natural wave the phase-1 mid pose
                // approximated with a half bend.
                new[] {  0.1f, -0.2f,  0.3f, -0.6f,  0f,   -0.8f, -0.3f,  0.1f, 0f }),
        };

        /// <summary>The animatable muscle-curve property names for one arm's
        /// muscle group, 9 total. Unlike the fingers (whose binding names
        /// diverge from <see cref="HumanTrait.MuscleName"/> — the
        /// "LeftHand.Thumb.1 Stretched" quirk), the arm names bind exactly
        /// as HumanTrait lists them ("Left Arm Down-Up" style, like the head
        /// muscles) — verified empirically via clip.humanMotion in
        /// MuscleNameValidityTests.</summary>
        internal static IEnumerable<string> ArmMuscleProperties(bool leftArm)
        {
            var side = leftArm ? "Left" : "Right";
            return ArmMuscles.Select(m => $"{side} {m}");
        }

        /// <summary>
        /// ONE Gesture-controller layer per arm (issue #28 phase 2,
        /// <c>OSC_VCO_L_ArmUpDown</c> / <c>OSC_VCO_R_ArmUpDown</c>) posing
        /// the whole arm from the tracker's MediaPipe-Pose arm parameters
        /// (<see cref="ArmParams"/>):
        ///
        /// - The state machine is gated by the <c>VCO_x_ArmTracked</c> Bool
        ///   (phase 2): Idle→Active on If, Active→Rest on IfNot,
        ///   Rest→Active on If (re-acquire mid-settle), Rest→Idle on exit
        ///   time. The phase-1 ±0.02 deadband-on-UpDown gate is retired —
        ///   it collided with "arm pointing at the camera", which
        ///   legitimately reads (0,0).
        /// - The default <b>Idle</b> state is EMPTY (no motion): while the
        ///   arm is untracked the layer writes nothing, so the avatar's own
        ///   idle/locomotion arm animation passes through untouched.
        /// - The <b>Active</b> state is a 2D Freeform Cartesian tree,
        ///   x = <c>VCO_x_ArmAcross</c>, y = <c>VCO_x_ArmUpDown</c>, with
        ///   five direction anchors (<see cref="ArmAnchorTable"/>: down /
        ///   up / across-chest / out-to-side / forward-at-camera). Each
        ///   anchor is itself a nested Simple1D tree on <c>VCO_x_Elbow</c>
        ///   (0 straight / 1 fully bent) — 10 leaf clips per arm, every
        ///   leaf writing ALL nine arm muscles as flat 1-second curves
        ///   (issue #27 group lesson + real-length convention: zero-length
        ///   clips never advance normalized time in the client, issue #25).
        /// - <b>Rest</b> carries the hanging pose: the release path travels
        ///   through it before handing over to the empty Idle state (issue
        ///   #28 live feedback — with write-defaults off, an
        ///   empty-state-only release froze the arm wherever tracking was
        ///   lost). Rest→Idle fires on exit time 1.2 so any WD-off residue
        ///   IS the rest pose when idle animation resumes.
        /// - Masked to that arm's body part only
        ///   (AvatarMaskBodyPart.Left/RightArm) — composes with the
        ///   fingers-masked hand layers, they touch disjoint groups.
        /// - Deliberately NO VRCAnimatorTrackingControl: Desktop arms are
        ///   animation-driven (the avatar's own idle animation moving them
        ///   proves it), unlike the IK-held head. If live testing ever shows
        ///   IK stealing the arms after all, the fallback is the head-saga
        ///   recipe: TrackingControl (Left/RightHand = Animation) on
        ///   ping-pong exit-time states (see AddCombinedHeadLayer).
        /// </summary>
        public static void AddArmLayer(AnimatorController controller, bool leftArm)
        {
            var key = leftArm ? ArmLeftParam : ArmRightParam;

            // Full sub-asset cleanup before rebuilding (idempotent re-Apply;
            // also migrates a phase-1 1-D layer in place — same layer name).
            RemoveLayer(controller, key);

            var armParams = ArmParams(leftArm);
            var trackedParam = armParams[0];
            var upDownParam = armParams[1];
            var acrossParam = armParams[2];
            var elbowParam = armParams[3];
            if (!controller.parameters.Any(p => p.name == trackedParam))
            {
                controller.AddParameter(trackedParam, AnimatorControllerParameterType.Bool);
            }
            EnsureFloatParameter(controller, upDownParam);
            EnsureFloatParameter(controller, acrossParam);
            EnsureFloatParameter(controller, elbowParam);

            var layerName = LayerNameFor(key);
            var stateMachine = new AnimatorStateMachine { name = layerName, hideFlags = HideFlags.HideInHierarchy };
            AssetDatabase.AddObjectToAsset(stateMachine, controller);

            var idle = stateMachine.AddState("Idle");
            idle.motion = null;
            idle.writeDefaultValues = false;
            stateMachine.defaultState = idle;

            // 2D direction tree with a nested elbow dimension per anchor.
            // useAutomaticThresholds stays false everywhere — automatic mode
            // silently rewrites Simple1D thresholds on AddChild (the eyelid
            // 0.75-neutral bug, issue #21).
            var top = new BlendTree
            {
                name = layerName,
                blendType = BlendTreeType.FreeformCartesian2D,
                blendParameter = acrossParam,
                blendParameterY = upDownParam,
                useAutomaticThresholds = false,
            };
            AssetDatabase.AddObjectToAsset(top, controller);
            foreach (var (name, across, upDown, straight, bent) in ArmAnchorTable)
            {
                var elbow = new BlendTree
                {
                    name = $"{layerName}_{name}",
                    blendType = BlendTreeType.Simple1D,
                    blendParameter = elbowParam,
                    useAutomaticThresholds = false,
                };
                AssetDatabase.AddObjectToAsset(elbow, controller);
                elbow.AddChild(ArmPoseClip(controller, $"{layerName}_{name}_straight", leftArm, straight), 0f);
                elbow.AddChild(ArmPoseClip(controller, $"{layerName}_{name}_bent", leftArm, bent), 1f);
                top.AddChild(elbow, new Vector2(across, upDown));
            }

            var active = stateMachine.AddState("Active");
            active.motion = top;
            active.writeDefaultValues = false;

            // Rest = the hanging pose (the "down"/straight-elbow anchor row).
            var rest = stateMachine.AddState("Rest");
            rest.motion = ArmPoseClip(controller, layerName + "_Rest", leftArm, ArmAnchorTable[0].straight);
            rest.writeDefaultValues = false;

            // Raise: leave Idle (or interrupt Rest) the moment the arm is
            // tracked again.
            ConfigureArmTransition(idle.AddTransition(active))
                .AddCondition(AnimatorConditionMode.If, 0f, trackedParam);
            ConfigureArmTransition(rest.AddTransition(active))
                .AddCondition(AnimatorConditionMode.If, 0f, trackedParam);
            // Lower: Active -> Rest as soon as tracking is lost.
            ConfigureArmTransition(active.AddTransition(rest))
                .AddCondition(AnimatorConditionMode.IfNot, 0f, trackedParam);
            // Settle: Rest -> Idle on exit time (the 1 s rest clip plays
            // once, then idle animation takes back over).
            var settle = rest.AddTransition(idle);
            settle.hasExitTime = true;
            settle.exitTime = 1.2f;
            settle.hasFixedDuration = true;
            settle.duration = ArmTransitionSeconds;

            var layers = controller.layers.ToList();
            layers.Add(new AnimatorControllerLayer
            {
                name = layerName,
                stateMachine = stateMachine,
                defaultWeight = 1f,
                blendingMode = AnimatorLayerBlendingMode.Override,
                avatarMask = GetOrCreateArmOnlyMask(controller, leftArm),
            });
            controller.layers = layers.ToArray();
        }

        static AnimatorStateTransition ConfigureArmTransition(AnimatorStateTransition transition)
        {
            transition.hasExitTime = false;
            transition.hasFixedDuration = true;
            transition.duration = ArmTransitionSeconds;
            return transition;
        }

        /// <summary>One arm pose clip writing ALL nine of that arm's muscles
        /// as flat 1-second curves — <paramref name="muscleValues"/> in
        /// <see cref="ArmMuscles"/> order (a row of
        /// <see cref="ArmAnchorTable"/>).</summary>
        static AnimationClip ArmPoseClip(
            AnimatorController controller,
            string clipName,
            bool leftArm,
            float[] muscleValues)
        {
            var clip = new AnimationClip { name = clipName };
            var side = leftArm ? "Left" : "Right";
            for (var i = 0; i < ArmMuscles.Length; i++)
            {
                var binding = EditorCurveBinding.FloatCurve(
                    string.Empty, typeof(Animator), $"{side} {ArmMuscles[i]}");
                var curve = new AnimationCurve(
                    new Keyframe(0f, muscleValues[i]),
                    new Keyframe(MuscleClipSeconds, muscleValues[i]));
                AnimationUtility.SetEditorCurve(clip, binding, curve);
            }
            AssetDatabase.AddObjectToAsset(clip, controller);
            return clip;
        }

        /// <summary>Per-arm mask (mirrors
        /// <see cref="GetOrCreateFingersOnlyMask"/>): contains the layer's
        /// written effect to that arm's body part so Humanoid retargeting
        /// can't perturb the torso — the same PhysBone-runaway class of bug
        /// the head-only mask exists for.</summary>
        static AvatarMask GetOrCreateArmOnlyMask(AnimatorController controller, bool leftArm)
        {
            var maskName = leftArm ? "OSC_LeftArmMask" : "OSC_RightArmMask";
            var allowed = leftArm ? AvatarMaskBodyPart.LeftArm : AvatarMaskBodyPart.RightArm;
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
                mask.SetHumanoidBodyPartActive(part, part == allowed);
            }
            AssetDatabase.AddObjectToAsset(mask, controller);
            return mask;
        }

        /// <summary>Pseudo-parameter key of the per-controller binary decode
        /// layer (issue #29), mirroring <see cref="CombinedHeadKey"/>: it
        /// derives the layer/asset name (<c>OSC_BinaryDecode</c>); the real
        /// Animator parameters are the face floats it reconstructs plus
        /// their bit params and <see cref="ConstantOneParam"/>.</summary>
        public const string BinaryDecodeKey = "BinaryDecode";

        /// <summary>Float animator parameter pinned at its default of 1 —
        /// the Direct Blend Tree weight for children that must always be
        /// fully on (the signed params' sign-select subtrees). Never synced,
        /// never written; the standard DBT constant-weight trick.</summary>
        public const string ConstantOneParam = "VCO_One";

        /// <summary>
        /// ONE layer per controller decoding the VRCFT binary Bool groups
        /// back into the float Animator parameters the existing driving
        /// layers read (issue #29) — those layers stay untouched, so the
        /// eyelid 0.75-neutral tree and the combined 27-leaf head tree keep
        /// their exact behavior; only the parameter's source changes
        /// (synced Float → decoded AAP).
        ///
        /// Construction (the VRCFT-community standard, verified against
        /// rrazgriz/VRCFTGenerator's decode layer and docs.vrcft.io):
        ///
        /// - VRChat casts a synced Bool expression parameter onto a
        ///   same-named <b>Float</b> animator parameter as 0.0/1.0
        ///   (creators.vrchat.com/avatars/animator-parameters, "Mismatched
        ///   parameter types": bool → float, true is 1.0) — so every bit
        ///   param is declared here as a Float.
        /// - The single state's motion is a Direct Blend Tree with one child
        ///   per bit: a clip writing the base parameter (an Animated
        ///   Animator Parameter) to that bit's contribution
        ///   <c>2^k/(2^N−1)</c>, weighted by the bit's 0/1 Float. Direct
        ///   children compose additively (unnormalized), so the children sum
        ///   to the decoded value — all bits set decodes to exactly 1.0,
        ///   matching the encoder's ≥0.99999 → all-ones saturation
        ///   (VRCFT BinaryBaseParameter, ported in src/mapping/avatar.rs).
        /// - Signed params (the -1..1 head axes) add a sign-select Simple1D
        ///   tree on <c>&lt;Name&gt;Negative</c> (0 → the positive bit tree,
        ///   1 → a mirrored tree whose clips write −2^k/(2^N−1)), under a
        ///   <see cref="ConstantOneParam"/> direct weight.
        /// - The state has write defaults ON — the known Direct-Blend-Tree /
        ///   AAP requirement (unlike every other wizard layer, which is WD
        ///   off by the issue #25/#28 lessons; this layer animates only
        ///   animator parameters, so WD on cannot leak into the pose).
        ///
        /// Call BEFORE the driving layers are (re)added so this layer sits
        /// above them: parameter writes propagate to later-evaluated layers.
        /// </summary>
        public static void AddBinaryDecodeLayer(
            AnimatorController controller, IEnumerable<OscParamSpec> faceSpecs, int binaryBits)
        {
            // Full sub-asset cleanup before rebuilding (idempotent re-Apply;
            // also drops stale bit params when the resolution changes).
            RemoveLayer(controller, BinaryDecodeKey);

            var specs = faceSpecs.ToList();
            if (specs.Count == 0)
            {
                return;
            }

            EnsureFloatParameter(controller, ConstantOneParam, defaultFloat: 1f);

            var layerName = LayerNameFor(BinaryDecodeKey);
            var root = new BlendTree
            {
                name = layerName,
                blendType = BlendTreeType.Direct,
                useAutomaticThresholds = false,
            };
            AssetDatabase.AddObjectToAsset(root, controller);

            var children = new List<ChildMotion>();
            foreach (var spec in specs)
            {
                // The decode target — the same Float the driving layers
                // blend on; with binary declarations there is no synced
                // Float of this name, so the AAP owns it.
                EnsureFloatParameter(controller, spec.Name);
                foreach (var bitName in BitParamNames(spec.Name, binaryBits))
                {
                    EnsureFloatParameter(controller, bitName);
                }

                if (!OscParameterSpec.IsSigned(spec))
                {
                    children.AddRange(BitChildren(controller, layerName, spec.Name, binaryBits, sign: +1f));
                }
                else
                {
                    var negativeParam = spec.Name + "Negative";
                    EnsureFloatParameter(controller, negativeParam);
                    var signTree = new BlendTree
                    {
                        name = $"{layerName}_{spec.Name.Replace('/', '_')}_sign",
                        blendType = BlendTreeType.Simple1D,
                        blendParameter = negativeParam,
                        useAutomaticThresholds = false,
                    };
                    AssetDatabase.AddObjectToAsset(signTree, controller);
                    signTree.AddChild(
                        BitTree(controller, layerName, spec.Name, binaryBits, sign: +1f), 0f);
                    signTree.AddChild(
                        BitTree(controller, layerName, spec.Name, binaryBits, sign: -1f), 1f);
                    children.Add(new ChildMotion
                    {
                        motion = signTree,
                        directBlendParameter = ConstantOneParam,
                        timeScale = 1f,
                    });
                }
            }
            root.children = children.ToArray();

            var stateMachine = new AnimatorStateMachine { name = layerName, hideFlags = HideFlags.HideInHierarchy };
            AssetDatabase.AddObjectToAsset(stateMachine, controller);
            var state = stateMachine.AddState(layerName);
            state.motion = root;
            // WD ON — see the doc comment; DBT/AAP layers require it.
            state.writeDefaultValues = true;
            stateMachine.defaultState = state;

            var layers = controller.layers.ToList();
            layers.Add(new AnimatorControllerLayer
            {
                name = layerName,
                stateMachine = stateMachine,
                defaultWeight = 1f,
                blendingMode = AnimatorLayerBlendingMode.Override,
                avatarMask = null, // animates only animator parameters
            });
            controller.layers = layers.ToArray();
        }

        /// <summary>The bit Float parameter names of one binarized face
        /// float, LSB first: <c>&lt;name&gt;1</c>, <c>&lt;name&gt;2</c>,
        /// <c>&lt;name&gt;4</c>, ...</summary>
        static IEnumerable<string> BitParamNames(string paramName, int binaryBits)
        {
            for (var k = 0; k < binaryBits; k++)
            {
                yield return paramName + (1 << k);
            }
        }

        /// <summary>Direct-tree children decoding one param's bits with the
        /// given sign: bit k's clip writes the AAP to
        /// <c>sign · 2^k/(2^N−1)</c>, weighted by that bit's 0/1 Float.</summary>
        static IEnumerable<ChildMotion> BitChildren(
            AnimatorController controller, string layerName, string paramName, int binaryBits, float sign)
        {
            var steps = (1 << binaryBits) - 1;
            for (var k = 0; k < binaryBits; k++)
            {
                var contribution = sign * (1 << k) / (float)steps;
                yield return new ChildMotion
                {
                    motion = AapClip(controller, layerName, paramName, 1 << k, contribution),
                    directBlendParameter = paramName + (1 << k),
                    timeScale = 1f,
                };
            }
        }

        /// <summary>One sign's bit-summing Direct tree (the sign-select
        /// Simple1D's children for the -1..1 head axes).</summary>
        static BlendTree BitTree(
            AnimatorController controller, string layerName, string paramName, int binaryBits, float sign)
        {
            var tree = new BlendTree
            {
                name = $"{layerName}_{paramName.Replace('/', '_')}_{(sign < 0f ? "neg" : "pos")}",
                blendType = BlendTreeType.Direct,
                useAutomaticThresholds = false,
            };
            AssetDatabase.AddObjectToAsset(tree, controller);
            tree.children = BitChildren(controller, layerName, paramName, binaryBits, sign).ToArray();
            return tree;
        }

        /// <summary>A clip writing one Animator Float parameter (an AAP) to
        /// a constant value — flat 1-second curve like every other wizard
        /// clip (<see cref="MuscleClipSeconds"/>).</summary>
        static AnimationClip AapClip(
            AnimatorController controller, string layerName, string paramName, int bitSuffix, float value)
        {
            var clip = new AnimationClip
            {
                name = $"{layerName}_{paramName.Replace('/', '_')}_b{bitSuffix}{(value < 0f ? "_neg" : "")}",
            };
            var binding = EditorCurveBinding.FloatCurve(string.Empty, typeof(Animator), paramName);
            var curve = new AnimationCurve(
                new Keyframe(0f, value),
                new Keyframe(MuscleClipSeconds, value));
            AnimationUtility.SetEditorCurve(clip, binding, curve);
            AssetDatabase.AddObjectToAsset(clip, controller);
            return clip;
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

                // The hand-gesture layers (issue #8) switch pose states via
                // any-state transitions — those live on the state machine, not
                // on any state's `transitions`, and are sub-assets like the
                // rest.
                foreach (var transition in layer.stateMachine.anyStateTransitions)
                {
                    if (transition == null)
                    {
                        continue;
                    }
                    AssetDatabase.RemoveObjectFromAsset(transition);
                    Object.DestroyImmediate(transition, true);
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

            // The combined head layer, the per-hand finger-curl layers, the
            // per-arm layers, and the binary decode layer are keyed by
            // pseudo-params; their real Animator parameters are the three
            // head axes / that hand's five curl floats / that arm's Bool
            // gate + three floats / the bit params + constant-1
            // (ArmLeftParam doubles as the UpDown parameter name, so the
            // arm mapping is a superset of the bare-name fallback). The
            // decode layer's mapping deliberately excludes the base float
            // names — the driving layers own those.
            var toDrop = paramName == CombinedHeadKey
                ? HeadAxes.Select(a => a.param).ToArray()
                : paramName == FingerCurlLeftKey
                    ? CurlParams(leftHand: true)
                    : paramName == FingerCurlRightKey
                        ? CurlParams(leftHand: false)
                        : paramName == ArmLeftParam
                            ? ArmParams(leftArm: true)
                            : paramName == ArmRightParam
                                ? ArmParams(leftArm: false)
                                : paramName == BinaryDecodeKey
                                    ? new[] { ConstantOneParam }
                                        .Concat(OscParameterSpec.AllPossibleBinaryNames())
                                        .ToArray()
                                    : new[] { paramName };
            controller.parameters = controller.parameters.Where(p => !toDrop.Contains(p.name)).ToArray();
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

        static void EnsureFloatParameter(AnimatorController controller, string paramName, float defaultFloat = 0f)
        {
            if (controller.parameters.Any(p => p.name == paramName))
            {
                return;
            }
            controller.AddParameter(new AnimatorControllerParameter
            {
                name = paramName,
                type = AnimatorControllerParameterType.Float,
                defaultFloat = defaultFloat,
            });
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

    }
}
