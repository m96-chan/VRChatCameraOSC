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

        /// <summary>Animator/expression parameter names for the arm-raise
        /// layers (issue #28): Float -1..1, +1 = straight up, -1 = hanging.
        /// The tracker sends exactly 0.0 while the hand is untracked — the
        /// layer's deadband rests in an empty Neutral state there.</summary>
        public const string ArmLeftParam = "VCO_L_ArmUpDown";
        public const string ArmRightParam = "VCO_R_ArmUpDown";

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

        /// <summary>Deadband around the tracker's exactly-0.0
        /// hand-untracked value: outside it the arm layer's Active state
        /// takes the arm, inside it the empty Neutral state hands the arm
        /// back to idle/locomotion.</summary>
        const float ArmDeadband = 0.02f;

        /// <summary>Seconds of cross-fade for the arm layer's Neutral ⇄
        /// Active transitions — long enough that gaining/losing hand
        /// tracking raises/lowers the arm smoothly instead of snapping.</summary>
        const float ArmTransitionSeconds = 0.25f;

        /// <summary>
        /// Per-arm muscle anchor poses for <see cref="AddArmLayer"/>:
        /// muscle values at parameter -1 (hanging), 0 (mid,
        /// forward-horizontal) and +1 (straight up). PROVISIONAL — tuned on
        /// paper, to be verified live (issue #28 expects a sign-flip round
        /// like the head yaw needed). Notes:
        /// - Muscle sign convention observed on the verified head muscles:
        ///   the name's SECOND word is +1 ("Down-Up" +1 = up, "Left-Right"
        ///   +1 = right) — so "Front-Back" +1 = back, and reaching forward
        ///   is NEGATIVE Front-Back. That's the least-certain guess here.
        /// - "Forearm Stretch" +1 = straight, -1 = fully bent (same
        ///   convention as the finger "Stretched" muscles).
        /// - Hanging uses Arm Down-Up -0.5 (not -1) plus a slightly bent
        ///   forearm — a relaxed hang, not a rigid pole.
        /// - Every anchor writes ALL nine arm-group muscles (issue #27
        ///   group lesson): wrist/twists ride at 0 rather than being left
        ///   unwritten.
        /// </summary>
        /// <summary>Revised after the first live round (issue #28): the
        /// original table kept the forearm near-straight throughout
        /// ("肘がきいていない"), reading as a stiff robot arm. The raise now
        /// travels through a strongly bent elbow at mid height (a natural
        /// wave: the bend does the lifting, hand near the face) and
        /// straightens back out overhead. Values remain provisional live
        /// tuning material.</summary>
        static readonly (string muscle, float hanging, float mid, float up)[] ArmPoseTable =
        {
            //  muscle                 -1 hang   0 mid   +1 up
            ("Shoulder Down-Up",       -0.1f,    0.1f,   0.5f),
            ("Shoulder Front-Back",     0f,     -0.2f,   0f),
            ("Arm Down-Up",            -0.5f,    0.1f,   0.95f),
            ("Arm Front-Back",          0f,     -0.5f,   0f),
            ("Arm Twist In-Out",        0f,      0f,     0f),
            ("Forearm Stretch",         0.85f,  -0.45f,  0.7f),
            ("Forearm Twist In-Out",    0f,      0f,     0f),
            ("Hand Down-Up",            0f,      0.1f,   0f),
            ("Hand In-Out",             0f,      0f,     0f),
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
            return ArmPoseTable.Select(e => $"{side} {e.muscle}");
        }

        /// <summary>
        /// ONE Gesture-controller layer per arm (issue #28 phase 1,
        /// <c>OSC_VCO_L_ArmUpDown</c> / <c>OSC_VCO_R_ArmUpDown</c>) raising
        /// and lowering that arm from the tracker's <c>VCO_*_ArmUpDown</c>
        /// float (-1 hanging .. +1 straight up):
        ///
        /// - The default <b>Neutral</b> state is EMPTY (no motion): while
        ///   the parameter sits inside the ±<see cref="ArmDeadband"/>
        ///   deadband — tracker off, or hand untracked (the tracker sends
        ///   exactly 0.0 then) — the layer writes nothing, so the avatar's
        ///   own idle/locomotion arm animation passes through untouched.
        /// - The <b>Active</b> state is a Simple1D tree on the parameter
        ///   with anchors at -1 / 0 / +1 (<see cref="ArmPoseTable"/>), every
        ///   anchor clip writing ALL nine arm muscles as flat 1-second
        ///   curves (issue #27 group lesson + real-length convention).
        /// - Neutral→Active on Greater +deadband OR Less -deadband (two
        ///   transitions); Active→Neutral when back inside the deadband
        ///   (ONE transition with both conditions AND-ed). 0.25 s fixed
        ///   duration both ways for a smooth raise/lower.
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
            var paramName = leftArm ? ArmLeftParam : ArmRightParam;

            // Full sub-asset cleanup before rebuilding (idempotent re-Apply).
            RemoveLayer(controller, paramName);
            EnsureFloatParameter(controller, paramName);

            var layerName = LayerNameFor(paramName);
            var stateMachine = new AnimatorStateMachine { name = layerName, hideFlags = HideFlags.HideInHierarchy };
            AssetDatabase.AddObjectToAsset(stateMachine, controller);

            // Three states (issue #28 live feedback: with an
            // empty-state-only release, write-defaults-off residue left the
            // arm frozen wherever it was when the hand left the frame —
            // "デフォルトの位置に戻らない"). The release path now travels
            // through an explicit Rest pose, THEN hands over to the empty
            // Idle state: the arm visibly returns to a defined hanging pose
            // within ~0.3 s, and once Idle takes over any WD-off residue IS
            // the rest pose, so idle/locomotion animation resumes from a
            // sane baseline either way.
            var idle = stateMachine.AddState("Idle");
            idle.motion = null;
            idle.writeDefaultValues = false;
            stateMachine.defaultState = idle;

            var tree = new BlendTree
            {
                name = layerName,
                blendType = BlendTreeType.Simple1D,
                blendParameter = paramName,
                useAutomaticThresholds = false,
            };
            AssetDatabase.AddObjectToAsset(tree, controller);
            foreach (var anchor in new[] { -1f, 0f, 1f })
            {
                tree.AddChild(ArmPoseClip(controller, layerName, leftArm, anchor, null), anchor);
            }

            var active = stateMachine.AddState("Active");
            active.motion = tree;
            active.writeDefaultValues = false;

            var rest = stateMachine.AddState("Rest");
            rest.motion = ArmPoseClip(controller, layerName, leftArm, -1f, "_Rest");
            rest.writeDefaultValues = false;

            // Raise: leave Idle (or interrupt Rest) as soon as the parameter
            // escapes the deadband in either direction (a condition can't
            // express OR, so two transitions each).
            foreach (var from in new[] { idle, rest })
            {
                ConfigureArmTransition(from.AddTransition(active))
                    .AddCondition(AnimatorConditionMode.Greater, ArmDeadband, paramName);
                ConfigureArmTransition(from.AddTransition(active))
                    .AddCondition(AnimatorConditionMode.Less, -ArmDeadband, paramName);
            }
            // Lower: Active -> Rest only when INSIDE the deadband — both
            // conditions on one transition AND together.
            var release = ConfigureArmTransition(active.AddTransition(rest));
            release.AddCondition(AnimatorConditionMode.Less, ArmDeadband, paramName);
            release.AddCondition(AnimatorConditionMode.Greater, -ArmDeadband, paramName);
            // Settle: Rest -> Idle on exit time (the 1 s rest clip plays
            // once, then idle animation takes back over).
            var settle = rest.AddTransition(idle);
            settle.hasExitTime = true;
            settle.exitTime = 1.2f;
            settle.hasFixedDuration = true;
            settle.duration = ArmTransitionSeconds;

            var layers = controller.layers.ToList();
            var existingIndex = layers.FindIndex(l => l.name == layerName);
            if (existingIndex >= 0)
            {
                layers.RemoveAt(existingIndex);
            }
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

        /// <summary>One arm anchor clip writing ALL nine of that arm's
        /// muscles as flat 1-second curves (<see cref="ArmPoseTable"/>).</summary>
        static AnimationClip ArmPoseClip(
            AnimatorController controller,
            string layerName,
            bool leftArm,
            float anchor,
            string nameSuffix)
        {
            var clip = new AnimationClip { name = $"{layerName}_{nameSuffix ?? anchor.ToString("0.#")}" };
            var side = leftArm ? "Left" : "Right";
            foreach (var (muscle, hanging, mid, up) in ArmPoseTable)
            {
                var value = anchor < 0f ? hanging : (anchor > 0f ? up : mid);
                var binding = EditorCurveBinding.FloatCurve(
                    string.Empty, typeof(Animator), $"{side} {muscle}");
                var curve = new AnimationCurve(
                    new Keyframe(0f, value),
                    new Keyframe(MuscleClipSeconds, value));
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

            // The combined head layer and the per-hand finger-curl layers are
            // keyed by pseudo-params; their real Animator parameters are the
            // three head axes / that hand's five curl floats.
            var toDrop = paramName == CombinedHeadKey
                ? HeadAxes.Select(a => a.param).ToArray()
                : paramName == FingerCurlLeftKey
                    ? CurlParams(leftHand: true)
                    : paramName == FingerCurlRightKey
                        ? CurlParams(leftHand: false)
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

    }
}
