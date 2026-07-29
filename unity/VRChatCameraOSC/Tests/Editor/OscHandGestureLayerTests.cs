using System.Linq;
using NUnit.Framework;
using UnityEditor;
using UnityEditor.Animations;
using UnityEngine;
using VRC.SDK3.Avatars.ScriptableObjects;

namespace VRChatCameraOsc.AvatarSetup.Tests
{
    /// <summary>
    /// Issue #8 phase 2: <c>VCO_GestureLeft/Right</c> Int parameters + one
    /// Gesture-controller hand-pose layer per hand
    /// (<see cref="OscAnimatorLayerBuilder.AddHandGestureLayer"/>).
    /// </summary>
    public class OscHandGestureLayerTests
    {
        const string TestDir = "Assets/_VRChatCameraOscHandTests";

        AnimatorController _controller;

        [SetUp]
        public void SetUp()
        {
            if (!AssetDatabase.IsValidFolder(TestDir))
            {
                AssetDatabase.CreateFolder("Assets", "_VRChatCameraOscHandTests");
            }
            _controller = AnimatorController.CreateAnimatorControllerAtPath(TestDir + "/Hands.controller");
        }

        [TearDown]
        public void TearDown()
        {
            AssetDatabase.DeleteAsset(TestDir);
        }

        [Test]
        public void Spec_DeclaresBothGestureInts_ModeGatedNotCore()
        {
            foreach (var name in new[] { "VCO_GestureLeft", "VCO_GestureRight" })
            {
                var spec = OscParameterSpec.All.Single(s => s.Name == name);
                Assert.AreEqual(OscParamKind.GestureInt, spec.Kind, name);
                Assert.AreEqual(0f, spec.Min, name);
                Assert.AreEqual(7f, spec.Max, name);
                Assert.AreEqual(0f, spec.DefaultValue, name);
                // Since the 3-way hand mode (issue #8 phase 3) the gesture
                // ints follow the mode instead of being always-core: they're
                // only declared in Gestures mode (see
                // AvatarSetupWindow.SpecsToDeclare), never alongside curls.
                Assert.IsTrue(OscParameterSpec.IsModeGated(spec.Kind), $"{name} must be mode-gated");
                Assert.IsFalse(OscParameterSpec.Core.Any(s => s.Name == name),
                    $"{name} must not be in the always-declared Core set");
            }
        }

        [Test]
        public void Merge_DeclaresGestureIntsAsIntParameters_EverythingElseFloat()
        {
            var asset = ScriptableObject.CreateInstance<VRCExpressionParameters>();
            asset.parameters = new VRCExpressionParameters.Parameter[0];

            VrcExpressionParametersMerger.Merge(asset, OscParameterSpec.All);

            foreach (var p in asset.parameters)
            {
                var expected = p.name.StartsWith("VCO_Gesture")
                    ? VRCExpressionParameters.ValueType.Int
                    : VRCExpressionParameters.ValueType.Float;
                Assert.AreEqual(expected, p.valueType, p.name);
            }
            var left = asset.parameters.Single(p => p.name == "VCO_GestureLeft");
            Assert.AreEqual(0f, left.defaultValue, 1e-6f);
            Assert.IsTrue(left.networkSynced);

            Object.DestroyImmediate(asset);
        }

        [Test]
        public void AddHandGestureLayer_Left_BuildsEmptyNeutralDefault_PlusSevenPoseStates()
        {
            OscAnimatorLayerBuilder.AddHandGestureLayer(_controller, leftHand: true);

            Assert.IsTrue(_controller.parameters.Any(p =>
                p.name == "VCO_GestureLeft" && p.type == AnimatorControllerParameterType.Int));

            var layer = _controller.layers.Single(l => l.name == "OSC_VCO_GestureLeft");
            Assert.AreEqual(AnimatorLayerBlendingMode.Override, layer.blendingMode);
            Assert.AreEqual(1f, layer.defaultWeight, 1e-6f);

            var sm = layer.stateMachine;
            Assert.AreEqual(8, sm.states.Length, "Neutral + 7 poses");

            // At VCO_GestureLeft == 0 the layer must NOT override: an empty
            // (motion-less) default state writes nothing, so VRChat's own
            // keyboard/controller gestures on the stock layers keep working.
            Assert.AreEqual("Neutral", sm.defaultState.name);
            Assert.IsNull(sm.defaultState.motion);

            CollectionAssert.AreEquivalent(
                new[] { "Neutral", "Fist", "HandOpen", "FingerPoint", "Victory", "RockNRoll", "HandGun", "ThumbsUp" },
                sm.states.Select(s => s.state.name).ToArray());
        }

        [Test]
        public void AddHandGestureLayer_AnyStateTransitions_EqualsConditionPerGestureValue()
        {
            OscAnimatorLayerBuilder.AddHandGestureLayer(_controller, leftHand: true);

            var sm = _controller.layers.Single(l => l.name == "OSC_VCO_GestureLeft").stateMachine;
            Assert.AreEqual(8, sm.anyStateTransitions.Length, "one per gesture value 0..7");

            var expectedValue = new System.Collections.Generic.Dictionary<string, float>
            {
                ["Neutral"] = 0f,
                ["Fist"] = 1f,
                ["HandOpen"] = 2f,
                ["FingerPoint"] = 3f,
                ["Victory"] = 4f,
                ["RockNRoll"] = 5f,
                ["HandGun"] = 6f,
                ["ThumbsUp"] = 7f,
            };
            foreach (var tr in sm.anyStateTransitions)
            {
                Assert.IsFalse(tr.hasExitTime, tr.destinationState.name);
                Assert.IsFalse(tr.canTransitionToSelf,
                    $"{tr.destinationState.name}: a held gesture must not re-trigger its own cross-fade");
                Assert.IsTrue(tr.hasFixedDuration, tr.destinationState.name);
                Assert.AreEqual(0.1f, tr.duration, 1e-4f, tr.destinationState.name);

                var condition = tr.conditions.Single();
                Assert.AreEqual(AnimatorConditionMode.Equals, condition.mode, tr.destinationState.name);
                Assert.AreEqual("VCO_GestureLeft", condition.parameter, tr.destinationState.name);
                Assert.AreEqual(expectedValue[tr.destinationState.name], condition.threshold, 1e-6f);
            }
        }

        [Test]
        public void AddHandGestureLayer_PoseClips_AnimateOnlyThatHandsFingerMuscles_WithRealDuration()
        {
            OscAnimatorLayerBuilder.AddHandGestureLayer(_controller, leftHand: true);

            var sm = _controller.layers.Single(l => l.name == "OSC_VCO_GestureLeft").stateMachine;
            var expectedProps = OscAnimatorLayerBuilder.FingerMuscleProperties(true).ToArray();
            foreach (var child in sm.states.Where(s => s.state.name != "Neutral"))
            {
                var clip = (AnimationClip)child.state.motion;
                Assert.IsNotNull(clip, child.state.name);
                // Zero-length clips broke exit-time machinery elsewhere in
                // this controller (issue #25) — real length by convention.
                Assert.Greater(clip.length, 0.5f, child.state.name);

                var bound = AnimationUtility.GetCurveBindings(clip).Select(b => b.propertyName).ToArray();
                CollectionAssert.AreEquivalent(expectedProps, bound,
                    $"{child.state.name}: every pose must write ALL 20 left-hand finger muscles and nothing else");
            }
        }

        [Test]
        public void AddHandGestureLayer_FistAndOpen_CarryTheExpectedMuscleValues()
        {
            OscAnimatorLayerBuilder.AddHandGestureLayer(_controller, leftHand: true);

            var sm = _controller.layers.Single(l => l.name == "OSC_VCO_GestureLeft").stateMachine;
            float ValueOf(AnimationClip clip, string prop) => AnimationUtility.GetEditorCurve(
                clip,
                AnimationUtility.GetCurveBindings(clip).Single(b => b.propertyName == prop)).Evaluate(0f);

            // Fist: every joint fully curled (-1 Stretched).
            var fist = (AnimationClip)sm.states.Single(s => s.state.name == "Fist").state.motion;
            foreach (var prop in OscAnimatorLayerBuilder.FingerMuscleProperties(true).Where(p => p.EndsWith("Stretched")))
            {
                Assert.AreEqual(-1f, ValueOf(fist, prop), 1e-6f, prop);
            }

            // HandOpen: every joint fully extended (+1), fingers fanned.
            var open = (AnimationClip)sm.states.Single(s => s.state.name == "HandOpen").state.motion;
            foreach (var prop in OscAnimatorLayerBuilder.FingerMuscleProperties(true))
            {
                var expected = prop.EndsWith("Stretched") ? 1f : 0.5f;
                Assert.AreEqual(expected, ValueOf(open, prop), 1e-6f, prop);
            }

            // FingerPoint: only the index extended.
            var point = (AnimationClip)sm.states.Single(s => s.state.name == "FingerPoint").state.motion;
            Assert.AreEqual(1f, ValueOf(point, "LeftHand.Index.1 Stretched"), 1e-6f);
            Assert.AreEqual(-1f, ValueOf(point, "LeftHand.Middle.1 Stretched"), 1e-6f);
            Assert.AreEqual(-1f, ValueOf(point, "LeftHand.Thumb.1 Stretched"), 1e-6f);

            // ThumbsUp: only the thumb extended, stuck out.
            var thumbs = (AnimationClip)sm.states.Single(s => s.state.name == "ThumbsUp").state.motion;
            Assert.AreEqual(1f, ValueOf(thumbs, "LeftHand.Thumb.1 Stretched"), 1e-6f);
            Assert.AreEqual(1f, ValueOf(thumbs, "LeftHand.Thumb.Spread"), 1e-6f);
            Assert.AreEqual(-1f, ValueOf(thumbs, "LeftHand.Index.1 Stretched"), 1e-6f);
        }

        [Test]
        public void AddHandGestureLayer_Right_UsesRightHandMusclesAndParameter()
        {
            OscAnimatorLayerBuilder.AddHandGestureLayer(_controller, leftHand: false);

            Assert.IsTrue(_controller.parameters.Any(p =>
                p.name == "VCO_GestureRight" && p.type == AnimatorControllerParameterType.Int));
            var sm = _controller.layers.Single(l => l.name == "OSC_VCO_GestureRight").stateMachine;
            var fist = (AnimationClip)sm.states.Single(s => s.state.name == "Fist").state.motion;
            var bound = AnimationUtility.GetCurveBindings(fist).Select(b => b.propertyName).ToArray();
            Assert.IsTrue(bound.All(p => p.StartsWith("RightHand.")), "right layer must animate RightHand.* only");
            CollectionAssert.AreEquivalent(OscAnimatorLayerBuilder.FingerMuscleProperties(false).ToArray(), bound);
        }

        [Test]
        public void AddHandGestureLayer_MaskRestrictedToThatHandsFingersOnly()
        {
            OscAnimatorLayerBuilder.AddHandGestureLayer(_controller, leftHand: true);
            OscAnimatorLayerBuilder.AddHandGestureLayer(_controller, leftHand: false);

            var left = _controller.layers.Single(l => l.name == "OSC_VCO_GestureLeft").avatarMask;
            var right = _controller.layers.Single(l => l.name == "OSC_VCO_GestureRight").avatarMask;
            Assert.IsNotNull(left);
            Assert.IsNotNull(right);
            for (var part = AvatarMaskBodyPart.Root; part < AvatarMaskBodyPart.LastBodyPart; part++)
            {
                Assert.AreEqual(part == AvatarMaskBodyPart.LeftFingers,
                    left.GetHumanoidBodyPartActive(part), $"left mask, {part}");
                Assert.AreEqual(part == AvatarMaskBodyPart.RightFingers,
                    right.GetHumanoidBodyPartActive(part), $"right mask, {part}");
            }
        }

        [Test]
        public void AddHandGestureLayer_NoTrackingControlBehaviours()
        {
            // Fingers aren't IK-held on Desktop — the stock gesture layers
            // prove muscle-driven hand poses need no VRCAnimatorTrackingControl.
            OscAnimatorLayerBuilder.AddHandGestureLayer(_controller, leftHand: true);

            var sm = _controller.layers.Single(l => l.name == "OSC_VCO_GestureLeft").stateMachine;
            foreach (var child in sm.states)
            {
                Assert.IsEmpty(child.state.behaviours, child.state.name);
            }
        }

        [Test]
        public void AddHandGestureLayer_ReApply_ReplacesWithoutDuplicatesOrLeakedSubAssets()
        {
            OscAnimatorLayerBuilder.AddHandGestureLayer(_controller, leftHand: true);
            OscAnimatorLayerBuilder.AddHandGestureLayer(_controller, leftHand: true);

            Assert.AreEqual(1, _controller.layers.Count(l => l.name == "OSC_VCO_GestureLeft"));
            Assert.AreEqual(1, _controller.parameters.Count(p => p.name == "VCO_GestureLeft"));

            var assets = AssetDatabase.LoadAllAssetsAtPath(AssetDatabase.GetAssetPath(_controller))
                .Where(a => a != null).ToArray();
            Assert.AreEqual(7, assets.OfType<AnimationClip>().Count(c => c.name.StartsWith("OSC_VCO_GestureLeft")),
                "re-Apply must not leak orphaned pose clips");
            Assert.AreEqual(8, assets.OfType<AnimatorStateTransition>().Count(),
                "re-Apply must not leak orphaned any-state transitions");
            Assert.AreEqual(1, assets.OfType<AvatarMask>().Count(m => m.name == "OSC_LeftFingersMask"));
        }

        [Test]
        public void RemoveLayer_HandGestureLayers_RoundTripToZero()
        {
            OscAnimatorLayerBuilder.AddHandGestureLayer(_controller, leftHand: true);
            OscAnimatorLayerBuilder.AddHandGestureLayer(_controller, leftHand: false);

            Assert.IsTrue(OscAnimatorLayerBuilder.RemoveLayer(_controller, "VCO_GestureLeft"));
            Assert.IsTrue(OscAnimatorLayerBuilder.RemoveLayer(_controller, "VCO_GestureRight"));

            Assert.IsFalse(OscAnimatorLayerBuilder.HasLayer(_controller, "VCO_GestureLeft"));
            Assert.IsFalse(OscAnimatorLayerBuilder.HasLayer(_controller, "VCO_GestureRight"));
            Assert.IsFalse(_controller.parameters.Any(p => p.name.StartsWith("VCO_Gesture")),
                "the Int animator parameters must be removed too");

            var assets = AssetDatabase.LoadAllAssetsAtPath(AssetDatabase.GetAssetPath(_controller))
                .Where(a => a != null).ToArray();
            Assert.IsFalse(assets.OfType<AnimationClip>().Any(), "orphaned pose clips");
            Assert.IsFalse(assets.OfType<AnimatorStateTransition>().Any(), "orphaned any-state transitions");
            Assert.IsFalse(assets.OfType<AnimatorState>().Any(), "orphaned states");
            Assert.IsFalse(assets.OfType<AvatarMask>().Any(m => m.name.StartsWith("OSC_")), "orphaned masks");
        }
    }
}
