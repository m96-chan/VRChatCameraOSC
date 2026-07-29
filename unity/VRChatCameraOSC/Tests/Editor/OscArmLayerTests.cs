using System.Linq;
using NUnit.Framework;
using UnityEditor;
using UnityEditor.Animations;
using UnityEngine;

namespace VRChatCameraOsc.AvatarSetup.Tests
{
    /// <summary>
    /// Issue #28 phase 1: <c>VCO_L/R_ArmUpDown</c> floats (-1 hanging ..
    /// +1 straight up; exactly 0.0 = hand untracked) driving one arm-raise
    /// layer per arm (<see cref="OscAnimatorLayerBuilder.AddArmLayer"/>):
    /// empty Neutral passthrough state inside the deadband, Simple1D anchor
    /// tree in the Active state outside it.
    /// </summary>
    public class OscArmLayerTests
    {
        const string TestDir = "Assets/_VRChatCameraOscArmTests";

        AnimatorController _controller;

        [SetUp]
        public void SetUp()
        {
            if (!AssetDatabase.IsValidFolder(TestDir))
            {
                AssetDatabase.CreateFolder("Assets", "_VRChatCameraOscArmTests");
            }
            _controller = AnimatorController.CreateAnimatorControllerAtPath(TestDir + "/Arms.controller");
        }

        [TearDown]
        public void TearDown()
        {
            AssetDatabase.DeleteAsset(TestDir);
        }

        [Test]
        public void Spec_DeclaresBothArmFloats_ModeGatedNotCore()
        {
            foreach (var name in new[] { "VCO_L_ArmUpDown", "VCO_R_ArmUpDown" })
            {
                var spec = OscParameterSpec.All.Single(s => s.Name == name);
                Assert.AreEqual(OscParamKind.ArmUpDown, spec.Kind, name);
                Assert.AreEqual(-1f, spec.Min, name);
                Assert.AreEqual(1f, spec.Max, name);
                Assert.AreEqual(0f, spec.DefaultValue, name);
                Assert.IsTrue(OscParameterSpec.IsModeGated(spec.Kind), name);
                Assert.IsFalse(OscParameterSpec.Core.Any(s => s.Name == name),
                    $"{name} is declared only while the arm toggle is on");
            }
        }

        [Test]
        public void AddArmLayer_Left_EmptyNeutralDefault_PlusActiveTreeState()
        {
            OscAnimatorLayerBuilder.AddArmLayer(_controller, leftArm: true);

            Assert.IsTrue(_controller.parameters.Any(p =>
                p.name == "VCO_L_ArmUpDown" && p.type == AnimatorControllerParameterType.Float));

            var layer = _controller.layers.Single(l => l.name == "OSC_VCO_L_ArmUpDown");
            Assert.AreEqual(AnimatorLayerBlendingMode.Override, layer.blendingMode);
            Assert.AreEqual(1f, layer.defaultWeight, 1e-6f);

            var sm = layer.stateMachine;
            CollectionAssert.AreEquivalent(
                new[] { "Neutral", "Active" },
                sm.states.Select(s => s.state.name).ToArray());

            // At 0.0 (hand untracked — the tracker's contract) the layer
            // must write nothing so idle/locomotion arm animation plays.
            Assert.AreEqual("Neutral", sm.defaultState.name);
            Assert.IsNull(sm.defaultState.motion);

            var active = sm.states.Single(s => s.state.name == "Active").state;
            var tree = (BlendTree)active.motion;
            Assert.AreEqual("VCO_L_ArmUpDown", tree.blendParameter);
            Assert.IsFalse(tree.useAutomaticThresholds);
            CollectionAssert.AreEqual(
                new[] { -1f, 0f, 1f },
                tree.children.Select(c => c.threshold).ToArray());
        }

        [Test]
        public void AddArmLayer_ThresholdTransitions_RaiseOnEitherSideOfDeadband_ReturnInsideIt()
        {
            OscAnimatorLayerBuilder.AddArmLayer(_controller, leftArm: true);

            var sm = _controller.layers.Single(l => l.name == "OSC_VCO_L_ArmUpDown").stateMachine;
            var neutral = sm.states.Single(s => s.state.name == "Neutral").state;
            var active = sm.states.Single(s => s.state.name == "Active").state;
            Assert.IsEmpty(sm.anyStateTransitions, "plain state transitions, no any-state");

            // Neutral -> Active: two transitions (conditions can't OR),
            // Greater +0.02 / Less -0.02.
            Assert.AreEqual(2, neutral.transitions.Length);
            foreach (var tr in neutral.transitions)
            {
                Assert.AreSame(active, tr.destinationState);
                Assert.IsFalse(tr.hasExitTime);
                Assert.IsTrue(tr.hasFixedDuration);
                Assert.AreEqual(0.25f, tr.duration, 1e-4f);
                Assert.AreEqual(1, tr.conditions.Length);
                Assert.AreEqual("VCO_L_ArmUpDown", tr.conditions[0].parameter);
            }
            var raise = neutral.transitions.Single(t => t.conditions[0].mode == AnimatorConditionMode.Greater);
            Assert.AreEqual(0.02f, raise.conditions[0].threshold, 1e-6f);
            var lowerSide = neutral.transitions.Single(t => t.conditions[0].mode == AnimatorConditionMode.Less);
            Assert.AreEqual(-0.02f, lowerSide.conditions[0].threshold, 1e-6f);

            // Active -> Neutral: ONE transition whose two conditions AND
            // together to "inside the deadband".
            var back = active.transitions.Single();
            Assert.AreSame(neutral, back.destinationState);
            Assert.IsFalse(back.hasExitTime);
            Assert.IsTrue(back.hasFixedDuration);
            Assert.AreEqual(0.25f, back.duration, 1e-4f);
            Assert.AreEqual(2, back.conditions.Length);
            var less = back.conditions.Single(c => c.mode == AnimatorConditionMode.Less);
            Assert.AreEqual(0.02f, less.threshold, 1e-6f);
            var greater = back.conditions.Single(c => c.mode == AnimatorConditionMode.Greater);
            Assert.AreEqual(-0.02f, greater.threshold, 1e-6f);
        }

        static float ValueOf(AnimationClip clip, string prop) => AnimationUtility.GetEditorCurve(
            clip,
            AnimationUtility.GetCurveBindings(clip).Single(b => b.propertyName == prop)).Evaluate(0f);

        [Test]
        public void AddArmLayer_AnchorClips_WriteAllNineArmMuscles_WithRealDuration()
        {
            OscAnimatorLayerBuilder.AddArmLayer(_controller, leftArm: true);

            var sm = _controller.layers.Single(l => l.name == "OSC_VCO_L_ArmUpDown").stateMachine;
            var tree = (BlendTree)sm.states.Single(s => s.state.name == "Active").state.motion;
            var expectedProps = OscAnimatorLayerBuilder.ArmMuscleProperties(true).ToArray();

            foreach (var child in tree.children)
            {
                var clip = (AnimationClip)child.motion;
                Assert.Greater(clip.length, 0.5f, clip.name);
                var bound = AnimationUtility.GetCurveBindings(clip).Select(b => b.propertyName).ToArray();
                CollectionAssert.AreEquivalent(expectedProps, bound,
                    $"anchor {child.threshold}: every anchor must write ALL nine arm muscles (issue #27 group lesson)");
            }

            // Provisional pose values (documented in ArmPoseTable): spot-check
            // the load-bearing ones per anchor.
            var hanging = (AnimationClip)tree.children[0].motion;
            Assert.AreEqual(-0.5f, ValueOf(hanging, "Left Arm Down-Up"), 1e-6f);
            Assert.AreEqual(0.7f, ValueOf(hanging, "Left Forearm Stretch"), 1e-6f, "slightly bent hang");

            var mid = (AnimationClip)tree.children[1].motion;
            Assert.AreEqual(0.3f, ValueOf(mid, "Left Arm Down-Up"), 1e-6f);
            Assert.AreEqual(-0.7f, ValueOf(mid, "Left Arm Front-Back"), 1e-6f, "reach forward at mid");

            var up = (AnimationClip)tree.children[2].motion;
            Assert.AreEqual(1f, ValueOf(up, "Left Arm Down-Up"), 1e-6f);
            Assert.AreEqual(1f, ValueOf(up, "Left Forearm Stretch"), 1e-6f, "straight arm overhead");
        }

        [Test]
        public void AddArmLayer_MaskRestrictedToThatArmOnly()
        {
            OscAnimatorLayerBuilder.AddArmLayer(_controller, leftArm: true);
            OscAnimatorLayerBuilder.AddArmLayer(_controller, leftArm: false);

            var left = _controller.layers.Single(l => l.name == "OSC_VCO_L_ArmUpDown").avatarMask;
            var right = _controller.layers.Single(l => l.name == "OSC_VCO_R_ArmUpDown").avatarMask;
            Assert.IsNotNull(left);
            Assert.IsNotNull(right);
            for (var part = AvatarMaskBodyPart.Root; part < AvatarMaskBodyPart.LastBodyPart; part++)
            {
                Assert.AreEqual(part == AvatarMaskBodyPart.LeftArm,
                    left.GetHumanoidBodyPartActive(part), $"left mask, {part}");
                Assert.AreEqual(part == AvatarMaskBodyPart.RightArm,
                    right.GetHumanoidBodyPartActive(part), $"right mask, {part}");
            }
        }

        [Test]
        public void AddArmLayer_Right_UsesRightParamAndMuscles()
        {
            OscAnimatorLayerBuilder.AddArmLayer(_controller, leftArm: false);

            Assert.IsTrue(_controller.parameters.Any(p =>
                p.name == "VCO_R_ArmUpDown" && p.type == AnimatorControllerParameterType.Float));
            var sm = _controller.layers.Single(l => l.name == "OSC_VCO_R_ArmUpDown").stateMachine;
            var tree = (BlendTree)sm.states.Single(s => s.state.name == "Active").state.motion;
            Assert.AreEqual("VCO_R_ArmUpDown", tree.blendParameter);
            var clip = (AnimationClip)tree.children[0].motion;
            var bound = AnimationUtility.GetCurveBindings(clip).Select(b => b.propertyName).ToArray();
            Assert.IsTrue(bound.All(p => p.StartsWith("Right ")), "right layer must animate Right * muscles only");
            CollectionAssert.AreEquivalent(OscAnimatorLayerBuilder.ArmMuscleProperties(false).ToArray(), bound);
        }

        [Test]
        public void AddArmLayer_NoTrackingControlBehaviours()
        {
            // Desktop arms are animation-driven (the avatar's own idle
            // animation moving them proves it) — no VRCAnimatorTrackingControl.
            // Fallback if live testing disproves this: the head-saga recipe
            // (TrackingControl + ping-pong states, see AddCombinedHeadLayer).
            OscAnimatorLayerBuilder.AddArmLayer(_controller, leftArm: true);

            var sm = _controller.layers.Single(l => l.name == "OSC_VCO_L_ArmUpDown").stateMachine;
            foreach (var child in sm.states)
            {
                Assert.IsEmpty(child.state.behaviours, child.state.name);
            }
        }

        [Test]
        public void AddArmLayer_ReApply_ReplacesWithoutDuplicatesOrLeakedSubAssets()
        {
            OscAnimatorLayerBuilder.AddArmLayer(_controller, leftArm: true);
            OscAnimatorLayerBuilder.AddArmLayer(_controller, leftArm: true);

            Assert.AreEqual(1, _controller.layers.Count(l => l.name == "OSC_VCO_L_ArmUpDown"));
            Assert.AreEqual(1, _controller.parameters.Count(p => p.name == "VCO_L_ArmUpDown"));

            var assets = AssetDatabase.LoadAllAssetsAtPath(AssetDatabase.GetAssetPath(_controller))
                .Where(a => a != null).ToArray();
            Assert.AreEqual(3, assets.OfType<AnimationClip>().Count(), "3 anchor clips, no orphans");
            Assert.AreEqual(1, assets.OfType<BlendTree>().Count(), "one Simple1D tree, no orphans");
            Assert.AreEqual(3, assets.OfType<AnimatorStateTransition>().Count(),
                "2 raise + 1 release transitions, no orphans");
            Assert.AreEqual(1, assets.OfType<AvatarMask>().Count(m => m.name == "OSC_LeftArmMask"));
        }

        [Test]
        public void RemoveLayer_ArmLayers_RoundTripToZero()
        {
            OscAnimatorLayerBuilder.AddArmLayer(_controller, leftArm: true);
            OscAnimatorLayerBuilder.AddArmLayer(_controller, leftArm: false);

            Assert.IsTrue(OscAnimatorLayerBuilder.RemoveLayer(_controller, OscAnimatorLayerBuilder.ArmLeftParam));
            Assert.IsTrue(OscAnimatorLayerBuilder.RemoveLayer(_controller, OscAnimatorLayerBuilder.ArmRightParam));

            Assert.IsFalse(OscAnimatorLayerBuilder.HasLayer(_controller, OscAnimatorLayerBuilder.ArmLeftParam));
            Assert.IsFalse(OscAnimatorLayerBuilder.HasLayer(_controller, OscAnimatorLayerBuilder.ArmRightParam));
            Assert.IsFalse(_controller.parameters.Any(p => p.name.Contains("ArmUpDown")),
                "the float animator parameters must be removed too");

            var assets = AssetDatabase.LoadAllAssetsAtPath(AssetDatabase.GetAssetPath(_controller))
                .Where(a => a != null).ToArray();
            Assert.IsFalse(assets.OfType<AnimationClip>().Any(), "orphaned anchor clips");
            Assert.IsFalse(assets.OfType<BlendTree>().Any(), "orphaned trees");
            Assert.IsFalse(assets.OfType<AnimatorStateTransition>().Any(), "orphaned transitions");
            Assert.IsFalse(assets.OfType<AnimatorState>().Any(), "orphaned states");
            Assert.IsFalse(assets.OfType<AvatarMask>().Any(m => m.name.StartsWith("OSC_")), "orphaned masks");
        }
    }
}
