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
    /// Idle/Active/Rest machine (empty Idle passthrough, deadband, Rest settle), Simple1D anchor
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
        public void AddArmLayer_Left_EmptyIdleDefault_ActiveTree_RestPose()
        {
            OscAnimatorLayerBuilder.AddArmLayer(_controller, leftArm: true);

            var layer = _controller.layers.Single(l => l.name == "OSC_VCO_L_ArmUpDown");
            var sm = layer.stateMachine;
            CollectionAssert.AreEquivalent(
                new[] { "Idle", "Active", "Rest" },
                sm.states.Select(s => s.state.name).ToArray());
            // Idle is the empty default (idle/locomotion passthrough); Rest
            // is an explicit hanging pose so the arm visibly returns when
            // the hand leaves the frame (issue #28 live feedback) before
            // handing back to Idle.
            Assert.AreEqual("Idle", sm.defaultState.name);
            Assert.IsNull(sm.defaultState.motion);
            var rest = sm.states.Single(s => s.state.name == "Rest").state;
            Assert.IsNotNull(rest.motion, "Rest carries the hanging pose clip");
            Assert.Greater(((AnimationClip)rest.motion).length, 0.5f);
        }

        [Test]
        public void AddArmLayer_TransitionGraph_DeadbandAndRestSettle()
        {
            OscAnimatorLayerBuilder.AddArmLayer(_controller, leftArm: true);

            var sm = _controller.layers.Single(l => l.name == "OSC_VCO_L_ArmUpDown").stateMachine;
            var idle = sm.states.Single(s => s.state.name == "Idle").state;
            var active = sm.states.Single(s => s.state.name == "Active").state;
            var rest = sm.states.Single(s => s.state.name == "Rest").state;

            // Idle -> Active: two condition-OR transitions escaping the deadband.
            Assert.AreEqual(2, idle.transitions.Length);
            Assert.IsTrue(idle.transitions.All(tr =>
                tr.destinationState.name == "Active" && !tr.hasExitTime && tr.conditions.Length == 1));

            // Rest -> Active (re-raise, 2) + Rest -> Idle (exit-time settle, 1).
            Assert.AreEqual(3, rest.transitions.Length);
            Assert.AreEqual(2, rest.transitions.Count(tr => tr.destinationState.name == "Active"));
            var settle = rest.transitions.Single(tr => tr.destinationState.name == "Idle");
            Assert.IsTrue(settle.hasExitTime, "rest hands over to idle on exit time");
            Assert.AreEqual(0, settle.conditions.Length);

            // Active -> Rest: ONE transition whose two deadband conditions AND.
            var release = active.transitions.Single();
            Assert.AreEqual("Rest", release.destinationState.name);
            Assert.AreEqual(2, release.conditions.Length);
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
            Assert.AreEqual(0.85f, ValueOf(hanging, "Left Forearm Stretch"), 1e-6f, "near-straight hang");

            var mid = (AnimationClip)tree.children[1].motion;
            Assert.AreEqual(0.1f, ValueOf(mid, "Left Arm Down-Up"), 1e-6f);
            Assert.AreEqual(-0.5f, ValueOf(mid, "Left Arm Front-Back"), 1e-6f, "reach forward at mid");
            // The elbow must actually articulate through the raise (issue
            // #28 live feedback: "肘がきいていない") — strongly bent at mid.
            Assert.AreEqual(-0.45f, ValueOf(mid, "Left Forearm Stretch"), 1e-6f, "bent elbow at mid");

            var up = (AnimationClip)tree.children[2].motion;
            Assert.AreEqual(0.95f, ValueOf(up, "Left Arm Down-Up"), 1e-6f);
            Assert.AreEqual(0.7f, ValueOf(up, "Left Forearm Stretch"), 1e-6f, "mostly straight overhead");
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
            Assert.AreEqual(4, assets.OfType<AnimationClip>().Count(), "3 anchor clips + 1 rest clip, no orphans");
            Assert.AreEqual(1, assets.OfType<BlendTree>().Count(), "one Simple1D tree, no orphans");
            Assert.AreEqual(6, assets.OfType<AnimatorStateTransition>().Count(),
                "2+2 raise + 1 release + 1 settle transitions, no orphans");
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
