using System.Linq;
using NUnit.Framework;
using UnityEditor;
using UnityEditor.Animations;
using UnityEngine;

namespace VRChatCameraOsc.AvatarSetup.Tests
{
    /// <summary>
    /// Issue #28 phase 2 — full-arm tracking. Per arm: a
    /// <c>VCO_x_ArmTracked</c> Bool gates the Idle/Active/Rest state
    /// machine (the phase-1 ±0.02 deadband-on-UpDown is retired: "arm at
    /// the camera" legitimately reads (0,0)); Active is a 2D Freeform
    /// Cartesian tree (x = <c>VCO_x_ArmAcross</c>, y =
    /// <c>VCO_x_ArmUpDown</c>) whose five direction anchors each nest a
    /// Simple1D elbow tree on <c>VCO_x_Elbow</c> — 10 leaf clips per arm
    /// (<see cref="OscAnimatorLayerBuilder.AddArmLayer"/>).
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
        public void Spec_DeclaresAllEightArmParams_ModeGatedNotCore()
        {
            foreach (var side in new[] { "L", "R" })
            {
                var tracked = OscParameterSpec.All.Single(s => s.Name == $"VCO_{side}_ArmTracked");
                Assert.AreEqual(OscParamKind.ArmTracked, tracked.Kind);
                Assert.AreEqual(0f, tracked.DefaultValue, "ArmTracked rests false");

                var upDown = OscParameterSpec.All.Single(s => s.Name == $"VCO_{side}_ArmUpDown");
                Assert.AreEqual(OscParamKind.ArmFloat, upDown.Kind);
                Assert.AreEqual(-1f, upDown.Min);
                Assert.AreEqual(1f, upDown.Max);
                Assert.AreEqual(0f, upDown.DefaultValue);

                var across = OscParameterSpec.All.Single(s => s.Name == $"VCO_{side}_ArmAcross");
                Assert.AreEqual(OscParamKind.ArmFloat, across.Kind);
                Assert.AreEqual(-1f, across.Min);
                Assert.AreEqual(1f, across.Max);
                Assert.AreEqual(0f, across.DefaultValue);

                var elbow = OscParameterSpec.All.Single(s => s.Name == $"VCO_{side}_Elbow");
                Assert.AreEqual(OscParamKind.ArmFloat, elbow.Kind);
                Assert.AreEqual(0f, elbow.Min);
                Assert.AreEqual(1f, elbow.Max);
                Assert.AreEqual(0f, elbow.DefaultValue);
            }

            var armSpecs = OscParameterSpec.All
                .Where(s => s.Kind == OscParamKind.ArmFloat || s.Kind == OscParamKind.ArmTracked)
                .ToList();
            Assert.AreEqual(8, armSpecs.Count, "2 Bool gates + 6 direction/elbow floats");
            foreach (var s in armSpecs)
            {
                Assert.IsTrue(OscParameterSpec.IsModeGated(s.Kind), s.Name);
                Assert.IsFalse(OscParameterSpec.Core.Any(c => c.Name == s.Name),
                    $"{s.Name} is declared only while the arm toggle is on");
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
            // tracking is lost (issue #28 live feedback) before handing
            // back to Idle.
            Assert.AreEqual("Idle", sm.defaultState.name);
            Assert.IsNull(sm.defaultState.motion);
            var rest = sm.states.Single(s => s.state.name == "Rest").state;
            Assert.IsNotNull(rest.motion, "Rest carries the hanging pose clip");
            Assert.Greater(((AnimationClip)rest.motion).length, 0.5f);
        }

        [Test]
        public void AddArmLayer_AddsBoolGateAndThreeFloatParams()
        {
            OscAnimatorLayerBuilder.AddArmLayer(_controller, leftArm: true);

            Assert.IsTrue(_controller.parameters.Any(p =>
                p.name == "VCO_L_ArmTracked" && p.type == AnimatorControllerParameterType.Bool),
                "ArmTracked must be a Bool animator parameter");
            foreach (var f in new[] { "VCO_L_ArmUpDown", "VCO_L_ArmAcross", "VCO_L_Elbow" })
            {
                Assert.IsTrue(_controller.parameters.Any(p =>
                    p.name == f && p.type == AnimatorControllerParameterType.Float), f);
            }
        }

        [Test]
        public void AddArmLayer_TransitionGraph_GatedByTrackedBool()
        {
            OscAnimatorLayerBuilder.AddArmLayer(_controller, leftArm: true);

            var sm = _controller.layers.Single(l => l.name == "OSC_VCO_L_ArmUpDown").stateMachine;
            var idle = sm.states.Single(s => s.state.name == "Idle").state;
            var active = sm.states.Single(s => s.state.name == "Active").state;
            var rest = sm.states.Single(s => s.state.name == "Rest").state;

            // Idle -> Active: ONE transition, If ArmTracked. No deadband
            // float conditions anywhere (phase-1 design retired).
            var raise = idle.transitions.Single();
            Assert.AreEqual("Active", raise.destinationState.name);
            Assert.IsFalse(raise.hasExitTime);
            var raiseCond = raise.conditions.Single();
            Assert.AreEqual(AnimatorConditionMode.If, raiseCond.mode);
            Assert.AreEqual("VCO_L_ArmTracked", raiseCond.parameter);

            // Active -> Rest: ONE transition, IfNot ArmTracked.
            var release = active.transitions.Single();
            Assert.AreEqual("Rest", release.destinationState.name);
            Assert.IsFalse(release.hasExitTime);
            var releaseCond = release.conditions.Single();
            Assert.AreEqual(AnimatorConditionMode.IfNot, releaseCond.mode);
            Assert.AreEqual("VCO_L_ArmTracked", releaseCond.parameter);

            // Rest -> Active (re-acquire, If) + Rest -> Idle (exit-time settle).
            Assert.AreEqual(2, rest.transitions.Length);
            var reacquire = rest.transitions.Single(tr => tr.destinationState.name == "Active");
            Assert.IsFalse(reacquire.hasExitTime);
            Assert.AreEqual(AnimatorConditionMode.If, reacquire.conditions.Single().mode);
            Assert.AreEqual("VCO_L_ArmTracked", reacquire.conditions.Single().parameter);
            var settle = rest.transitions.Single(tr => tr.destinationState.name == "Idle");
            Assert.IsTrue(settle.hasExitTime, "rest hands over to idle on exit time");
            Assert.AreEqual(1.2f, settle.exitTime, 1e-6f);
            Assert.AreEqual(0, settle.conditions.Length);
        }

        [Test]
        public void AddArmLayer_ActiveIs2DFreeformWithNestedElbowTrees()
        {
            OscAnimatorLayerBuilder.AddArmLayer(_controller, leftArm: true);

            var sm = _controller.layers.Single(l => l.name == "OSC_VCO_L_ArmUpDown").stateMachine;
            var top = (BlendTree)sm.states.Single(s => s.state.name == "Active").state.motion;

            Assert.AreEqual(BlendTreeType.FreeformCartesian2D, top.blendType);
            Assert.AreEqual("VCO_L_ArmAcross", top.blendParameter, "x-axis = Across");
            Assert.AreEqual("VCO_L_ArmUpDown", top.blendParameterY, "y-axis = UpDown");
            Assert.IsFalse(top.useAutomaticThresholds);

            // Five direction anchors at (Across, UpDown).
            CollectionAssert.AreEquivalent(
                new[]
                {
                    new Vector2(0f, -1f),  // down
                    new Vector2(0f, 1f),   // up
                    new Vector2(1f, 0f),   // across the chest
                    new Vector2(-1f, 0f),  // out to the side
                    new Vector2(0f, 0f),   // forward, toward the camera
                },
                top.children.Select(c => c.position).ToArray());

            // Each anchor nests a Simple1D elbow tree: 0 = straight, 1 = bent.
            foreach (var child in top.children)
            {
                var elbow = (BlendTree)child.motion;
                Assert.AreEqual(BlendTreeType.Simple1D, elbow.blendType, elbow.name);
                Assert.AreEqual("VCO_L_Elbow", elbow.blendParameter, elbow.name);
                Assert.IsFalse(elbow.useAutomaticThresholds, elbow.name);
                Assert.AreEqual(2, elbow.children.Length, elbow.name);
                Assert.AreEqual(0f, elbow.children[0].threshold, 1e-6f, elbow.name);
                Assert.AreEqual(1f, elbow.children[1].threshold, 1e-6f, elbow.name);
                Assert.IsInstanceOf<AnimationClip>(elbow.children[0].motion, elbow.name);
                Assert.IsInstanceOf<AnimationClip>(elbow.children[1].motion, elbow.name);
            }
        }

        static float ValueOf(AnimationClip clip, string prop) => AnimationUtility.GetEditorCurve(
            clip,
            AnimationUtility.GetCurveBindings(clip).Single(b => b.propertyName == prop)).Evaluate(0f);

        static AnimationClip Leaf(AnimatorController controller, Vector2 anchor, float elbow)
        {
            var sm = controller.layers.Single(l => l.name.StartsWith("OSC_VCO_")).stateMachine;
            var top = (BlendTree)sm.states.Single(s => s.state.name == "Active").state.motion;
            var tree = (BlendTree)top.children.Single(c => c.position == anchor).motion;
            return (AnimationClip)tree.children.Single(c => Mathf.Approximately(c.threshold, elbow)).motion;
        }

        [Test]
        public void AddArmLayer_AllTenLeafClips_WriteAllNineArmMuscles_WithRealDuration()
        {
            OscAnimatorLayerBuilder.AddArmLayer(_controller, leftArm: true);

            var sm = _controller.layers.Single(l => l.name == "OSC_VCO_L_ArmUpDown").stateMachine;
            var top = (BlendTree)sm.states.Single(s => s.state.name == "Active").state.motion;
            var expectedProps = OscAnimatorLayerBuilder.ArmMuscleProperties(true).ToArray();

            var leaves = top.children
                .SelectMany(c => ((BlendTree)c.motion).children)
                .Select(c => (AnimationClip)c.motion)
                .ToArray();
            Assert.AreEqual(10, leaves.Length, "5 direction anchors x elbow straight/bent");
            foreach (var clip in leaves)
            {
                // Real duration: zero-length single-key clips never advance
                // normalized time in the VRChat client (issue #25 trap).
                Assert.Greater(clip.length, 0.5f, clip.name);
                var bound = AnimationUtility.GetCurveBindings(clip).Select(b => b.propertyName).ToArray();
                CollectionAssert.AreEquivalent(expectedProps, bound,
                    $"{clip.name}: every leaf must write ALL nine arm muscles (issue #27 group lesson)");
            }
        }

        [Test]
        public void AddArmLayer_PoseTable_SpotChecks()
        {
            OscAnimatorLayerBuilder.AddArmLayer(_controller, leftArm: true);

            // down/straight = the phase-1 live-tuned relaxed hang.
            var downStraight = Leaf(_controller, new Vector2(0f, -1f), 0f);
            Assert.AreEqual(-0.5f, ValueOf(downStraight, "Left Arm Down-Up"), 1e-6f);
            Assert.AreEqual(0.85f, ValueOf(downStraight, "Left Forearm Stretch"), 1e-6f, "near-straight hang");

            // up/straight = the phase-1 live-tuned overhead pose.
            var upStraight = Leaf(_controller, new Vector2(0f, 1f), 0f);
            Assert.AreEqual(0.95f, ValueOf(upStraight, "Left Arm Down-Up"), 1e-6f);
            Assert.AreEqual(0.7f, ValueOf(upStraight, "Left Forearm Stretch"), 1e-6f, "mostly straight overhead");

            // across = horizontal reach toward the opposite shoulder: strong
            // arm-forward (Front-Back negative per the second-word=+1
            // convention) + shoulder protraction.
            var acrossStraight = Leaf(_controller, new Vector2(1f, 0f), 0f);
            Assert.AreEqual(-0.9f, ValueOf(acrossStraight, "Left Arm Front-Back"), 1e-6f);
            Assert.AreEqual(-0.5f, ValueOf(acrossStraight, "Left Shoulder Front-Back"), 1e-6f);

            // out = horizontal T-pose-ish: Down-Up in the horizontal range,
            // no front-back component.
            var outStraight = Leaf(_controller, new Vector2(-1f, 0f), 0f);
            Assert.AreEqual(0.4f, ValueOf(outStraight, "Left Arm Down-Up"), 1e-6f);
            Assert.AreEqual(0f, ValueOf(outStraight, "Left Arm Front-Back"), 1e-6f);

            // forward = pointing at the camera: arm forward, elbow straight.
            var forwardStraight = Leaf(_controller, new Vector2(0f, 0f), 0f);
            Assert.AreEqual(-0.6f, ValueOf(forwardStraight, "Left Arm Front-Back"), 1e-6f);
            Assert.AreEqual(0.85f, ValueOf(forwardStraight, "Left Forearm Stretch"), 1e-6f);

            // Every bent variant folds the forearm hard (issue #28 live
            // feedback: the elbow must actually articulate).
            foreach (var anchor in new[]
                     {
                         new Vector2(0f, -1f), new Vector2(0f, 1f), new Vector2(1f, 0f),
                         new Vector2(-1f, 0f), new Vector2(0f, 0f),
                     })
            {
                var bent = Leaf(_controller, anchor, 1f);
                Assert.AreEqual(-0.8f, ValueOf(bent, "Left Forearm Stretch"), 1e-6f, $"bent at {anchor}");
            }
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
        public void AddArmLayer_Right_UsesRightParamsAndMuscles()
        {
            OscAnimatorLayerBuilder.AddArmLayer(_controller, leftArm: false);

            Assert.IsTrue(_controller.parameters.Any(p =>
                p.name == "VCO_R_ArmTracked" && p.type == AnimatorControllerParameterType.Bool));
            var sm = _controller.layers.Single(l => l.name == "OSC_VCO_R_ArmUpDown").stateMachine;
            var top = (BlendTree)sm.states.Single(s => s.state.name == "Active").state.motion;
            Assert.AreEqual("VCO_R_ArmAcross", top.blendParameter);
            Assert.AreEqual("VCO_R_ArmUpDown", top.blendParameterY);
            var elbow = (BlendTree)top.children[0].motion;
            Assert.AreEqual("VCO_R_Elbow", elbow.blendParameter);
            var clip = (AnimationClip)elbow.children[0].motion;
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
            foreach (var p in OscAnimatorLayerBuilder.ArmParams(leftArm: true))
            {
                Assert.AreEqual(1, _controller.parameters.Count(x => x.name == p), p);
            }

            var assets = AssetDatabase.LoadAllAssetsAtPath(AssetDatabase.GetAssetPath(_controller))
                .Where(a => a != null).ToArray();
            Assert.AreEqual(11, assets.OfType<AnimationClip>().Count(), "10 leaf clips + 1 rest clip, no orphans");
            Assert.AreEqual(6, assets.OfType<BlendTree>().Count(), "one 2D tree + 5 nested elbow trees, no orphans");
            Assert.AreEqual(4, assets.OfType<AnimatorStateTransition>().Count(),
                "raise + release + re-acquire + settle transitions, no orphans");
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
            Assert.IsFalse(_controller.parameters.Any(p =>
                p.name.Contains("ArmTracked") || p.name.Contains("ArmUpDown") ||
                p.name.Contains("ArmAcross") || p.name.Contains("Elbow")),
                "the Bool gate and all three float animator parameters must be removed too");

            var assets = AssetDatabase.LoadAllAssetsAtPath(AssetDatabase.GetAssetPath(_controller))
                .Where(a => a != null).ToArray();
            Assert.IsFalse(assets.OfType<AnimationClip>().Any(), "orphaned pose clips");
            Assert.IsFalse(assets.OfType<BlendTree>().Any(), "orphaned trees (incl. nested elbow trees)");
            Assert.IsFalse(assets.OfType<AnimatorStateTransition>().Any(), "orphaned transitions");
            Assert.IsFalse(assets.OfType<AnimatorState>().Any(), "orphaned states");
            Assert.IsFalse(assets.OfType<AvatarMask>().Any(m => m.name.StartsWith("OSC_")), "orphaned masks");
        }
    }
}
