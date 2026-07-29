using System.Linq;
using NUnit.Framework;
using UnityEditor;
using UnityEditor.Animations;
using UnityEngine;
using VRC.SDK3.Avatars.Components;
using VRC.SDKBase;

namespace VRChatCameraOsc.AvatarSetup.Tests
{
    public class OscAnimatorLayerBuilderTests
    {
        const string TestDir = "Assets/_VRChatCameraOscTests";

        AnimatorController _controller;
        GameObject _avatarRoot;
        SkinnedMeshRenderer _renderer;

        [SetUp]
        public void SetUp()
        {
            if (!AssetDatabase.IsValidFolder(TestDir))
            {
                AssetDatabase.CreateFolder("Assets", "_VRChatCameraOscTests");
            }
            _controller = AnimatorController.CreateAnimatorControllerAtPath(TestDir + "/Test.controller");

            _avatarRoot = new GameObject("Avatar");
            var meshGo = new GameObject("Face");
            meshGo.transform.SetParent(_avatarRoot.transform);
            _renderer = meshGo.AddComponent<SkinnedMeshRenderer>();

            var mesh = new Mesh();
            mesh.vertices = new[] { Vector3.zero, Vector3.up, Vector3.right };
            mesh.triangles = new[] { 0, 1, 2 };
            mesh.AddBlendShapeFrame("Smile", 100f, new Vector3[3], null, null);
            mesh.AddBlendShapeFrame("Blink", 100f, new Vector3[3], null, null);
            mesh.AddBlendShapeFrame("Wide", 100f, new Vector3[3], null, null);
            _renderer.sharedMesh = mesh;
        }

        [TearDown]
        public void TearDown()
        {
            Object.DestroyImmediate(_avatarRoot);
            AssetDatabase.DeleteAsset(TestDir);
        }

        [Test]
        public void AddBlendShapeLayer_AddsParameterAndLayer()
        {
            OscAnimatorLayerBuilder.AddBlendShapeLayer(_controller, _avatarRoot.transform, "v2/JawOpen", _renderer, "Smile");

            Assert.IsTrue(_controller.parameters.Any(p =>
                p.name == "v2/JawOpen" && p.type == AnimatorControllerParameterType.Float));
            // The Animator *parameter* keeps its slash; layer/asset names are
            // sanitized (a "/" in an asset name is a path separator).
            Assert.IsTrue(_controller.layers.Any(l => l.name == "OSC_v2_JawOpen"));
        }

        [Test]
        public void AddBlendShapeLayer_ReRunning_ReplacesRatherThanDuplicates()
        {
            OscAnimatorLayerBuilder.AddBlendShapeLayer(_controller, _avatarRoot.transform, "v2/JawOpen", _renderer, "Smile");
            OscAnimatorLayerBuilder.AddBlendShapeLayer(_controller, _avatarRoot.transform, "v2/JawOpen", _renderer, "Smile");

            Assert.AreEqual(1, _controller.layers.Count(l => l.name == "OSC_v2_JawOpen"));
            Assert.AreEqual(1, _controller.parameters.Count(p => p.name == "v2/JawOpen"));
        }

        [Test]
        public void AddBlendShapeLayer_WithFullScale_ReachesWeight100AtThatThreshold()
        {
            // Issue #23: a deliberate brow raise only reaches ~0.3-0.5 on the
            // wire (VRCFT-semantics values, measured live), so the brow trees
            // rescale avatar-side: weight 100 at fullScale (0.5), clamped
            // above it. Real FT avatars are unaffected — this is wizard-only.
            OscAnimatorLayerBuilder.AddBlendShapeLayer(
                _controller, _avatarRoot.transform, "v2/BrowUpLeft", _renderer, "Smile", 0.5f);

            var layer = _controller.layers.First(l => l.name == "OSC_v2_BrowUpLeft");
            var tree = (BlendTree)layer.stateMachine.states.Single().state.motion;
            Assert.IsFalse(tree.useAutomaticThresholds);
            Assert.AreEqual(2, tree.children.Length);
            Assert.AreEqual(0f, tree.children[0].threshold, 1e-6f);
            Assert.AreEqual(0.5f, tree.children[1].threshold, 1e-6f);
            StringAssert.EndsWith("_0", tree.children[0].motion.name);
            StringAssert.EndsWith("_100", tree.children[1].motion.name);
        }

        [Test]
        public void Spec_WeakChannelsCarryReducedFullScale_OthersFullRange()
        {
            // Webcam-measured/reported weak channels rescale avatar-side
            // (issues #23/#24); everything else keeps full range.
            var expected = new System.Collections.Generic.Dictionary<string, float>
            {
                ["v2/BrowUpLeft"] = 0.5f,
                ["v2/BrowUpRight"] = 0.5f,
                ["v2/CheekPuffLeft"] = 0.4f,
                ["v2/CheekPuffRight"] = 0.4f,
                ["v2/JawLeft"] = 0.5f,
                ["v2/JawRight"] = 0.5f,
                ["v2/MouthFrownLeft"] = 0.5f,
                ["v2/MouthFrownRight"] = 0.5f,
                ["v2/NoseSneerLeft"] = 0.5f,
                ["v2/NoseSneerRight"] = 0.5f,
            };
            foreach (var spec in OscParameterSpec.All)
            {
                var want = expected.TryGetValue(spec.Name, out var v) ? v : spec.Max;
                Assert.AreEqual(want, spec.FullScale, 1e-6f, spec.Name);
            }
        }

        [Test]
        public void AddEyeLidLayer_AddsInvertedBlendTree_Closed100AtZero_Open0AtNeutral()
        {
            // VRCFT v2/EyeLid* semantics: 0 = closed, 0.75 = relaxed open.
            // The blink shape must be at weight 100 for parameter 0 and reach
            // weight 0 at the neutral threshold (clamped to 0 above it).
            OscAnimatorLayerBuilder.AddEyeLidLayer(_controller, _avatarRoot.transform, "v2/EyeLidLeft", _renderer, "Blink");

            var layer = _controller.layers.First(l => l.name == "OSC_v2_EyeLidLeft");
            var tree = (BlendTree)layer.stateMachine.states.Single().state.motion;

            Assert.AreEqual("v2/EyeLidLeft", tree.blendParameter);
            Assert.AreEqual(2, tree.children.Length);
            Assert.AreEqual(0f, tree.children[0].threshold, 1e-6f);
            Assert.AreEqual(OscParameterSpec.EyeLidNeutral, tree.children[1].threshold, 1e-6f);

            // Child 0 (param = 0, closed) carries the weight-100 clip; child 1
            // (param = neutral) the weight-0 clip. Clip naming encodes weight.
            StringAssert.EndsWith("_100", tree.children[0].motion.name);
            StringAssert.EndsWith("_0", tree.children[1].motion.name);
        }

        [Test]
        public void AddEyeLidLayer_WithWideShape_DrivesItOverTheUpperSegment()
        {
            // Issue #24: an optional eye-wide shape rides the 0.75..1 range
            // the two-child tree previously clamped — no extra parameters.
            OscAnimatorLayerBuilder.AddEyeLidLayer(
                _controller, _avatarRoot.transform, "v2/EyeLidLeft", _renderer, "Blink", _renderer, "Wide");

            var layer = _controller.layers.First(l => l.name == "OSC_v2_EyeLidLeft");
            var tree = (BlendTree)layer.stateMachine.states.Single().state.motion;
            Assert.AreEqual(3, tree.children.Length);
            CollectionAssert.AreEqual(
                new[] { 0f, OscParameterSpec.EyeLidNeutral, 1f },
                tree.children.Select(c => c.threshold).ToArray());
            // Every child animates both curves (blink AND wide), so the
            // blend never leaves one shape at an unanimated default.
            foreach (var child in tree.children)
            {
                var bindings = AnimationUtility.GetCurveBindings((AnimationClip)child.motion)
                    .Select(b => b.propertyName).ToArray();
                CollectionAssert.AreEquivalent(
                    new[] { "blendShape.Blink", "blendShape.Wide" }, bindings);
            }
        }

        [Test]
        public void AddCombinedHeadLayer_OneOverrideLayer_NestedThreeAxisTree()
        {
            // Issue #27: per-axis Override layers group-override each other
            // in-client (the last layer pinned yaw/pitch to 0). All three
            // muscles must be written by ONE layer's tree.
            OscAnimatorLayerBuilder.AddCombinedHeadLayer(_controller);

            foreach (var p in new[] { "v2/Head/Yaw", "v2/Head/Pitch", "v2/Head/Roll" })
            {
                Assert.IsTrue(_controller.parameters.Any(x => x.name == p), p);
            }
            var layer = _controller.layers.Single(l => l.name.StartsWith("OSC_v2_Head"));
            Assert.AreEqual("OSC_v2_Head", layer.name);
            Assert.AreEqual(AnimatorLayerBlendingMode.Override, layer.blendingMode);

            var top = (BlendTree)layer.stateMachine.defaultState.motion;
            Assert.AreEqual("v2/Head/Yaw", top.blendParameter);
            Assert.IsFalse(top.useAutomaticThresholds);
            CollectionAssert.AreEqual(new[] { -1f, 0f, 1f }, top.children.Select(c => c.threshold).ToArray());
            var mid = (BlendTree)top.children[2].motion;
            Assert.AreEqual("v2/Head/Pitch", mid.blendParameter);
            var leafTree = (BlendTree)mid.children[0].motion;
            Assert.AreEqual("v2/Head/Roll", leafTree.blendParameter);
            // Every leaf clip animates ALL THREE muscles with real duration
            // (group-override then always carries all axes; real length keeps
            // the ping-pong clock ticking).
            var clip = (AnimationClip)leafTree.children[1].motion;
            var muscles = AnimationUtility.GetCurveBindings(clip).Select(b => b.propertyName).ToArray();
            CollectionAssert.AreEquivalent(
                new[] { "Head Turn Left-Right", "Head Nod Down-Up", "Head Tilt Left-Right" }, muscles);
            Assert.Greater(clip.length, 0.5f);
            // Leaf values encode the branch (y=+1, p=-1, r=0), with the
            // yaw sign flipped: param +1 = subject's LEFT (VRCFT), muscle
            // +1 = right — live-confirmed mirrored without the flip.
            float ValueOf(string muscle) => AnimationUtility.GetEditorCurve(
                clip, AnimationUtility.GetCurveBindings(clip).Single(b => b.propertyName == muscle)).Evaluate(0f);
            Assert.AreEqual(-1f, ValueOf("Head Turn Left-Right"), 1e-6f);
            Assert.AreEqual(-1f, ValueOf("Head Nod Down-Up"), 1e-6f);
            Assert.AreEqual(0f, ValueOf("Head Tilt Left-Right"), 1e-6f);
        }

        [Test]
        public void AddCombinedHeadLayer_PingPongStates_EachCarryTrackingControlAndMotion()
        {
            OscAnimatorLayerBuilder.AddCombinedHeadLayer(_controller);

            var layer = _controller.layers.Single(l => l.name == "OSC_v2_Head");
            Assert.AreEqual(2, layer.stateMachine.states.Length);
            foreach (var child in layer.stateMachine.states)
            {
                Assert.IsNotNull(child.state.motion, child.state.name);
                var b = child.state.behaviours.OfType<VRCAnimatorTrackingControl>().Single();
                Assert.AreEqual(VRC_AnimatorTrackingControl.TrackingType.Animation, b.trackingHead);
                Assert.AreEqual(VRC_AnimatorTrackingControl.TrackingType.NoChange, b.trackingLeftHand);
                Assert.AreEqual(VRC_AnimatorTrackingControl.TrackingType.NoChange, b.trackingEyes);
                Assert.AreEqual(VRC_AnimatorTrackingControl.TrackingType.NoChange, b.trackingMouth);
                var tr = child.state.transitions.Single();
                Assert.IsTrue(tr.hasExitTime);
            }
        }

        [Test]
        public void AddCombinedHeadLayer_RestrictsToHeadOnlyMask()
        {
            // Real bug (issue #16): Humanoid retargeting can leak tiny
            // perturbations into Chest/Spine from Head-only muscle clips —
            // a chest-mounted VRCPhysBone amplifies that into visible spin.
            OscAnimatorLayerBuilder.AddCombinedHeadLayer(_controller);

            var layer = _controller.layers.Single(l => l.name == "OSC_v2_Head");
            Assert.IsNotNull(layer.avatarMask);
            Assert.IsTrue(layer.avatarMask.GetHumanoidBodyPartActive(AvatarMaskBodyPart.Head));
            Assert.IsFalse(layer.avatarMask.GetHumanoidBodyPartActive(AvatarMaskBodyPart.Body));
            Assert.IsFalse(layer.avatarMask.GetHumanoidBodyPartActive(AvatarMaskBodyPart.LeftArm));
        }

        [Test]
        public void RemoveLayer_CombinedHeadLayer_CleansTrackingControlsMaskAndNestedTrees()
        {
            OscAnimatorLayerBuilder.AddCombinedHeadLayer(_controller);

            Assert.IsTrue(OscAnimatorLayerBuilder.RemoveLayer(_controller, OscAnimatorLayerBuilder.CombinedHeadKey));
            Assert.IsFalse(OscAnimatorLayerBuilder.HasLayer(_controller, OscAnimatorLayerBuilder.CombinedHeadKey));

            var assets = AssetDatabase.LoadAllAssetsAtPath(AssetDatabase.GetAssetPath(_controller));
            Assert.IsFalse(assets.OfType<VRCAnimatorTrackingControl>().Any(), "orphaned tracking control");
            Assert.IsFalse(assets.OfType<AvatarMask>().Any(m => m.name == "OSC_HeadOnlyMask"), "orphaned mask");
            Assert.IsFalse(assets.OfType<BlendTree>().Any(m => m.name.StartsWith("OSC_v2_Head")), "orphaned nested trees");
        }

        [Test]
        public void AddBlendShapeLayer_DoesNotAttachTrackingControl()
        {
            // Only head-pose layers need VRCAnimatorTrackingControl; blend
            // shape layers must be left alone.
            OscAnimatorLayerBuilder.AddBlendShapeLayer(_controller, _avatarRoot.transform, "v2/JawOpen", _renderer, "Smile");

            var layer = _controller.layers.First(l => l.name == "OSC_v2_JawOpen");
            var state = layer.stateMachine.states.Single().state;
            Assert.IsEmpty(state.behaviours);
        }

        [Test]
        public void RemoveLayer_IsTheInverseOfAdd_LeavesOtherLayersAlone()
        {
            OscAnimatorLayerBuilder.AddBlendShapeLayer(_controller, _avatarRoot.transform, "v2/JawOpen", _renderer, "Smile");
            OscAnimatorLayerBuilder.AddBlendShapeLayer(_controller, _avatarRoot.transform, "v2/MouthSmileLeft", _renderer, "Smile");

            Assert.IsTrue(OscAnimatorLayerBuilder.HasLayer(_controller, "v2/JawOpen"));
            var removed = OscAnimatorLayerBuilder.RemoveLayer(_controller, "v2/JawOpen");

            Assert.IsTrue(removed);
            Assert.IsFalse(OscAnimatorLayerBuilder.HasLayer(_controller, "v2/JawOpen"));
            Assert.IsFalse(_controller.parameters.Any(p => p.name == "v2/JawOpen"));
            // The other parameter's layer must survive untouched.
            Assert.IsTrue(OscAnimatorLayerBuilder.HasLayer(_controller, "v2/MouthSmileLeft"));
            Assert.IsTrue(_controller.parameters.Any(p => p.name == "v2/MouthSmileLeft"));
        }

        [Test]
        public void RemoveLayer_WhenAbsent_ReturnsFalseAndDoesNotThrow()
        {
            Assert.IsFalse(OscAnimatorLayerBuilder.HasLayer(_controller, "v2/JawOpen"));
            Assert.IsFalse(OscAnimatorLayerBuilder.RemoveLayer(_controller, "v2/JawOpen"));
        }

        [Test]
        public void RemoveLayer_WorksForEyeLidLayer()
        {
            OscAnimatorLayerBuilder.AddEyeLidLayer(_controller, _avatarRoot.transform, "v2/EyeLidLeft", _renderer, "Blink");

            Assert.IsTrue(OscAnimatorLayerBuilder.RemoveLayer(_controller, "v2/EyeLidLeft"));
            Assert.IsFalse(OscAnimatorLayerBuilder.HasLayer(_controller, "v2/EyeLidLeft"));
        }

        [Test]
        public void RemoveLayer_LegacyCustom10Name_StillWorks_ForMigration()
        {
            // RemoveLegacyCustom10 (issue #21) relies on RemoveLayer working
            // for the retired slash-less names.
            OscAnimatorLayerBuilder.AddBlendShapeLayer(_controller, _avatarRoot.transform, "MouthOpen", _renderer, "Smile");

            Assert.IsTrue(OscAnimatorLayerBuilder.HasLayer(_controller, "MouthOpen"));
            Assert.IsTrue(OscAnimatorLayerBuilder.RemoveLayer(_controller, "MouthOpen"));
            Assert.IsFalse(OscAnimatorLayerBuilder.HasLayer(_controller, "MouthOpen"));
        }

        static void WireAll(AnimatorController controller, Transform root, SkinnedMeshRenderer renderer)
        {
            foreach (var spec in OscParameterSpec.All)
            {
                switch (spec.Kind)
                {
                    case OscParamKind.BlendShape:
                        OscAnimatorLayerBuilder.AddBlendShapeLayer(controller, root, spec.Name, renderer, "Smile");
                        break;
                    case OscParamKind.EyeLid:
                        OscAnimatorLayerBuilder.AddEyeLidLayer(controller, root, spec.Name, renderer, "Blink");
                        break;
                    case OscParamKind.HeadPose:
                        if (!OscAnimatorLayerBuilder.HasLayer(controller, OscAnimatorLayerBuilder.CombinedHeadKey))
                        {
                            OscAnimatorLayerBuilder.AddCombinedHeadLayer(controller);
                        }
                        break;
                    case OscParamKind.GestureInt:
                        OscAnimatorLayerBuilder.AddHandGestureLayer(
                            controller, spec.Name == OscAnimatorLayerBuilder.GestureLeftParam);
                        break;
                    case OscParamKind.FingerCurl:
                        // Five specs share one per-hand layer (the wizard
                        // never wires curls AND gestures together, but the
                        // builder allows it — exclusivity is ApplyHandMode's
                        // job, tested in OscFingerCurlLayerTests).
                        var left = spec.Name.StartsWith("VCO_L_");
                        var key = left
                            ? OscAnimatorLayerBuilder.FingerCurlLeftKey
                            : OscAnimatorLayerBuilder.FingerCurlRightKey;
                        if (!OscAnimatorLayerBuilder.HasLayer(controller, key))
                        {
                            OscAnimatorLayerBuilder.AddFingerCurlLayer(controller, left);
                        }
                        break;
                    case OscParamKind.ArmTracked:
                    case OscParamKind.ArmFloat:
                        // Four specs (Bool gate + three floats) share one
                        // per-arm layer (issue #28 phase 2).
                        var leftArm = spec.Name.StartsWith("VCO_L_");
                        var armKey = leftArm
                            ? OscAnimatorLayerBuilder.ArmLeftParam
                            : OscAnimatorLayerBuilder.ArmRightParam;
                        if (!OscAnimatorLayerBuilder.HasLayer(controller, armKey))
                        {
                            OscAnimatorLayerBuilder.AddArmLayer(controller, leftArm);
                        }
                        break;
                }
            }
        }

        static string LayerKeyFor(OscParamSpec spec)
        {
            switch (spec.Kind)
            {
                case OscParamKind.HeadPose:
                    return OscAnimatorLayerBuilder.CombinedHeadKey;
                case OscParamKind.FingerCurl:
                    return spec.Name.StartsWith("VCO_L_")
                        ? OscAnimatorLayerBuilder.FingerCurlLeftKey
                        : OscAnimatorLayerBuilder.FingerCurlRightKey;
                case OscParamKind.ArmTracked:
                case OscParamKind.ArmFloat:
                    return spec.Name.StartsWith("VCO_L_")
                        ? OscAnimatorLayerBuilder.ArmLeftParam
                        : OscAnimatorLayerBuilder.ArmRightParam;
                default:
                    return spec.Name;
            }
        }

        [Test]
        public void AllParameters_CanBeWiredWithoutError()
        {
            WireAll(_controller, _avatarRoot.transform, _renderer);

            Assert.AreEqual(OscParameterSpec.All.Count, _controller.parameters.Length);
            // CreateAnimatorControllerAtPath seeds one default "Base Layer";
            // AddLayer only ever replaces layers it owns (named "OSC_*").
            // The three head axes share ONE combined layer (issue #27),
            // each hand's five curl floats share one per-hand layer, and
            // each arm's four params (Bool gate + three floats) share one
            // per-arm layer (issue #28 phase 2).
            var headCount = OscParameterSpec.All.Count(s => s.Kind == OscParamKind.HeadPose);
            var curlCount = OscParameterSpec.All.Count(s => s.Kind == OscParamKind.FingerCurl);
            var armCount = OscParameterSpec.All.Count(s =>
                s.Kind == OscParamKind.ArmTracked || s.Kind == OscParamKind.ArmFloat);
            Assert.AreEqual(
                OscParameterSpec.All.Count - headCount + 1 - curlCount + 2 - armCount + 2,
                _controller.layers.Count(l => l.name.StartsWith("OSC_")));
            foreach (var spec in OscParameterSpec.All)
            {
                Assert.IsTrue(OscAnimatorLayerBuilder.HasLayer(_controller, LayerKeyFor(spec)),
                    $"missing layer for {spec.Name}");
            }
        }

        [Test]
        public void WireThenRemoveAll_RoundTripsBackToZero()
        {
            WireAll(_controller, _avatarRoot.transform, _renderer);

            var keys = OscParameterSpec.All
                .Select(LayerKeyFor)
                .Distinct();
            foreach (var key in keys)
            {
                Assert.IsTrue(OscAnimatorLayerBuilder.RemoveLayer(_controller, key), key);
            }

            Assert.AreEqual(0, _controller.parameters.Length);
            Assert.AreEqual(0, _controller.layers.Count(l => l.name.StartsWith("OSC_")));
        }
    }
}
