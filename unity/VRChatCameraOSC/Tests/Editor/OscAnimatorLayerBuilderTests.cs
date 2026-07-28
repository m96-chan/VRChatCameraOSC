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
        public void Spec_BrowUpParamsCarryHalfFullScale_OthersFullRange()
        {
            foreach (var spec in OscParameterSpec.All)
            {
                var expected = spec.Name.StartsWith("v2/BrowUp") ? 0.5f : spec.Max;
                Assert.AreEqual(expected, spec.FullScale, 1e-6f, spec.Name);
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
        public void AddHeadPoseLayer_UsesAdditiveBlending_WithExplicitThresholds()
        {
            OscAnimatorLayerBuilder.AddHeadPoseLayer(_controller, "v2/Head/Roll", "Head Tilt Left-Right");

            var layer = _controller.layers.First(l => l.name == "OSC_v2_Head_Roll");
            Assert.AreEqual(AnimatorLayerBlendingMode.Additive, layer.blendingMode);
            Assert.IsTrue(_controller.parameters.Any(p => p.name == "v2/Head/Roll"));

            // Guard against useAutomaticThresholds rewriting the -1/0/1
            // spread (same failure mode the eyelid tree hit — issue #21).
            var tree = (BlendTree)layer.stateMachine.states.Single().state.motion;
            Assert.IsFalse(tree.useAutomaticThresholds);
            CollectionAssert.AreEqual(
                new[] { -1f, 0f, 1f },
                tree.children.Select(c => c.threshold).ToArray());
        }

        [Test]
        public void AddHeadPoseLayer_RestrictsToHeadOnlyMask_ToStopChestBonePerturbationLeaking()
        {
            // Real bug (issue #16): Humanoid retargeting can leak tiny
            // perturbations into Chest/Spine from a Head-only muscle clip,
            // which a chest-mounted VRCPhysBone (e.g. wings) can amplify into
            // a visible spin under real (continuous, jittery) OSC head data
            // even though a single static parameter scrub looks harmless.
            // The mask must contain the layer's effect to Head only.
            OscAnimatorLayerBuilder.AddHeadPoseLayer(_controller, "v2/Head/Roll", "Head Tilt Left-Right");

            var layer = _controller.layers.First(l => l.name == "OSC_v2_Head_Roll");
            Assert.IsNotNull(layer.avatarMask, "head-pose layer must have a mask");
            Assert.IsTrue(layer.avatarMask.GetHumanoidBodyPartActive(AvatarMaskBodyPart.Head));
            Assert.IsFalse(layer.avatarMask.GetHumanoidBodyPartActive(AvatarMaskBodyPart.Body));
            Assert.IsFalse(layer.avatarMask.GetHumanoidBodyPartActive(AvatarMaskBodyPart.LeftArm));
            Assert.IsFalse(layer.avatarMask.GetHumanoidBodyPartActive(AvatarMaskBodyPart.RightArm));
            Assert.IsFalse(layer.avatarMask.GetHumanoidBodyPartActive(AvatarMaskBodyPart.LeftLeg));
            Assert.IsFalse(layer.avatarMask.GetHumanoidBodyPartActive(AvatarMaskBodyPart.RightLeg));
        }

        [Test]
        public void AddHeadPoseLayer_SharesOneMaskAcrossAllThreeAxes()
        {
            OscAnimatorLayerBuilder.AddHeadPoseLayer(_controller, "v2/Head/Roll", "Head Tilt Left-Right");
            OscAnimatorLayerBuilder.AddHeadPoseLayer(_controller, "v2/Head/Yaw", "Head Turn Left-Right");
            OscAnimatorLayerBuilder.AddHeadPoseLayer(_controller, "v2/Head/Pitch", "Head Nod Down-Up");

            var masks = _controller.layers
                .Where(l => l.name.StartsWith("OSC_v2_Head"))
                .Select(l => l.avatarMask)
                .ToArray();
            Assert.AreEqual(3, masks.Length);
            Assert.AreSame(masks[0], masks[1]);
            Assert.AreSame(masks[1], masks[2]);
        }

        [Test]
        public void AddHeadPoseLayer_AttachesExactlyOneTrackingControl_HeadAnimationOthersNoChange()
        {
            // VRChat fact (creators.vrchat.com/avatars/state-behaviors/): the
            // Head bone is IK-driven on Desktop; only a
            // VRCAnimatorTrackingControl with trackingHead = Animation makes
            // this layer's muscle curves win. Every other tracked part must
            // stay NoChange so this layer never stomps on hands/eyes/etc.
            OscAnimatorLayerBuilder.AddHeadPoseLayer(_controller, "v2/Head/Roll", "Head Tilt Left-Right");

            var layer = _controller.layers.First(l => l.name == "OSC_v2_Head_Roll");
            var state = layer.stateMachine.states.Single().state;
            var behaviours = state.behaviours.OfType<VRCAnimatorTrackingControl>().ToArray();

            Assert.AreEqual(1, behaviours.Length, "exactly one VRCAnimatorTrackingControl must be attached");
            var behaviour = behaviours[0];
            Assert.AreEqual(VRC_AnimatorTrackingControl.TrackingType.Animation, behaviour.trackingHead);
            Assert.AreEqual(VRC_AnimatorTrackingControl.TrackingType.NoChange, behaviour.trackingLeftHand);
            Assert.AreEqual(VRC_AnimatorTrackingControl.TrackingType.NoChange, behaviour.trackingRightHand);
            Assert.AreEqual(VRC_AnimatorTrackingControl.TrackingType.NoChange, behaviour.trackingHip);
            Assert.AreEqual(VRC_AnimatorTrackingControl.TrackingType.NoChange, behaviour.trackingLeftFoot);
            Assert.AreEqual(VRC_AnimatorTrackingControl.TrackingType.NoChange, behaviour.trackingRightFoot);
            Assert.AreEqual(VRC_AnimatorTrackingControl.TrackingType.NoChange, behaviour.trackingLeftFingers);
            Assert.AreEqual(VRC_AnimatorTrackingControl.TrackingType.NoChange, behaviour.trackingRightFingers);
            Assert.AreEqual(VRC_AnimatorTrackingControl.TrackingType.NoChange, behaviour.trackingEyes);
            Assert.AreEqual(VRC_AnimatorTrackingControl.TrackingType.NoChange, behaviour.trackingMouth);
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
        public void RemoveLayer_HeadPoseLayer_LeavesNoOrphanedTrackingControlAsset()
        {
            OscAnimatorLayerBuilder.AddHeadPoseLayer(_controller, "v2/Head/Roll", "Head Tilt Left-Right");

            OscAnimatorLayerBuilder.RemoveLayer(_controller, "v2/Head/Roll");

            var orphaned = AssetDatabase.LoadAllAssetsAtPath(AssetDatabase.GetAssetPath(_controller))
                .OfType<VRCAnimatorTrackingControl>()
                .Any();
            Assert.IsFalse(orphaned, "removing a head-pose layer must not leave a StateMachineBehaviour sub-asset behind");
        }

        [Test]
        public void RemoveLayer_KeepsSharedMask_WhileOtherHeadLayersStillUseIt()
        {
            OscAnimatorLayerBuilder.AddHeadPoseLayer(_controller, "v2/Head/Roll", "Head Tilt Left-Right");
            OscAnimatorLayerBuilder.AddHeadPoseLayer(_controller, "v2/Head/Yaw", "Head Turn Left-Right");

            OscAnimatorLayerBuilder.RemoveLayer(_controller, "v2/Head/Roll");

            var remaining = _controller.layers.First(l => l.name == "OSC_v2_Head_Yaw");
            Assert.IsNotNull(remaining.avatarMask, "surviving layer's mask must not have been destroyed");
        }

        [Test]
        public void RemoveLayer_CleansUpSharedMask_OnceNoLayerReferencesIt()
        {
            OscAnimatorLayerBuilder.AddHeadPoseLayer(_controller, "v2/Head/Roll", "Head Tilt Left-Right");
            OscAnimatorLayerBuilder.RemoveLayer(_controller, "v2/Head/Roll");

            var maskStillInAsset = AssetDatabase.LoadAllAssetsAtPath(AssetDatabase.GetAssetPath(_controller))
                .OfType<AvatarMask>()
                .Any(m => m.name == "OSC_HeadOnlyMask");
            Assert.IsFalse(maskStillInAsset, "orphaned mask sub-asset must be cleaned up");
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
        public void RemoveLayer_WorksForHeadPoseLayer()
        {
            OscAnimatorLayerBuilder.AddHeadPoseLayer(_controller, "v2/Head/Roll", "Head Tilt Left-Right");

            Assert.IsTrue(OscAnimatorLayerBuilder.RemoveLayer(_controller, "v2/Head/Roll"));
            Assert.IsFalse(OscAnimatorLayerBuilder.HasLayer(_controller, "v2/Head/Roll"));
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
                        var muscle = spec.Name == "v2/Head/Roll" ? "Head Tilt Left-Right"
                            : spec.Name == "v2/Head/Yaw" ? "Head Turn Left-Right"
                            : "Head Nod Down-Up";
                        OscAnimatorLayerBuilder.AddHeadPoseLayer(controller, spec.Name, muscle);
                        break;
                }
            }
        }

        [Test]
        public void AllParameters_CanBeWiredWithoutError()
        {
            WireAll(_controller, _avatarRoot.transform, _renderer);

            Assert.AreEqual(OscParameterSpec.All.Count, _controller.parameters.Length);
            // CreateAnimatorControllerAtPath seeds one default "Base Layer";
            // AddLayer only ever replaces layers it owns (named "OSC_*").
            Assert.AreEqual(OscParameterSpec.All.Count, _controller.layers.Count(l => l.name.StartsWith("OSC_")));
            foreach (var spec in OscParameterSpec.All)
            {
                Assert.IsTrue(
                    OscAnimatorLayerBuilder.HasLayer(_controller, spec.Name),
                    $"missing layer for {spec.Name}");
            }
        }

        [Test]
        public void WireThenRemoveAll_RoundTripsBackToZero()
        {
            WireAll(_controller, _avatarRoot.transform, _renderer);

            foreach (var spec in OscParameterSpec.All)
            {
                Assert.IsTrue(OscAnimatorLayerBuilder.RemoveLayer(_controller, spec.Name));
            }

            Assert.AreEqual(0, _controller.parameters.Length);
            Assert.AreEqual(0, _controller.layers.Count(l => l.name.StartsWith("OSC_")));
        }
    }
}
