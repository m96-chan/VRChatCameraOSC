using System.Linq;
using NUnit.Framework;
using UnityEditor;
using UnityEditor.Animations;
using UnityEngine;
using VRC.SDK3.Avatars.Components;

namespace VRChatCameraOsc.AvatarSetup.Tests
{
    /// <summary>
    /// Covers <see cref="AvatarSetupWindow.EnsureGestureController"/> /
    /// <see cref="AvatarSetupWindow.TryGetGestureController"/> (issue #16
    /// head-pose fix): head-pose layers moved from FX to the Gesture
    /// playable layer, and creating a Gesture controller from scratch must
    /// preserve the avatar's default hand gestures by copying the VRC SDK's
    /// stock hands controller rather than starting blank.
    /// </summary>
    public class AvatarSetupWindowGestureControllerTests
    {
        const string TestDir = "Assets/_VRChatCameraOscGestureTests";

        GameObject _avatarRoot;
        VRCAvatarDescriptor _descriptor;

        [SetUp]
        public void SetUp()
        {
            if (!AssetDatabase.IsValidFolder(TestDir))
            {
                AssetDatabase.CreateFolder("Assets", "_VRChatCameraOscGestureTests");
            }

            _avatarRoot = new GameObject("Avatar");
            _descriptor = _avatarRoot.AddComponent<VRCAvatarDescriptor>();
        }

        [TearDown]
        public void TearDown()
        {
            Object.DestroyImmediate(_avatarRoot);
            AssetDatabase.DeleteAsset(TestDir);
        }

        static VRCAvatarDescriptor.CustomAnimLayer[] SingleGestureLayer(bool isDefault, AnimatorController controller = null)
        {
            return new[]
            {
                new VRCAvatarDescriptor.CustomAnimLayer
                {
                    type = VRCAvatarDescriptor.AnimLayerType.Gesture,
                    isDefault = isDefault,
                    animatorController = controller,
                    isEnabled = true,
                },
            };
        }

        [Test]
        public void TryGetGestureController_WhenLayerIsDefault_ReturnsNull()
        {
            _descriptor.baseAnimationLayers = SingleGestureLayer(isDefault: true);

            Assert.IsNull(AvatarSetupWindow.TryGetGestureController(_descriptor));
        }

        [Test]
        public void TryGetGestureController_WhenCustomControllerAssigned_ReturnsIt()
        {
            var controller = AnimatorController.CreateAnimatorControllerAtPath(TestDir + "/Existing.controller");
            _descriptor.baseAnimationLayers = SingleGestureLayer(isDefault: false, controller);

            Assert.AreSame(controller, AvatarSetupWindow.TryGetGestureController(_descriptor));
        }

        [Test]
        public void EnsureGestureController_WithExistingCustomController_ReturnsItUnchanged()
        {
            var controller = AnimatorController.CreateAnimatorControllerAtPath(TestDir + "/Existing.controller");
            _descriptor.baseAnimationLayers = SingleGestureLayer(isDefault: false, controller);

            var result = AvatarSetupWindow.EnsureGestureController(_descriptor, out var copiedDefault);

            Assert.AreSame(controller, result);
            Assert.IsTrue(copiedDefault, "an already-custom controller counts as success, not a fallback");
        }

        [Test]
        public void EnsureGestureController_WhenDefault_CopiesStockHandsControllerAndPreservesItsMask()
        {
            // Simulate the VRC SDK's stock "vrc_AvatarV3HandsLayer" asset
            // (normally under the SDK's optional "AV3 Demo Assets" sample) by
            // creating one with that exact name under the test folder —
            // AvatarSetupWindow.FindDefaultHandsControllerAssetPath finds it
            // by asset name project-wide, regardless of location.
            var stockHands = AnimatorController.CreateAnimatorControllerAtPath(
                TestDir + "/" + AvatarSetupWindow.DefaultHandsControllerAssetName + ".controller");
            var handMask = new AvatarMask { name = "StockHandMask" };
            AssetDatabase.AddObjectToAsset(handMask, stockHands);
            var layers = stockHands.layers;
            layers[0].avatarMask = handMask;
            stockHands.layers = layers;

            _descriptor.baseAnimationLayers = SingleGestureLayer(isDefault: true);

            var result = AvatarSetupWindow.EnsureGestureController(_descriptor, out var copiedDefault);

            Assert.IsTrue(copiedDefault);
            Assert.IsNotNull(result);
            Assert.AreNotSame(stockHands, result, "must be a copy, not the stock asset itself");

            var gestureLayer = _descriptor.baseAnimationLayers
                .Single(l => l.type == VRCAvatarDescriptor.AnimLayerType.Gesture);
            Assert.IsFalse(gestureLayer.isDefault);
            Assert.AreSame(result, gestureLayer.animatorController);
            Assert.IsNotNull(gestureLayer.mask, "gesture layer mask should follow the copied controller's first sub-layer mask");
            Assert.AreEqual("StockHandMask", gestureLayer.mask.name);
        }

        /// <summary>A vrc_HandsOnly-like mask: fingers allowed, everything
        /// else (crucially Head) denied.</summary>
        AvatarMask CreateHandsOnlyLikeMask(string assetName)
        {
            var mask = new AvatarMask { name = assetName };
            for (var part = AvatarMaskBodyPart.Root; part < AvatarMaskBodyPart.LastBodyPart; part++)
            {
                mask.SetHumanoidBodyPartActive(
                    part,
                    part == AvatarMaskBodyPart.LeftFingers || part == AvatarMaskBodyPart.RightFingers);
            }
            AssetDatabase.CreateAsset(mask, TestDir + "/" + assetName + ".mask");
            return mask;
        }

        static AnimatorController ControllerWithFirstLayerMask(string path, AvatarMask mask)
        {
            var controller = AnimatorController.CreateAnimatorControllerAtPath(path);
            var layers = controller.layers;
            layers[0].avatarMask = mask;
            controller.layers = layers;
            return controller;
        }

        // Issue #25: VRChat ANDs the Gesture controller's FIRST layer mask
        // over the entire Gesture playable layer — a hands-only first mask
        // silently disables the head-pose layers in the client (while the
        // Unity editor preview plays them fine).
        [Test]
        public void EnsureGestureMaskAllowsHead_HandsOnlyFirstMask_SwapsInCombinedMask()
        {
            var handsOnly = CreateHandsOnlyLikeMask("FakeHandsOnly");
            var controller = ControllerWithFirstLayerMask(TestDir + "/Gesture.controller", handsOnly);
            _descriptor.baseAnimationLayers = SingleGestureLayer(isDefault: false, controller);

            AvatarSetupWindow.EnsureGestureMaskAllowsHead(_descriptor, controller);

            var first = controller.layers[0].avatarMask;
            Assert.IsNotNull(first);
            Assert.AreEqual(AvatarSetupWindow.GestureMaskName, first.name);
            Assert.IsTrue(first.GetHumanoidBodyPartActive(AvatarMaskBodyPart.Head), "Head must be allowed");
            Assert.IsTrue(first.GetHumanoidBodyPartActive(AvatarMaskBodyPart.LeftFingers), "original parts preserved");
            Assert.IsFalse(first.GetHumanoidBodyPartActive(AvatarMaskBodyPart.Body), "denied parts stay denied");
            // The original (SDK-owned in real projects) mask asset is untouched.
            Assert.IsFalse(handsOnly.GetHumanoidBodyPartActive(AvatarMaskBodyPart.Head));
            // Descriptor's Gesture slot follows (VRChat reads it too).
            var gestureLayer = _descriptor.baseAnimationLayers
                .Single(l => l.type == VRCAvatarDescriptor.AnimLayerType.Gesture);
            Assert.AreSame(first, gestureLayer.mask);
        }

        [Test]
        public void EnsureGestureMaskAllowsHead_IsIdempotent_SingleCombinedMaskSubAsset()
        {
            var handsOnly = CreateHandsOnlyLikeMask("FakeHandsOnly");
            var controller = ControllerWithFirstLayerMask(TestDir + "/Gesture.controller", handsOnly);
            _descriptor.baseAnimationLayers = SingleGestureLayer(isDefault: false, controller);

            AvatarSetupWindow.EnsureGestureMaskAllowsHead(_descriptor, controller);
            AvatarSetupWindow.EnsureGestureMaskAllowsHead(_descriptor, controller);

            var combined = AssetDatabase.LoadAllAssetsAtPath(AssetDatabase.GetAssetPath(controller))
                .OfType<AvatarMask>()
                .Count(m => m.name == AvatarSetupWindow.GestureMaskName);
            Assert.AreEqual(1, combined);
        }

        [Test]
        public void EnsureGestureMaskAllowsHead_NoOp_WhenHeadAlreadyAllowedOrNoMask()
        {
            // Head already allowed: mask reference must be kept as-is.
            var headOk = CreateHandsOnlyLikeMask("HeadOk");
            headOk.SetHumanoidBodyPartActive(AvatarMaskBodyPart.Head, true);
            var controller = ControllerWithFirstLayerMask(TestDir + "/Gesture.controller", headOk);
            _descriptor.baseAnimationLayers = SingleGestureLayer(isDefault: false, controller);
            AvatarSetupWindow.EnsureGestureMaskAllowsHead(_descriptor, controller);
            Assert.AreSame(headOk, controller.layers[0].avatarMask);

            // No first-layer mask: nothing is denied, nothing to fix.
            var bare = ControllerWithFirstLayerMask(TestDir + "/Gesture2.controller", null);
            _descriptor.baseAnimationLayers = SingleGestureLayer(isDefault: false, bare);
            AvatarSetupWindow.EnsureGestureMaskAllowsHead(_descriptor, bare);
            Assert.IsNull(bare.layers[0].avatarMask);
        }

        [Test]
        public void RestoreGestureMask_SwapsBackAndDeletesCombinedSubAsset()
        {
            var handsOnly = CreateHandsOnlyLikeMask("FakeHandsOnly");
            var controller = ControllerWithFirstLayerMask(TestDir + "/Gesture.controller", handsOnly);
            _descriptor.baseAnimationLayers = SingleGestureLayer(isDefault: false, controller);
            AvatarSetupWindow.EnsureGestureMaskAllowsHead(_descriptor, controller);

            AvatarSetupWindow.RestoreGestureMask(_descriptor, controller);

            var first = controller.layers[0].avatarMask;
            // Restored by equivalence: the stock vrc_HandsOnly if findable in
            // this project (it ships inside the SDK package), else null.
            Assert.IsTrue(first == null || first.name == "vrc_HandsOnly", $"got: {first?.name}");
            var lingering = AssetDatabase.LoadAllAssetsAtPath(AssetDatabase.GetAssetPath(controller))
                .OfType<AvatarMask>()
                .Any(m => m.name == AvatarSetupWindow.GestureMaskName);
            Assert.IsFalse(lingering, "combined mask sub-asset must be cleaned up");
        }

        [Test]
        public void EnsureGestureController_WhenDefaultAndNoStockAssetFound_CreatesBlankControllerAndReportsFallback()
        {
            // This scenario needs a project where no "vrc_AvatarV3HandsLayer"
            // asset exists at all. Newer VRChat SDK packages ship it inside
            // Packages/com.vrchat.avatars/Samples (visible to
            // AssetDatabase.FindAssets without any sample import), and a test
            // cannot delete package contents — skip there instead of
            // asserting an unreachable state (issue #21 EditMode run).
            if (AvatarSetupWindow.FindDefaultHandsControllerAssetPath() != null)
            {
                Assert.Ignore(
                    "stock hands controller ships inside this SDK package — " +
                    "the no-asset fallback path is unreachable in this project");
            }

            _descriptor.baseAnimationLayers = SingleGestureLayer(isDefault: true);

            var result = AvatarSetupWindow.EnsureGestureController(_descriptor, out var copiedDefault);

            Assert.IsFalse(copiedDefault);
            Assert.IsNotNull(result);
            var gestureLayer = _descriptor.baseAnimationLayers
                .Single(l => l.type == VRCAvatarDescriptor.AnimLayerType.Gesture);
            Assert.IsFalse(gestureLayer.isDefault);
            Assert.AreSame(result, gestureLayer.animatorController);
        }
        [Test]
        public void RestoreGestureMask_NoOp_WhenFirstLayerMaskIsNotOurs()
        {
            var controller = AnimatorController.CreateAnimatorControllerAtPath(TestDir + "/Gesture2.controller");
            var foreign = new AvatarMask { name = "SomeoneElsesMask" };
            AssetDatabase.AddObjectToAsset(foreign, controller);
            var layers = controller.layers;
            layers[0].avatarMask = foreign;
            controller.layers = layers;
            _descriptor.baseAnimationLayers = SingleGestureLayer(isDefault: false, controller);

            AvatarSetupWindow.RestoreGestureMask(_descriptor, controller);

            Assert.AreSame(foreign, controller.layers[0].avatarMask, "foreign masks must be left alone");
        }
    }
}
