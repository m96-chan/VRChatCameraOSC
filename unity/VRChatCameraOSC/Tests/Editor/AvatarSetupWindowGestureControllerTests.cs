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
    }
}
