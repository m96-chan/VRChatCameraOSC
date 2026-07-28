using System.Linq;
using NUnit.Framework;
using UnityEngine;

namespace VRChatCameraOsc.AvatarSetup.Tests
{
    public class BlendShapeAutoMatcherTests
    {
        GameObject _root;

        [TearDown]
        public void TearDown()
        {
            if (_root != null)
            {
                Object.DestroyImmediate(_root);
                _root = null;
            }
        }

        SkinnedMeshRenderer MakeRenderer(string name, params string[] blendShapeNames)
        {
            _root ??= new GameObject("Root");
            var go = new GameObject(name);
            go.transform.SetParent(_root.transform);
            var renderer = go.AddComponent<SkinnedMeshRenderer>();

            var mesh = new Mesh { name = name + "Mesh" };
            mesh.vertices = new[] { Vector3.zero, Vector3.up, Vector3.right };
            mesh.triangles = new[] { 0, 1, 2 };
            foreach (var shape in blendShapeNames)
            {
                mesh.AddBlendShapeFrame(shape, 100f, new Vector3[3], null, null);
            }
            renderer.sharedMesh = mesh;
            return renderer;
        }

        [Test]
        public void FindBodyRenderer_PrefersExactlyNamedBody()
        {
            var other = MakeRenderer("Face", "smile");
            var body = MakeRenderer("Body", "mouth_open");
            var found = BlendShapeAutoMatcher.FindBodyRenderer(new[] { other, body });
            Assert.AreSame(body, found);
        }

        [Test]
        public void FindBodyRenderer_FallsBackToNameContainingBody()
        {
            var other = MakeRenderer("Hair");
            var bodyish = MakeRenderer("MainBodyMesh");
            var found = BlendShapeAutoMatcher.FindBodyRenderer(new[] { other, bodyish });
            Assert.AreSame(bodyish, found);
        }

        [Test]
        public void FindBodyRenderer_FallsBackToMostBlendShapes_WhenNoneNamedBody()
        {
            var small = MakeRenderer("Cloth", "toggle");
            var big = MakeRenderer("Face", "a", "b", "c");
            var found = BlendShapeAutoMatcher.FindBodyRenderer(new[] { small, big });
            Assert.AreSame(big, found);
        }

        [Test]
        public void FindBlendShapeForParam_MatchesCommonNamingConventions()
        {
            var body = MakeRenderer(
                "Body",
                "vrc.blink_left", "vrc.blink_right",
                "Fcl_MTH_A",
                "Mouth_Smile_L", "Mouth_Smile_R",
                "BrowUp_L", "BrowUp_R");

            Assert.AreEqual("Fcl_MTH_A", BlendShapeAutoMatcher.FindBlendShapeForParam(body, "v2/JawOpen"));
            Assert.AreEqual("vrc.blink_left", BlendShapeAutoMatcher.FindBlendShapeForParam(body, "v2/EyeLidLeft"));
            Assert.AreEqual("vrc.blink_right", BlendShapeAutoMatcher.FindBlendShapeForParam(body, "v2/EyeLidRight"));
            Assert.AreEqual("Mouth_Smile_L", BlendShapeAutoMatcher.FindBlendShapeForParam(body, "v2/MouthSmileLeft"));
            Assert.AreEqual("Mouth_Smile_R", BlendShapeAutoMatcher.FindBlendShapeForParam(body, "v2/MouthSmileRight"));
            Assert.AreEqual("BrowUp_L", BlendShapeAutoMatcher.FindBlendShapeForParam(body, "v2/BrowUpLeft"));
            Assert.AreEqual("BrowUp_R", BlendShapeAutoMatcher.FindBlendShapeForParam(body, "v2/BrowUpRight"));
        }

        [Test]
        public void FindBlendShapeForParam_FallsBackToSharedShape_WhenNoLRSplit()
        {
            var body = MakeRenderer("Body", "BrowUp", "Fcl_MTH_Fun", "Mouth_Wide");
            Assert.AreEqual("BrowUp", BlendShapeAutoMatcher.FindBlendShapeForParam(body, "v2/BrowUpLeft"));
            Assert.AreEqual("BrowUp", BlendShapeAutoMatcher.FindBlendShapeForParam(body, "v2/BrowUpRight"));
            // A single (non-split) smile / wide shape drives both sides.
            Assert.AreEqual("Fcl_MTH_Fun", BlendShapeAutoMatcher.FindBlendShapeForParam(body, "v2/MouthSmileLeft"));
            Assert.AreEqual("Fcl_MTH_Fun", BlendShapeAutoMatcher.FindBlendShapeForParam(body, "v2/MouthSmileRight"));
            Assert.AreEqual("Mouth_Wide", BlendShapeAutoMatcher.FindBlendShapeForParam(body, "v2/MouthStretchLeft"));
            Assert.AreEqual("Mouth_Wide", BlendShapeAutoMatcher.FindBlendShapeForParam(body, "v2/MouthStretchRight"));
        }

        [Test]
        public void FindBlendShapeForParam_ReturnsNull_WhenNothingMatches()
        {
            var body = MakeRenderer("Body", "SomeUnrelatedShape");
            Assert.IsNull(BlendShapeAutoMatcher.FindBlendShapeForParam(body, "v2/JawOpen"));
        }

        [Test]
        public void FindBodyRenderer_EmptyList_ReturnsNull()
        {
            Assert.IsNull(BlendShapeAutoMatcher.FindBodyRenderer(new SkinnedMeshRenderer[0]));
        }

        // Real-world regression (Chocolat avatar, issue #16 live test):
        // "smile"/"wide" keywords wired the smile param -> eye_smile_1 and
        // the stretch param -> eyelid_inner_wide, driving EYE shapes from
        // mouth params (eyes permanently half-closed, blink stacking deeper).
        // Mouth params must never match another face region's shapes.
        [Test]
        public void MouthSmile_NeverMatchesEyeShapes_EvenWhenNoMouthShapeExists()
        {
            var body = MakeRenderer("Body", "eye_smile_1", "eye_blink_1_L", "eyelid_inner_wide");
            Assert.IsNull(BlendShapeAutoMatcher.FindBlendShapeForParam(body, "v2/MouthSmileLeft"));
            Assert.IsNull(BlendShapeAutoMatcher.FindBlendShapeForParam(body, "v2/MouthSmileRight"));
        }

        [Test]
        public void MouthSmile_PrefersMouthShape_OverEyeShapeListedFirst()
        {
            var body = MakeRenderer("Body", "eye_smile_1", "mouth_smile_1");
            Assert.AreEqual("mouth_smile_1", BlendShapeAutoMatcher.FindBlendShapeForParam(body, "v2/MouthSmileLeft"));
        }

        [Test]
        public void MouthStretch_NeverMatchesEyelidShapes()
        {
            var body = MakeRenderer("Body", "eyelid_inner_wide", "eye_wide");
            Assert.IsNull(BlendShapeAutoMatcher.FindBlendShapeForParam(body, "v2/MouthStretchLeft"));
            var body2 = MakeRenderer("Body2", "eyelid_inner_wide", "mouth_wide_1");
            Assert.AreEqual("mouth_wide_1", BlendShapeAutoMatcher.FindBlendShapeForParam(body2, "v2/MouthStretchLeft"));
        }

        [Test]
        public void BrowUp_StillMatchesEyebrowShapes_RegionGuardMustNotOverreach()
        {
            var body = MakeRenderer("Body", "eyebrow_up_L", "eyebrow_up_R");
            Assert.AreEqual("eyebrow_up_L", BlendShapeAutoMatcher.FindBlendShapeForParam(body, "v2/BrowUpLeft"));
            Assert.AreEqual("eyebrow_up_R", BlendShapeAutoMatcher.FindBlendShapeForParam(body, "v2/BrowUpRight"));
        }

        [Test]
        public void EyeLid_NeverMatchesEyebrowShapes()
        {
            // "eyebrow_wink_l" is a brow shape; the eyelid param must not grab it.
            var body = MakeRenderer("Body", "eyebrow_wink_l");
            Assert.IsNull(BlendShapeAutoMatcher.FindBlendShapeForParam(body, "v2/EyeLidLeft"));
        }
    }
}
