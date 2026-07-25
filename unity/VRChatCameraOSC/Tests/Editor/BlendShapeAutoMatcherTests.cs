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
                "Fcl_MTH_A", "Fcl_MTH_Fun",
                "BrowUp_L", "BrowUp_R");

            Assert.AreEqual("Fcl_MTH_A", BlendShapeAutoMatcher.FindBlendShapeForParam(body, "MouthOpen"));
            Assert.AreEqual("vrc.blink_left", BlendShapeAutoMatcher.FindBlendShapeForParam(body, "EyeBlinkLeft"));
            Assert.AreEqual("vrc.blink_right", BlendShapeAutoMatcher.FindBlendShapeForParam(body, "EyeBlinkRight"));
            Assert.AreEqual("Fcl_MTH_Fun", BlendShapeAutoMatcher.FindBlendShapeForParam(body, "MouthSmile"));
            Assert.AreEqual("BrowUp_L", BlendShapeAutoMatcher.FindBlendShapeForParam(body, "BrowUpLeft"));
            Assert.AreEqual("BrowUp_R", BlendShapeAutoMatcher.FindBlendShapeForParam(body, "BrowUpRight"));
        }

        [Test]
        public void FindBlendShapeForParam_BrowFallsBackToSharedShape_WhenNoLRSplit()
        {
            var body = MakeRenderer("Body", "BrowUp");
            Assert.AreEqual("BrowUp", BlendShapeAutoMatcher.FindBlendShapeForParam(body, "BrowUpLeft"));
            Assert.AreEqual("BrowUp", BlendShapeAutoMatcher.FindBlendShapeForParam(body, "BrowUpRight"));
        }

        [Test]
        public void FindBlendShapeForParam_ReturnsNull_WhenNothingMatches()
        {
            var body = MakeRenderer("Body", "SomeUnrelatedShape");
            Assert.IsNull(BlendShapeAutoMatcher.FindBlendShapeForParam(body, "MouthOpen"));
        }

        [Test]
        public void MouthWideKeywords_MatchWideAndPucker()
        {
            var body = MakeRenderer("Body", "Mouth_Wide", "Mouth_Pucker");
            Assert.AreEqual("Mouth_Wide", BlendShapeAutoMatcher.FindBlendShape(body, BlendShapeAutoMatcher.MouthWidePositiveKeywords));
            Assert.AreEqual("Mouth_Pucker", BlendShapeAutoMatcher.FindBlendShape(body, BlendShapeAutoMatcher.MouthWideNegativeKeywords));
        }

        [Test]
        public void FindBodyRenderer_EmptyList_ReturnsNull()
        {
            Assert.IsNull(BlendShapeAutoMatcher.FindBodyRenderer(new SkinnedMeshRenderer[0]));
        }
    }
}
