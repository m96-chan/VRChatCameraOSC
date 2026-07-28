using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.Animations;
using UnityEngine;
using VRC.SDK3.Avatars.Components;
using VRC.SDK3.Avatars.ScriptableObjects;

// EnsureGestureController/EnsureGestureMaskAllowsHead/TryGet*Controller/RestoreGestureMask
// are `internal` (rather than the `static`-in-a-window-class-implies-private
// default) specifically so the EditMode test assembly can exercise the
// Gesture-controller copy-default-hands-layer logic directly — see
// Tests/Editor/AvatarSetupWindowGestureControllerTests.cs.
[assembly: System.Runtime.CompilerServices.InternalsVisibleTo("VRChatCameraOsc.AvatarSetup.Editor.Tests")]

namespace VRChatCameraOsc.AvatarSetup
{
    /// <summary>
    /// Editor wizard (issue #16): given a Humanoid avatar with existing blend
    /// shapes, generates/merges the VRC Expression Parameters and FX Animator
    /// Controller layers so VRChat's own OSC-to-parameter routing has
    /// something to drive. Does not create blend shapes, does not run at
    /// avatar runtime — see the package README.
    /// </summary>
    public class AvatarSetupWindow : EditorWindow
    {
        VRCAvatarDescriptor _avatar;
        readonly Dictionary<string, SkinnedMeshRenderer> _positiveRenderer = new Dictionary<string, SkinnedMeshRenderer>();
        readonly Dictionary<string, string> _positiveBlendShape = new Dictionary<string, string>();
        /// <summary>Optional eye-wide shape per v2/EyeLid* param (issue #24)
        /// — drives the 0.75..1 segment of the eyelid tree, no extra bits.</summary>
        readonly Dictionary<string, SkinnedMeshRenderer> _wideRenderer = new Dictionary<string, SkinnedMeshRenderer>();
        readonly Dictionary<string, string> _wideBlendShape = new Dictionary<string, string>();
        bool _includeHeadPose = true;
        /// <summary>Wire the VCO_GestureLeft/Right Int hand-pose layers on
        /// the Gesture controller (issue #8). No blend shape involved —
        /// muscle-driven, so no picker; just this toggle.</summary>
        bool _includeHandGestures = true;
        bool _disableNativeEyelids = true;
        bool _pendingAutoFill;
        Vector2 _scroll;

        [MenuItem("VRChatCameraOSC/Avatar Setup Wizard")]
        public static void Open() => GetWindow<AvatarSetupWindow>("VRChatCameraOSC Setup");

        void OnGUI()
        {
            EditorGUILayout.HelpBox(
                "Wires the standard VRCFT (Unified Expressions) v2/* parameters this " +
                "app sends to blend shapes/head bone you already have on this " +
                "Humanoid avatar — a \"lite\" face-tracking avatar that also works " +
                "with VRCFaceTracking itself. Doesn't create blend shapes, doesn't " +
                "add any runtime script.",
                MessageType.Info);

            var newAvatar = (VRCAvatarDescriptor)EditorGUILayout.ObjectField(
                "Avatar", _avatar, typeof(VRCAvatarDescriptor), true);
            if (newAvatar != _avatar)
            {
                _avatar = newAvatar;
                _positiveRenderer.Clear();
                _positiveBlendShape.Clear();
                _wideRenderer.Clear();
                _wideBlendShape.Clear();
                _pendingAutoFill = true;
            }

            if (_avatar == null)
            {
                return;
            }

            var animator = _avatar.GetComponent<Animator>();
            if (animator == null || !animator.isHuman)
            {
                EditorGUILayout.HelpBox("This avatar isn't Humanoid — out of scope (issue #16).", MessageType.Error);
                return;
            }

            var renderers = _avatar.GetComponentsInChildren<SkinnedMeshRenderer>(true)
                .Where(r => r.sharedMesh != null && r.sharedMesh.blendShapeCount > 0)
                .ToArray();
            if (renderers.Length == 0)
            {
                EditorGUILayout.HelpBox("No SkinnedMeshRenderer with blend shapes found under this avatar.", MessageType.Warning);
            }

            if (_pendingAutoFill)
            {
                _pendingAutoFill = false;
                AutoFillFromBody(renderers);
            }

            EditorGUILayout.Space();
            if (GUILayout.Button("Auto-fill from Body mesh (best-effort guess, review before Apply)"))
            {
                AutoFillFromBody(renderers);
            }

            _scroll = EditorGUILayout.BeginScrollView(_scroll);
            foreach (var spec in OscParameterSpec.All)
            {
                // Head pose and hand gestures are muscle-driven (no blend
                // shape to pick) — their wiring is the toggles below, not a
                // picker row.
                if (spec.Kind == OscParamKind.HeadPose || spec.Kind == OscParamKind.GestureInt)
                {
                    continue;
                }
                DrawBlendShapePicker(spec, renderers);
            }
            EditorGUILayout.EndScrollView();

            EditorGUILayout.Space();
            _includeHeadPose = EditorGUILayout.ToggleLeft(
                "Wire v2/Head/Yaw / Pitch / Roll to the Humanoid head bone (additive layer)",
                _includeHeadPose);
            _includeHandGestures = EditorGUILayout.ToggleLeft(
                "Wire hand gestures (VCO_GestureLeft/Right) to Humanoid finger muscles (Gesture layer)",
                _includeHandGestures);

            var eyelidType = _avatar.customEyeLookSettings.eyelidType;
            if (eyelidType != VRCAvatarDescriptor.EyelidType.None &&
                (_positiveRenderer.ContainsKey("v2/EyeLidLeft") || _positiveRenderer.ContainsKey("v2/EyeLidRight")))
            {
                _disableNativeEyelids = EditorGUILayout.ToggleLeft(
                    "Disable VRChat's native auto-blink/eye-tracking (Eyelids is currently " +
                    $"{eyelidType}) — otherwise it fights with OSC-driven blinking",
                    _disableNativeEyelids);
            }

            EditorGUILayout.Space();
            var wired = VrcExpressionParametersMerger.IsFullyWired(_avatar.expressionParameters, OscParameterSpec.Core);
            var activeLayers = CountActiveLayers(_avatar);
            EditorGUILayout.LabelField(
                "Status",
                (wired ? "ON — parameters declared" : "OFF — not applied yet") +
                $"  |  {activeLayers}/{OscParameterSpec.All.Count} actually wired (have an Animator layer)");
            if (wired && activeLayers < OscParameterSpec.All.Count)
            {
                EditorGUILayout.HelpBox(
                    $"{OscParameterSpec.All.Count - activeLayers} parameter(s) are declared but not driving " +
                    "anything — their blend shape picker above is on (skip). That's fine if intentional.",
                    MessageType.Info);
            }

            var newWired = GUILayout.Toggle(wired, wired ? "ON (click to remove)" : "OFF (click to apply)", "Button");
            if (newWired != wired)
            {
                if (newWired)
                {
                    RunSetup();
                }
                else
                {
                    RunRemove();
                }
            }
        }

        /// <summary>How many of the parameters currently have an <c>OSC_*</c>
        /// layer actually driving them (vs. just being declared). Head-pose
        /// parameters live on the Gesture controller (issue #16 head-pose
        /// fix) but are also checked against FX so avatars set up by an
        /// older version of this wizard still show as wired.</summary>
        static int CountActiveLayers(VRCAvatarDescriptor avatar)
        {
            var fx = TryGetFxController(avatar);
            var additive = TryGetAdditiveController(avatar);
            var gesture = TryGetGestureController(avatar);
            return OscParameterSpec.All.Count(s =>
                (s.Kind == OscParamKind.HeadPose && gesture != null &&
                 OscAnimatorLayerBuilder.HasLayer(gesture, OscAnimatorLayerBuilder.CombinedHeadKey)) ||
                (fx != null && OscAnimatorLayerBuilder.HasLayer(fx, s.Name)) ||
                (additive != null && OscAnimatorLayerBuilder.HasLayer(additive, s.Name)) ||
                (gesture != null && OscAnimatorLayerBuilder.HasLayer(gesture, s.Name)));
        }

        /// <summary>
        /// Best-effort auto-suggestion (issue #16 follow-up): guesses blend
        /// shapes from the avatar's main face mesh (see
        /// <see cref="BlendShapeAutoMatcher.FindBodyRenderer"/>) by common
        /// naming convention. Only fills parameters that aren't already
        /// picked, so it never clobbers a manual choice — safe to click
        /// again after changing renderers/blend shapes.
        /// </summary>
        void AutoFillFromBody(SkinnedMeshRenderer[] renderers)
        {
            var body = BlendShapeAutoMatcher.FindBodyRenderer(renderers);
            if (body == null)
            {
                return;
            }

            foreach (var spec in OscParameterSpec.All)
            {
                if (spec.Kind == OscParamKind.HeadPose || spec.Kind == OscParamKind.GestureInt)
                {
                    continue;
                }

                if (!_positiveRenderer.ContainsKey(spec.Name))
                {
                    var shape = BlendShapeAutoMatcher.FindBlendShapeForParam(body, spec.Name);
                    if (shape != null)
                    {
                        _positiveRenderer[spec.Name] = body;
                        _positiveBlendShape[spec.Name] = shape;
                    }
                }

                if (spec.Kind == OscParamKind.EyeLid && !_wideRenderer.ContainsKey(spec.Name))
                {
                    var widePseudo = spec.Name == "v2/EyeLidLeft" ? "EyeWideLeft" : "EyeWideRight";
                    var wide = BlendShapeAutoMatcher.FindBlendShapeForParam(body, widePseudo);
                    if (wide != null)
                    {
                        _wideRenderer[spec.Name] = body;
                        _wideBlendShape[spec.Name] = wide;
                    }
                }
            }
        }

        void DrawBlendShapePicker(OscParamSpec spec, SkinnedMeshRenderer[] renderers)
        {
            EditorGUILayout.LabelField(
                spec.Optional ? spec.Name + "  (optional — declared only when wired)" : spec.Name,
                EditorStyles.boldLabel);
            EditorGUI.indentLevel++;
            DrawOneShapePicker(
                spec.Kind == OscParamKind.EyeLid ? "Blink shape (inverted)" : "Blend shape",
                spec.Name, renderers, _positiveRenderer, _positiveBlendShape);
            if (spec.Kind == OscParamKind.EyeLid)
            {
                DrawOneShapePicker("Eye-wide (optional)", spec.Name, renderers, _wideRenderer, _wideBlendShape);
            }
            EditorGUI.indentLevel--;
        }

        void DrawOneShapePicker(
            string label,
            string paramName,
            SkinnedMeshRenderer[] renderers,
            Dictionary<string, SkinnedMeshRenderer> rendererMap,
            Dictionary<string, string> shapeMap)
        {
            var rendererOptions = new[] { "(skip)" }.Concat(renderers.Select(r => r.name)).ToArray();
            rendererMap.TryGetValue(paramName, out var currentRenderer);
            var rendererIndex = currentRenderer == null ? 0 : System.Array.IndexOf(renderers, currentRenderer) + 1;

            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField(label, GUILayout.Width(140));
            var newRendererIndex = EditorGUILayout.Popup(rendererIndex, rendererOptions, GUILayout.Width(160));

            if (newRendererIndex == 0)
            {
                rendererMap.Remove(paramName);
                shapeMap.Remove(paramName);
                EditorGUILayout.EndHorizontal();
                return;
            }

            var renderer = renderers[newRendererIndex - 1];
            rendererMap[paramName] = renderer;

            var shapeNames = System.Linq.Enumerable.Range(0, renderer.sharedMesh.blendShapeCount)
                .Select(renderer.sharedMesh.GetBlendShapeName)
                .ToArray();
            shapeMap.TryGetValue(paramName, out var currentShape);
            var shapeIndex = System.Array.IndexOf(shapeNames, currentShape);
            if (shapeIndex < 0)
            {
                shapeIndex = 0;
            }
            var newShapeIndex = EditorGUILayout.Popup(shapeIndex, shapeNames);
            if (shapeNames.Length > 0)
            {
                shapeMap[paramName] = shapeNames[newShapeIndex];
            }
            EditorGUILayout.EndHorizontal();
        }

        void RunSetup()
        {
            var expressionParameters = EnsureExpressionParameters(_avatar);
            // Migrate a pre-issue-#21 (custom10) setup in place: strip the
            // retired parameters and their layers before wiring the v2/* set.
            RemoveLegacyCustom10(_avatar);
            var toDeclare = SpecsToDeclare(_positiveRenderer.Keys);
            var added = VrcExpressionParametersMerger.Merge(expressionParameters, toDeclare);
            // Optional params the user un-wired since a previous Apply: drop
            // their declaration (stop paying bits) and their stale layer.
            var unwiredOptional = OscParameterSpec.All
                .Where(s => s.Optional && !_positiveRenderer.ContainsKey(s.Name))
                .ToList();
            VrcExpressionParametersMerger.RemoveByName(
                expressionParameters, unwiredOptional.Select(s => s.Name));

            var controller = EnsureFxController(_avatar);
            AnimatorController gestureController = null;
            var gestureCopiedDefaultHands = true;
            foreach (var spec in OscParameterSpec.All)
            {
                switch (spec.Kind)
                {
                    case OscParamKind.BlendShape:
                        if (_positiveRenderer.TryGetValue(spec.Name, out var r) &&
                            _positiveBlendShape.TryGetValue(spec.Name, out var shape))
                        {
                            OscAnimatorLayerBuilder.AddBlendShapeLayer(
                                controller, _avatar.transform, spec.Name, r, shape, spec.FullScale);
                        }
                        break;
                    case OscParamKind.EyeLid:
                        if (_positiveRenderer.TryGetValue(spec.Name, out var eyeR) &&
                            _positiveBlendShape.TryGetValue(spec.Name, out var eyeShape))
                        {
                            _wideRenderer.TryGetValue(spec.Name, out var wideR);
                            _wideBlendShape.TryGetValue(spec.Name, out var wideShape);
                            OscAnimatorLayerBuilder.AddEyeLidLayer(
                                controller, _avatar.transform, spec.Name, eyeR, eyeShape, wideR, wideShape);
                        }
                        break;
                    case OscParamKind.HeadPose:
                        // All three axes are wired below as ONE combined
                        // layer (issue #27: per-axis Override layers
                        // group-override each other in-client).
                        break;
                }
            }

            foreach (var stale in unwiredOptional)
            {
                OscAnimatorLayerBuilder.RemoveLayer(controller, stale.Name);
            }

            if (_includeHeadPose || _includeHandGestures)
            {
                gestureController = EnsureGestureController(_avatar, out gestureCopiedDefaultHands);
            }

            if (_includeHeadPose)
            {
                // Migrate the per-axis head layers a previous version added —
                // left in place they would group-override the combined layer
                // (issue #27), the very bug being fixed.
                foreach (var headSpec in OscParameterSpec.All.Where(s => s.Kind == OscParamKind.HeadPose))
                {
                    OscAnimatorLayerBuilder.RemoveLayer(gestureController, headSpec.Name);
                }
                OscAnimatorLayerBuilder.AddCombinedHeadLayer(gestureController);
                // Without this the head layer is inert in the client: the
                // Gesture first-layer mask (stock: vrc_HandsOnly) is AND-ed
                // over the whole playable layer (issue #25).
                EnsureGestureMaskAllowsHead(_avatar, gestureController);
            }

            if (_includeHandGestures)
            {
                // Hand-pose Int layers (issue #8). No first-layer-mask work
                // needed here, unlike head pose: both the stock vrc_HandsOnly
                // mask and the wizard's combined OSC_GestureMask already
                // allow the Left/RightFingers parts these layers animate.
                OscAnimatorLayerBuilder.AddHandGestureLayer(gestureController, leftHand: true);
                OscAnimatorLayerBuilder.AddHandGestureLayer(gestureController, leftHand: false);
            }

            // Migrate away from the Additive-playable-layer experiment
            // (issue #25 attempts 2-3): VRChat applies that whole playable
            // layer additively, so even Override muscle layers zero out
            // against its reference pose. Head is back on Gesture — the same
            // home, blending mode, and clip construction the (working) hand
            // gestures use.
            var migrateAdditive = TryGetAdditiveController(_avatar);
            if (migrateAdditive != null)
            {
                foreach (var headSpec in OscParameterSpec.All.Where(s => s.Kind == OscParamKind.HeadPose))
                {
                    OscAnimatorLayerBuilder.RemoveLayer(migrateAdditive, headSpec.Name);
                }
                EditorUtility.SetDirty(migrateAdditive);
            }

            var disabledEyelids = false;
            if (_disableNativeEyelids &&
                (_positiveRenderer.ContainsKey("v2/EyeLidLeft") || _positiveRenderer.ContainsKey("v2/EyeLidRight")) &&
                _avatar.customEyeLookSettings.eyelidType != VRCAvatarDescriptor.EyelidType.None)
            {
                Undo.RecordObject(_avatar, "Disable native VRChat auto-blink (VRChatCameraOSC drives blink)");
                _avatar.customEyeLookSettings.eyelidType = VRCAvatarDescriptor.EyelidType.None;
                EditorUtility.SetDirty(_avatar);
                disabledEyelids = true;
            }

            EditorUtility.SetDirty(controller);
            if (gestureController != null)
            {
                EditorUtility.SetDirty(gestureController);
            }
            AssetDatabase.SaveAssets();
            EditorUtility.DisplayDialog(
                "VRChatCameraOSC Setup",
                $"Done. Added {added} new expression parameter(s). FX layers wired for the selected blend shapes" +
                (_includeHeadPose ? " and head pose wired on the Gesture layer (Head = Animation via a" +
                                     " VRCAnimatorTrackingControl)." : ".") +
                (_includeHandGestures
                    ? "\n\nHand gestures wired: VCO_GestureLeft/Right (Int 0-7) pose the fingers on the " +
                      "Gesture layer; at 0 (Neutral) the layers write nothing, so VRChat's own " +
                      "keyboard/controller hand gestures keep working."
                    : "") +
                (_includeHeadPose && !gestureCopiedDefaultHands
                    ? "\n\nCouldn't find the VRC SDK's default hand-gesture controller " +
                      $"(\"{DefaultHandsControllerAssetName}\", normally imported via the SDK's " +
                      "\"AV3 Demo Assets\" sample) to copy as a starting point, so a blank Gesture " +
                      "controller was created instead — any default VRChat hand gestures (fist, point, " +
                      "etc.) will need to be set up manually if you want them."
                    : "") +
                (disabledEyelids
                    ? "\n\nAlso disabled VRChat's native Eyelids (was fighting with OSC-driven blinking) — " +
                      "re-enable it yourself in the avatar's Eye Look settings if you ever remove OSC blink wiring."
                    : ""),
                "OK");
        }

        /// <summary>
        /// The "OFF" side of the toggle: removes every VRChatCameraOSC
        /// parameter and <c>OSC_*</c> layer this wizard could have added.
        /// Leaves the Expression Parameters asset and the FX/Gesture Animator
        /// Controllers themselves in place (only their VRChatCameraOSC-owned
        /// contents are removed) since they may hold unrelated setup — the
        /// Gesture controller in particular may be carrying the avatar's real
        /// hand-gesture animations.
        ///
        /// Head-pose layers are stripped from <em>both</em> controllers:
        /// their current home (Gesture, issue #16 head-pose fix) and their
        /// old one (FX, from an older version of this wizard) — an avatar set
        /// up before the fix only has the FX copy, and this toggle must still
        /// show/act correctly for it.
        /// </summary>
        void RunRemove()
        {
            var removedParams = 0;
            if (_avatar.expressionParameters != null)
            {
                removedParams = VrcExpressionParametersMerger.Remove(_avatar.expressionParameters, OscParameterSpec.All);
            }
            removedParams += RemoveLegacyCustom10(_avatar);

            var removedLayers = 0;
            var controller = TryGetFxController(_avatar);
            var additiveController = TryGetAdditiveController(_avatar);
            if (additiveController != null)
            {
                foreach (var headSpec in OscParameterSpec.All.Where(s => s.Kind == OscParamKind.HeadPose))
                {
                    if (OscAnimatorLayerBuilder.RemoveLayer(additiveController, headSpec.Name))
                    {
                        removedLayers++;
                    }
                }
                EditorUtility.SetDirty(additiveController);
            }
            var gestureController = TryGetGestureController(_avatar);
            if (gestureController != null)
            {
                if (OscAnimatorLayerBuilder.RemoveLayer(gestureController, OscAnimatorLayerBuilder.CombinedHeadKey))
                {
                    removedLayers++;
                }
                // Pre-combined-era homes: strip per-axis head layers a
                // previous version put on Gesture, and undo its
                // first-layer-mask swap.
                foreach (var headSpec in OscParameterSpec.All.Where(s => s.Kind == OscParamKind.HeadPose))
                {
                    if (OscAnimatorLayerBuilder.RemoveLayer(gestureController, headSpec.Name))
                    {
                        removedLayers++;
                    }
                }
                // Hand-gesture pose layers (issue #8) — RemoveLayer also
                // drops their Int animator parameter and per-hand masks.
                foreach (var gestureSpec in OscParameterSpec.All.Where(s => s.Kind == OscParamKind.GestureInt))
                {
                    if (OscAnimatorLayerBuilder.RemoveLayer(gestureController, gestureSpec.Name))
                    {
                        removedLayers++;
                    }
                }
                RestoreGestureMask(_avatar, gestureController);
                EditorUtility.SetDirty(gestureController);
            }
            if (controller != null)
            {
                foreach (var spec in OscParameterSpec.All)
                {
                    if (OscAnimatorLayerBuilder.RemoveLayer(controller, spec.Name))
                    {
                        removedLayers++;
                    }
                }
                EditorUtility.SetDirty(controller);
            }

            AssetDatabase.SaveAssets();
            EditorUtility.DisplayDialog(
                "VRChatCameraOSC Setup",
                $"Removed {removedParams} expression parameter(s) and {removedLayers} Animator layer(s).",
                "OK");
        }

        /// <summary>The parameters to declare on the avatar this Apply: the
        /// always-on core set plus whichever optional extras actually have a
        /// blend shape wired (issue #24 — unwired optionals cost no
        /// expression-parameter bits).</summary>
        internal static List<OscParamSpec> SpecsToDeclare(IEnumerable<string> wiredParamNames)
        {
            var wired = new HashSet<string>(wiredParamNames);
            return OscParameterSpec.All
                .Where(s => !s.Optional || wired.Contains(s.Name))
                .ToList();
        }

        /// <summary>Name of the combined (original-parts + Head) first-layer
        /// Gesture mask this wizard swaps in — see
        /// <see cref="EnsureGestureMaskAllowsHead"/> (issue #25).</summary>
        internal const string GestureMaskName = "OSC_GestureMask";

        /// <summary>
        /// VRChat ANDs the Gesture controller's <b>first layer's</b> avatar
        /// mask over the entire Gesture playable layer (community-documented:
        /// vrc.school "Avatar Masks", vrclibrary.com; there is a VRChat
        /// feedback ticket asking for per-layer masks). The stock hand-gesture
        /// setups put <c>vrc_HandsOnly</c> there, which denies Head — so the
        /// head-pose layers play in the Unity editor preview but are silently
        /// inert in the VRChat client (issue #25, observed live).
        ///
        /// When the first-layer mask denies Head, swap in a combined mask:
        /// a copy of the original's humanoid parts + transforms with Head
        /// enabled, stored as a sub-asset of the gesture controller. The hand
        /// layers keep their own hand masks, so the AND still restricts them
        /// to hands. The descriptor's Gesture slot mask is pointed at the
        /// same combined mask (the SDK inspector mirrors the first sub-layer
        /// there, and VRChat reads it too). No-op when Head is already
        /// allowed or there is no first-layer mask (nothing denied).
        /// </summary>
        internal static void EnsureGestureMaskAllowsHead(VRCAvatarDescriptor avatar, AnimatorController gesture)
        {
            var layers = gesture.layers;
            if (layers.Length == 0)
            {
                return;
            }
            var original = layers[0].avatarMask;
            if (original == null || original.GetHumanoidBodyPartActive(AvatarMaskBodyPart.Head))
            {
                return;
            }

            var combined = AssetDatabase.LoadAllAssetsAtPath(AssetDatabase.GetAssetPath(gesture))
                .OfType<AvatarMask>()
                .FirstOrDefault(m => m.name == GestureMaskName);
            if (combined == null)
            {
                combined = new AvatarMask { name = GestureMaskName };
                for (var part = AvatarMaskBodyPart.Root; part < AvatarMaskBodyPart.LastBodyPart; part++)
                {
                    combined.SetHumanoidBodyPartActive(
                        part,
                        part == AvatarMaskBodyPart.Head || original.GetHumanoidBodyPartActive(part));
                }
                combined.transformCount = original.transformCount;
                for (var i = 0; i < original.transformCount; i++)
                {
                    combined.SetTransformPath(i, original.GetTransformPath(i));
                    combined.SetTransformActive(i, original.GetTransformActive(i));
                }
                AssetDatabase.AddObjectToAsset(combined, gesture);
            }

            layers[0].avatarMask = combined;
            gesture.layers = layers;
            SetDescriptorGestureMask(avatar, combined);
        }

        /// <summary>
        /// Inverse of <see cref="EnsureGestureMaskAllowsHead"/> for the
        /// wizard's Remove path: if the first-layer mask is the wizard's
        /// combined mask, swap the stock <c>vrc_HandsOnly</c> back in (found
        /// by asset name, like the stock hands controller; null if the
        /// project doesn't have it) and delete the combined sub-asset.
        /// Restores by <em>equivalence</em>, not identity — an avatar whose
        /// original first-layer mask was something other than vrc_HandsOnly
        /// gets the stock mask instead (documented caveat, issue #25).
        /// </summary>
        internal static void RestoreGestureMask(VRCAvatarDescriptor avatar, AnimatorController gesture)
        {
            var layers = gesture.layers;
            if (layers.Length == 0 || layers[0].avatarMask == null ||
                layers[0].avatarMask.name != GestureMaskName)
            {
                return;
            }
            var ours = layers[0].avatarMask;

            AvatarMask stock = null;
            foreach (var guid in AssetDatabase.FindAssets("vrc_HandsOnly t:AvatarMask"))
            {
                var path = AssetDatabase.GUIDToAssetPath(guid);
                if (Path.GetFileNameWithoutExtension(path) == "vrc_HandsOnly")
                {
                    stock = AssetDatabase.LoadAssetAtPath<AvatarMask>(path);
                    break;
                }
            }

            layers[0].avatarMask = stock;
            gesture.layers = layers;
            SetDescriptorGestureMask(avatar, stock);

            AssetDatabase.RemoveObjectFromAsset(ours);
            Object.DestroyImmediate(ours, true);
        }

        static void SetDescriptorGestureMask(VRCAvatarDescriptor avatar, AvatarMask mask)
        {
            var layers = avatar.baseAnimationLayers;
            var gestureIndex = System.Array.FindIndex(layers, l => l.type == VRCAvatarDescriptor.AnimLayerType.Gesture);
            if (gestureIndex < 0)
            {
                return;
            }
            Undo.RecordObject(avatar, "Update VRChatCameraOSC Gesture layer mask");
            layers[gestureIndex].mask = mask;
            avatar.baseAnimationLayers = layers;
            EditorUtility.SetDirty(avatar);
        }

        /// <summary>
        /// Strips everything a pre-issue-#21 wizard added under the retired
        /// custom10 names (<see cref="OscParameterSpec.LegacyNames"/>):
        /// expression parameters, FX layers, and the old Gesture/FX head-pose
        /// layers. Run on both Apply (migrate in place) and Remove (clean old
        /// avatars). Returns how many legacy expression parameters went away.
        /// </summary>
        static int RemoveLegacyCustom10(VRCAvatarDescriptor avatar)
        {
            var removed = 0;
            if (avatar.expressionParameters != null)
            {
                removed = VrcExpressionParametersMerger.RemoveByName(
                    avatar.expressionParameters, OscParameterSpec.LegacyNames);
            }

            var fx = TryGetFxController(avatar);
            var gesture = TryGetGestureController(avatar);
            foreach (var name in OscParameterSpec.LegacyNames)
            {
                if (fx != null && OscAnimatorLayerBuilder.RemoveLayer(fx, name))
                {
                    EditorUtility.SetDirty(fx);
                }
                if (gesture != null && OscAnimatorLayerBuilder.RemoveLayer(gesture, name))
                {
                    EditorUtility.SetDirty(gesture);
                }
            }
            return removed;
        }

        static VRCExpressionParameters EnsureExpressionParameters(VRCAvatarDescriptor avatar)
        {
            if (avatar.expressionParameters != null)
            {
                return avatar.expressionParameters;
            }
            var asset = ScriptableObject.CreateInstance<VRCExpressionParameters>();
            asset.parameters = new VRCExpressionParameters.Parameter[0];
            var path = AssetPathNextToAvatar(avatar, "ExpressionParameters.asset");
            AssetDatabase.CreateAsset(asset, path);
            Undo.RecordObject(avatar, "Assign VRC Expression Parameters");
            avatar.expressionParameters = asset;
            EditorUtility.SetDirty(avatar);
            return asset;
        }

        /// <summary>Read-only: the avatar's FX controller if it already has one, else null.</summary>
        static AnimatorController TryGetFxController(VRCAvatarDescriptor avatar)
        {
            var layers = avatar.baseAnimationLayers;
            var fxIndex = System.Array.FindIndex(layers, l => l.type == VRCAvatarDescriptor.AnimLayerType.FX);
            return fxIndex >= 0 ? layers[fxIndex].animatorController as AnimatorController : null;
        }

        static AnimatorController EnsureFxController(VRCAvatarDescriptor avatar)
        {
            var existing = TryGetFxController(avatar);
            if (existing != null)
            {
                return existing;
            }

            var layers = avatar.baseAnimationLayers;
            var fxIndex = System.Array.FindIndex(layers, l => l.type == VRCAvatarDescriptor.AnimLayerType.FX);
            var path = AssetPathNextToAvatar(avatar, "FX.controller");
            var controller = AnimatorController.CreateAnimatorControllerAtPath(path);

            Undo.RecordObject(avatar, "Assign VRChatCameraOSC FX Controller");
            if (fxIndex >= 0)
            {
                layers[fxIndex].animatorController = controller;
                layers[fxIndex].isDefault = false;
            }
            avatar.baseAnimationLayers = layers;
            EditorUtility.SetDirty(avatar);
            return controller;
        }

        /// <summary>Read-only: the avatar's Gesture controller if it already
        /// has a custom (non-default) one assigned, else null. Used only for
        /// migration cleanup — head pose lived on Gesture in a previous
        /// version (issue #25).</summary>
        internal static AnimatorController TryGetGestureController(VRCAvatarDescriptor avatar)
        {
            var layers = avatar.baseAnimationLayers;
            var gestureIndex = System.Array.FindIndex(layers, l => l.type == VRCAvatarDescriptor.AnimLayerType.Gesture);
            if (gestureIndex < 0)
            {
                return null;
            }
            var layer = layers[gestureIndex];
            return !layer.isDefault ? layer.animatorController as AnimatorController : null;
        }

        /// <summary>Read-only: the avatar's Additive controller if it already
        /// has a custom (non-default) one assigned, else null.</summary>
        internal static AnimatorController TryGetAdditiveController(VRCAvatarDescriptor avatar)
        {
            var layers = avatar.baseAnimationLayers;
            var index = System.Array.FindIndex(layers, l => l.type == VRCAvatarDescriptor.AnimLayerType.Additive);
            if (index < 0)
            {
                return null;
            }
            var layer = layers[index];
            return !layer.isDefault ? layer.animatorController as AnimatorController : null;
        }

        /// <summary>Asset name of the VRC SDK's stock default-hand-gestures
        /// controller (from the SDK's "AV3 Demo Assets" sample), used as the
        /// starting point for a new Gesture controller so existing default
        /// hand gestures (fist, point, etc.) aren't lost when this wizard has
        /// to create one — see <see cref="EnsureGestureController"/>.</summary>
        internal const string DefaultHandsControllerAssetName = "vrc_AvatarV3HandsLayer";

        /// <summary>
        /// Head-pose layers (issue #16 head-pose fix) go on the Gesture
        /// playable layer, not FX — see <see cref="OscAnimatorLayerBuilder.AddHeadPoseLayer"/>
        /// for why. If the avatar still has VRChat's default Gesture layer
        /// (<c>isDefault == true</c>, no controller of its own), this creates
        /// a real one by copying the VRC SDK's stock hand-gestures controller
        /// (<see cref="DefaultHandsControllerAssetName"/>) so the avatar's
        /// existing default hand gestures survive the switch away from
        /// "default". <paramref name="copiedDefaultHandsController"/> is
        /// false if that stock asset couldn't be found (e.g. the SDK's "AV3
        /// Demo Assets" sample was never imported into this project) — in
        /// that case a blank controller is created instead and the caller
        /// should warn the user that default hand gestures need manual setup.
        /// </summary>
        internal static AnimatorController EnsureGestureController(VRCAvatarDescriptor avatar, out bool copiedDefaultHandsController)
        {
            var existing = TryGetGestureController(avatar);
            if (existing != null)
            {
                copiedDefaultHandsController = true;
                return existing;
            }

            var layers = avatar.baseAnimationLayers;
            var gestureIndex = System.Array.FindIndex(layers, l => l.type == VRCAvatarDescriptor.AnimLayerType.Gesture);
            var path = AssetPathNextToAvatar(avatar, "Gesture.controller");

            AnimatorController controller;
            var defaultHandsPath = FindDefaultHandsControllerAssetPath();
            if (defaultHandsPath != null)
            {
                AssetDatabase.CopyAsset(defaultHandsPath, path);
                controller = AssetDatabase.LoadAssetAtPath<AnimatorController>(path);
                copiedDefaultHandsController = true;
            }
            else
            {
                controller = AnimatorController.CreateAnimatorControllerAtPath(path);
                copiedDefaultHandsController = false;
            }

            Undo.RecordObject(avatar, "Assign VRChatCameraOSC Gesture Controller");
            if (gestureIndex >= 0)
            {
                layers[gestureIndex].animatorController = controller;
                layers[gestureIndex].isDefault = false;
                layers[gestureIndex].isEnabled = true;
                // Mirrors what the VRC SDK's own inspector does when you
                // manually assign a controller to a non-default layer slot
                // (AvatarDescriptorEditor3AnimLayerInit.SetLayerMaskFromController):
                // the layer's mask follows its first sub-layer's AvatarMask,
                // so a copied vrc_AvatarV3HandsLayer keeps animating hands
                // only, not the whole body.
                layers[gestureIndex].mask = controller.layers.Length > 0 ? controller.layers[0].avatarMask : null;
            }
            avatar.baseAnimationLayers = layers;
            EditorUtility.SetDirty(avatar);
            return controller;
        }

        /// <summary>
        /// Finds the VRC SDK's stock default-hand-gestures controller by
        /// asset name, wherever it happens to live in the project (normally
        /// under the SDK package's Samples — an *optional* Package Manager
        /// sample import, so it may not be present in every project).
        /// </summary>
        internal static string FindDefaultHandsControllerAssetPath()
        {
            foreach (var guid in AssetDatabase.FindAssets($"{DefaultHandsControllerAssetName} t:AnimatorController"))
            {
                var candidatePath = AssetDatabase.GUIDToAssetPath(guid);
                if (Path.GetFileNameWithoutExtension(candidatePath) == DefaultHandsControllerAssetName)
                {
                    return candidatePath;
                }
            }
            return null;
        }

        static string AssetPathNextToAvatar(VRCAvatarDescriptor avatar, string fileName)
        {
            var scenePath = avatar.gameObject.scene.IsValid() ? "Assets" : null;
            var prefabPath = PrefabUtility.GetPrefabAssetPathOfNearestInstanceRoot(avatar.gameObject);
            var dir = !string.IsNullOrEmpty(prefabPath)
                ? Path.GetDirectoryName(prefabPath)
                : (scenePath ?? "Assets");
            var path = $"{dir}/VRChatCameraOSC_{avatar.gameObject.name}_{fileName}".Replace("\\", "/");
            return AssetDatabase.GenerateUniqueAssetPath(path);
        }
    }
}
