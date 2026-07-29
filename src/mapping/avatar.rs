//! Avatar-aware VRCFT parameter gating + binary encoding (issue #18 phases
//! 2+3).
//!
//! VRChat writes a per-avatar OSC config JSON (avatar id/name + every
//! parameter with its input/output address and type) under
//! `.../AppData/LocalLow/VRChat/VRChat/OSC/usr_*/Avatars/<avatarid>.json`,
//! and announces avatar switches with an `/avatar/change` OSC message on its
//! output port (9001). VRCFaceTracking reads that JSON and sends **only** the
//! parameters the avatar declares, at their exact declared addresses,
//! expanding each conceptual value into up to three wire forms (float, bool,
//! binary bits). This module ports those semantics so the `vrcft` mapping
//! profile behaves the same way.
//!
//! # Sources (CLAUDE.md rule 3 — ported from upstream GitHub, not memory)
//!
//! - **Suffix matching + typed relevancy:**
//!   [`BaseParam.cs`](https://github.com/benaclejames/VRCFaceTracking/blob/master/VRCFaceTracking.Core/OSC/DataTypes/BaseParameter.cs)
//!   — regex `(?<!(v\d+))(/<name>)$|^(<name>)$` against the declared address,
//!   with the declared type required to match, and the **declared address
//!   used verbatim** for sending (arbitrary user prefixes work).
//! - **Binary bit encoding:**
//!   [`BinaryBaseParameter.cs`](https://github.com/benaclejames/VRCFaceTracking/blob/master/VRCFaceTracking.Core/OSC/DataTypes/BinaryBaseParameter.cs)
//!   — bool params `<Name>1/2/4/8...` (powers of two) plus optional
//!   `<Name>Negative`; see [`binary_bit`] for the exact rules.
//! - **Float+bool+binary expansion per name:** `EParam` in
//!   [`ParamContainers.cs`](https://github.com/benaclejames/VRCFaceTracking/blob/master/VRCFaceTracking.Core/Params/DataTypes/ParamContainers.cs)
//!   — note the bool form is `value < 0.5` (`minBoolThreshold`) upstream,
//!   ported as-is.
//! - **Config file model + `input != null` filter:**
//!   `AvatarConfigFileParser.cs` / `AvatarConfigFile.cs` (types are the
//!   strings `"Bool"`/`"Float"`/`"Int"`, `OSCUtils.cs`).
//!
//! # Fallback
//!
//! When no avatar config is available (nothing discovered, or an
//! `/avatar/change` to an avatar we can't find a JSON for), the
//! [`super::unified::UnifiedMapper`] keeps its phase-1 behavior: the full
//! float set sent blind under each configured prefix.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::osc::oscquery::OscQueryClient;
use crate::osc::{AvatarChangeListener, OscParam};

// ---------------------------------------------------------------------------
// VRChat avatar OSC config JSON model
// ---------------------------------------------------------------------------

/// One parsed `OSC/usr_*/Avatars/<avatarid>.json` file.
#[derive(Debug, Clone, Deserialize)]
pub struct AvatarOscConfig {
    pub id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub parameters: Vec<AvatarParamDecl>,
}

/// One declared avatar parameter. Only `input` (the writable direction)
/// matters for sending — output-only parameters (velocity, gesture feedback,
/// ...) are ignored, matching VRCFT's `param.input != null` filter.
#[derive(Debug, Clone, Deserialize)]
pub struct AvatarParamDecl {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub input: Option<AvatarIoDecl>,
    #[serde(default)]
    pub output: Option<AvatarIoDecl>,
}

/// The address/type pair of one parameter direction.
#[derive(Debug, Clone, Deserialize)]
pub struct AvatarIoDecl {
    pub address: String,
    /// `"Float"`, `"Bool"`, or `"Int"` (VRChat's config type strings).
    #[serde(rename = "type")]
    pub osc_type: String,
}

impl AvatarOscConfig {
    /// Parse one avatar config JSON document.
    pub fn from_json(text: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(text)?)
    }

    /// Read and parse one avatar config file.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("reading {}: {e}", path.display()))?;
        Self::from_json(&text).map_err(|e| anyhow::anyhow!("parsing {}: {e}", path.display()))
    }
}

// ---------------------------------------------------------------------------
// Config discovery
// ---------------------------------------------------------------------------

/// The directories to search for avatar configs: the explicit
/// `[mapping] avatar_config_dir` override when set, else the per-platform
/// VRChat defaults ([`default_osc_dirs`]).
pub fn config_dirs(override_dir: Option<&str>) -> Vec<PathBuf> {
    match override_dir {
        Some(dir) if !dir.is_empty() => vec![PathBuf::from(dir)],
        _ => default_osc_dirs(),
    }
}

/// Platform-default locations of VRChat's `OSC` directory:
/// `%USERPROFILE%\AppData\LocalLow\VRChat\VRChat\OSC` on Windows, and the
/// Steam/Proton prefix (`compatdata/438100`) equivalent on Linux. macOS has
/// no native VRChat client, so nothing is returned there — use the
/// `avatar_config_dir` override for exotic setups.
pub fn default_osc_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if cfg!(windows) {
        if let Some(profile) = std::env::var_os("USERPROFILE") {
            dirs.push(
                PathBuf::from(profile)
                    .join("AppData")
                    .join("LocalLow")
                    .join("VRChat")
                    .join("VRChat")
                    .join("OSC"),
            );
        }
    } else if cfg!(target_os = "linux") {
        if let Some(home) = std::env::var_os("HOME") {
            dirs.push(PathBuf::from(home).join(
                ".steam/steam/steamapps/compatdata/438100/pfx/drive_c/users/steamuser/AppData/LocalLow/VRChat/VRChat/OSC",
            ));
        }
    }
    dirs
}

/// Every candidate avatar json under `dir`. Layouts accepted:
/// `dir/<user>/Avatars/*.json` (the real `OSC/usr_*/...` layout — upstream
/// scans every subdirectory containing an `Avatars` folder, not only
/// `usr_*`), plus `dir/Avatars/*.json` and `dir/*.json` so an explicit
/// override can point at any level of the tree.
fn avatar_json_files(dir: &Path) -> Vec<PathBuf> {
    fn push_jsons(d: &Path, files: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(d) else {
            return;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_file() && p.extension().is_some_and(|x| x == "json") {
                files.push(p);
            }
        }
    }
    let mut files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for e in entries.flatten() {
            let sub = e.path().join("Avatars");
            if sub.is_dir() {
                push_jsons(&sub, &mut files);
            }
        }
    }
    let direct = dir.join("Avatars");
    if direct.is_dir() {
        push_jsons(&direct, &mut files);
    }
    push_jsons(dir, &mut files);
    files
}

/// Find the config whose `id` matches, scanning `dirs`. Malformed files are
/// skipped (mirrors upstream's tolerant scan).
pub fn find_config_by_id(dirs: &[PathBuf], id: &str) -> Option<AvatarOscConfig> {
    for dir in dirs {
        for file in avatar_json_files(dir) {
            if let Ok(cfg) = AvatarOscConfig::load(&file) {
                if cfg.id == id {
                    return Some(cfg);
                }
            }
        }
    }
    None
}

/// Best-effort startup pick before any `/avatar/change` arrives: the most
/// recently modified parseable avatar json — VRChat rewrites the file when
/// an avatar is loaded, so the newest one is almost always the avatar
/// currently worn.
pub fn find_most_recent_config(dirs: &[PathBuf]) -> Option<AvatarOscConfig> {
    let mut candidates: Vec<(std::time::SystemTime, PathBuf)> = Vec::new();
    for dir in dirs {
        for file in avatar_json_files(dir) {
            if let Ok(mtime) = std::fs::metadata(&file).and_then(|m| m.modified()) {
                candidates.push((mtime, file));
            }
        }
    }
    candidates.sort_by_key(|(mtime, _)| std::cmp::Reverse(*mtime)); // newest first
    candidates
        .into_iter()
        .find_map(|(_, file)| AvatarOscConfig::load(&file).ok())
}

// ---------------------------------------------------------------------------
// VRCFT name matching (BaseParam / BinaryBaseParameter regex ports)
// ---------------------------------------------------------------------------

/// `BaseParam` address matching: the declared address must end with
/// `/<name>` — with the text before that slash **not** ending in `v<digits>`
/// (upstream's `(?<!(v\d+))` lookbehind) — or equal `<name>` outright.
pub fn address_matches(address: &str, name: &str) -> bool {
    if address == name {
        return true;
    }
    let Some(prefix) = address.strip_suffix(name) else {
        return false;
    };
    let Some(prefix) = prefix.strip_suffix('/') else {
        return false;
    };
    !ends_with_v_digits(prefix)
}

/// The `(?<!(v\d+))` negative lookbehind from VRCFT's parameter regexes.
fn ends_with_v_digits(s: &str) -> bool {
    let no_digits = s.trim_end_matches(|c: char| c.is_ascii_digit());
    no_digits.len() < s.len() && no_digits.ends_with('v')
}

/// `BinaryBaseParameter` bit-name matching (regex
/// `(?<!(v\d+))/<name>\d+$|^<name>\d+$` + the trailing-digits parse):
/// returns the declared bit index — e.g. `Some(4)` for `.../<name>4`.
pub fn binary_address_index(address: &str, name: &str) -> Option<u32> {
    let head = address.trim_end_matches(|c: char| c.is_ascii_digit());
    if head.len() == address.len() {
        return None; // no trailing digits
    }
    let index: u32 = address[head.len()..].parse().ok()?;
    if head == name {
        return Some(index);
    }
    let prefix = head.strip_suffix(name)?;
    let prefix = prefix.strip_suffix('/')?;
    (!ends_with_v_digits(prefix)).then_some(index)
}

/// `GetBinarySteps` port: a declared index participates only if it is a
/// power of two, and its step count is the bit position (1→0, 2→1, 4→2,
/// 8→3, ...). 0 and non-powers return `None`.
pub fn binary_steps(index: u32) -> Option<u32> {
    (index != 0 && index.is_power_of_two()).then_some(index.trailing_zeros())
}

/// One bit of the binary encoding — a faithful port of
/// `BinaryBaseParameter.ProcessBinary`:
///
/// - no declared `<Name>Negative` → negative values read as all-bits-clear;
/// - `|v| > 0.99999` → every bit set (upstream's platform-independent fix
///   for `floor(1.0 × max_int)` overflowing the bit range);
/// - otherwise bit = `(⌊|v| · max_int⌋ >> shift) & 1`, where `max_int` is
///   `2^(number of declared bit params)`.
pub fn binary_bit(value: f32, max_int: i32, shift: u32, has_negative: bool) -> bool {
    if !has_negative && value < 0.0 {
        return false;
    }
    let v = value.abs();
    if v > 0.99999 {
        return true;
    }
    let big = (v * max_int as f32) as i32;
    (big >> shift) & 1 == 1
}

// ---------------------------------------------------------------------------
// Resolved per-avatar parameter set
// ---------------------------------------------------------------------------

/// The declared binary expansion of one name: bit addresses with their
/// shifts, plus the optional `<Name>Negative` bool. Upstream sends the
/// Negative param even when no bit params are declared, and vice versa.
#[derive(Debug, Clone)]
struct BinaryTarget {
    /// `(declared address, bit shift)`, one per declared power-of-two index.
    bits: Vec<(String, u32)>,
    /// `2^bits.len()` — `_maxPossibleBinaryInt`.
    max_int: i32,
    /// Declared `<Name>Negative` address, if any.
    negative: Option<String>,
}

impl BinaryTarget {
    fn resolve(name: &str, declared_bools: &[&str]) -> Option<Self> {
        // (address, declared index, shift); first declaration of an index wins.
        let mut bits: Vec<(String, u32, u32)> = Vec::new();
        for addr in declared_bools {
            let Some(index) = binary_address_index(addr, name) else {
                continue;
            };
            let Some(shift) = binary_steps(index) else {
                continue;
            };
            if !bits.iter().any(|(_, i, _)| *i == index) {
                bits.push((addr.to_string(), index, shift));
            }
        }
        let negative_name = format!("{name}Negative");
        let negative = declared_bools
            .iter()
            .find(|a| address_matches(a, &negative_name))
            .map(|a| a.to_string());
        if bits.is_empty() && negative.is_none() {
            return None;
        }
        bits.sort_by_key(|(_, i, _)| *i);
        let max_int = 1i32 << bits.len().min(30) as u32;
        Some(Self {
            bits: bits.into_iter().map(|(a, _, s)| (a, s)).collect(),
            max_int,
            negative,
        })
    }

    fn emit(&self, value: f32, out: &mut Vec<OscParam>) {
        let has_negative = self.negative.is_some();
        for (addr, shift) in &self.bits {
            out.push(OscParam::bool(
                addr.clone(),
                binary_bit(value, self.max_int, *shift, has_negative),
            ));
        }
        if let Some(addr) = &self.negative {
            out.push(OscParam::bool(addr.clone(), value < 0.0));
        }
    }
}

/// Everything the avatar declares for one canonical name (`EParam` port:
/// float, thresholded bool, and binary forms — each present only if the
/// avatar declares it).
#[derive(Debug, Clone, Default)]
struct ResolvedParam {
    float_addr: Option<String>,
    bool_addr: Option<String>,
    binary: Option<BinaryTarget>,
}

/// VRCFT `EParam` `minBoolThreshold` — the plain declared-bool form of a
/// float value is sent as `value < 0.5`, ported verbatim (yes, `<`: that is
/// what upstream ships in `ParamContainers.cs`).
const BOOL_THRESHOLD: f32 = 0.5;

/// The active avatar's declared VRCFT parameters, resolved once per avatar
/// change: canonical name → the exact declared wire addresses/forms.
#[derive(Debug, Clone, Default)]
pub struct AvatarParamSet {
    pub avatar_id: String,
    pub avatar_name: String,
    entries: HashMap<String, ResolvedParam>,
    /// Status bools (`EyeTrackingActive`, ...) → declared address.
    status: HashMap<String, String>,
}

impl AvatarParamSet {
    /// Resolve `float_names` (each expandable to float/bool/binary forms)
    /// and `status_names` (plain bools) against the avatar's declared input
    /// parameters. Names nothing was declared for get no entry — they are
    /// simply never sent.
    pub fn resolve(cfg: &AvatarOscConfig, float_names: &[&str], status_names: &[&str]) -> Self {
        let mut float_addrs = Vec::new();
        let mut bool_addrs = Vec::new();
        for p in &cfg.parameters {
            let Some(io) = &p.input else { continue };
            match io.osc_type.as_str() {
                "Float" => float_addrs.push(io.address.as_str()),
                "Bool" => bool_addrs.push(io.address.as_str()),
                _ => {} // Int declarations have no VRCFT expression form here
            }
        }

        let mut entries = HashMap::new();
        for &name in float_names {
            let resolved = ResolvedParam {
                float_addr: float_addrs
                    .iter()
                    .find(|a| address_matches(a, name))
                    .map(|a| a.to_string()),
                bool_addr: bool_addrs
                    .iter()
                    .find(|a| address_matches(a, name))
                    .map(|a| a.to_string()),
                binary: BinaryTarget::resolve(name, &bool_addrs),
            };
            if resolved.float_addr.is_some()
                || resolved.bool_addr.is_some()
                || resolved.binary.is_some()
            {
                entries.insert(name.to_string(), resolved);
            }
        }

        let mut status = HashMap::new();
        for &name in status_names {
            if let Some(a) = bool_addrs.iter().find(|a| address_matches(a, name)) {
                status.insert(name.to_string(), a.to_string());
            }
        }

        Self {
            avatar_id: cfg.id.clone(),
            avatar_name: cfg.name.clone(),
            entries,
            status,
        }
    }

    /// Number of canonical names the avatar declares at least one form of.
    pub fn len(&self) -> usize {
        self.entries.len() + self.status.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Emit `name = value` in every form the avatar declares: float
    /// verbatim, plain bool as `value < 0.5`, binary bits + Negative.
    /// Undeclared names emit nothing (the gating).
    pub fn emit_value(&self, name: &str, value: f32, out: &mut Vec<OscParam>) {
        let Some(e) = self.entries.get(name) else {
            return;
        };
        if let Some(addr) = &e.float_addr {
            out.push(OscParam::float(addr.clone(), value));
        }
        if let Some(addr) = &e.bool_addr {
            out.push(OscParam::bool(addr.clone(), value < BOOL_THRESHOLD));
        }
        if let Some(b) = &e.binary {
            b.emit(value, out);
        }
    }

    /// Emit a status bool (`EyeTrackingActive`, ...) if declared.
    pub fn emit_status(&self, name: &str, value: bool, out: &mut Vec<OscParam>) {
        if let Some(addr) = self.status.get(name) {
            out.push(OscParam::bool(addr.clone(), value));
        }
    }
}

// ---------------------------------------------------------------------------
// Avatar watcher: /avatar/change listener + config lookup
// ---------------------------------------------------------------------------

/// Tracks which avatar is worn: an optional nonblocking `/avatar/change`
/// listener plus the config-dir scan, with an optional OSCQuery fallback
/// (issue #18 phase 3b). Owned by the pipeline and polled once per step
/// (see `crate::osc::listen` module docs for why polling, not a thread —
/// only the single nonblocking recv is on the hot path; the filesystem and
/// the network are touched exclusively when the avatar actually changes).
///
/// Source priority: **local config file → OSCQuery fetch → `None`** (the
/// caller's blind-prefix fallback). The local file is exact (id + name +
/// declared params) and free; OSCQuery covers a VRChat on another machine.
#[derive(Debug)]
pub struct AvatarWatcher {
    listener: Option<AvatarChangeListener>,
    dirs: Vec<PathBuf>,
    current_id: Option<String>,
    oscquery: Option<OscQueryClient>,
}

impl AvatarWatcher {
    pub fn new(dirs: Vec<PathBuf>, listener: Option<AvatarChangeListener>) -> Self {
        Self {
            listener,
            dirs,
            current_id: None,
            oscquery: None,
        }
    }

    /// Attach an OSCQuery client, consulted whenever the local config scan
    /// comes up empty (issue #18 phase 3b).
    pub fn with_oscquery(mut self, client: OscQueryClient) -> Self {
        self.oscquery = Some(client);
        self
    }

    /// Startup best-effort pick: the most recently modified local avatar
    /// config (see [`find_most_recent_config`]), else an OSCQuery fetch
    /// (waiting briefly for mDNS discovery). `None` when neither source has
    /// anything — the mapper then stays in blind-prefix fallback mode.
    pub fn initial_config(&mut self) -> Option<AvatarOscConfig> {
        if let Some(cfg) = find_most_recent_config(&self.dirs) {
            self.current_id = Some(cfg.id.clone());
            return Some(cfg);
        }
        let cfg = self.oscquery.as_mut()?.fetch_initial()?;
        if !cfg.id.is_empty() {
            self.current_id = Some(cfg.id.clone());
        }
        Some(cfg)
    }

    /// Nonblocking poll (an OSCQuery fetch only happens when the avatar
    /// actually changed and no local config matches — a rare, human-scale
    /// event, never the frame hot path). `None` = no avatar change;
    /// `Some(Some(cfg))` = a new avatar whose config was found;
    /// `Some(None)` = a new avatar with no discoverable config (the caller
    /// falls back to blind mode).
    pub fn poll(&mut self) -> Option<Option<AvatarOscConfig>> {
        let id = self.listener.as_mut()?.poll_avatar_change()?;
        if self.current_id.as_deref() == Some(id.as_str()) {
            return None; // same avatar re-announced — keep the active set
        }
        let cfg = find_config_by_id(&self.dirs, &id)
            .or_else(|| self.oscquery.as_mut().and_then(|c| c.fetch(&id)));
        self.current_id = Some(id);
        Some(cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::osc::OscValue;

    fn fixture_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/avatar_configs")
    }

    // ---- JSON parsing (fixture files) ----

    #[test]
    fn parses_vrchat_avatar_config_fixture() {
        let path = fixture_dir().join("usr_test-user/Avatars/avtr_float.json");
        let cfg = AvatarOscConfig::load(&path).unwrap();
        assert_eq!(cfg.id, "avtr_00000000-0000-4000-8000-00000000f10a");
        assert_eq!(cfg.name, "Fixture Float FT Avatar");
        assert_eq!(cfg.parameters.len(), 8);
        let jaw = &cfg.parameters[0];
        assert_eq!(jaw.name, "FT/v2/JawOpen");
        let input = jaw.input.as_ref().unwrap();
        assert_eq!(input.address, "/avatar/parameters/FT/v2/JawOpen");
        assert_eq!(input.osc_type, "Float");
        // Output-only params parse with input == None.
        let velocity = cfg
            .parameters
            .iter()
            .find(|p| p.name == "VelocityX")
            .unwrap();
        assert!(velocity.input.is_none());
        assert!(velocity.output.is_some());
    }

    #[test]
    fn malformed_json_is_an_error_but_discovery_skips_it() {
        let path = fixture_dir().join("usr_test-user/Avatars/malformed.json");
        assert!(AvatarOscConfig::load(&path).is_err());
        // Discovery still finds the valid configs around it.
        let found = find_config_by_id(
            &[fixture_dir()],
            "avtr_00000000-0000-4000-8000-00000000b14a",
        )
        .expect("valid config found despite malformed sibling");
        assert_eq!(found.name, "Fixture Binary FT Avatar");
    }

    #[test]
    fn config_dirs_prefers_explicit_override() {
        let dirs = config_dirs(Some("/tmp/custom-osc"));
        assert_eq!(dirs, vec![PathBuf::from("/tmp/custom-osc")]);
        // Empty override string falls back to the platform defaults.
        assert_eq!(config_dirs(Some("")), default_osc_dirs());
        assert_eq!(config_dirs(None), default_osc_dirs());
    }

    // ---- Suffix matching (BaseParam regex port) ----

    #[test]
    fn address_suffix_matching_follows_vrcft_semantics() {
        // Bare and prefixed declarations match by suffix.
        assert!(address_matches(
            "/avatar/parameters/v2/JawOpen",
            "v2/JawOpen"
        ));
        assert!(address_matches(
            "/avatar/parameters/FT/v2/JawOpen",
            "v2/JawOpen"
        ));
        assert!(address_matches(
            "/avatar/parameters/My/Custom/Prefix/v2/JawOpen",
            "v2/JawOpen"
        ));
        // Whole-string equality branch (`^name$`).
        assert!(address_matches("v2/JawOpen", "v2/JawOpen"));
        // Suffix must be a whole path segment...
        assert!(!address_matches(
            "/avatar/parameters/FTv2/JawOpen",
            "v2/JawOpen"
        ));
        // ...and exact to the end.
        assert!(!address_matches(
            "/avatar/parameters/v2/JawOpenLeft",
            "v2/JawOpen"
        ));
        assert!(!address_matches(
            "/avatar/parameters/v2/JawOpen1",
            "v2/JawOpen"
        ));
        // The (?<!(v\d+)) lookbehind: a v<digits> segment before the name
        // rejects the match.
        assert!(!address_matches(
            "/avatar/parameters/v1/v2/JawOpen",
            "v2/JawOpen"
        ));
        // Status bools (no v2/ prefix in the name).
        assert!(address_matches(
            "/avatar/parameters/EyeTrackingActive",
            "EyeTrackingActive"
        ));
        assert!(address_matches(
            "/avatar/parameters/FT/EyeTrackingActive",
            "EyeTrackingActive"
        ));
        // Multi-segment names (head params) match as a unit.
        assert!(address_matches(
            "/avatar/parameters/FT/v2/Head/Yaw",
            "v2/Head/Yaw"
        ));
    }

    // ---- Binary name matching ----

    #[test]
    fn binary_index_parses_trailing_digits_with_vrcft_regex() {
        assert_eq!(
            binary_address_index("/avatar/parameters/FT/v2/JawOpen4", "v2/JawOpen"),
            Some(4)
        );
        assert_eq!(
            binary_address_index("/avatar/parameters/v2/JawOpen16", "v2/JawOpen"),
            Some(16)
        );
        // Equality branch.
        assert_eq!(binary_address_index("v2/JawOpen8", "v2/JawOpen"), Some(8));
        // No digits → not a bit param.
        assert_eq!(
            binary_address_index("/avatar/parameters/FT/v2/JawOpen", "v2/JawOpen"),
            None
        );
        // Different name.
        assert_eq!(
            binary_address_index("/avatar/parameters/FT/v2/JawForward4", "v2/JawOpen"),
            None
        );
        // Lookbehind applies here too.
        assert_eq!(
            binary_address_index("/avatar/parameters/v9/v2/JawOpen4", "v2/JawOpen"),
            None
        );
    }

    #[test]
    fn binary_steps_accepts_only_powers_of_two() {
        assert_eq!(binary_steps(1), Some(0));
        assert_eq!(binary_steps(2), Some(1));
        assert_eq!(binary_steps(4), Some(2));
        assert_eq!(binary_steps(8), Some(3));
        assert_eq!(binary_steps(16), Some(4));
        assert_eq!(binary_steps(0), None);
        assert_eq!(binary_steps(3), None);
        assert_eq!(binary_steps(6), None);
        assert_eq!(binary_steps(12), None);
    }

    // ---- Binary bit encoding (ProcessBinary port) ----

    /// Collect the bit pattern for `value` under `num_bits` declared bits
    /// (shifts 0..num_bits), LSB first.
    fn bits_of(value: f32, num_bits: u32, has_negative: bool) -> Vec<bool> {
        let max_int = 1i32 << num_bits;
        (0..num_bits)
            .map(|shift| binary_bit(value, max_int, shift, has_negative))
            .collect()
    }

    fn as_int(bits: &[bool]) -> i32 {
        bits.iter().enumerate().map(|(i, &b)| (b as i32) << i).sum()
    }

    // Exhaustive: for 1/2/4/8-bit configs, a value in the middle of each
    // quantization step must encode exactly that step index.
    #[test]
    fn binary_bits_quantize_exhaustively_for_1_2_4_8_bits() {
        for num_bits in [1u32, 2, 4, 8] {
            let max_int = 1i32 << num_bits;
            for step in 0..max_int {
                let v = (step as f32 + 0.5) / max_int as f32;
                let bits = bits_of(v, num_bits, false);
                assert_eq!(
                    as_int(&bits),
                    step,
                    "{num_bits}-bit encoding of {v} should be step {step}"
                );
            }
        }
    }

    // The v = 1.0 edge case: floor(1.0 * max_int) == max_int would overflow
    // every bit position to 0; upstream special-cases > 0.99999 to all-ones.
    #[test]
    fn binary_bits_saturate_to_all_ones_at_one() {
        for num_bits in [1u32, 2, 4, 8] {
            for v in [1.0f32, 0.999995, 1.5] {
                let bits = bits_of(v, num_bits, false);
                assert!(
                    bits.iter().all(|&b| b),
                    "{num_bits}-bit encoding of {v} should be all ones: {bits:?}"
                );
            }
        }
        // Just below the threshold: the normal floor path (max_int - 1).
        let bits = bits_of(0.99, 4, false);
        assert_eq!(as_int(&bits), 15);
    }

    #[test]
    fn binary_bits_without_negative_param_clamp_negatives_to_zero() {
        for num_bits in [1u32, 2, 4, 8] {
            let bits = bits_of(-0.7, num_bits, false);
            assert!(
                bits.iter().all(|&b| !b),
                "negative value without a Negative param must read 0"
            );
        }
    }

    #[test]
    fn binary_bits_with_negative_param_encode_magnitude() {
        // -0.7 with 4 bits: |v| = 0.7 → floor(0.7 * 16) = 11 = 0b1011.
        let bits = bits_of(-0.7, 4, true);
        assert_eq!(as_int(&bits), 11);
        // -1.0 saturates to all ones like +1.0.
        assert!(bits_of(-1.0, 4, true).iter().all(|&b| b));
    }

    // ---- Resolution + emission ----

    fn decl(name: &str, ty: &str) -> AvatarParamDecl {
        AvatarParamDecl {
            name: name.to_string(),
            input: Some(AvatarIoDecl {
                address: format!("/avatar/parameters/{name}"),
                osc_type: ty.to_string(),
            }),
            output: None,
        }
    }

    fn cfg_with(params: Vec<AvatarParamDecl>) -> AvatarOscConfig {
        AvatarOscConfig {
            id: "avtr_test".to_string(),
            name: "test".to_string(),
            parameters: params,
        }
    }

    #[test]
    fn resolve_gates_to_declared_names_and_exact_addresses() {
        let cfg = cfg_with(vec![
            decl("Odd/Prefix/v2/JawOpen", "Float"),
            decl("EyeTrackingActive", "Bool"),
        ]);
        let set = AvatarParamSet::resolve(
            &cfg,
            &["v2/JawOpen", "v2/MouthClosed"],
            &["EyeTrackingActive", "LipTrackingActive"],
        );
        assert_eq!(set.len(), 2);

        let mut out = Vec::new();
        set.emit_value("v2/JawOpen", 0.8, &mut out);
        set.emit_value("v2/MouthClosed", 0.8, &mut out); // undeclared → gated
        set.emit_status("EyeTrackingActive", true, &mut out);
        set.emit_status("LipTrackingActive", true, &mut out); // gated
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].name, "/avatar/parameters/Odd/Prefix/v2/JawOpen");
        assert_eq!(out[0].value, OscValue::Float(0.8));
        assert_eq!(out[1].name, "/avatar/parameters/EyeTrackingActive");
        assert_eq!(out[1].value, OscValue::Bool(true));
    }

    #[test]
    fn resolve_expands_declared_bool_and_binary_forms() {
        let cfg = cfg_with(vec![
            decl("FT/v2/JawOpen", "Bool"),
            decl("FT/v2/JawOpen1", "Bool"),
            decl("FT/v2/JawOpen2", "Bool"),
            decl("FT/v2/JawOpen4", "Bool"),
            decl("FT/v2/JawOpen3", "Bool"), // not a power of two → ignored
            decl("FT/v2/JawOpenNegative", "Bool"),
        ]);
        let set = AvatarParamSet::resolve(&cfg, &["v2/JawOpen"], &[]);

        // 3 valid bits → max_int = 8; 0.6 → floor(0.6*8) = 4 = 0b100.
        let mut out = Vec::new();
        set.emit_value("v2/JawOpen", 0.6, &mut out);
        let get = |addr: &str| -> bool {
            match out.iter().find(|p| p.name == addr) {
                Some(OscParam {
                    value: OscValue::Bool(b),
                    ..
                }) => *b,
                other => panic!("{addr}: expected bool, got {other:?}"),
            }
        };
        // Plain bool form: EParam semantics `value < 0.5` → false at 0.6.
        assert!(!get("/avatar/parameters/FT/v2/JawOpen"));
        assert!(!get("/avatar/parameters/FT/v2/JawOpen1"));
        assert!(!get("/avatar/parameters/FT/v2/JawOpen2"));
        assert!(get("/avatar/parameters/FT/v2/JawOpen4"));
        assert!(!get("/avatar/parameters/FT/v2/JawOpenNegative"));
        assert!(
            !out.iter().any(|p| p.name.ends_with("JawOpen3")),
            "non-power-of-two bit names must be ignored"
        );
        // 5 messages: plain bool + 3 bits + Negative. No floats.
        assert_eq!(out.len(), 5);
        assert!(out.iter().all(|p| matches!(p.value, OscValue::Bool(_))));
    }

    // The exact declaration shape the unity/ wizard's binary mode emits
    // (issue #29): un-prefixed `<name>1/2/4/8` Bool bits per face float,
    // plus `<name>Negative` for the signed head axes — including the
    // nested-slash `v2/Head/*` names. Pins the Rust↔wizard contract.
    #[test]
    fn resolve_matches_wizard_binary_declarations() {
        let cfg = cfg_with(vec![
            decl("v2/JawOpen1", "Bool"),
            decl("v2/JawOpen2", "Bool"),
            decl("v2/JawOpen4", "Bool"),
            decl("v2/JawOpen8", "Bool"),
            decl("v2/Head/Yaw1", "Bool"),
            decl("v2/Head/Yaw2", "Bool"),
            decl("v2/Head/Yaw4", "Bool"),
            decl("v2/Head/Yaw8", "Bool"),
            decl("v2/Head/YawNegative", "Bool"),
        ]);
        let set = AvatarParamSet::resolve(&cfg, &["v2/JawOpen", "v2/Head/Yaw"], &[]);
        assert_eq!(set.len(), 2);

        let mut out = Vec::new();
        // 4 bits → max_int = 16; 0.6 → floor(0.6*16) = 9 = 0b1001.
        set.emit_value("v2/JawOpen", 0.6, &mut out);
        // Signed axis: |-0.5| → floor(0.5*16) = 8 = 0b1000, Negative true.
        set.emit_value("v2/Head/Yaw", -0.5, &mut out);
        let get = |addr: &str| -> bool {
            match out.iter().find(|p| p.name == addr) {
                Some(OscParam {
                    value: OscValue::Bool(b),
                    ..
                }) => *b,
                other => panic!("{addr}: expected bool, got {other:?}"),
            }
        };
        assert!(get("/avatar/parameters/v2/JawOpen1"));
        assert!(!get("/avatar/parameters/v2/JawOpen2"));
        assert!(!get("/avatar/parameters/v2/JawOpen4"));
        assert!(get("/avatar/parameters/v2/JawOpen8"));
        assert!(!get("/avatar/parameters/v2/Head/Yaw1"));
        assert!(!get("/avatar/parameters/v2/Head/Yaw2"));
        assert!(!get("/avatar/parameters/v2/Head/Yaw4"));
        assert!(get("/avatar/parameters/v2/Head/Yaw8"));
        assert!(get("/avatar/parameters/v2/Head/YawNegative"));
        // 4 bits + 4 bits + Negative, all Bool, no floats.
        assert_eq!(out.len(), 9);
        assert!(out.iter().all(|p| matches!(p.value, OscValue::Bool(_))));
    }

    #[test]
    fn int_declarations_are_ignored() {
        let cfg = cfg_with(vec![decl("FT/v2/JawOpen", "Int")]);
        let set = AvatarParamSet::resolve(&cfg, &["v2/JawOpen"], &[]);
        assert!(set.is_empty());
    }

    // ---- Most-recent pick ----

    #[test]
    fn most_recent_config_wins_and_malformed_newest_is_skipped() {
        let dir = std::env::temp_dir().join(format!("vco-avatar-unit-{}", std::process::id()));
        let avatars = dir.join("usr_a").join("Avatars");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&avatars).unwrap();
        let write = |name: &str, body: &str| {
            std::fs::write(avatars.join(name), body).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(30));
        };
        write(
            "old.json",
            r#"{"id":"avtr_old","name":"old","parameters":[]}"#,
        );
        write(
            "new.json",
            r#"{"id":"avtr_new","name":"new","parameters":[]}"#,
        );
        write("broken.json", "{ not json");
        // Newest file is malformed → fall back to the newest parseable one.
        let cfg = find_most_recent_config(std::slice::from_ref(&dir)).unwrap();
        assert_eq!(cfg.id, "avtr_new");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---- Watcher ----

    #[test]
    fn watcher_without_listener_never_reports_changes() {
        let mut w = AvatarWatcher::new(vec![fixture_dir()], None);
        assert!(w.initial_config().is_some());
        assert!(w.poll().is_none());
    }
}
