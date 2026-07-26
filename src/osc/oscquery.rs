//! OSCQuery discovery + advertisement (issue #18 phase 3b).
//!
//! Local-file avatar discovery (`mapping::avatar`, phases 2+3) assumes the
//! tracker and VRChat share a machine. VRChat has implemented **OSCQuery**
//! since 2023.3.1: it advertises an HTTP endpoint over mDNS
//! (`_oscjson._tcp.local.`, instance `VRChat-Client-*`) that serves its OSC
//! address tree as JSON, and it *browses* the same service types to find
//! trackers to send its own OSC output (`/avatar/change`, parameters) to.
//! This module implements both directions:
//!
//! - **Client** ([`OscQueryClient`] + [`MdnsVrchatDiscovery`]): browse mDNS
//!   for the VRChat client, then HTTP-GET its `/avatar/parameters` node tree
//!   (falling back to `/` and walking) and convert it into the existing
//!   [`AvatarOscConfig`] model — the phase-2/3 gating/binary machinery is
//!   reused untouched.
//! - **Advertise** ([`OscQueryAdvertiser`]): register
//!   `VRChatCameraOSC-<hex>` on `_osc._udp.local.` (our `[osc] listen_port`)
//!   and `_oscjson._tcp.local.` (a minimal std-`TcpListener` HTTP responder
//!   serving `?HOST_INFO` and a root node declaring `/avatar/change`). This
//!   is what makes VRChat send us `/avatar/change` with zero VRChat-side
//!   setup, locally and remotely.
//!
//! # Sources (CLAUDE.md rule 3 — verified upstream, not memory)
//!
//! - VRCFaceTracking's working implementation:
//!   [`OscQueryService.cs`](https://github.com/benaclejames/VRCFaceTracking/blob/master/VRCFaceTracking.Core/Services/OscQueryService.cs),
//!   its `mDNS/` folder (query of `_oscjson._tcp.local`, `VRChat-Client`
//!   instance filter), `OscQueryConfigParser.cs` (HTTP fetch + node
//!   flattening by `FULL_PATH`), and `HttpHandler.cs` /
//!   `OscQueryHostInfo.cs` (the exact HOST_INFO + root-node JSON we mirror).
//! - The [OSCQuery spec proposal](https://github.com/Vidvox/OSCQueryProposal)
//!   (`CONTENTS`/`FULL_PATH`/`TYPE`/`ACCESS`, `?HOST_INFO`) and VRChat's
//!   [vrc-oscquery-lib](https://github.com/vrchat-community/vrc-oscquery-lib).
//! - The `mdns-sd` crate API, verified against the 0.20 source
//!   (`ServiceDaemon::browse`/`register`, `ServiceEvent::ServiceResolved`,
//!   `ServiceInfo::enable_addr_auto`).
//!
//! # Threading / hot path (CLAUDE.md rule 8, mirroring `osc::listen`)
//!
//! Nothing here runs per frame. mDNS browsing and the HTTP responder each
//! live on their own background thread; the pipeline only reads a shared
//! `Option<SocketAddr>` (via [`VrchatEndpointSource::endpoint`], never
//! blocking) and performs an HTTP fetch **only when the avatar actually
//! changes** — the same rationale as the filesystem scan in
//! `mapping::avatar`. Every network failure degrades gracefully: log once,
//! fall back (local file / blind prefixes), never crash the loop.
//!
//! # Known limitation (documented, from the VRCFT source)
//!
//! VRChat may advertise `127.0.0.1` as its mDNS A record even on a remote
//! machine; VRCFT substitutes the packet's sender address in that case.
//! `mdns-sd` does not expose the sender, so a remote VRChat that only
//! advertises loopback cannot be resolved here — the local-file path (or a
//! mounted config dir) still covers that setup.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::{IpAddr, SocketAddr, TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Deserialize;

use crate::mapping::avatar::{AvatarIoDecl, AvatarOscConfig, AvatarParamDecl};

/// Overall timeout for one OSCQuery HTTP fetch. Short: the endpoint is on
/// the local machine or LAN, and a fetch only happens at startup or on an
/// avatar change.
pub const HTTP_TIMEOUT: Duration = Duration::from_secs(2);

/// How long [`OscQueryClient::fetch_initial`] waits for mDNS discovery at
/// startup before giving up (the browse keeps running in the background, so
/// a VRChat started later is still picked up on the next avatar change).
pub const STARTUP_DISCOVERY_WAIT: Duration = Duration::from_secs(2);

// ---------------------------------------------------------------------------
// OSCQuery node JSON -> AvatarOscConfig
// ---------------------------------------------------------------------------

/// One node of an OSCQuery address tree (the subset of the spec VRChat
/// serves): containers carry `CONTENTS`, leaves carry `TYPE`/`ACCESS`/
/// `VALUE`, and every node knows its `FULL_PATH`.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct OscQueryNode {
    #[serde(rename = "FULL_PATH", default)]
    pub full_path: Option<String>,
    #[serde(rename = "TYPE", default)]
    pub osc_type: Option<String>,
    #[serde(rename = "ACCESS", default)]
    pub access: Option<u8>,
    #[serde(rename = "CONTENTS", default)]
    pub contents: Option<HashMap<String, OscQueryNode>>,
    #[serde(rename = "VALUE", default)]
    pub value: Option<Vec<serde_json::Value>>,
}

impl OscQueryNode {
    pub fn from_json(text: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(text)?)
    }

    fn child(&self, name: &str) -> Option<&OscQueryNode> {
        self.contents.as_ref()?.get(name)
    }

    /// OSCQuery `ACCESS` is a bitmask: 1 = readable, 2 = writable. Only
    /// write-capable leaves are OSC *inputs* we can send to; a missing
    /// `ACCESS` is treated as writable (spec default is full access).
    fn writable(&self) -> bool {
        self.access.is_none_or(|a| a & 2 != 0)
    }
}

/// Map an OSCQuery `TYPE` tag to VRChat's avatar-config type string
/// (`"Float"`/`"Bool"`/`"Int"` — what [`AvatarOscConfig`] carries). VRChat
/// serves `f` for floats, `T` (or `F`) for bools, and `i` for ints; `d`/`h`
/// are accepted as the wide variants. Anything else (e.g. `s`) has no
/// avatar-parameter form and is skipped.
fn config_type_for_tag(tag: &str) -> Option<&'static str> {
    match tag.chars().next()? {
        'f' | 'd' => Some("Float"),
        'T' | 'F' => Some("Bool"),
        'i' | 'h' => Some("Int"),
        _ => None,
    }
}

/// Recursively collect every writable, typed leaf under `node` as a declared
/// input parameter (mirrors VRCFT's `OscQueryAvatarInfo` flattening, plus
/// the write-`ACCESS` filter that stands in for the config file's
/// `input != null`).
fn collect_params(node: &OscQueryNode, out: &mut Vec<AvatarParamDecl>) {
    if let Some(children) = &node.contents {
        if !children.is_empty() {
            for child in children.values() {
                collect_params(child, out);
            }
            return;
        }
    }
    let Some(full_path) = node.full_path.as_deref() else {
        return;
    };
    let Some(cfg_type) = node.osc_type.as_deref().and_then(config_type_for_tag) else {
        return;
    };
    if !node.writable() {
        return;
    }
    let name = full_path
        .strip_prefix("/avatar/parameters/")
        .unwrap_or(full_path)
        .to_string();
    out.push(AvatarParamDecl {
        name,
        input: Some(AvatarIoDecl {
            address: full_path.to_string(),
            osc_type: cfg_type.to_string(),
        }),
        output: None,
    });
}

/// Build an [`AvatarOscConfig`] from a fetched `/avatar/parameters` subtree.
/// The avatar id is not part of that subtree, so the caller supplies it
/// (from `/avatar/change`, or `""` at startup); the name is unknown over
/// OSCQuery (VRCFT hits the same limitation and falls back to the local
/// config file for it).
pub fn avatar_config_from_parameters_node(node: &OscQueryNode, avatar_id: &str) -> AvatarOscConfig {
    let mut parameters = Vec::new();
    collect_params(node, &mut parameters);
    AvatarOscConfig {
        id: avatar_id.to_string(),
        name: String::new(),
        parameters,
    }
}

/// Build an [`AvatarOscConfig`] by walking a fetched root (`/`) node down to
/// `avatar/parameters`. The id is taken from the `/avatar/change` node's
/// `VALUE` when present, else `fallback_id`. `None` when the tree has no
/// `avatar/parameters` subtree.
pub fn avatar_config_from_root(root: &OscQueryNode, fallback_id: &str) -> Option<AvatarOscConfig> {
    let avatar = root.child("avatar")?;
    let parameters = avatar.child("parameters")?;
    let id = avatar
        .child("change")
        .and_then(|c| c.value.as_ref())
        .and_then(|v| v.first())
        .and_then(|v| v.as_str())
        .unwrap_or(fallback_id);
    Some(avatar_config_from_parameters_node(parameters, id))
}

// ---------------------------------------------------------------------------
// HTTP client fetch
// ---------------------------------------------------------------------------

/// Fetch the avatar's declared parameters from a VRChat OSCQuery endpoint:
/// `GET /avatar/parameters`, falling back to `GET /` + walking the tree when
/// the direct path is not served. `avatar_id` seeds the config id (see
/// [`avatar_config_from_parameters_node`]).
pub fn fetch_avatar_config(
    endpoint: SocketAddr,
    avatar_id: &str,
) -> anyhow::Result<AvatarOscConfig> {
    let base = format!("http://{endpoint}");
    let url = format!("{base}/avatar/parameters");
    match ureq::get(&url).timeout(HTTP_TIMEOUT).call() {
        Ok(resp) => {
            let node = OscQueryNode::from_json(&resp.into_string()?)
                .map_err(|e| anyhow::anyhow!("parsing {url}: {e}"))?;
            Ok(avatar_config_from_parameters_node(&node, avatar_id))
        }
        // Any HTTP status (404, ...) -> try the root node and walk instead.
        Err(ureq::Error::Status(_, _)) => {
            let resp = ureq::get(&base).timeout(HTTP_TIMEOUT).call()?;
            let node = OscQueryNode::from_json(&resp.into_string()?)
                .map_err(|e| anyhow::anyhow!("parsing {base}/: {e}"))?;
            avatar_config_from_root(&node, avatar_id)
                .ok_or_else(|| anyhow::anyhow!("{base}/ has no avatar/parameters subtree"))
        }
        Err(e) => Err(e.into()),
    }
}

// ---------------------------------------------------------------------------
// VRChat endpoint discovery (mDNS behind a trait, for testability)
// ---------------------------------------------------------------------------

/// Where the VRChat OSCQuery HTTP endpoint currently is, if known. The real
/// implementation is [`MdnsVrchatDiscovery`]; tests use [`FixedEndpoint`].
pub trait VrchatEndpointSource: Send + std::fmt::Debug {
    /// Latest known endpoint. Must never block — this is consulted from the
    /// pipeline's avatar-change handling.
    fn endpoint(&self) -> Option<SocketAddr>;

    /// Wait up to `timeout` for an endpoint to appear (startup only).
    fn wait_for_endpoint(&self, timeout: Duration) -> Option<SocketAddr> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(ep) = self.endpoint() {
                return Some(ep);
            }
            if Instant::now() >= deadline {
                return None;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}

/// A fixed, already-known endpoint — for tests and for a future explicit
/// `oscquery_host` override.
#[derive(Debug)]
pub struct FixedEndpoint(pub SocketAddr);

impl VrchatEndpointSource for FixedEndpoint {
    fn endpoint(&self) -> Option<SocketAddr> {
        Some(self.0)
    }
}

/// Continuous background mDNS browse of `_oscjson._tcp.local.` for a
/// `VRChat-Client-*` instance (VRCFT's filter). The most recently resolved
/// endpoint is kept in shared state; the browse thread ends when the daemon
/// is shut down on drop.
pub struct MdnsVrchatDiscovery {
    endpoint: Arc<Mutex<Option<SocketAddr>>>,
    daemon: Option<mdns_sd::ServiceDaemon>,
}

impl std::fmt::Debug for MdnsVrchatDiscovery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // ServiceDaemon has no Debug impl; show the resolved endpoint.
        f.debug_struct("MdnsVrchatDiscovery")
            .field("endpoint", &self.endpoint())
            .finish_non_exhaustive()
    }
}

/// The service type VRChat advertises its OSCQuery HTTP server on.
pub const OSCJSON_SERVICE: &str = "_oscjson._tcp.local.";
/// The service type plain OSC receivers advertise on.
pub const OSC_SERVICE: &str = "_osc._udp.local.";
/// VRChat's OSCQuery instance-name prefix (from the VRCFT source).
pub const VRCHAT_INSTANCE_PREFIX: &str = "VRChat-Client";

impl MdnsVrchatDiscovery {
    pub fn start() -> anyhow::Result<Self> {
        let daemon = mdns_sd::ServiceDaemon::new()?;
        let receiver = daemon.browse(OSCJSON_SERVICE)?;
        let endpoint = Arc::new(Mutex::new(None));
        let shared = Arc::clone(&endpoint);
        std::thread::Builder::new()
            .name("oscquery-mdns-browse".into())
            .spawn(move || {
                while let Ok(event) = receiver.recv() {
                    if let mdns_sd::ServiceEvent::ServiceResolved(info) = event {
                        if !info.fullname.starts_with(VRCHAT_INSTANCE_PREFIX) {
                            continue;
                        }
                        if let Some(ip) = pick_address(&info.addresses) {
                            *shared.lock().unwrap() = Some(SocketAddr::new(ip, info.port));
                        }
                    }
                }
            })?;
        Ok(Self {
            endpoint,
            daemon: Some(daemon),
        })
    }
}

/// Prefer a routable (non-loopback) IPv4 address, then any non-loopback,
/// then loopback (the same-machine case — see the module-docs limitation).
fn pick_address(addresses: &std::collections::HashSet<mdns_sd::ScopedIp>) -> Option<IpAddr> {
    let ips: Vec<IpAddr> = addresses.iter().map(|a| a.to_ip_addr()).collect();
    ips.iter()
        .find(|ip| !ip.is_loopback() && ip.is_ipv4())
        .or_else(|| ips.iter().find(|ip| !ip.is_loopback()))
        .or_else(|| ips.first())
        .copied()
}

impl VrchatEndpointSource for MdnsVrchatDiscovery {
    fn endpoint(&self) -> Option<SocketAddr> {
        *self.endpoint.lock().unwrap()
    }
}

impl Drop for MdnsVrchatDiscovery {
    fn drop(&mut self) {
        if let Some(daemon) = self.daemon.take() {
            let _ = daemon.shutdown();
        }
    }
}

// ---------------------------------------------------------------------------
// Client: endpoint source + fetch + graceful degradation
// ---------------------------------------------------------------------------

/// The OSCQuery avatar-config source used by `mapping::avatar::AvatarWatcher`
/// when the local config file is not found. Failures log **once** (repeat
/// failures are silent until a fetch succeeds again) and return `None`, so
/// the caller falls back to blind-prefix mode — the loop never crashes on
/// network trouble.
#[derive(Debug)]
pub struct OscQueryClient {
    source: Box<dyn VrchatEndpointSource>,
    failure_logged: bool,
}

impl OscQueryClient {
    pub fn new(source: Box<dyn VrchatEndpointSource>) -> Self {
        Self {
            source,
            failure_logged: false,
        }
    }

    /// Startup fetch: waits up to [`STARTUP_DISCOVERY_WAIT`] for mDNS to
    /// resolve VRChat, then fetches. `None` -> no OSCQuery source available
    /// (not discovered, or fetch failed).
    pub fn fetch_initial(&mut self) -> Option<AvatarOscConfig> {
        let endpoint = self.source.wait_for_endpoint(STARTUP_DISCOVERY_WAIT)?;
        self.fetch_from(endpoint, "")
    }

    /// Avatar-change fetch (triggered-on-change only — never on the frame
    /// hot path). Non-blocking when no endpoint is known yet.
    pub fn fetch(&mut self, avatar_id: &str) -> Option<AvatarOscConfig> {
        let endpoint = self.source.endpoint()?;
        self.fetch_from(endpoint, avatar_id)
    }

    fn fetch_from(&mut self, endpoint: SocketAddr, avatar_id: &str) -> Option<AvatarOscConfig> {
        match fetch_avatar_config(endpoint, avatar_id) {
            Ok(cfg) => {
                self.failure_logged = false;
                eprintln!(
                    "oscquery: avatar parameters fetched from http://{endpoint} \
                     ({} declared inputs)",
                    cfg.parameters.len()
                );
                Some(cfg)
            }
            Err(e) => {
                if !self.failure_logged {
                    eprintln!(
                        "oscquery: fetch from http://{endpoint} failed: {e:#} — \
                         falling back (further failures logged silently)"
                    );
                    self.failure_logged = true;
                }
                None
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Advertisement: HOST_INFO / root-node JSON + HTTP responder + mDNS register
// ---------------------------------------------------------------------------

/// The `?HOST_INFO` document (mirrors VRCFT's `OscQueryHostInfo`): who we
/// are, where our OSC server (the `/avatar/change` listener) is, and which
/// OSCQuery extensions we answer.
pub fn host_info_json(name: &str, osc_ip: IpAddr, osc_port: u16) -> serde_json::Value {
    serde_json::json!({
        "NAME": name,
        "EXTENSIONS": {
            "ACCESS": true,
            "CLIPMODE": false,
            "RANGE": true,
            "TYPE": true,
            "VALUE": true,
        },
        "OSC_IP": osc_ip.to_string(),
        "OSC_PORT": osc_port,
        "OSC_TRANSPORT": "UDP",
    })
}

/// Our root address-tree node: like VRCFT, we only declare `/avatar/change`
/// (write-only string) — enough for VRChat to know where to send its OSC
/// output.
pub fn root_node_json() -> serde_json::Value {
    serde_json::json!({
        "FULL_PATH": "/",
        "ACCESS": 0,
        "CONTENTS": {
            "avatar": {
                "FULL_PATH": "/avatar",
                "ACCESS": 0,
                "CONTENTS": {
                    "change": {
                        "FULL_PATH": "/avatar/change",
                        "ACCESS": 2,
                        "TYPE": "s",
                    }
                }
            }
        }
    })
}

/// Advertises this tracker over OSCQuery: a minimal HTTP/1.1 responder on an
/// ephemeral TCP port (background thread) plus mDNS registrations of
/// `_oscjson._tcp` (the HTTP port) and `_osc._udp` (our OSC listen port).
///
/// mDNS registration failing (no multicast — containers, locked-down
/// networks) is non-fatal: the HTTP responder still runs, and a warning is
/// printed once. Threads are detached; when the process exits they die with
/// it, so TUI/monitor shutdown is never blocked. Dropping the advertiser
/// unregisters the services and shuts the mDNS daemon down.
pub struct OscQueryAdvertiser {
    instance: String,
    http_port: u16,
    daemon: Option<mdns_sd::ServiceDaemon>,
    fullnames: Vec<String>,
}

impl std::fmt::Debug for OscQueryAdvertiser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // ServiceDaemon has no Debug impl; show the advertised identity.
        f.debug_struct("OscQueryAdvertiser")
            .field("instance", &self.instance)
            .field("http_port", &self.http_port)
            .field("mdns_active", &self.daemon.is_some())
            .finish_non_exhaustive()
    }
}

impl OscQueryAdvertiser {
    /// Start the responder and advertise `osc_listen_port` as our OSC input.
    pub fn start(osc_listen_port: u16) -> anyhow::Result<Self> {
        let instance = format!("VRChatCameraOSC-{}", random_hex());
        let http_port = spawn_http_responder(instance.clone(), osc_listen_port)?;

        // mDNS registration is best-effort (see the struct docs).
        let mut fullnames = Vec::new();
        let daemon = match register_services(&instance, http_port, osc_listen_port) {
            Ok((daemon, names)) => {
                fullnames = names;
                Some(daemon)
            }
            Err(e) => {
                eprintln!(
                    "oscquery: mDNS advertisement failed: {e:#} — \
                     VRChat will not auto-discover this tracker \
                     (the OSCQuery HTTP endpoint on port {http_port} still runs)"
                );
                None
            }
        };

        Ok(Self {
            instance,
            http_port,
            daemon,
            fullnames,
        })
    }

    /// The advertised instance name (`VRChatCameraOSC-<hex>`).
    pub fn instance_name(&self) -> &str {
        &self.instance
    }

    /// The TCP port the OSCQuery HTTP responder is bound to.
    pub fn http_port(&self) -> u16 {
        self.http_port
    }

    /// Whether the mDNS registration itself succeeded.
    pub fn mdns_active(&self) -> bool {
        self.daemon.is_some()
    }
}

impl Drop for OscQueryAdvertiser {
    fn drop(&mut self) {
        if let Some(daemon) = self.daemon.take() {
            for fullname in &self.fullnames {
                let _ = daemon.unregister(fullname);
            }
            let _ = daemon.shutdown();
        }
    }
}

fn register_services(
    instance: &str,
    http_port: u16,
    osc_port: u16,
) -> anyhow::Result<(mdns_sd::ServiceDaemon, Vec<String>)> {
    let daemon = mdns_sd::ServiceDaemon::new()?;
    let hostname = format!("{instance}.local.");
    let props: HashMap<String, String> = [("txtvers".to_string(), "1".to_string())].into();
    let mut fullnames = Vec::new();
    for (ty, port) in [(OSCJSON_SERVICE, http_port), (OSC_SERVICE, osc_port)] {
        // `enable_addr_auto`: the daemon fills in (and tracks) the host's
        // interface addresses, so the advertisement carries routable IPs —
        // required for a VRChat on another machine to reach us.
        let info = mdns_sd::ServiceInfo::new(ty, instance, &hostname, (), port, props.clone())?
            .enable_addr_auto();
        fullnames.push(info.get_fullname().to_string());
        daemon.register(info)?;
    }
    Ok((daemon, fullnames))
}

/// A dependency-free instance suffix: unique enough per process/run to avoid
/// mDNS name collisions between trackers on one LAN (mirrors VRCFT's random
/// suffix; cryptographic quality is not needed here).
fn random_hex() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    format!("{:08x}", nanos ^ std::process::id().rotate_left(16))
}

/// Bind `0.0.0.0:0` and serve the two OSCQuery documents on a detached
/// thread. Dependency-free HTTP/1.1, one request per connection
/// (`Connection: close`) — exactly what VRChat's client does.
fn spawn_http_responder(name: String, osc_port: u16) -> anyhow::Result<u16> {
    let listener = TcpListener::bind(("0.0.0.0", 0))
        .map_err(|e| anyhow::anyhow!("binding the OSCQuery HTTP port: {e}"))?;
    let port = listener.local_addr()?.port();
    std::thread::Builder::new()
        .name("oscquery-http".into())
        .spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                let _ = handle_http_request(&mut stream, &name, osc_port);
            }
        })?;
    Ok(port)
}

fn handle_http_request(stream: &mut TcpStream, name: &str, osc_port: u16) -> std::io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_millis(500)))?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request_line = String::new();
    reader.read_line(&mut request_line)?;
    // Drain the headers so the client sees a clean close.
    let mut line = String::new();
    while reader.read_line(&mut line)? > 2 {
        line.clear();
    }

    let target = request_line.split_whitespace().nth(1).unwrap_or("/");
    let (path, query) = target.split_once('?').unwrap_or((target, ""));

    let (status, body) = if query.contains("HOST_INFO") {
        // OSC_IP: the address the client reached this responder on is, by
        // construction, an address of ours it can also send OSC to.
        let osc_ip = stream
            .local_addr()
            .map(|a| a.ip())
            .unwrap_or(IpAddr::from([127, 0, 0, 1]));
        ("200 OK", host_info_json(name, osc_ip, osc_port).to_string())
    } else if path == "/" {
        ("200 OK", root_node_json().to_string())
    } else {
        ("404 Not Found", String::new())
    };

    let response = format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(response.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mapping::avatar::AvatarParamSet;
    use std::path::Path;

    fn fixture(name: &str) -> String {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/oscquery")
            .join(name);
        std::fs::read_to_string(path).unwrap()
    }

    // ---- Node JSON -> AvatarOscConfig ----

    #[test]
    fn parameters_fixture_flattens_to_writable_typed_leaves() {
        let node = OscQueryNode::from_json(&fixture("avatar_parameters.json")).unwrap();
        let cfg = avatar_config_from_parameters_node(&node, "avtr_x");
        assert_eq!(cfg.id, "avtr_x");
        let names: Vec<&str> = cfg.parameters.iter().map(|p| p.name.as_str()).collect();
        // Nested CONTENTS are walked (v2 subtree, multi-segment Head/Yaw).
        assert!(names.contains(&"FT/v2/JawOpen"));
        assert!(names.contains(&"FT/v2/Head/Yaw"));
        // Binary bit bools + Negative come through as Bool declarations.
        assert!(names.contains(&"FT/v2/JawOpen4"));
        assert!(names.contains(&"FT/v2/JawOpenNegative"));
        // Non-FT writable params are kept in the config (gating drops them
        // later, exactly like a local config file's declarations).
        assert!(names.contains(&"GestureLeftWeight"));
        // ACCESS = 1 (read-only output) is not an input.
        assert!(!names.contains(&"VelocityX"));
        // Every collected declaration is input-capable with a mapped type.
        for p in &cfg.parameters {
            let input = p.input.as_ref().expect("all collected params are inputs");
            assert!(matches!(input.osc_type.as_str(), "Float" | "Bool" | "Int"));
            assert!(input.address.starts_with("/avatar/parameters/"));
        }
        // Type-tag mapping: f -> Float, T -> Bool, i -> Int.
        let ty = |n: &str| {
            cfg.parameters
                .iter()
                .find(|p| p.name == n)
                .unwrap()
                .input
                .as_ref()
                .unwrap()
                .osc_type
                .clone()
        };
        assert_eq!(ty("FT/v2/JawOpen"), "Float");
        assert_eq!(ty("EyeTrackingActive"), "Bool");
        assert_eq!(ty("GestureLeft"), "Int");
    }

    #[test]
    fn oscquery_config_drives_the_phase23_gating_and_binary_machinery() {
        let node = OscQueryNode::from_json(&fixture("avatar_parameters.json")).unwrap();
        let cfg = avatar_config_from_parameters_node(&node, "avtr_x");
        let set = AvatarParamSet::resolve(
            &cfg,
            &["v2/JawOpen", "v2/Head/Yaw", "v2/MouthClosed"],
            &["EyeTrackingActive", "LipTrackingActive"],
        );
        // JawOpen (float + 3 bits + Negative), Head/Yaw, EyeTrackingActive;
        // MouthClosed / LipTrackingActive are undeclared -> gated.
        assert_eq!(set.len(), 3);
        let mut out = Vec::new();
        set.emit_value("v2/JawOpen", 0.6, &mut out);
        // float + 3 bits + Negative = 5 messages; 0.6 with max_int 8 -> 0b100.
        assert_eq!(out.len(), 5);
        assert!(out
            .iter()
            .any(|p| p.name == "/avatar/parameters/FT/v2/JawOpen4"
                && p.value == crate::osc::OscValue::Bool(true)));
        out.clear();
        set.emit_value("v2/MouthClosed", 0.9, &mut out);
        assert!(out.is_empty(), "undeclared params stay gated");
    }

    #[test]
    fn root_fixture_walks_to_parameters_and_takes_id_from_change_value() {
        let root = OscQueryNode::from_json(&fixture("root.json")).unwrap();
        let cfg = avatar_config_from_root(&root, "fallback").unwrap();
        assert_eq!(cfg.id, "avtr_00000000-0000-4000-8000-00000000c0de");
        assert!(cfg
            .parameters
            .iter()
            .any(|p| p.name == "v2/SmileFrownRight"));
        assert!(cfg
            .parameters
            .iter()
            .any(|p| p.name == "ExpressionTrackingActive"));
    }

    #[test]
    fn root_without_avatar_subtree_is_none() {
        let root = OscQueryNode::from_json(r#"{"FULL_PATH":"/","CONTENTS":{}}"#).unwrap();
        assert!(avatar_config_from_root(&root, "x").is_none());
    }

    #[test]
    fn unknown_type_tags_and_missing_access_are_handled() {
        let json = r#"{
            "FULL_PATH": "/avatar/parameters",
            "CONTENTS": {
                "NoAccess": {"FULL_PATH": "/avatar/parameters/NoAccess", "TYPE": "f"},
                "StringParam": {"FULL_PATH": "/avatar/parameters/StringParam", "ACCESS": 3, "TYPE": "s"},
                "Untyped": {"FULL_PATH": "/avatar/parameters/Untyped", "ACCESS": 3}
            }
        }"#;
        let node = OscQueryNode::from_json(json).unwrap();
        let cfg = avatar_config_from_parameters_node(&node, "");
        // Missing ACCESS -> treated writable; s-typed and untyped leaves drop.
        assert_eq!(cfg.parameters.len(), 1);
        assert_eq!(cfg.parameters[0].name, "NoAccess");
    }

    // ---- Our served documents, parsed back and shape-asserted ----

    #[test]
    fn host_info_document_matches_the_oscquery_shape() {
        let v = host_info_json("VRChatCameraOSC-abc", IpAddr::from([192, 168, 1, 2]), 9001);
        assert_eq!(v["NAME"], "VRChatCameraOSC-abc");
        assert_eq!(v["OSC_IP"], "192.168.1.2");
        assert_eq!(v["OSC_PORT"], 9001);
        assert_eq!(v["OSC_TRANSPORT"], "UDP");
        for (ext, on) in [
            ("ACCESS", true),
            ("CLIPMODE", false),
            ("RANGE", true),
            ("TYPE", true),
            ("VALUE", true),
        ] {
            assert_eq!(v["EXTENSIONS"][ext], on, "extension {ext}");
        }
    }

    #[test]
    fn root_node_document_declares_avatar_change_write_only_string() {
        let v = root_node_json();
        // Parseable by our own client-side node model (shape sanity).
        let node = OscQueryNode::from_json(&v.to_string()).unwrap();
        let change = node.child("avatar").unwrap().child("change").unwrap();
        assert_eq!(change.full_path.as_deref(), Some("/avatar/change"));
        assert_eq!(change.osc_type.as_deref(), Some("s"));
        assert_eq!(change.access, Some(2));
    }

    #[test]
    fn advertiser_instance_names_are_prefixed_and_vary() {
        let a = random_hex();
        assert_eq!(a.len(), 8);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
