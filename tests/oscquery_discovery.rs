//! Integration tests for OSCQuery avatar-parameter discovery (issue #18
//! phase 3b): the HTTP client against a loopback stub serving fixture node
//! JSON, our own HTTP responder round-tripped through `serde_json`, and the
//! `AvatarWatcher` source priority (local file -> OSCQuery -> blind
//! fallback). The mDNS layer is exercised through the `VrchatEndpointSource`
//! trait with a fixed-endpoint fake; real multicast is environment-dependent
//! and covered by the `#[ignore]`d test at the bottom.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener};
use std::path::{Path, PathBuf};
use std::time::Duration;

use vrchat_camera_osc::mapping::avatar::{AvatarParamSet, AvatarWatcher};
use vrchat_camera_osc::osc::oscquery::{
    fetch_avatar_config, FixedEndpoint, OscQueryAdvertiser, OscQueryClient,
};
use vrchat_camera_osc::osc::{AvatarChangeListener, OscValue};

fn fixture(name: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/oscquery")
        .join(name);
    std::fs::read_to_string(path).unwrap()
}

/// A minimal HTTP/1.1 stub: serves `responses` (path-prefix, status, body)
/// on a loopback ephemeral port until dropped; unknown paths get a 404.
struct StubServer {
    addr: SocketAddr,
}

impl StubServer {
    fn serve(responses: Vec<(&'static str, u16, String)>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                let mut buf = [0u8; 2048];
                let n = stream.read(&mut buf).unwrap_or(0);
                let req = String::from_utf8_lossy(&buf[..n]).to_string();
                let target = req.split_whitespace().nth(1).unwrap_or("/").to_string();
                let (status, body) = responses
                    .iter()
                    .find(|(path, _, _)| target == *path)
                    .map(|(_, s, b)| (*s, b.clone()))
                    .unwrap_or((404, String::new()));
                let reason = if status == 200 { "OK" } else { "Not Found" };
                let resp = format!(
                    "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\n\
                     Content-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(resp.as_bytes());
            }
        });
        Self { addr }
    }
}

// ---------------------------------------------------------------------------
// HTTP client fetch against a loopback stub
// ---------------------------------------------------------------------------

#[test]
fn fetches_and_parses_avatar_parameters_endpoint() {
    let stub = StubServer::serve(vec![(
        "/avatar/parameters",
        200,
        fixture("avatar_parameters.json"),
    )]);
    let cfg = fetch_avatar_config(stub.addr, "avtr_from_change").unwrap();
    assert_eq!(cfg.id, "avtr_from_change");
    // Writable (ACCESS & 2) leaves only: the ACCESS=1 VelocityX is dropped.
    assert!(!cfg.parameters.iter().any(|p| p.name == "VelocityX"));
    let jaw = cfg
        .parameters
        .iter()
        .find(|p| p.name == "FT/v2/JawOpen")
        .expect("nested float leaf");
    let input = jaw.input.as_ref().unwrap();
    assert_eq!(input.address, "/avatar/parameters/FT/v2/JawOpen");
    assert_eq!(input.osc_type, "Float");
    // The whole phase-2/3 machinery reuses this config unchanged: binary
    // bits + Negative resolve exactly as from a local config file.
    let set = AvatarParamSet::resolve(&cfg, &["v2/JawOpen"], &["EyeTrackingActive"]);
    let mut out = Vec::new();
    set.emit_value("v2/JawOpen", 0.6, &mut out);
    set.emit_status("EyeTrackingActive", true, &mut out);
    // float + 3 bits + Negative + status bool.
    assert_eq!(out.len(), 6);
    assert!(out
        .iter()
        .any(|p| p.name == "/avatar/parameters/FT/v2/JawOpen4" && p.value == OscValue::Bool(true)));
}

#[test]
fn falls_back_to_root_when_parameters_endpoint_404s() {
    let stub = StubServer::serve(vec![("/", 200, fixture("root.json"))]);
    let cfg = fetch_avatar_config(stub.addr, "").unwrap();
    // The id comes from the /avatar/change VALUE in the tree.
    assert_eq!(cfg.id, "avtr_00000000-0000-4000-8000-00000000c0de");
    assert!(cfg
        .parameters
        .iter()
        .any(|p| p.name == "v2/SmileFrownRight"));
}

#[test]
fn malformed_json_is_an_error_not_a_panic() {
    let stub = StubServer::serve(vec![("/avatar/parameters", 200, "{ not json".to_string())]);
    assert!(fetch_avatar_config(stub.addr, "x").is_err());
}

// ---------------------------------------------------------------------------
// Watcher priority: local file -> OSCQuery -> blind fallback
// ---------------------------------------------------------------------------

fn local_fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/avatar_configs")
}

#[test]
fn watcher_prefers_local_config_file_over_oscquery() {
    let stub = StubServer::serve(vec![(
        "/avatar/parameters",
        200,
        fixture("avatar_parameters.json"),
    )]);
    let client = OscQueryClient::new(Box::new(FixedEndpoint(stub.addr)));
    let mut w = AvatarWatcher::new(vec![local_fixture_dir()], None).with_oscquery(client);
    let cfg = w.initial_config().expect("local config found");
    // The local fixture wins; the OSCQuery stub's tree has no name/id match.
    assert!(cfg.id.starts_with("avtr_00000000-0000-4000-8000-00000000"));
    assert_ne!(cfg.id, "avtr_from_change");
}

#[test]
fn watcher_uses_oscquery_when_no_local_config_exists() {
    let stub = StubServer::serve(vec![(
        "/avatar/parameters",
        200,
        fixture("avatar_parameters.json"),
    )]);
    let client = OscQueryClient::new(Box::new(FixedEndpoint(stub.addr)));
    let empty = std::env::temp_dir().join(format!("vco-oscquery-none-{}", std::process::id()));
    let mut w = AvatarWatcher::new(vec![empty], None).with_oscquery(client);
    let cfg = w.initial_config().expect("OSCQuery config fetched");
    assert!(cfg.parameters.iter().any(|p| p.name == "FT/v2/JawOpen"));
}

#[test]
fn avatar_change_retriggers_oscquery_fetch() {
    let stub = StubServer::serve(vec![(
        "/avatar/parameters",
        200,
        fixture("avatar_parameters.json"),
    )]);
    let client = OscQueryClient::new(Box::new(FixedEndpoint(stub.addr)));
    let listener = AvatarChangeListener::bind(0).unwrap();
    let port = listener.local_port();
    let empty = std::env::temp_dir().join(format!("vco-oscquery-none2-{}", std::process::id()));
    let mut w = AvatarWatcher::new(vec![empty], Some(listener)).with_oscquery(client);

    let msg = rosc::encoder::encode(&rosc::OscPacket::Message(rosc::OscMessage {
        addr: "/avatar/change".to_string(),
        args: vec![rosc::OscType::String("avtr_new".to_string())],
    }))
    .unwrap();
    let sock = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();
    sock.send_to(&msg, ("127.0.0.1", port)).unwrap();

    // Poll until the datagram lands; the change must resolve via OSCQuery.
    let mut update = None;
    for _ in 0..100 {
        if let Some(u) = w.poll() {
            update = Some(u);
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    let cfg = update
        .expect("avatar change seen")
        .expect("config resolved via OSCQuery");
    assert_eq!(cfg.id, "avtr_new");
    assert!(cfg.parameters.iter().any(|p| p.name == "FT/v2/JawOpen"));
}

#[test]
fn unreachable_endpoint_degrades_to_blind_fallback() {
    // A bound-then-dropped port: connection refused, quickly.
    let dead = {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap()
    };
    let client = OscQueryClient::new(Box::new(FixedEndpoint(dead)));
    let empty = std::env::temp_dir().join(format!("vco-oscquery-none3-{}", std::process::id()));
    let mut w = AvatarWatcher::new(vec![empty], None).with_oscquery(client);
    assert!(
        w.initial_config().is_none(),
        "fetch failure -> blind fallback"
    );
}

// ---------------------------------------------------------------------------
// Our own advertisement: HTTP responder (HOST_INFO + root node)
// ---------------------------------------------------------------------------

fn http_get(addr: SocketAddr, target: &str) -> (u16, String) {
    let mut stream = std::net::TcpStream::connect(addr).unwrap();
    stream
        .write_all(
            format!("GET {target} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n").as_bytes(),
        )
        .unwrap();
    let mut resp = String::new();
    stream.read_to_string(&mut resp).unwrap();
    let status: u16 = resp
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap();
    let body = resp
        .split_once("\r\n\r\n")
        .map(|(_, b)| b.to_string())
        .unwrap_or_default();
    (status, body)
}

#[test]
fn advertiser_http_serves_host_info_and_root_node() {
    // mDNS registration may fail in a sandbox without multicast; the HTTP
    // responder must come up regardless (graceful degradation).
    let adv = OscQueryAdvertiser::start(9001).unwrap();
    let addr: SocketAddr = ([127, 0, 0, 1], adv.http_port()).into();

    let (status, body) = http_get(addr, "/?HOST_INFO");
    assert_eq!(status, 200);
    let host_info: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(host_info["NAME"]
        .as_str()
        .unwrap()
        .starts_with("VRChatCameraOSC-"));
    assert_eq!(host_info["OSC_PORT"], 9001);
    assert_eq!(host_info["OSC_TRANSPORT"], "UDP");
    assert_eq!(host_info["OSC_IP"], "127.0.0.1");
    assert_eq!(host_info["EXTENSIONS"]["ACCESS"], true);

    let (status, body) = http_get(addr, "/");
    assert_eq!(status, 200);
    let root: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(root["FULL_PATH"], "/");
    let change = &root["CONTENTS"]["avatar"]["CONTENTS"]["change"];
    assert_eq!(change["FULL_PATH"], "/avatar/change");
    assert_eq!(change["TYPE"], "s");
    assert_eq!(change["ACCESS"], 2);

    let (status, _) = http_get(addr, "/nonexistent");
    assert_eq!(status, 404);
}

// ---------------------------------------------------------------------------
// Real multicast (environment-dependent -> ignored by default)
// ---------------------------------------------------------------------------

/// Requires working multicast on the host: our advertiser must be
/// discoverable via a plain mdns-sd browse of `_oscjson._tcp.local.`.
/// Run with `cargo test -- --ignored oscquery_advertisement`.
#[test]
#[ignore]
fn oscquery_advertisement_is_visible_over_real_mdns() {
    let adv = OscQueryAdvertiser::start(9001).unwrap();
    let name = adv.instance_name().to_string();
    let daemon = mdns_sd::ServiceDaemon::new().unwrap();
    let rx = daemon.browse("_oscjson._tcp.local.").unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    let mut found = false;
    while std::time::Instant::now() < deadline {
        match rx.recv_timeout(Duration::from_millis(500)) {
            Ok(mdns_sd::ServiceEvent::ServiceResolved(info)) => {
                if info.fullname.starts_with(&name) {
                    assert_eq!(info.port, adv.http_port());
                    found = true;
                    break;
                }
            }
            Ok(_) => {}
            Err(_) => {}
        }
    }
    let _ = daemon.shutdown();
    assert!(found, "our _oscjson._tcp advertisement was not resolved");
}
