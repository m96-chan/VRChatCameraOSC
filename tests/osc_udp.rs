//! End-to-end OSC/UDP integration test over loopback (issue #3).
//!
//! Binds a receiver `UdpSocket` on `127.0.0.1:0`, points a `UdpOscSender` at
//! it, sends a batch of parameters, then decodes each received datagram and
//! asserts the address + argument round-trip. Loopback works in CI.

use std::net::UdpSocket;
use std::time::Duration;

use rosc::{OscPacket, OscType};
use vrchat_camera_osc::osc::{OscParam, OscSink, UdpOscSender};

use vrchat_camera_osc::mapping::arkit::ArkitMappingConfig;
use vrchat_camera_osc::mapping::unified::UnifiedMapper;
use vrchat_camera_osc::tracking::arkit::{
    ArkitBlendshapes, ArkitFaceFrame, HeadPose, NUM_BLENDSHAPES,
};

#[test]
fn udp_sender_delivers_decodable_osc_over_loopback() {
    let receiver = UdpSocket::bind("127.0.0.1:0").expect("bind receiver");
    receiver
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    let port = receiver.local_addr().unwrap().port();

    let mut sender = UdpOscSender::new("127.0.0.1", port).expect("build sender");
    assert_eq!(sender.target().port(), port);

    let params = vec![
        OscParam::float("MouthOpen", 0.42),
        OscParam::bool("Jump", true),
        OscParam::int("VRCEmote", 3),
    ];
    sender.send(&params).expect("send params");

    let mut buf = [0u8; 1024];
    for expected in &params {
        let (len, _from) = receiver.recv_from(&mut buf).expect("recv datagram");
        let (rest, packet) = rosc::decoder::decode_udp(&buf[..len]).expect("decode");
        assert!(rest.is_empty());

        let msg = match packet {
            OscPacket::Message(m) => m,
            OscPacket::Bundle(_) => panic!("expected single message"),
        };
        assert_eq!(msg.addr, expected.address());
        assert_eq!(msg.args.len(), 1);

        match (&msg.args[0], &expected.value) {
            (OscType::Float(a), vrchat_camera_osc::osc::OscValue::Float(b)) => assert_eq!(a, b),
            (OscType::Bool(a), vrchat_camera_osc::osc::OscValue::Bool(b)) => assert_eq!(a, b),
            (OscType::Int(a), vrchat_camera_osc::osc::OscValue::Int(b)) => assert_eq!(a, b),
            (got, want) => panic!("arg mismatch: got {got:?}, want {want:?}"),
        }
    }
}

// VRCFT profile end-to-end (issue #18): UnifiedMapper output reaches the wire
// as decodable `/avatar/parameters/<prefix>v2/...` OSC messages, floats and
// the three *TrackingActive bools included. Single prefix: the default
// two-prefix set (~300 datagrams) overflows an unread loopback receive buffer
// in this synchronous test — VRChat reads concurrently, the test can't.
#[test]
fn vrcft_profile_params_reach_the_wire_as_v2_addresses() {
    let receiver = UdpSocket::bind("127.0.0.1:0").expect("bind receiver");
    receiver
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    let port = receiver.local_addr().unwrap().port();
    let mut sender = UdpOscSender::new("127.0.0.1", port).expect("build sender");

    let mut mapper = UnifiedMapper::new(ArkitMappingConfig::default(), vec![String::new()]);
    let mut bs = [0.0f32; NUM_BLENDSHAPES];
    bs[25] = 0.6; // jawOpen
    let params = mapper.map(&ArkitFaceFrame {
        blendshapes: ArkitBlendshapes(bs),
        head: HeadPose::default(),
    });
    sender.send(&params).expect("send params");

    let mut buf = [0u8; 1024];
    let mut seen = std::collections::HashSet::new();
    for _ in 0..params.len() {
        let (len, _from) = receiver.recv_from(&mut buf).expect("recv datagram");
        let (rest, packet) = rosc::decoder::decode_udp(&buf[..len]).expect("decode");
        assert!(rest.is_empty());
        let msg = match packet {
            OscPacket::Message(m) => m,
            OscPacket::Bundle(_) => panic!("expected single message"),
        };
        assert!(
            msg.addr.starts_with("/avatar/parameters/"),
            "unexpected address {}",
            msg.addr
        );
        seen.insert(msg.addr);
    }
    for addr in [
        "/avatar/parameters/v2/JawOpen",
        "/avatar/parameters/v2/EyeLidLeft",
        "/avatar/parameters/v2/Head/Yaw",
        "/avatar/parameters/EyeTrackingActive",
        "/avatar/parameters/ExpressionTrackingActive",
    ] {
        assert!(seen.contains(addr), "missing on the wire: {addr}");
    }
}
