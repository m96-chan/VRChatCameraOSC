//! End-to-end OSC/UDP integration test over loopback (issue #3).
//!
//! Binds a receiver `UdpSocket` on `127.0.0.1:0`, points a `UdpOscSender` at
//! it, sends a batch of parameters, then decodes each received datagram and
//! asserts the address + argument round-trip. Loopback works in CI.

use std::net::UdpSocket;
use std::time::Duration;

use rosc::{OscPacket, OscType};
use vrchat_camera_osc::osc::{OscParam, OscSink, UdpOscSender};

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
