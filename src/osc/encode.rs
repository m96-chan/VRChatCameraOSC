//! Converting [`OscParam`]s into `rosc` messages and OSC/UDP wire bytes.
//!
//! See the module-level docs in [`crate::osc`] for the address/type design.

use anyhow::Context;
use rosc::{OscMessage, OscPacket, OscType};

use super::{OscParam, OscValue};

/// Map an [`OscValue`] to its single `rosc` OSC argument.
pub(crate) fn to_osc_type(value: OscValue) -> OscType {
    match value {
        OscValue::Float(f) => OscType::Float(f),
        OscValue::Bool(b) => OscType::Bool(b),
        OscValue::Int(i) => OscType::Int(i),
    }
}

/// Build the `rosc::OscMessage` for a parameter: address plus one argument.
pub fn to_message(param: &OscParam) -> OscMessage {
    OscMessage {
        addr: param.address(),
        args: vec![to_osc_type(param.value)],
    }
}

/// Encode a parameter into an OSC/UDP datagram (one message per packet).
pub fn encode_param(param: &OscParam) -> anyhow::Result<Vec<u8>> {
    let packet = OscPacket::Message(to_message(param));
    rosc::encoder::encode(&packet)
        .with_context(|| format!("failed to OSC-encode {}", param.address()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_conversions_cover_every_variant() {
        assert_eq!(to_osc_type(OscValue::Float(0.5)), OscType::Float(0.5));
        assert_eq!(to_osc_type(OscValue::Bool(true)), OscType::Bool(true));
        assert_eq!(to_osc_type(OscValue::Bool(false)), OscType::Bool(false));
        assert_eq!(to_osc_type(OscValue::Int(-7)), OscType::Int(-7));
    }

    #[test]
    fn message_uses_avatar_parameter_address_and_single_arg() {
        let msg = to_message(&OscParam::float("MouthOpen", 0.42));
        assert_eq!(msg.addr, "/avatar/parameters/MouthOpen");
        assert_eq!(msg.args, vec![OscType::Float(0.42)]);
    }

    #[test]
    fn encode_matches_hand_built_packet_bytes() {
        let param = OscParam::float("MouthOpen", 0.42);
        let bytes = encode_param(&param).unwrap();

        // Hand-built equivalent packet must encode to identical bytes.
        let expected = rosc::encoder::encode(&OscPacket::Message(OscMessage {
            addr: "/avatar/parameters/MouthOpen".to_string(),
            args: vec![OscType::Float(0.42)],
        }))
        .unwrap();

        assert_eq!(bytes, expected);
        // OSC datagrams are always 4-byte aligned.
        assert_eq!(bytes.len() % 4, 0);
    }

    #[test]
    fn encode_round_trips_through_decoder_for_each_type() {
        for param in [
            OscParam::float("Smile", 0.75),
            OscParam::bool("Jump", true),
            OscParam::int("VRCEmote", 3),
        ] {
            let bytes = encode_param(&param).unwrap();
            let (rest, packet) = rosc::decoder::decode_udp(&bytes).unwrap();
            assert!(rest.is_empty(), "no trailing bytes expected");
            match packet {
                OscPacket::Message(m) => {
                    assert_eq!(m.addr, param.address());
                    assert_eq!(m.args, vec![to_osc_type(param.value)]);
                }
                OscPacket::Bundle(_) => panic!("expected a single message, not a bundle"),
            }
        }
    }
}
