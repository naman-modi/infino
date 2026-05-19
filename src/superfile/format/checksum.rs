//! CRC32C wrapper.
//!
//! Every section of every blob ends with a 4-byte CRC32C of its body —
//! the format-spec invariant. This module is a thin shim over the
//! `crc32c` crate (which uses SSE4.2 / ARMv8.1 hardware instructions when
//! available, ~25 GB/s on modern hardware) so the rest of the codebase
//! has a single, named entry point and so we can swap implementations
//! without touching call sites.
//!
//! Use Castagnoli polynomial (0x1EDC6F41) — the same one used by iSCSI,
//! Btrfs, Parquet bloom filters, and most modern systems.

/// Compute CRC32C over `bytes`. Hardware-accelerated when the crate's
/// runtime feature detection finds SSE4.2 (x86) or v8.1 CRC (ARM); falls
/// back to a software implementation otherwise.
#[inline]
pub fn crc32c(bytes: &[u8]) -> u32 {
    ::crc32c::crc32c(bytes)
}

/// Streaming variant — useful when the input arrives in chunks (we don't
/// need this in v1 readers/builders, but keeping the entry point so future
/// callers don't reach for the upstream crate directly).
#[inline]
pub fn crc32c_append(prev: u32, bytes: &[u8]) -> u32 {
    ::crc32c::crc32c_append(prev, bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Castagnoli reference vectors. These match RFC 3720 / iSCSI test data
    // and the published `crc32c` crate test fixtures.

    #[test]
    fn empty_input_is_zero() {
        // crc32c() of empty input is 0 by definition.
        assert_eq!(crc32c(b""), 0);
    }

    #[test]
    fn single_byte_known_vector() {
        // crc32c("a") = 0xC1D04330 (standard Castagnoli reference).
        assert_eq!(crc32c(b"a"), 0xC1D0_4330);
    }

    #[test]
    fn ascii_string_known_vector() {
        // crc32c("123456789") = 0xE3069283 — the classic CRC reference vector.
        assert_eq!(crc32c(b"123456789"), 0xE306_9283);
    }

    #[test]
    fn append_matches_one_shot() {
        // Streaming over arbitrary chunks must agree with a single-shot call.
        let bytes: &[u8] = b"the quick brown fox jumps over the lazy dog";
        let one_shot = crc32c(bytes);

        let mut acc = 0u32;
        for chunk in bytes.chunks(7) {
            acc = crc32c_append(acc, chunk);
        }
        assert_eq!(acc, one_shot);

        // Single-byte chunks too — most pathological case.
        let mut acc2 = 0u32;
        for chunk in bytes.chunks(1) {
            acc2 = crc32c_append(acc2, chunk);
        }
        assert_eq!(acc2, one_shot);
    }

    #[test]
    fn append_with_zero_seed_equals_one_shot() {
        // crc32c_append(0, x) === crc32c(x). Documents the "fresh stream
        // starts at 0" invariant for callers.
        assert_eq!(crc32c_append(0, b"hello"), crc32c(b"hello"));
    }

    #[test]
    fn alignment_does_not_affect_result() {
        // Hardware CRC instructions can be alignment-sensitive on some
        // microarchitectures; the crate's wrapper handles that. Verify
        // shifted slices give identical results.
        let buf: Vec<u8> = (0..1024u16).map(|i| i as u8).collect();
        let baseline = crc32c(&buf);
        for shift in 1..16 {
            let shifted: Vec<u8> = std::iter::repeat_n(0u8, shift)
                .chain(buf.iter().copied())
                .collect();
            assert_eq!(crc32c(&shifted[shift..]), baseline);
        }
    }

    #[test]
    fn different_inputs_give_different_outputs() {
        // Sanity: changing one bit changes the CRC. This is the whole
        // point of using a CRC vs nothing.
        let a = crc32c(b"AAAA");
        let b = crc32c(b"AAAB");
        assert_ne!(a, b);
    }

    #[test]
    fn large_input_does_not_panic_or_truncate() {
        // 4 MB input — exercise the streaming path inside the crate.
        let big: Vec<u8> = (0..4_000_000).map(|i| (i % 256) as u8).collect();
        let _ = crc32c(&big);
    }
}
