//! JPEG helpers.
//!
//! UVC cameras commonly send MJPEG frames without a DHT (Huffman table)
//! segment: the AVI1 MJPEG convention lets the decoder assume the standard
//! tables. Standalone JPEG decoders need them spelled out, which
//! [`ensure_huffman`] does without touching the image data.

use std::borrow::Cow;

/// The default Huffman tables of ITU-T T.81 Annex K.3, as one DHT segment.
///
/// Many webcams send MJPEG frames without a DHT segment and rely on the decoder
/// assuming these tables (the AVI1 MJPEG convention). Standalone `.jpg` files
/// need them spelled out, so [`ensure_huffman`] inserts this segment.
const STANDARD_DHT: &[u8] = &{
    const DC_LUMA_BITS: [u8; 16] = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
    const DC_CHROMA_BITS: [u8; 16] = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
    const DC_VALUES: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    const AC_LUMA_BITS: [u8; 16] = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7D];
    const AC_LUMA_VALUES: [u8; 162] = [
        0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61,
        0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52,
        0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25,
        0x26, 0x27, 0x28, 0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45,
        0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64,
        0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83,
        0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99,
        0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6,
        0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3,
        0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8,
        0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA,
    ];
    const AC_CHROMA_BITS: [u8; 16] = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77];
    const AC_CHROMA_VALUES: [u8; 162] = [
        0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61,
        0x71, 0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xA1, 0xB1, 0xC1, 0x09, 0x23, 0x33,
        0x52, 0xF0, 0x15, 0x62, 0x72, 0xD1, 0x0A, 0x16, 0x24, 0x34, 0xE1, 0x25, 0xF1, 0x17, 0x18,
        0x19, 0x1A, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44,
        0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63,
        0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A,
        0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97,
        0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4,
        0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA,
        0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7,
        0xE8, 0xE9, 0xEA, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA,
    ];
    // Segment: FFC4, length, then (class/id, 16 counts, values) per table.
    const LEN: usize = 2 + 4 * 17 + 12 + 12 + 162 + 162;
    let mut out = [0u8; LEN + 2];
    out[0] = 0xFF;
    out[1] = 0xC4;
    // LEN is 418, so both halves fit a byte.
    #[allow(clippy::cast_possible_truncation)]
    {
        out[2] = (LEN >> 8) as u8;
        out[3] = (LEN & 0xFF) as u8;
    }
    let mut i = 4;
    let tables: [(u8, &[u8; 16], &[u8]); 4] = [
        (0x00, &DC_LUMA_BITS, &DC_VALUES),
        (0x10, &AC_LUMA_BITS, &AC_LUMA_VALUES),
        (0x01, &DC_CHROMA_BITS, &DC_VALUES),
        (0x11, &AC_CHROMA_BITS, &AC_CHROMA_VALUES),
    ];
    let mut t = 0;
    while t < tables.len() {
        let (class, bits, values) = tables[t];
        out[i] = class;
        i += 1;
        let mut k = 0;
        while k < 16 {
            out[i] = bits[k];
            i += 1;
            k += 1;
        }
        let mut k = 0;
        while k < values.len() {
            out[i] = values[k];
            i += 1;
            k += 1;
        }
        t += 1;
    }
    out
};

/// Returns the JPEG with the standard Huffman tables inserted before the first
/// SOS marker if it has no DHT segment of its own. The entropy-coded image data
/// is never touched. Data that isn't a parseable JPEG is returned unchanged.
///
/// ```
/// use std::borrow::Cow;
/// use stance::jpeg::ensure_huffman;
///
/// // SOI, then SOS straight away: no Huffman tables.
/// let frame = [0xFF, 0xD8, 0xFF, 0xDA, 0x00, 0x02, 0x00, 0xFF, 0xD9];
/// let fixed = ensure_huffman(&frame);
/// assert!(matches!(fixed, Cow::Owned(_)));
/// assert_eq!(&fixed[2..4], &[0xFF, 0xC4]);
/// // Already complete (or not a JPEG): borrowed, unchanged.
/// assert!(matches!(ensure_huffman(&fixed), Cow::Borrowed(_)));
/// assert!(matches!(ensure_huffman(b"not a jpeg"), Cow::Borrowed(_)));
/// ```
#[must_use]
pub fn ensure_huffman(jpeg: &[u8]) -> Cow<'_, [u8]> {
    if jpeg.len() < 4 || jpeg[0] != 0xFF || jpeg[1] != 0xD8 {
        return Cow::Borrowed(jpeg);
    }
    let mut i = 2;
    while i + 4 <= jpeg.len() {
        if jpeg[i] != 0xFF {
            return Cow::Borrowed(jpeg);
        }
        let marker = jpeg[i + 1];
        match marker {
            0xFF => {
                i += 1; // fill byte
                continue;
            }
            0xC4 => return Cow::Borrowed(jpeg),
            0xDA => {
                let mut out = Vec::with_capacity(jpeg.len() + STANDARD_DHT.len());
                out.extend_from_slice(&jpeg[..i]);
                out.extend_from_slice(STANDARD_DHT);
                out.extend_from_slice(&jpeg[i..]);
                return Cow::Owned(out);
            }
            0x01 | 0xD0..=0xD7 => {
                i += 2; // markers without a length
                continue;
            }
            _ => {}
        }
        let len = usize::from(u16::from_be_bytes([jpeg[i + 2], jpeg[i + 3]]));
        i += 2 + len;
    }
    Cow::Borrowed(jpeg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use zune_jpeg::JpegDecoder;
    use zune_jpeg::zune_core::bytestream::ZCursor;

    fn encode(w: u16, h: u16) -> Vec<u8> {
        let mut rgb = Vec::with_capacity(usize::from(w) * usize::from(h) * 3);
        for y in 0..h {
            for x in 0..w {
                rgb.extend_from_slice(&[
                    u8::try_from(u32::from(x) * 255 / u32::from(w)).unwrap(),
                    u8::try_from(u32::from(y) * 255 / u32::from(h)).unwrap(),
                    128,
                ]);
            }
        }
        let mut out = Vec::new();
        jpeg_encoder::Encoder::new(&mut out, 90)
            .encode(&rgb, w, h, jpeg_encoder::ColorType::Rgb)
            .unwrap();
        out
    }

    fn decode(jpeg: &[u8]) -> (usize, usize, Vec<u8>) {
        let mut dec = JpegDecoder::new(ZCursor::new(jpeg));
        let pixels = dec.decode().unwrap();
        let (w, h) = dec.dimensions().unwrap();
        (w, h, pixels)
    }

    /// Removes every DHT segment, as a DHT-less MJPEG camera frame would be.
    fn strip_dht(jpeg: &[u8]) -> Vec<u8> {
        let mut out = jpeg[..2].to_vec();
        let mut i = 2;
        loop {
            let marker = jpeg[i + 1];
            if marker == 0xDA {
                out.extend_from_slice(&jpeg[i..]);
                return out;
            }
            let len = usize::from(u16::from_be_bytes([jpeg[i + 2], jpeg[i + 3]]));
            if marker != 0xC4 {
                out.extend_from_slice(&jpeg[i..i + 2 + len]);
            }
            i += 2 + len;
        }
    }

    #[test]
    fn standard_dht_is_well_formed() {
        assert_eq!(&STANDARD_DHT[..2], &[0xFF, 0xC4]);
        let len = usize::from(u16::from_be_bytes([STANDARD_DHT[2], STANDARD_DHT[3]]));
        assert_eq!(len + 2, STANDARD_DHT.len());
        // Each table's counts sum to its number of values.
        let mut i = 4;
        for (class, values) in [(0x00, 12), (0x10, 162), (0x01, 12), (0x11, 162)] {
            assert_eq!(STANDARD_DHT[i], class);
            let total: usize = STANDARD_DHT[i + 1..i + 17]
                .iter()
                .map(|&b| usize::from(b))
                .sum();
            assert_eq!(total, values);
            i += 17 + values;
        }
        assert_eq!(i, STANDARD_DHT.len());
    }

    #[test]
    fn restores_a_frame_without_huffman_tables() {
        // jpeg-encoder writes the standard tables; strip them and put ours back.
        let original = encode(64, 48);
        let stripped = strip_dht(&original);
        assert!(stripped.len() < original.len());
        assert!(
            JpegDecoder::new(ZCursor::new(&stripped)).decode().is_err(),
            "a strict decoder rejects the stripped frame"
        );
        let fixed = ensure_huffman(&stripped);
        assert!(matches!(fixed, Cow::Owned(_)));
        let (w, h, pixels) = decode(&fixed);
        assert_eq!((w, h), (64, 48));
        assert_eq!(pixels, decode(&original).2, "same pixels as the original");
        // The entropy-coded data after SOS is byte-identical.
        let sos = |j: &[u8]| j.windows(2).position(|w| w == [0xFF, 0xDA]).unwrap();
        assert_eq!(&original[sos(&original)..], &fixed[sos(&fixed)..]);
    }

    #[test]
    fn leaves_complete_frames_alone() {
        let jpeg = encode(16, 16);
        assert!(matches!(ensure_huffman(&jpeg), Cow::Borrowed(_)));
        assert!(matches!(ensure_huffman(&[]), Cow::Borrowed(_)));
        assert!(matches!(
            ensure_huffman(&[0xFF, 0xD8, 0x00]),
            Cow::Borrowed(_)
        ));
        // A truncated header never reads past the end.
        assert!(matches!(
            ensure_huffman(&[0xFF, 0xD8, 0xFF, 0xE0, 0xFF, 0xFF]),
            Cow::Borrowed(_)
        ));
    }
}
