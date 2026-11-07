//! Keypoint layouts: names, bones for drawing and left/right pairs.

use serde::{Deserialize, Serialize};

/// A keypoint layout.
///
/// Every table is indexed by keypoint index, the order the model emits.
/// "Left" and "right" are the person's own sides.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Skeleton {
    /// Halpe-26: the 17 COCO body points plus head top, neck, hip center and
    /// big toe, small toe and heel on each foot.
    Halpe26,
}

/// Which side of the body a keypoint is on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Side {
    /// The person's left.
    Left,
    /// The person's right.
    Right,
    /// On the midline (nose, head, neck, hip center).
    Center,
}

const HALPE26_NAMES: [&str; 26] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "head",
    "neck",
    "hip",
    "left_big_toe",
    "right_big_toe",
    "left_small_toe",
    "right_small_toe",
    "left_heel",
    "right_heel",
];

const HALPE26_BONES: [(usize, usize); 27] = [
    (15, 13),
    (13, 11),
    (11, 19),
    (16, 14),
    (14, 12),
    (12, 19),
    (17, 18),
    (18, 19),
    (18, 5),
    (5, 7),
    (7, 9),
    (18, 6),
    (6, 8),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
    (15, 20),
    (15, 22),
    (15, 24),
    (16, 21),
    (16, 23),
    (16, 25),
];

const HALPE26_MIRROR: [usize; 26] = [
    0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 17, 18, 19, 21, 20, 23, 22, 25, 24,
];

const HALPE26_PAIRS: [(usize, usize); 11] = [
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
    (20, 21),
    (22, 23),
    (24, 25),
];

impl Skeleton {
    /// Number of keypoints.
    #[must_use]
    #[allow(
        clippy::len_without_is_empty,
        reason = "every layout has keypoints, so `is_empty` would always be false"
    )]
    pub fn len(self) -> usize {
        self.names().len()
    }

    /// Keypoint names (`snake_case`), by index.
    #[must_use]
    pub fn names(self) -> &'static [&'static str] {
        match self {
            Self::Halpe26 => &HALPE26_NAMES,
        }
    }

    /// The index of a keypoint by name.
    #[must_use]
    pub fn index(self, name: &str) -> Option<usize> {
        self.names().iter().position(|n| *n == name)
    }

    /// Pairs of keypoint indices to join with a line when drawing.
    #[must_use]
    pub fn bones(self) -> &'static [(usize, usize)] {
        match self {
            Self::Halpe26 => &HALPE26_BONES,
        }
    }

    /// `(left, right)` index pairs of the keypoints that come in pairs.
    #[must_use]
    pub fn pairs(self) -> &'static [(usize, usize)] {
        match self {
            Self::Halpe26 => &HALPE26_PAIRS,
        }
    }

    /// For each keypoint, its left/right partner (itself on the midline):
    /// the index permutation that mirrors a pose.
    #[must_use]
    pub fn mirror(self) -> &'static [usize] {
        match self {
            Self::Halpe26 => &HALPE26_MIRROR,
        }
    }

    /// The side of the body keypoint `i` is on (`Center` for an index out of
    /// range).
    #[must_use]
    pub fn side(self, i: usize) -> Side {
        match self.names().get(i) {
            Some(n) if n.starts_with("left_") => Side::Left,
            Some(n) if n.starts_with("right_") => Side::Right,
            _ => Side::Center,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_are_consistent() {
        let s = Skeleton::Halpe26;
        assert_eq!(s.len(), 26);
        assert_eq!(s.mirror().len(), s.len());
        for &(a, b) in s.bones() {
            assert!(a < s.len() && b < s.len() && a != b);
        }
        for (i, &m) in s.mirror().iter().enumerate() {
            assert_eq!(s.mirror()[m], i, "mirror is an involution");
            let (n, mn) = (s.names()[i], s.names()[m]);
            match s.side(i) {
                Side::Left => assert_eq!(mn, n.replace("left_", "right_")),
                Side::Right => assert_eq!(mn, n.replace("right_", "left_")),
                Side::Center => assert_eq!(m, i),
            }
        }
        // Pairs are exactly the left/right mirror pairs.
        let mut from_mirror: Vec<(usize, usize)> = (0..s.len())
            .filter(|&i| s.side(i) == Side::Left)
            .map(|i| (i, s.mirror()[i]))
            .collect();
        from_mirror.sort_unstable();
        let mut pairs = s.pairs().to_vec();
        pairs.sort_unstable();
        assert_eq!(pairs, from_mirror);
        for &(l, r) in s.pairs() {
            assert_eq!((s.side(l), s.side(r)), (Side::Left, Side::Right));
        }
        assert_eq!(s.index("left_heel"), Some(24));
        assert_eq!(s.index("club_head"), None);
        assert_eq!(s.side(99), Side::Center);
    }

    #[test]
    fn serializes_by_name() {
        assert_eq!(
            serde_json::to_string(&Skeleton::Halpe26).unwrap(),
            "\"halpe26\""
        );
    }
}
