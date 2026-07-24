// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Clustering-key ranges and the global range-disjointness chain.
//!
//! One place owns the predicate "do these superfiles' clustering-key
//! ranges form a single, globally non-overlapping, key-ordered chain?"
//! Two callers share it so their answers can never drift apart:
//!
//!   * the **SQL scan** ([`crate::supertable::query::provider`]) declares
//!     the writer's sort order on a clustered table only when the chain
//!     holds — a false declaration would be a wrong-results bug, so the
//!     check is a correctness gate, not an optimization; and
//!   * **compaction** ([`crate::supertable::compaction`]) uses the same
//!     predicate after its convergence rounds to decide whether a final
//!     full-table merge is needed to make the surviving data superfiles
//!     globally disjoint.
//!
//! The ranges are lifted from each superfile's manifest skip statistics,
//! so the check is manifest-only: no superfile bytes are read.

use std::{cmp::Ordering, sync::Arc};

use datafusion::scalar::ScalarValue;

use crate::supertable::manifest::SuperfileEntry;

/// One superfile's clustering-key range, lifted from the manifest's
/// per-column skip stats for the range-disjointness decision.
///
/// Column-wise bounds are valid lexicographic row bounds: every row has
/// each column inside its per-column `[min, max]`, so the tuple of
/// minima compares `<=` every row and the tuple of maxima `>=` every
/// non-null row. The bounds may be loose (not attained by any row),
/// which only makes the disjointness test more conservative — a safe
/// direction, since looseness can at worst force the unordered
/// fallback, never a false ordering.
pub(crate) struct ClusterKeyRange {
    /// Column-wise minima, in key order. Valid even for files with
    /// nulls in the key: nulls sort last, so the min over non-null
    /// values still lower-bounds every row.
    pub(crate) min: Vec<ScalarValue>,
    /// Column-wise maxima, in key order. An upper bound on the
    /// *non-null* rows only — see `may_have_nulls`.
    pub(crate) max: Vec<ScalarValue>,
    /// Whether any key column holds nulls (or its null count is
    /// unknown). Under the writer's nulls-last order a null key sorts
    /// after every value, i.e. beyond `max` — so a null-bearing file
    /// is ordered only as the *last* file of its scan partition, and
    /// the scan's grouping forces a partition break after it.
    pub(crate) may_have_nulls: bool,
}

/// Extract `entry`'s clustering-key range from its manifest stats.
/// `None` when any key column lacks min/max (no stat recorded, or an
/// all-null column) — the caller then falls back to the unordered scan.
pub(crate) fn cluster_key_range(entry: &SuperfileEntry, key: &[String]) -> Option<ClusterKeyRange> {
    let mut min = Vec::with_capacity(key.len());
    let mut max = Vec::with_capacity(key.len());
    let mut may_have_nulls = false;
    for column in key {
        let agg = entry.scalar_stats.get(column)?;
        let lo = ScalarValue::try_from_array(&agg.min, 0).ok()?;
        let hi = ScalarValue::try_from_array(&agg.max, 0).ok()?;
        if lo.is_null() || hi.is_null() {
            return None;
        }
        may_have_nulls |= agg.null_count.is_none_or(|n| n > 0);
        min.push(lo);
        max.push(hi);
    }
    Some(ClusterKeyRange {
        min,
        max,
        may_have_nulls,
    })
}

/// Lexicographic comparison of two key-bound tuples, column by column
/// in key order. `None` when a pair of scalars isn't comparable (e.g.
/// mismatched stat types across superfiles) — callers treat that as
/// "cannot prove ordered" and fall back.
pub(crate) fn cmp_key_bounds(a: &[ScalarValue], b: &[ScalarValue]) -> Option<Ordering> {
    for (x, y) in a.iter().zip(b) {
        match x.partial_cmp(y)? {
            Ordering::Equal => {}
            decided => return Some(decided),
        }
    }
    Some(Ordering::Equal)
}

/// Order `ranges` into a single globally non-overlapping key chain and
/// return the index permutation that realizes it, or `None` when no such
/// chain exists.
///
/// This is the pure core the SQL scan's file grouping and compaction's
/// final-pass decision both consume, so "provably range-disjoint" means
/// exactly one thing across the engine.
///
/// # How
/// 1. Sort the range indices by `(min, max)` bound. Any incomparable
///    pair → `None`.
/// 2. Verify the chain: each range's max must compare `<=` the next
///    range's min (touching bounds are fine — a duplicate key value may
///    legitimately span a shard boundary and the concatenation is still
///    non-decreasing). Any overlap → `None`.
///
/// Null-bearing ranges (`may_have_nulls`) are ordered by their non-null
/// max here; the scan grouping is what forces a break after them (their
/// null rows sort past the recorded max). Disjointness of the non-null
/// spans is unaffected, so the chain check ignores the flag.
pub(crate) fn disjoint_chain_order(ranges: &[ClusterKeyRange]) -> Option<Vec<usize>> {
    let mut order: Vec<usize> = (0..ranges.len()).collect();
    let mut comparable = true;
    order.sort_by(|&a, &b| {
        let decided = match cmp_key_bounds(&ranges[a].min, &ranges[b].min) {
            Some(Ordering::Equal) => cmp_key_bounds(&ranges[a].max, &ranges[b].max),
            decided => decided,
        };
        decided.unwrap_or_else(|| {
            comparable = false;
            Ordering::Equal
        })
    });
    if !comparable {
        return None;
    }

    // Chain check over the min-sorted order: any overlap disproves a
    // global key order across files, so no partitioning of these files
    // (other than trivial reshuffles) is declared ordered — fall back.
    for pair in order.windows(2) {
        if cmp_key_bounds(&ranges[pair[0]].max, &ranges[pair[1]].min)? == Ordering::Greater {
            return None;
        }
    }
    Some(order)
}

/// The global range-disjointness verdict for a set of superfiles under a
/// clustering key — the exact precondition the ordered scan declares its
/// sort order under, plus the extra bit compaction needs to tell a
/// fixable overlap apart from an unfixable missing-stat case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ChainStatus {
    /// Every entry has a usable key range and they chain without
    /// overlap. The scan declares its ordering; compaction's final pass
    /// is not needed.
    Holds,
    /// Every entry has a usable key range but the ranges overlap (or
    /// aren't mutually comparable). A single full-table re-sort can cut
    /// these into globally disjoint outputs, so this is the state
    /// compaction's final pass exists to repair.
    Broken,
    /// At least one entry lacks a usable key range (a key column with no
    /// recorded stat, or an all-null key column). The scan can't be
    /// ordered for a reason no merge is guaranteed to fix, so callers
    /// leave the table on its existing unordered fallback rather than
    /// rewriting it.
    Indeterminate,
}

/// Classify how `entries`' clustering-key ranges chain under `key`. The
/// `Holds` verdict is exactly the scan's ordered-path precondition;
/// `Broken` vs `Indeterminate` splits "fixable by a re-sort" from "no
/// usable stats" for compaction's final-pass gate.
pub(crate) fn cluster_chain_status(entries: &[Arc<SuperfileEntry>], key: &[String]) -> ChainStatus {
    if key.is_empty() {
        return ChainStatus::Indeterminate;
    }
    let mut ranges = Vec::with_capacity(entries.len());
    for entry in entries {
        match cluster_key_range(entry, key) {
            Some(range) => ranges.push(range),
            None => return ChainStatus::Indeterminate,
        }
    }
    if disjoint_chain_order(&ranges).is_some() {
        ChainStatus::Holds
    } else {
        ChainStatus::Broken
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use datafusion::scalar::ScalarValue;

    use crate::supertable::manifest::cluster_range::{
        ClusterKeyRange, cmp_key_bounds, disjoint_chain_order,
    };

    /// Single-column i64 range with no nulls — the common shape.
    fn range(lo: i64, hi: i64) -> ClusterKeyRange {
        ClusterKeyRange {
            min: vec![ScalarValue::Int64(Some(lo))],
            max: vec![ScalarValue::Int64(Some(hi))],
            may_have_nulls: false,
        }
    }

    #[test]
    fn empty_and_singleton_chain_trivially() {
        assert_eq!(disjoint_chain_order(&[]), Some(vec![]));
        assert_eq!(disjoint_chain_order(&[range(0, 10)]), Some(vec![0]));
    }

    #[test]
    fn already_ordered_disjoint_ranges_chain() {
        let ranges = [range(0, 10), range(20, 30), range(40, 50)];
        assert_eq!(disjoint_chain_order(&ranges), Some(vec![0, 1, 2]));
    }

    #[test]
    fn scrambled_disjoint_ranges_chain_in_sorted_order() {
        // Same ranges, shuffled: the returned permutation must sort them.
        let ranges = [range(40, 50), range(0, 10), range(20, 30)];
        assert_eq!(disjoint_chain_order(&ranges), Some(vec![1, 2, 0]));
    }

    #[test]
    fn touching_bounds_are_allowed() {
        // A duplicate key value may legitimately straddle a shard boundary;
        // max == next.min keeps the concatenation non-decreasing.
        let ranges = [range(0, 10), range(10, 20), range(20, 30)];
        assert_eq!(disjoint_chain_order(&ranges), Some(vec![0, 1, 2]));
    }

    #[test]
    fn partial_overlap_breaks_the_chain() {
        // [0,15] and [10,20] share (10,15] — no global order exists.
        assert_eq!(disjoint_chain_order(&[range(0, 15), range(10, 20)]), None);
    }

    #[test]
    fn nested_range_breaks_the_chain() {
        // [10,20] sits entirely inside [0,100]; the field failure's shape.
        assert_eq!(disjoint_chain_order(&[range(0, 100), range(10, 20)]), None);
    }

    #[test]
    fn incomparable_bound_types_fall_back() {
        // Mismatched stat types across superfiles (Int64 vs Utf8) are not
        // comparable — treated as "cannot prove ordered", never a lie.
        let a = ClusterKeyRange {
            min: vec![ScalarValue::Int64(Some(0))],
            max: vec![ScalarValue::Int64(Some(10))],
            may_have_nulls: false,
        };
        let b = ClusterKeyRange {
            min: vec![ScalarValue::Utf8(Some("x".into()))],
            max: vec![ScalarValue::Utf8(Some("z".into()))],
            may_have_nulls: false,
        };
        assert_eq!(disjoint_chain_order(&[a, b]), None);
    }

    #[test]
    fn null_bearing_ranges_still_chain_on_non_null_bounds() {
        // may_have_nulls does not affect the chain — the scan grouping,
        // not this predicate, forces the partition break after a null file.
        let mut a = range(0, 10);
        a.may_have_nulls = true;
        let ranges = [a, range(20, 30)];
        assert_eq!(disjoint_chain_order(&ranges), Some(vec![0, 1]));
    }

    #[test]
    fn multi_column_key_orders_lexicographically() {
        // Key (col0, col1): equal col0, disjoint on col1 -> chains.
        let a = ClusterKeyRange {
            min: vec![ScalarValue::Int64(Some(1)), ScalarValue::Int64(Some(0))],
            max: vec![ScalarValue::Int64(Some(1)), ScalarValue::Int64(Some(10))],
            may_have_nulls: false,
        };
        let b = ClusterKeyRange {
            min: vec![ScalarValue::Int64(Some(1)), ScalarValue::Int64(Some(20))],
            max: vec![ScalarValue::Int64(Some(1)), ScalarValue::Int64(Some(30))],
            may_have_nulls: false,
        };
        assert_eq!(disjoint_chain_order(&[b, a]), Some(vec![1, 0]));
    }

    #[test]
    fn multi_column_second_col_overlap_breaks_chain() {
        // Equal col0, overlapping col1 -> no order.
        let a = ClusterKeyRange {
            min: vec![ScalarValue::Int64(Some(1)), ScalarValue::Int64(Some(0))],
            max: vec![ScalarValue::Int64(Some(1)), ScalarValue::Int64(Some(25))],
            may_have_nulls: false,
        };
        let b = ClusterKeyRange {
            min: vec![ScalarValue::Int64(Some(1)), ScalarValue::Int64(Some(20))],
            max: vec![ScalarValue::Int64(Some(1)), ScalarValue::Int64(Some(30))],
            may_have_nulls: false,
        };
        assert_eq!(disjoint_chain_order(&[a, b]), None);
    }

    #[test]
    fn cmp_key_bounds_decides_on_first_differing_column() {
        let a = [ScalarValue::Int64(Some(1)), ScalarValue::Int64(Some(9))];
        let b = [ScalarValue::Int64(Some(1)), ScalarValue::Int64(Some(2))];
        assert_eq!(cmp_key_bounds(&a, &b), Some(Ordering::Greater));
        assert_eq!(cmp_key_bounds(&a, &a), Some(Ordering::Equal));
    }
}
