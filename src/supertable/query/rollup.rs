// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Grouped `COUNT(*)` rollup partials: the pre-aggregated `(key -> count)`
//! payload that lets a `GROUP BY key, COUNT(*)` be answered without
//! re-aggregating the base rows.
//!
//! This is the uncapped sibling of the manifest's `ScalarValueCounts`
//! (`manifest/list.rs`). Both count exact per-value frequencies over a
//! column's non-null rows through the one shared counting authority
//! [`count_values_by`]; the manifest stat caps at `MAX_EXACT_VALUE_COUNTS`
//! and lives inline in the manifest, whereas a `GroupedCount` has no cap
//! and is meant to be materialized as the rows of a rollup superfile
//! (`(key, count)`), because a high-cardinality key (e.g. a URL column with
//! tens of millions of distinct values) cannot ride the manifest.
//!
//! Two operations define it:
//!   * **build** ([`GroupedCount::from_column`]) — the per-source partial,
//!     one grouped count over one superfile's rows;
//!   * **merge** ([`GroupedCount::merge`]) — combine partials by summing
//!     counts per key, the combine a query performs across the rollup
//!     superfiles (plus a freshly-counted undrained tail).
//!
//! Entries are kept sorted by key so a merge of already-sorted runs stays
//! cheap and emission into a rollup superfile is deterministic.

use std::{cmp::Ordering, collections::HashMap, sync::Arc};

use arrow::ipc::{reader::StreamReader, writer::StreamWriter};
use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use datafusion::common::ScalarValue;

use crate::{
    superfile::format::{CRC_BYTES, checksum::crc32c},
    supertable::manifest::list::count_values_by,
};

/// Column name of the count column in a rollup record batch.
const COUNT_COLUMN: &str = "__rollup_count";

/// A grouped `COUNT(*)` partial: exact non-null `(key, count)` pairs over
/// one source, sorted by key, with no cardinality cap. See the module doc.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GroupedCount {
    /// `(key, count)` sorted ascending by key; keys are distinct and
    /// non-null, counts are strictly positive.
    entries: Vec<(ScalarValue, u64)>,
}

impl GroupedCount {
    /// Exact grouped count over `column`'s non-null rows. `None` when the
    /// column type is not one [`count_values_by`] supports. Unlike the
    /// manifest stat, there is no distinct-value cap.
    pub(crate) fn from_column(column: &ArrayRef) -> Option<Self> {
        Self::from_entries(count_values_by(column, None)?)
    }

    /// Merge partials by summing counts per key. `None` on count overflow
    /// or a mixed-type key across partials (an internal invariant break).
    pub(crate) fn merge(parts: impl IntoIterator<Item = Self>) -> Option<Self> {
        let mut merged: HashMap<ScalarValue, u64> = HashMap::new();
        for part in parts {
            for (value, count) in part.entries {
                let total = merged.entry(value).or_default();
                *total = total.checked_add(count)?;
            }
        }
        Self::from_entries(merged.into_iter().collect())
    }

    /// Assemble from raw `(key, count)` pairs: drop zero counts, reject
    /// nulls and mixed key types, sum duplicates, then sort by key.
    fn from_entries(entries: Vec<(ScalarValue, u64)>) -> Option<Self> {
        let mut merged: HashMap<ScalarValue, u64> = HashMap::with_capacity(entries.len());
        for (value, count) in entries {
            if value.is_null() {
                return None;
            }
            if count == 0 {
                continue;
            }
            let total = merged.entry(value).or_default();
            *total = total.checked_add(count)?;
        }
        let mut entries: Vec<(ScalarValue, u64)> = merged.into_iter().collect();
        if let Some((first, _)) = entries.first() {
            let key_type = first.data_type();
            if entries
                .iter()
                .any(|(value, _)| value.data_type() != key_type)
            {
                return None;
            }
        }
        entries.sort_by(|left, right| left.0.partial_cmp(&right.0).unwrap_or(Ordering::Equal));
        // A key type whose values are not totally ordered cannot back a
        // deterministic rollup: bail rather than emit an arbitrary order.
        if entries
            .windows(2)
            .any(|pair| pair[0].0.partial_cmp(&pair[1].0).is_none())
        {
            return None;
        }
        Some(Self { entries })
    }

    /// Sorted `(key, count)` view. Test-only: production materializes via
    /// [`to_record_batch`] / [`to_blob_bytes`]; exposed for assertions.
    #[cfg(test)]
    pub(crate) fn entries(&self) -> &[(ScalarValue, u64)] {
        &self.entries
    }

    /// Distinct-key count. Test-only: production reaches the rows via
    /// [`to_record_batch`] or checks [`is_empty`]; exposed for assertions.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Materialize as a two-column record batch `(key_name: key_type,
    /// __rollup_count: Int64)` — the on-disk rows of a rollup superfile.
    /// `None` if the key array cannot be built or a count exceeds `i64`.
    pub(crate) fn to_record_batch(&self, key_name: &str) -> Option<RecordBatch> {
        let keys =
            ScalarValue::iter_to_array(self.entries.iter().map(|(value, _)| value.clone())).ok()?;
        let counts: Vec<i64> = self
            .entries
            .iter()
            .map(|(_, count)| i64::try_from(*count))
            .collect::<Result<_, _>>()
            .ok()?;
        let counts = Int64Array::from(counts);
        let schema = Arc::new(Schema::new(vec![
            Field::new(key_name, keys.data_type().clone(), false),
            Field::new(COUNT_COLUMN, DataType::Int64, false),
        ]));
        RecordBatch::try_new(schema, vec![keys, Arc::new(counts)]).ok()
    }

    /// Inverse of [`to_record_batch`]: read a rollup superfile's rows back
    /// into a partial. Expects `(key, __rollup_count: Int64)`; `None` on a
    /// shape mismatch, a null, or a negative count.
    pub(crate) fn from_record_batch(batch: &RecordBatch) -> Option<Self> {
        if batch.num_columns() != 2 {
            return None;
        }
        let keys = batch.column(0);
        let counts = batch.column(1).as_any().downcast_ref::<Int64Array>()?;
        let mut entries = Vec::with_capacity(batch.num_rows());
        for row in 0..batch.num_rows() {
            if keys.is_null(row) || counts.is_null(row) {
                return None;
            }
            let value = ScalarValue::try_from_array(keys, row).ok()?;
            let count = u64::try_from(counts.value(row)).ok()?;
            entries.push((value, count));
        }
        Self::from_entries(entries)
    }

    /// Serialize to the self-framed bytes of a grouped-count superfile blob:
    /// the `(key, __rollup_count)` record batch as an Arrow IPC stream,
    /// followed by a trailing 4-byte CRC-32C of that stream (the format's
    /// per-blob checksum discipline — the splice layer does not checksum).
    /// `None` if the batch cannot be built or encoded.
    pub(crate) fn to_blob_bytes(&self, key_name: &str) -> Option<Vec<u8>> {
        let batch = self.to_record_batch(key_name)?;
        let mut body: Vec<u8> = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut body, batch.schema().as_ref()).ok()?;
            writer.write(&batch).ok()?;
            writer.finish().ok()?;
        }
        let crc = crc32c(&body);
        body.extend_from_slice(&crc.to_le_bytes());
        Some(body)
    }

    /// Inverse of [`to_blob_bytes`]: verify the trailing CRC-32C, then decode
    /// the IPC stream back into a partial. `None` on a truncated blob, a CRC
    /// mismatch (corruption / tamper), or a malformed batch.
    pub(crate) fn from_blob_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < CRC_BYTES {
            return None;
        }
        let (body, crc_bytes) = bytes.split_at(bytes.len() - CRC_BYTES);
        let stored = u32::from_le_bytes(crc_bytes.try_into().ok()?);
        if crc32c(body) != stored {
            return None;
        }
        let mut reader = StreamReader::try_new(body, None).ok()?;
        let batch = reader.next()?.ok()?;
        Self::from_record_batch(&batch)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int64Array, StringArray};
    use datafusion::common::ScalarValue;

    use crate::supertable::query::rollup::GroupedCount;

    fn str_col(values: &[&str]) -> ArrayRef {
        Arc::new(StringArray::from(values.to_vec()))
    }

    fn utf8(value: &str, count: u64) -> (ScalarValue, u64) {
        (ScalarValue::Utf8(Some(value.to_string())), count)
    }

    #[test]
    fn from_column_counts_and_sorts_by_key() {
        let col = str_col(&["b", "a", "b", "c", "a", "b"]);
        let gc = GroupedCount::from_column(&col).expect("grouped count");
        assert_eq!(gc.entries(), &[utf8("a", 2), utf8("b", 3), utf8("c", 1)]);
    }

    #[test]
    fn from_column_has_no_cardinality_cap() {
        // 1000 distinct keys: the manifest stat would bail at 256; the
        // rollup partial must not.
        let owned: Vec<String> = (0..1000).map(|i| format!("k{i:04}")).collect();
        let refs: Vec<&str> = owned.iter().map(String::as_str).collect();
        let col = str_col(&refs);
        let gc = GroupedCount::from_column(&col).expect("uncapped grouped count");
        assert_eq!(gc.len(), 1000);
        assert!(gc.entries().iter().all(|(_, count)| *count == 1));
    }

    #[test]
    fn from_column_ignores_nulls() {
        let col: ArrayRef = Arc::new(StringArray::from(vec![Some("a"), None, Some("a"), None]));
        let gc = GroupedCount::from_column(&col).expect("grouped count");
        assert_eq!(gc.entries(), &[utf8("a", 2)]);
    }

    #[test]
    fn merge_sums_counts_per_key() {
        let a = GroupedCount::from_column(&str_col(&["x", "y", "x"])).expect("a");
        let b = GroupedCount::from_column(&str_col(&["y", "z", "y"])).expect("b");
        let merged = GroupedCount::merge([a, b]).expect("merged");
        assert_eq!(
            merged.entries(),
            &[utf8("x", 2), utf8("y", 3), utf8("z", 1)]
        );
    }

    #[test]
    fn merge_of_disjoint_partials_is_concatenation_by_key() {
        // Clustered case: disjoint key slices merge to the concatenation,
        // sorted, with no key colliding (030.11 composition).
        let a = GroupedCount::from_column(&str_col(&["a", "a", "b"])).expect("a");
        let b = GroupedCount::from_column(&str_col(&["c", "d", "d"])).expect("b");
        let merged = GroupedCount::merge([a, b]).expect("merged");
        assert_eq!(
            merged.entries(),
            &[utf8("a", 2), utf8("b", 1), utf8("c", 1), utf8("d", 2)]
        );
    }

    #[test]
    fn record_batch_roundtrip() {
        let gc = GroupedCount::from_column(&str_col(&["a", "b", "b", "c"])).expect("gc");
        let batch = gc.to_record_batch("url").expect("batch");
        assert_eq!(batch.num_columns(), 2);
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.schema().field(0).name(), "url");
        let back = GroupedCount::from_record_batch(&batch).expect("roundtrip");
        assert_eq!(back, gc);
    }

    #[test]
    fn int64_key_roundtrip() {
        let col: ArrayRef = Arc::new(Int64Array::from(vec![10, 20, 10, 30, 20, 10]));
        let gc = GroupedCount::from_column(&col).expect("gc");
        assert_eq!(
            gc.entries(),
            &[
                (ScalarValue::Int64(Some(10)), 3),
                (ScalarValue::Int64(Some(20)), 2),
                (ScalarValue::Int64(Some(30)), 1),
            ]
        );
        let batch = gc.to_record_batch("region_id").expect("batch");
        let back = GroupedCount::from_record_batch(&batch).expect("roundtrip");
        assert_eq!(back, gc);
    }

    #[test]
    fn blob_bytes_roundtrip() {
        let gc = GroupedCount::from_column(&str_col(&["a", "b", "b", "c", "c", "c"])).expect("gc");
        let blob = gc.to_blob_bytes("url").expect("blob");
        let back = GroupedCount::from_blob_bytes(&blob).expect("roundtrip");
        assert_eq!(back, gc);
    }

    #[test]
    fn blob_crc_detects_corruption() {
        let gc = GroupedCount::from_column(&str_col(&["a", "b", "b"])).expect("gc");
        let mut blob = gc.to_blob_bytes("url").expect("blob");
        // Flip a bit in the IPC body (before the trailing CRC): must fail.
        blob[0] ^= 0x01;
        assert!(GroupedCount::from_blob_bytes(&blob).is_none());
    }

    #[test]
    fn blob_truncated_is_rejected() {
        assert!(GroupedCount::from_blob_bytes(&[]).is_none());
        assert!(GroupedCount::from_blob_bytes(&[0, 1]).is_none());
    }
}
