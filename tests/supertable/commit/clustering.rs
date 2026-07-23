// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Clustering-key write path.
//!
//! A table created with a clustering key physically sorts each
//! commit's rows by the key (lexicographic on the column list,
//! ascending, nulls last) before the shard split, so:
//!
//! - every superfile a commit produces is internally sorted by the
//!   key (including a multi-column key with duplicates and nulls);
//! - a multi-shard commit stays sorted per shard AND the shards
//!   partition the key space contiguously (global commit order);
//! - vector payloads stay row-aligned through the permutation;
//! - the default no-key path preserves append order exactly;
//! - BM25 / vector results are unaffected by clustered ingestion.
//!
//! Row order is asserted by opening each committed superfile's
//! bytes directly and decoding all rows in file order.

#![deny(clippy::unwrap_used)]

use std::{fs, path::Path, sync::Arc};

use arrow_array::{
    Array, ArrayRef, Decimal128Array, FixedSizeListArray, Float32Array, Int64Array,
    LargeStringArray, RecordBatch,
};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use infino::{
    VectorSearchOptions,
    superfile::{SuperfileReader, builder::FtsConfig},
    supertable::{
        Supertable, SupertableOptions,
        query::fts::BoolMode,
        storage::{LocalFsStorageProvider, StorageProvider},
    },
    test_helpers::{default_tokenizer, default_vector_config},
};
use rayon::ThreadPoolBuilder;
use tempfile::TempDir;

/// Vector fixture dimensionality (`default_vector_config` shape).
const DIM: usize = 16;
/// Top-k for the search smoke checks.
const SEARCH_TOP_K: usize = 5;
/// Rows in the multi-shard commit; comfortably above the shard count
/// so every shard gets a non-trivial slice.
const MULTI_SHARD_ROWS: usize = 40;
/// Writer threads (= shards) for the multi-shard commit.
const MULTI_SHARD_THREADS: usize = 4;

/// User schema `[category: LargeUtf8?, rank: Int64?]` — two sortable
/// scalar columns, both nullable so the nulls-last contract is
/// observable.
fn schema_category_rank() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("category", DataType::LargeUtf8, true),
        Field::new("rank", DataType::Int64, true),
    ]))
}

/// Options over [`schema_category_rank`] with the given clustering
/// key (empty = unclustered) and writer-pool width.
fn options_with_key(cluster_by: &[&str], writer_threads: usize) -> SupertableOptions {
    let pool = Arc::new(
        ThreadPoolBuilder::new()
            .num_threads(writer_threads)
            .build()
            .expect("rayon pool builds"),
    );
    SupertableOptions::new(schema_category_rank(), vec![], vec![], None)
        .expect("valid options")
        .with_cluster_by(cluster_by.iter().map(|c| c.to_string()).collect())
        .expect("valid clustering key")
        .with_writer_pool(pool)
}

fn batch_category_rank(rows: &[(Option<&str>, Option<i64>)]) -> RecordBatch {
    let categories = LargeStringArray::from(rows.iter().map(|(c, _)| *c).collect::<Vec<_>>());
    let ranks = Int64Array::from(rows.iter().map(|(_, r)| *r).collect::<Vec<_>>());
    RecordBatch::try_new(
        schema_category_rank(),
        vec![Arc::new(categories), Arc::new(ranks)],
    )
    .expect("batch matches schema")
}

/// Decode every committed superfile under `<root>/data/` to one
/// RecordBatch each, rows in file order.
fn read_committed_superfiles(root: &Path) -> Vec<RecordBatch> {
    let data_dir = root.join("data");
    let mut paths: Vec<_> = fs::read_dir(&data_dir)
        .expect("read data dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .collect();
    paths.sort();
    paths
        .iter()
        .map(|p| {
            let bytes = Bytes::from(fs::read(p).expect("read superfile bytes"));
            SuperfileReader::open(bytes)
                .expect("open superfile")
                .get_record_batch(None)
                .expect("decode rows")
        })
        .collect()
}

/// `(category, rank)` per row, in the batch's physical order.
fn category_rank_rows(batch: &RecordBatch) -> Vec<(Option<String>, Option<i64>)> {
    let categories = batch
        .column_by_name("category")
        .expect("category column")
        .as_any()
        .downcast_ref::<LargeStringArray>()
        .expect("category is LargeUtf8");
    let ranks = batch
        .column_by_name("rank")
        .expect("rank column")
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("rank is Int64");
    (0..batch.num_rows())
        .map(|i| {
            (
                categories.is_valid(i).then(|| categories.value(i).into()),
                ranks.is_valid(i).then(|| ranks.value(i)),
            )
        })
        .collect()
}

/// Comparable form of one row under the clustering contract:
/// per column ascending with nulls LAST — `(is_null, value)` tuples
/// order exactly that way.
type NullsLastKey = (bool, Option<String>, bool, Option<i64>);

fn nulls_last_key(row: &(Option<String>, Option<i64>)) -> NullsLastKey {
    (row.0.is_none(), row.0.clone(), row.1.is_none(), row.1)
}

fn assert_rows_sorted(rows: &[(Option<String>, Option<i64>)], context: &str) {
    for pair in rows.windows(2) {
        assert!(
            nulls_last_key(&pair[0]) <= nulls_last_key(&pair[1]),
            "{context}: rows out of key order: {:?} then {:?}",
            pair[0],
            pair[1]
        );
    }
}

#[test]
fn single_column_key_sorts_rows_within_superfile() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st = Supertable::create(options_with_key(&["category"], 1).with_storage(storage))
        .expect("create");

    // Two appends, both out of key order — the sort spans the whole
    // buffered commit, not just one append call.
    let mut w = st.writer().expect("writer");
    w.append(&batch_category_rank(&[
        (Some("delta"), Some(4)),
        (Some("alpha"), Some(1)),
        (Some("echo"), Some(5)),
        (Some("bravo"), Some(2)),
    ]))
    .expect("append 1");
    w.append(&batch_category_rank(&[
        (Some("charlie"), Some(3)),
        (Some("aardvark"), Some(0)),
    ]))
    .expect("append 2");
    w.commit().expect("commit");
    drop(w);

    let superfiles = read_committed_superfiles(dir.path());
    assert_eq!(superfiles.len(), 1, "1-thread pool ⇒ one superfile");
    let rows = category_rank_rows(&superfiles[0]);
    let categories: Vec<_> = rows.iter().map(|(c, _)| c.clone()).collect();
    assert_eq!(
        categories,
        ["aardvark", "alpha", "bravo", "charlie", "delta", "echo"]
            .iter()
            .map(|s| Some(s.to_string()))
            .collect::<Vec<_>>(),
        "rows must be sorted by the key across all buffered appends"
    );
}

#[test]
fn multi_column_key_orders_duplicates_and_puts_nulls_last() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st = Supertable::create(options_with_key(&["category", "rank"], 1).with_storage(storage))
        .expect("create");

    let mut w = st.writer().expect("writer");
    w.append(&batch_category_rank(&[
        (Some("m"), None), // null rank sorts after m's present ranks
        (None, Some(0)),   // null category sorts after every category
        (Some("m"), Some(9)),
        (Some("a"), Some(2)),
        (Some("m"), Some(1)),
        (None, None),
        (Some("a"), Some(1)),
    ]))
    .expect("append");
    w.commit().expect("commit");
    drop(w);

    let superfiles = read_committed_superfiles(dir.path());
    assert_eq!(superfiles.len(), 1);
    let rows = category_rank_rows(&superfiles[0]);
    let expected: Vec<(Option<String>, Option<i64>)> = vec![
        (Some("a".into()), Some(1)),
        (Some("a".into()), Some(2)),
        (Some("m".into()), Some(1)),
        (Some("m".into()), Some(9)),
        (Some("m".into()), None),
        (None, Some(0)),
        (None, None),
    ];
    assert_eq!(
        rows, expected,
        "lexicographic on (category, rank), nulls last per column"
    );
}

#[test]
fn default_no_key_path_preserves_append_order() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st = Supertable::create(options_with_key(&[], 1).with_storage(storage)).expect("create");

    // Deliberately out-of-key-order input: with no clustering key the
    // commit must keep exactly this order.
    let input: Vec<(Option<&str>, Option<i64>)> = vec![
        (Some("zulu"), Some(3)),
        (Some("alpha"), None),
        (None, Some(7)),
        (Some("mike"), Some(1)),
    ];
    let mut w = st.writer().expect("writer");
    w.append(&batch_category_rank(&input)).expect("append");
    w.commit().expect("commit");
    drop(w);

    let superfiles = read_committed_superfiles(dir.path());
    assert_eq!(superfiles.len(), 1);
    let rows = category_rank_rows(&superfiles[0]);
    let expected: Vec<(Option<String>, Option<i64>)> = input
        .iter()
        .map(|(c, r)| (c.map(String::from), *r))
        .collect();
    assert_eq!(rows, expected, "unclustered commit keeps append order");

    // The injected ids are minted in append order, so order
    // preservation is also visible as an ascending `_id` run.
    let ids = superfiles[0]
        .column_by_name("_id")
        .expect("_id column")
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .expect("_id is Decimal128")
        .values()
        .to_vec();
    assert!(
        ids.windows(2).all(|w| w[0] < w[1]),
        "append order ⇒ strictly ascending ids"
    );
}

#[test]
fn multi_shard_commit_stays_sorted_per_shard_and_across_shards() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st = Supertable::create(
        options_with_key(&["category"], MULTI_SHARD_THREADS).with_storage(storage),
    )
    .expect("create");

    // A deterministic scramble of c00..c39 (17 is coprime with 40, so
    // the walk visits every value once, far from sorted order).
    let scrambled: Vec<String> = (0..MULTI_SHARD_ROWS)
        .map(|i| format!("c{:02}", (i * 17) % MULTI_SHARD_ROWS))
        .collect();
    let rows: Vec<(Option<&str>, Option<i64>)> = scrambled
        .iter()
        .map(|c| (Some(c.as_str()), Some(0)))
        .collect();
    let mut w = st.writer().expect("writer");
    w.append(&batch_category_rank(&rows)).expect("append");
    w.commit().expect("commit");
    drop(w);

    let superfiles = read_committed_superfiles(dir.path());
    assert_eq!(
        superfiles.len(),
        MULTI_SHARD_THREADS,
        "one superfile per writer-pool shard"
    );

    // Each shard is internally sorted, and the shards' key ranges are
    // pairwise disjoint — the contiguous split of one globally-sorted
    // run. Together the two properties pin the global commit order.
    let mut ranges: Vec<(String, String)> = Vec::new();
    let mut total_rows = 0;
    for (i, superfile) in superfiles.iter().enumerate() {
        let rows = category_rank_rows(superfile);
        assert!(!rows.is_empty(), "shard {i} must carry rows");
        assert_rows_sorted(&rows, &format!("shard {i}"));
        total_rows += rows.len();
        let first = rows[0].0.clone().expect("no nulls in this fixture");
        let last = rows[rows.len() - 1]
            .0
            .clone()
            .expect("no nulls in this fixture");
        ranges.push((first, last));
    }
    assert_eq!(total_rows, MULTI_SHARD_ROWS, "no rows lost or duplicated");
    ranges.sort();
    for pair in ranges.windows(2) {
        assert!(
            pair[0].1 < pair[1].0,
            "shard key ranges must not overlap: {:?} vs {:?}",
            pair[0],
            pair[1]
        );
    }
}

#[test]
fn bm25_results_unaffected_by_clustered_ingestion() {
    // FTS schema clustered by its text column: ingestion reorders the
    // rows, search results must be the same as an unclustered table's.
    let schema = Arc::new(Schema::new(vec![Field::new(
        "title",
        DataType::LargeUtf8,
        false,
    )]));
    let opts = SupertableOptions::new(
        Arc::clone(&schema),
        vec![FtsConfig {
            column: "title".into(),
            positions: false,
        }],
        vec![],
        Some(default_tokenizer()),
    )
    .expect("valid options")
    .with_cluster_by(vec!["title".into()])
    .expect("valid clustering key");
    let st = Supertable::create(opts).expect("create");

    let titles = ["zebra stripes", "nimblefox special token", "apple orchard"];
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(LargeStringArray::from(titles.to_vec()))],
    )
    .expect("batch");
    let mut w = st.writer().expect("writer");
    w.append(&batch).expect("append");
    w.commit().expect("commit");
    drop(w);

    let hits = st
        .reader()
        .bm25_hits("title", "nimblefox", SEARCH_TOP_K, BoolMode::Or)
        .expect("bm25 over clustered table");
    assert_eq!(hits.len(), 1, "the unique term matches exactly one doc");
}

#[test]
fn vector_search_unaffected_and_vectors_stay_row_aligned() {
    // Vector table clustered by a scalar column. The sort permutes the
    // commit's rows, so this doubles as the row-alignment check: each
    // row's one-hot embedding must still land NEXT TO its own scalar
    // values, or the top-1 lookup below would return the wrong category.
    let schema = Arc::new(Schema::new(vec![
        Field::new("category", DataType::LargeUtf8, false),
        Field::new(
            "emb",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                DIM as i32,
            ),
            false,
        ),
    ]));
    let opts = SupertableOptions::new(
        Arc::clone(&schema),
        vec![],
        vec![default_vector_config("emb", 0)],
        None,
    )
    .expect("valid options")
    .with_cluster_by(vec!["category".into()])
    .expect("valid clustering key");
    let st = Supertable::create(opts).expect("create");

    // Reverse-alphabetical categories so the clustering sort reverses
    // the append order; row i carries the one-hot embedding at dim i.
    let categories = ["zulu", "yankee", "xray", "whiskey"];
    let n = categories.len();
    let mut flat = vec![0.0f32; n * DIM];
    for (i, row) in flat.chunks_mut(DIM).enumerate() {
        row[i] = 1.0;
    }
    let fsl = FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, true)),
        DIM as i32,
        Arc::new(Float32Array::from(flat)) as ArrayRef,
        None,
    )
    .expect("FSL");
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(LargeStringArray::from(categories.to_vec())),
            Arc::new(fsl),
        ],
    )
    .expect("batch");
    let mut w = st.writer().expect("writer");
    w.append(&batch).expect("append");
    w.commit().expect("commit");
    drop(w);

    // One-hot queries are mutually orthogonal, so top-1 for query i is
    // exactly the row appended with category[i].
    for (i, expected_category) in categories.iter().enumerate() {
        let mut query = vec![0.0f32; DIM];
        query[i] = 1.0;
        let batches = st
            .reader()
            .vector_search(
                "emb",
                &query,
                1,
                VectorSearchOptions::new().with_nprobe(DIM),
                None,
                Some(&["category"]),
            )
            .expect("vector search over clustered table");
        let top = batches.first().expect("one batch");
        assert_eq!(top.num_rows(), 1);
        let got = top
            .column_by_name("category")
            .expect("projected category")
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("category is LargeUtf8")
            .value(0)
            .to_string();
        assert_eq!(
            got, *expected_category,
            "row {i}'s vector must stay aligned with its scalar row"
        );
    }
}
