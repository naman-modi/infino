// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Clustering-key query path: the SQL scan's ordering declaration.
//!
//! On a clustered table the scan declares the writer's sort order
//! (ascending, nulls last) so DataFusion runs matching `GROUP BY`s in
//! sorted / partially-sorted input mode — but only when the surviving
//! superfiles' key ranges are provably disjoint. The core contract
//! under test is the *conditionality*: a false ordering declaration is
//! a wrong-results bug, so overlapping ranges (multi-commit tables
//! before an optimize) must plan exactly like today's unordered scan.
//!
//! - **Oracle**: grouped queries on a clustered table return the same
//!   rows as on an unclustered copy of the same data — duplicates,
//!   nulls in the key, multi-superfile commits, both key shapes
//!   (exact key and prefix + extra column), ordered path and fallback.
//! - **EXPLAIN**: a single-commit clustered table aggregates with
//!   `ordering_mode=Sorted` / `PartiallySorted`; an overlapping
//!   two-commit table and an unclustered table show no ordering mode.
//! - **Deletes**: tombstoned rows stay excluded under the ordered path
//!   (the per-file row selection skips rows *within* a file, which
//!   preserves the file's internal key order).

#![deny(clippy::unwrap_used)]

use std::{
    collections::HashSet,
    sync::Arc,
    time::{Duration, Instant},
};

use arrow_array::{Array, Int64Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use datafusion::{
    prelude::{col, lit},
    scalar::ScalarValue,
};
use infino::{
    CompactionSettings, OptimizeOptions,
    storage::{LocalFsStorageProvider, StorageProvider},
    supertable::{
        Supertable, SupertableOptions,
        reader_cache::{ColdFetchMode, DiskCacheConfig, DiskCacheStore, LruPolicy},
    },
};
use rayon::ThreadPoolBuilder;
use tempfile::TempDir;

/// Writer threads (= superfiles per commit) for multi-superfile tables.
const SHARD_THREADS: usize = 4;
/// Rows per commit in the oracle fixtures — enough that every shard
/// gets a non-trivial slice.
const ORACLE_ROWS: usize = 48;
/// Disk-cache byte budget for the delete fixture.
const DISK_CACHE_BUDGET_BYTES: u64 = 1 << 30;
/// Parallel cold-fetch streams for the test disk cache.
const COLD_FETCH_STREAMS: usize = 4;
/// Cold-fetch range chunk size (1 MiB).
const COLD_FETCH_CHUNK_BYTES: u64 = 1 << 20;
/// Background prefetch concurrency for the hybrid cache.
const PREFETCH_CONCURRENCY: usize = 8;
/// Mmap promotion timers disabled in tests (no idle eviction).
const MMAP_TIMER_DISABLED_SECS: u64 = 0;

/// `[category: LargeUtf8?, rank: Int64?, val: Int64?]` — a sortable
/// key column, a secondary key / extra grouping column, and a measure.
fn schema_category_rank_val() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("category", DataType::LargeUtf8, true),
        Field::new("rank", DataType::Int64, true),
        Field::new("val", DataType::Int64, true),
    ]))
}

/// Options over [`schema_category_rank_val`] with the given clustering
/// key (empty = unclustered) and writer-pool width.
fn options_with_key(cluster_by: &[&str], writer_threads: usize) -> SupertableOptions {
    let pool = Arc::new(
        ThreadPoolBuilder::new()
            .num_threads(writer_threads)
            .build()
            .expect("rayon pool builds"),
    );
    SupertableOptions::new(schema_category_rank_val(), vec![], vec![], None)
        .expect("valid options")
        .with_cluster_by(cluster_by.iter().map(|c| c.to_string()).collect())
        .expect("valid clustering key")
        .with_writer_pool(pool)
}

type Row = (Option<String>, Option<i64>, i64);

fn batch_rows(rows: &[Row]) -> RecordBatch {
    let categories = LargeStringArray::from(
        rows.iter()
            .map(|(c, _, _)| c.as_deref())
            .collect::<Vec<_>>(),
    );
    let ranks = Int64Array::from(rows.iter().map(|(_, r, _)| *r).collect::<Vec<_>>());
    let vals = Int64Array::from(rows.iter().map(|(_, _, v)| *v).collect::<Vec<_>>());
    RecordBatch::try_new(
        schema_category_rank_val(),
        vec![Arc::new(categories), Arc::new(ranks), Arc::new(vals)],
    )
    .expect("batch matches schema")
}

/// Commit each slice of `commits` as one commit on a fresh table.
fn table_with_commits(options: SupertableOptions, commits: &[Vec<Row>]) -> Supertable {
    let st = Supertable::create(options).expect("create");
    let mut w = st.writer().expect("writer");
    for rows in commits {
        w.append(&batch_rows(rows)).expect("append");
        w.commit().expect("commit");
    }
    drop(w);
    st
}

/// Flatten an `EXPLAIN` result into one searchable string.
fn explain(st: &Supertable, sql: &str) -> String {
    let batches = st
        .reader()
        .query_sql(&format!("EXPLAIN {sql}"))
        .expect("explain");
    let mut out = String::new();
    for batch in &batches {
        for column in batch.columns() {
            if let Some(strings) = column.as_any().downcast_ref::<arrow_array::StringArray>() {
                for i in 0..strings.len() {
                    if !strings.is_null(i) {
                        out.push_str(strings.value(i));
                        out.push('\n');
                    }
                }
            }
        }
    }
    out
}

/// All result rows of `sql`, each rendered `cell|cell|…`, sorted — an
/// order-insensitive result fingerprint for the oracle comparisons.
fn sorted_rows(st: &Supertable, sql: &str) -> Vec<String> {
    let batches = st.reader().query_sql(sql).expect("sql");
    let mut rows = Vec::new();
    for batch in &batches {
        for r in 0..batch.num_rows() {
            let cells: Vec<String> = batch
                .columns()
                .iter()
                .map(|c| {
                    ScalarValue::try_from_array(c, r)
                        .expect("printable cell")
                        .to_string()
                })
                .collect();
            rows.push(cells.join("|"));
        }
    }
    rows.sort();
    rows
}

/// Grouped-query shapes the ordering declaration targets: the exact
/// clustering key, and a prefix + extra column.
const ORACLE_QUERIES: &[&str] = &[
    "SELECT category, COUNT(*), SUM(val) FROM supertable GROUP BY category",
    "SELECT category, rank, COUNT(*), SUM(val) FROM supertable GROUP BY category, rank",
];

/// Assert every oracle query returns identical rows on both tables.
fn assert_same_results(clustered: &Supertable, unclustered: &Supertable) {
    for sql in ORACLE_QUERIES {
        let got = sorted_rows(clustered, sql);
        let expected = sorted_rows(unclustered, sql);
        assert!(!expected.is_empty(), "{sql}: oracle must produce rows");
        assert_eq!(got, expected, "{sql}: clustered result diverged");
    }
}

/// One commit's worth of oracle rows: duplicates (every category
/// repeats), nulls in both key columns, deterministic scramble so the
/// clustering sort actually permutes.
fn oracle_rows(commit: usize) -> Vec<Row> {
    (0..ORACLE_ROWS)
        .map(|i| {
            // 17 is coprime with 48: visits every slot, far from sorted.
            let slot = (i * 17) % ORACLE_ROWS;
            let category = match slot % 8 {
                7 => None, // nulls in the key
                c => Some(format!("cat{c}")),
            };
            let rank = match slot % 5 {
                4 => None, // nulls in the secondary column
                r => Some(r as i64),
            };
            (category, rank, (commit * 1000 + slot) as i64)
        })
        .collect()
}

/// Ordered path: a single clustered multi-superfile commit (disjoint
/// shard ranges by the writer contract) returns exactly the rows an
/// unclustered copy of the same data returns — duplicates, nulls in
/// the key, both grouping shapes.
#[test]
fn oracle_single_commit_matches_unclustered_copy() {
    let commits = vec![oracle_rows(0)];
    let clustered = table_with_commits(options_with_key(&["category"], SHARD_THREADS), &commits);
    let unclustered = table_with_commits(options_with_key(&[], SHARD_THREADS), &commits);
    assert_same_results(&clustered, &unclustered);
}

/// Fallback path: three commits with interleaved key ranges (every
/// commit spans all categories) overlap, so the scan must stay
/// unordered — and still return exactly the unclustered results.
#[test]
fn oracle_overlapping_commits_match_unclustered_copy() {
    let commits: Vec<Vec<Row>> = (0..3).map(oracle_rows).collect();
    let clustered = table_with_commits(options_with_key(&["category"], SHARD_THREADS), &commits);
    let unclustered = table_with_commits(options_with_key(&[], SHARD_THREADS), &commits);
    assert_same_results(&clustered, &unclustered);
}

/// Multi-column key: clustering by (category, rank) serves both the
/// full-key and the prefix grouping, results identical to unclustered.
#[test]
fn oracle_multi_column_key_matches_unclustered_copy() {
    let commits = vec![oracle_rows(0)];
    let clustered = table_with_commits(
        options_with_key(&["category", "rank"], SHARD_THREADS),
        &commits,
    );
    let unclustered = table_with_commits(options_with_key(&[], SHARD_THREADS), &commits);
    assert_same_results(&clustered, &unclustered);
}

/// Distinct, null-free keys for the EXPLAIN fixtures: `k00 … k15`,
/// scrambled so the writer sort does real work.
fn explain_rows(offset: usize) -> Vec<Row> {
    (0..16)
        .map(|i| {
            let slot = (i * 5) % 16; // coprime scramble
            (
                Some(format!("k{:02}", offset + slot)),
                Some(slot as i64),
                slot as i64,
            )
        })
        .collect()
}

/// EXPLAIN evidence, ordered path: on a single-commit clustered table
/// the scan declares its ordering and the aggregate runs in sorted
/// input mode — `Sorted` when grouping by the exact key,
/// `PartiallySorted` when the key is a prefix of the grouping.
#[test]
fn explain_single_commit_clustered_aggregates_in_sorted_mode() {
    let st = table_with_commits(
        options_with_key(&["category"], SHARD_THREADS),
        &[explain_rows(0)],
    );

    let plan = explain(
        &st,
        "SELECT category, SUM(val) FROM supertable GROUP BY category",
    );
    eprintln!("PLAN_EVIDENCE_SORTED:\n{plan}");
    assert!(
        plan.contains("output_ordering"),
        "scan must declare its sort order; plan was:\n{plan}"
    );
    assert!(
        plan.contains("ordering_mode=Sorted"),
        "exact-key GROUP BY must aggregate in sorted input mode; plan was:\n{plan}"
    );

    let plan = explain(
        &st,
        "SELECT category, rank, SUM(val) FROM supertable GROUP BY category, rank",
    );
    assert!(
        plan.contains("ordering_mode=PartiallySorted"),
        "prefix + extra-column GROUP BY must aggregate partially sorted; plan was:\n{plan}"
    );
}

/// EXPLAIN evidence, fallback: two commits with interleaved key ranges
/// overlap, so no ordering may be declared — the sorted input mode must
/// be absent (hash aggregation, exactly today's plan shape).
#[test]
fn explain_overlapping_commits_stay_unordered() {
    let st = table_with_commits(
        options_with_key(&["category"], SHARD_THREADS),
        // Same key universe in both commits → ranges interleave.
        &[explain_rows(0), explain_rows(0)],
    );
    let plan = explain(
        &st,
        "SELECT category, SUM(val) FROM supertable GROUP BY category",
    );
    assert!(
        !plan.contains("ordering_mode"),
        "overlapping ranges must not aggregate in sorted mode; plan was:\n{plan}"
    );
    assert!(
        !plan.contains("output_ordering"),
        "overlapping ranges must not declare a scan ordering; plan was:\n{plan}"
    );
}

/// No-regression guard: an unclustered table's plan carries neither an
/// ordering declaration nor a sorted aggregation mode.
#[test]
fn explain_unclustered_table_stays_unordered() {
    let st = table_with_commits(options_with_key(&[], SHARD_THREADS), &[explain_rows(0)]);
    let plan = explain(
        &st,
        "SELECT category, SUM(val) FROM supertable GROUP BY category",
    );
    assert!(
        !plan.contains("ordering_mode") && !plan.contains("output_ordering"),
        "unclustered plan must stay unordered; plan was:\n{plan}"
    );
}

fn make_disk_cache(
    storage: Arc<dyn StorageProvider>,
    cache_root: &std::path::Path,
) -> Arc<DiskCacheStore> {
    let cfg = DiskCacheConfig {
        cache_root: cache_root.to_path_buf(),
        disk_budget_bytes: DISK_CACHE_BUDGET_BYTES,
        cold_fetch_mode: ColdFetchMode::HybridWithPrefetch,
        cold_fetch_streams: COLD_FETCH_STREAMS,
        cold_fetch_chunk_bytes: COLD_FETCH_CHUNK_BYTES,
        prefetch_concurrency: PREFETCH_CONCURRENCY,
        mmap_cold_threshold_secs: MMAP_TIMER_DISABLED_SECS,
        mmap_sweep_interval_secs: MMAP_TIMER_DISABLED_SECS,
        eviction: Box::new(LruPolicy::new()),
        verify_crc_on_open: true,
    };
    let pinned: Arc<dyn Fn() -> HashSet<_> + Send + Sync> = Arc::new(HashSet::new);
    DiskCacheStore::new(storage, cfg, pinned).expect("cache")
}

/// Deletes under the ordered path: tombstoned rows are excluded from a
/// grouped query on a clustered table while the plan keeps its sorted
/// input mode. The tombstone row selection skips rows *within* each
/// superfile, which preserves the file's internal key order — so the
/// ordering declaration stays truthful with deletes applied.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn deletes_excluded_under_ordered_path() {
    let dir = TempDir::new().expect("tempdir");
    let cache_dir = TempDir::new().expect("cache");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let disk_cache = make_disk_cache(Arc::clone(&storage), cache_dir.path());
    let st = Supertable::create(
        options_with_key(&["category"], SHARD_THREADS)
            .with_storage(storage)
            .with_disk_cache(disk_cache),
    )
    .expect("create");

    let mut w = st.writer().expect("writer");
    w.append(&batch_rows(&explain_rows(0))).expect("append");
    w.commit().expect("commit");

    // Tombstone one whole key ("k03") plus assert the count drop.
    let pending = w.delete(col("category").eq(lit("k03"))).expect("delete");
    assert_eq!(pending.matched, 1);
    w.commit().expect("commit delete");
    drop(w);

    let sql = "SELECT category, COUNT(*), SUM(val) FROM supertable GROUP BY category";
    let rows = sorted_rows(&st, sql);
    assert_eq!(rows.len(), 15, "16 keys − 1 tombstoned key");
    assert!(
        rows.iter().all(|r| !r.starts_with("k03|")),
        "tombstoned key must not appear: {rows:?}"
    );

    // The ordered path must still be active with tombstones applied.
    let plan = explain(&st, sql);
    assert!(
        plan.contains("ordering_mode=Sorted"),
        "deletes must not disable the ordered scan; plan was:\n{plan}"
    );
}

/// Filters and LIMIT behave unchanged under the ordered path: a
/// predicate + limit query on a clustered table matches the unclustered
/// copy (modulo which rows a LIMIT admits — compare a deterministic
/// filtered aggregate and a bounded row count instead of exact rows).
#[test]
fn filters_and_limit_unchanged_under_ordered_path() {
    let commits = vec![oracle_rows(0)];
    let clustered = table_with_commits(options_with_key(&["category"], SHARD_THREADS), &commits);
    let unclustered = table_with_commits(options_with_key(&[], SHARD_THREADS), &commits);

    let filtered = "SELECT category, COUNT(*), SUM(val) FROM supertable \
                    WHERE rank >= 2 GROUP BY category";
    assert_eq!(
        sorted_rows(&clustered, filtered),
        sorted_rows(&unclustered, filtered),
        "filtered grouped results must match the unclustered copy"
    );

    let limited = "SELECT category FROM supertable WHERE rank >= 2 LIMIT 5";
    assert_eq!(
        sorted_rows(&clustered, limited).len(),
        5,
        "LIMIT must admit exactly its bound"
    );
}

// ---- optimize: global clustering across commits ---------------------

/// Commits per table in the optimize fixtures.
const OPTIMIZE_COMMITS: usize = 3;
/// Single-file commits in the multi-output optimize fixture.
const MULTI_OUTPUT_COMMITS: usize = 12;
/// Rows per commit in the multi-output fixture.
const MULTI_OUTPUT_ROWS_PER_COMMIT: usize = 3200;
/// Length of the incompressible key strings in the multi-output fixture.
const MULTI_OUTPUT_KEY_LEN: usize = 128;

/// Compaction sized for test superfiles: 1 MiB target, ~10 KiB fill floor.
fn small_optimize_opts() -> OptimizeOptions {
    OptimizeOptions::compact(CompactionSettings {
        target_superfile_size_mb: 1,
        min_fill_percent: 1,
        ..CompactionSettings::default()
    })
}

fn local_storage(dir: &TempDir) -> Arc<dyn StorageProvider> {
    Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"))
}

/// Optimize upgrades the fallback table to the ordered path: overlapping
/// commits plan unordered; the compaction merge re-sorts ALL live rows by
/// the key, so the surviving superfiles' ranges chain and the same query
/// now scans with a declared ordering and aggregates in sorted input mode.
#[test]
fn optimize_upgrades_overlapping_commits_to_the_ordered_path() {
    let dir = TempDir::new().expect("tempdir");
    // Same key universe in every commit → ranges interleave pre-optimize.
    let commits: Vec<Vec<Row>> = (0..OPTIMIZE_COMMITS).map(|_| explain_rows(0)).collect();
    let clustered = table_with_commits(
        options_with_key(&["category"], SHARD_THREADS).with_storage(local_storage(&dir)),
        &commits,
    );
    let unclustered = table_with_commits(options_with_key(&[], SHARD_THREADS), &commits);

    let sql = "SELECT category, SUM(val) FROM supertable GROUP BY category";
    let plan = explain(&clustered, sql);
    assert!(
        !plan.contains("ordering_mode") && !plan.contains("output_ordering"),
        "overlapping commits must plan unordered before optimize; plan was:\n{plan}"
    );

    clustered
        .optimize(&small_optimize_opts())
        .expect("optimize");

    let plan = explain(&clustered, sql);
    assert!(
        plan.contains("output_ordering"),
        "post-optimize scan must declare its sort order; plan was:\n{plan}"
    );
    assert!(
        plan.contains("ordering_mode=Sorted"),
        "post-optimize exact-key GROUP BY must aggregate sorted; plan was:\n{plan}"
    );
    let plan = explain(
        &clustered,
        "SELECT category, rank, SUM(val) FROM supertable GROUP BY category, rank",
    );
    assert!(
        plan.contains("ordering_mode=PartiallySorted"),
        "post-optimize prefix GROUP BY must aggregate partially sorted; plan was:\n{plan}"
    );

    assert_same_results(&clustered, &unclustered);
}

/// The optimize oracle under the messy shapes: duplicate keys, nulls in
/// the key, multi-superfile commits. Results after the global re-sort are
/// exactly the unclustered copy's.
#[test]
fn optimize_oracle_with_nulls_and_duplicates_matches_unclustered_copy() {
    let dir = TempDir::new().expect("tempdir");
    let commits: Vec<Vec<Row>> = (0..OPTIMIZE_COMMITS).map(oracle_rows).collect();
    let clustered = table_with_commits(
        options_with_key(&["category"], SHARD_THREADS).with_storage(local_storage(&dir)),
        &commits,
    );
    let unclustered = table_with_commits(options_with_key(&[], SHARD_THREADS), &commits);

    clustered
        .optimize(&small_optimize_opts())
        .expect("optimize");
    assert_same_results(&clustered, &unclustered);
}

/// Tombstones and the optimize-time sort compose: the merged table holds
/// only live rows, and the rewritten superfile still carries the ordering
/// the scan declares.
#[test]
fn optimize_with_deletes_keeps_only_live_rows_on_the_ordered_path() {
    let dir = TempDir::new().expect("tempdir");
    let st = table_with_commits(
        options_with_key(&["category"], SHARD_THREADS).with_storage(local_storage(&dir)),
        &[explain_rows(0), explain_rows(0)],
    );

    let mut w = st.writer().expect("writer");
    let pending = w.delete(col("category").eq(lit("k03"))).expect("delete");
    assert_eq!(pending.matched, 2, "one k03 row per commit");
    w.commit().expect("commit delete");
    drop(w);

    let before = st.reader().n_superfiles();
    st.optimize(&small_optimize_opts()).expect("optimize");
    assert!(
        st.reader().n_superfiles() < before,
        "optimize must have merged the commits"
    );

    let sql = "SELECT category, COUNT(*), SUM(val) FROM supertable GROUP BY category";
    let rows = sorted_rows(&st, sql);
    assert_eq!(rows.len(), 15, "16 keys − 1 tombstoned key");
    assert!(
        rows.iter().all(|r| !r.starts_with("k03|")),
        "tombstoned key must not survive the merge: {rows:?}"
    );
    let plan = explain(&st, sql);
    assert!(
        plan.contains("ordering_mode=Sorted"),
        "merged live rows must still scan ordered; plan was:\n{plan}"
    );
}

/// Seed scrambler for [`incompressible_key`] (the splitmix64 increment,
/// a fixed-point-free odd multiplier that spreads consecutive seeds).
const KEY_SEED_MULTIPLIER: u64 = 0x9E37_79B9_7F4A_7C15;

/// Deterministic pseudo-random hex so parquet can't compress the key
/// column much: the multi-output fixture needs file sizes near their raw
/// size so optimize actually packs several target-sized outputs.
fn incompressible_key(seed: u64) -> String {
    // xorshift64 over the scrambled seed (forced odd so state is nonzero).
    let mut x = seed.wrapping_mul(KEY_SEED_MULTIPLIER) | 1;
    let mut out = String::with_capacity(MULTI_OUTPUT_KEY_LEN + 16);
    while out.len() < MULTI_OUTPUT_KEY_LEN {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        out.push_str(&format!("{x:016x}"));
    }
    out.truncate(MULTI_OUTPUT_KEY_LEN);
    out
}

/// When the merged rows exceed one target-sized superfile, optimize slices
/// ONE globally sorted run into consecutive outputs — so even the
/// multi-file result carries chained, non-overlapping key ranges and the
/// ordered path fires. Independently sorted outputs would overlap and pin
/// the plan to the unordered fallback.
#[test]
fn optimize_multiple_outputs_carry_disjoint_ranges() {
    let dir = TempDir::new().expect("tempdir");
    let commits: Vec<Vec<Row>> = (0..MULTI_OUTPUT_COMMITS)
        .map(|commit| {
            (0..MULTI_OUTPUT_ROWS_PER_COMMIT)
                .map(|i| {
                    let seed = (commit * MULTI_OUTPUT_ROWS_PER_COMMIT + i) as u64;
                    (Some(incompressible_key(seed)), Some(i as i64), i as i64)
                })
                .collect()
        })
        .collect();
    // One writer thread → one superfile per commit, ~0.4 MiB of raw key
    // bytes each; a 1 MiB target forces several outputs.
    let st = table_with_commits(
        options_with_key(&["category"], 1).with_storage(local_storage(&dir)),
        &commits,
    );

    let before = st.reader().n_superfiles();
    assert_eq!(before, MULTI_OUTPUT_COMMITS);
    st.optimize(&small_optimize_opts()).expect("optimize");
    let after = st.reader().n_superfiles();
    assert!(
        (2..before).contains(&after),
        "expected several merged outputs, got {after} superfiles"
    );

    // The ordered declaration only fires when EVERY surviving file's key
    // range chains without overlap — the plan is the disjointness proof.
    let plan = explain(
        &st,
        "SELECT category, SUM(val) FROM supertable GROUP BY category",
    );
    assert!(
        plan.contains("output_ordering") && plan.contains("ordering_mode=Sorted"),
        "multi-output optimize must keep the ordered path; plan was:\n{plan}"
    );

    let rows = sorted_rows(&st, "SELECT COUNT(*) FROM supertable");
    let expected_rows = (MULTI_OUTPUT_COMMITS * MULTI_OUTPUT_ROWS_PER_COMMIT).to_string();
    assert_eq!(rows, vec![expected_rows], "no rows lost across the split");
}

// ---- optimize past the compaction memory ceiling ---------------------

/// Rows per commit in the streaming-optimize fixtures.
const STREAMING_ROWS_PER_COMMIT: usize = 3200;
/// Single-file commits in the streaming-optimize fixtures.
const STREAMING_COMMITS: usize = 6;
/// Distinct key values: far fewer than the row count, so every key
/// repeats across rows AND commits, while the dictionary of
/// incompressible 128-byte keys keeps each file's size well past what
/// the 1 MiB ceiling admits for an in-memory sort.
const STREAMING_DISTINCT_KEYS: u64 = 1600;
/// One row in this many carries a null key.
const STREAMING_NULL_STRIDE: usize = 97;

/// Optimize options whose compaction memory ceiling (1 MiB) every
/// fixture job exceeds — raw inputs total several MiB, so the clustered
/// merge must take the streaming route (its in-memory reserve of three
/// times the input bytes cannot fit), and with per-file reserves near
/// the ceiling the fan-in admission also forces a multi-fold cascade.
fn streaming_optimize_opts() -> OptimizeOptions {
    OptimizeOptions::compact(CompactionSettings {
        target_superfile_size_mb: 1,
        min_fill_percent: 1,
        max_memory_mb: 1,
        ..CompactionSettings::default()
    })
}

/// One commit of the streaming fixture: duplicate incompressible keys,
/// nulls in the key, unique measures, scrambled order. Every commit
/// spans the same key universe, so pre-optimize ranges interleave.
fn streaming_rows(commit: usize) -> Vec<Row> {
    (0..STREAMING_ROWS_PER_COMMIT)
        .map(|i| {
            // Coprime stride visits every slot far from sorted order.
            let slot = (i * 2011) % STREAMING_ROWS_PER_COMMIT;
            let key = if slot.is_multiple_of(STREAMING_NULL_STRIDE) {
                None
            } else {
                Some(incompressible_key(slot as u64 % STREAMING_DISTINCT_KEYS))
            };
            (
                key,
                Some((slot % 5) as i64),
                (commit * STREAMING_ROWS_PER_COMMIT + slot) as i64,
            )
        })
        .collect()
}

/// Optimize past the ceiling: the streaming merge produces globally
/// ordered, range-disjoint outputs, so the same grouped queries that
/// planned unordered before now scan with a declared ordering and
/// aggregate in sorted input mode — and still return exactly what an
/// unclustered copy of the same data returns, duplicates and null keys
/// included.
#[test]
fn optimize_streams_over_ceiling_jobs_onto_the_ordered_path() {
    let dir = TempDir::new().expect("tempdir");
    let commits: Vec<Vec<Row>> = (0..STREAMING_COMMITS).map(streaming_rows).collect();
    // One writer thread → one superfile per commit → six overlapping
    // sorted runs for the streaming merge to combine.
    let clustered = table_with_commits(
        options_with_key(&["category"], 1).with_storage(local_storage(&dir)),
        &commits,
    );
    let unclustered = table_with_commits(options_with_key(&[], 1), &commits);

    let sql = "SELECT category, SUM(val) FROM supertable GROUP BY category";
    let plan = explain(&clustered, sql);
    assert!(
        !plan.contains("ordering_mode") && !plan.contains("output_ordering"),
        "overlapping commits must plan unordered before optimize; plan was:\n{plan}"
    );

    let before = clustered.reader().n_superfiles();
    assert_eq!(before, STREAMING_COMMITS);
    clustered
        .optimize(&streaming_optimize_opts())
        .expect("optimize past the compaction memory ceiling");
    let after = clustered.reader().n_superfiles();
    assert!(
        (1..before).contains(&after),
        "expected merged outputs, got {after} superfiles"
    );

    // The ordered declaration only fires when EVERY surviving file's
    // key range chains without overlap — the plan is the proof that the
    // streaming outputs are globally ordered and pairwise disjoint.
    let plan = explain(&clustered, sql);
    assert!(
        plan.contains("output_ordering") && plan.contains("ordering_mode=Sorted"),
        "streamed optimize must land on the ordered path; plan was:\n{plan}"
    );

    let rows = sorted_rows(&clustered, "SELECT COUNT(*) FROM supertable");
    let expected_rows = (STREAMING_COMMITS * STREAMING_ROWS_PER_COMMIT).to_string();
    assert_eq!(
        rows,
        vec![expected_rows],
        "no rows lost through the cascade"
    );
    assert_same_results(&clustered, &unclustered);
}

/// Tombstones compose with the streaming merge: rows deleted before the
/// over-ceiling optimize are absent from every output, and the
/// survivors still scan on the ordered path.
#[test]
fn optimize_streaming_excludes_tombstones_on_the_ordered_path() {
    let dir = TempDir::new().expect("tempdir");
    let commits: Vec<Vec<Row>> = (0..STREAMING_COMMITS).map(streaming_rows).collect();
    let st = table_with_commits(
        options_with_key(&["category"], 1).with_storage(local_storage(&dir)),
        &commits,
    );

    // Every commit carries this key once (slot 1 of the key universe).
    let doomed = incompressible_key(1);
    let mut w = st.writer().expect("writer");
    let pending = w
        .delete(col("category").eq(lit(doomed.as_str())))
        .expect("delete");
    assert!(
        pending.matched as usize >= STREAMING_COMMITS,
        "the doomed key must appear in every commit"
    );
    w.commit().expect("commit delete");
    drop(w);

    st.optimize(&streaming_optimize_opts())
        .expect("optimize past the compaction memory ceiling");

    let sql = "SELECT category, COUNT(*), SUM(val) FROM supertable GROUP BY category";
    let rows = sorted_rows(&st, sql);
    assert!(
        rows.iter().all(|r| !r.starts_with(&format!("{doomed}|"))),
        "tombstoned key must not survive the streaming merge"
    );
    let plan = explain(&st, sql);
    assert!(
        plan.contains("ordering_mode=Sorted"),
        "survivors must still scan ordered; plan was:\n{plan}"
    );
}

// ---- optimize: convergence at commit scale ---------------------------

/// Single-file commits in the convergence fixture: enough that the one
/// coalesced merge job spans many more inputs than one streaming pass
/// admits, like a bulk load's per-commit superfile trail.
const CONVERGENCE_COMMITS: usize = 64;
/// Rows per commit in the convergence fixture.
const CONVERGENCE_ROWS_PER_COMMIT: usize = 800;
/// The field shape's ceiling-to-target ratio (256 MiB target,
/// 256+2048 MiB ceiling), scaled to the 1 MiB test target.
const CONVERGENCE_MAX_MEMORY_MB: u64 = 9;

/// The field OptimizeOptions shape at test scale: 1 MiB target (256 MiB
/// in the field), 1% fill floor, ceiling 9x the target (2304/256).
fn convergence_optimize_opts() -> OptimizeOptions {
    OptimizeOptions::compact(CompactionSettings {
        target_superfile_size_mb: 1,
        min_fill_percent: 1,
        max_memory_mb: CONVERGENCE_MAX_MEMORY_MB,
        ..CompactionSettings::default()
    })
}

/// One commit of the convergence fixture: every commit spans the same
/// key universe (ranges fully interleave), keys are incompressible so
/// file sizes track raw size, measures are globally unique.
fn convergence_rows(commit: usize) -> Vec<Row> {
    (0..CONVERGENCE_ROWS_PER_COMMIT)
        .map(|i| {
            let slot = (i * 389) % CONVERGENCE_ROWS_PER_COMMIT; // coprime scramble
            (
                Some(incompressible_key(slot as u64)),
                Some((slot % 7) as i64),
                (commit * CONVERGENCE_ROWS_PER_COMMIT + slot) as i64,
            )
        })
        .collect()
}

/// A bulk load's per-commit superfile trail must converge in ONE
/// optimize call: every small clustered file merged into a handful of
/// target-sized outputs whose key ranges chain without overlap, so the
/// ordered scan fires. Non-convergence (many small files left, plan
/// unordered) is the field failure this test pins.
#[test]
fn optimize_converges_many_small_commits_in_one_call() {
    let dir = TempDir::new().expect("tempdir");
    let commits: Vec<Vec<Row>> = (0..CONVERGENCE_COMMITS).map(convergence_rows).collect();
    let clustered = table_with_commits(
        options_with_key(&["category"], 1).with_storage(local_storage(&dir)),
        &commits,
    );
    let unclustered = table_with_commits(options_with_key(&[], 1), &commits);

    let before = clustered.reader().n_superfiles();
    assert_eq!(before, CONVERGENCE_COMMITS);

    clustered
        .optimize(&convergence_optimize_opts())
        .expect("one optimize call");

    // Convergence: nothing mergeable is left behind. The merged rows
    // total a few MiB against a 1 MiB target, so the surviving file
    // count must be a small multiple of the target count, not the
    // commit count.
    let after = clustered.reader().n_superfiles();
    eprintln!("CONVERGENCE_EVIDENCE: before={before} after={after}");
    let max_converged = CONVERGENCE_COMMITS / 4;
    assert!(
        after <= max_converged,
        "one optimize call must converge {before} small files, got {after} \
         (> {max_converged})"
    );

    // Disjointness proof: the ordered declaration only fires when EVERY
    // surviving file's key range chains without overlap.
    let sql = "SELECT category, SUM(val) FROM supertable GROUP BY category";
    let plan = explain(&clustered, sql);
    assert!(
        plan.contains("output_ordering") && plan.contains("ordering_mode=Sorted"),
        "converged table must scan on the ordered path; plan was:\n{plan}"
    );

    let rows = sorted_rows(&clustered, "SELECT COUNT(*) FROM supertable");
    let expected_rows = (CONVERGENCE_COMMITS * CONVERGENCE_ROWS_PER_COMMIT).to_string();
    assert_eq!(rows, vec![expected_rows], "no rows lost by the merge");
    assert_same_results(&clustered, &unclustered);
}

/// Rows per commit in the big-commit breakdown shape — same total rows
/// as the convergence fixture packed into far fewer commits, so the
/// per-commit sort works on a field-like larger buffer.
const BREAKDOWN_BIG_COMMIT_ROWS: usize = 6400;
/// Commits in the big-commit breakdown shape.
const BREAKDOWN_BIG_COMMITS: usize = 8;

/// Load the given commits into a fresh table and return the total
/// append+commit wall time.
fn timed_load(st: &Supertable, commits: &[Vec<Row>]) -> Duration {
    let started = Instant::now();
    let mut w = st.writer().expect("writer");
    for rows in commits {
        w.append(&batch_rows(rows)).expect("append");
        w.commit().expect("commit");
    }
    drop(w);
    started.elapsed()
}

/// One breakdown shape: load the same data clustered and unclustered,
/// optimize the clustered table, print the timings, and check no rows
/// were lost.
fn run_load_breakdown(label: &str, n_commits: usize, rows_per_commit: usize) {
    let commits: Vec<Vec<Row>> = (0..n_commits)
        .map(|commit| {
            (0..rows_per_commit)
                .map(|i| {
                    let slot = (i * 389) % rows_per_commit; // coprime scramble
                    (
                        Some(incompressible_key(slot as u64)),
                        Some((slot % 7) as i64),
                        (commit * rows_per_commit + slot) as i64,
                    )
                })
                .collect()
        })
        .collect();

    let dir = TempDir::new().expect("tempdir");
    let clustered =
        Supertable::create(options_with_key(&["category"], 1).with_storage(local_storage(&dir)))
            .expect("create clustered");
    let clustered_load = timed_load(&clustered, &commits);

    let unclustered_dir = TempDir::new().expect("tempdir");
    let unclustered =
        Supertable::create(options_with_key(&[], 1).with_storage(local_storage(&unclustered_dir)))
            .expect("create unclustered");
    let unclustered_load = timed_load(&unclustered, &commits);

    let started = Instant::now();
    clustered
        .optimize(&convergence_optimize_opts())
        .expect("optimize");
    let optimize_elapsed = started.elapsed();

    eprintln!(
        "LOAD_BREAKDOWN[{label}]: commits={n_commits} rows_per_commit={rows_per_commit} \
         clustered_load={clustered_load:?} unclustered_load={unclustered_load:?} \
         per_commit_clustering_overhead={:?} optimize={optimize_elapsed:?}",
        clustered_load
            .saturating_sub(unclustered_load)
            .div_f64(n_commits as f64),
    );

    let rows = sorted_rows(&clustered, "SELECT COUNT(*) FROM supertable");
    let expected_rows = (n_commits * rows_per_commit).to_string();
    assert_eq!(rows, vec![expected_rows], "no rows lost across the load");
}

/// Diagnostic breakdown of where a clustered bulk load's time goes at
/// commit scale: total commit wall time on a clustered table vs an
/// unclustered copy of the same data (the delta is the per-commit
/// clustering sort and its stats), plus the one-call optimize cost —
/// measured at two commit shapes (many small commits, few big ones) so
/// the sort cost's scaling with commit size is visible. Timings are
/// printed as evidence (run with `--nocapture`), never asserted — only
/// the row-count invariant is.
#[test]
fn load_time_breakdown_clustered_vs_unclustered_commits() {
    run_load_breakdown(
        "many-small",
        CONVERGENCE_COMMITS,
        CONVERGENCE_ROWS_PER_COMMIT,
    );
    run_load_breakdown("few-big", BREAKDOWN_BIG_COMMITS, BREAKDOWN_BIG_COMMIT_ROWS);
}

/// The tiny ceiling never bends the unclustered path: optimize still
/// merges (concatenation order, no sort), the plan stays exactly as
/// unordered as before, and no rows are lost — the streaming route is a
/// clustered-table concern only.
#[test]
fn optimize_with_tiny_ceiling_leaves_unclustered_path_unchanged() {
    let dir = TempDir::new().expect("tempdir");
    let commits: Vec<Vec<Row>> = (0..STREAMING_COMMITS).map(streaming_rows).collect();
    let st = table_with_commits(
        options_with_key(&[], 1).with_storage(local_storage(&dir)),
        &commits,
    );

    let before = st.reader().n_superfiles();
    assert_eq!(before, STREAMING_COMMITS);
    st.optimize(&streaming_optimize_opts())
        .expect("unclustered optimize under the tiny ceiling");
    assert!(
        st.reader().n_superfiles() < before,
        "unclustered optimize must still merge superfiles"
    );

    let sql = "SELECT category, SUM(val) FROM supertable GROUP BY category";
    let plan = explain(&st, sql);
    assert!(
        !plan.contains("ordering_mode") && !plan.contains("output_ordering"),
        "unclustered plans must stay unordered after optimize; plan was:\n{plan}"
    );
    let rows = sorted_rows(&st, "SELECT COUNT(*) FROM supertable");
    let expected_rows = (STREAMING_COMMITS * STREAMING_ROWS_PER_COMMIT).to_string();
    assert_eq!(rows, vec![expected_rows], "no rows lost by the merge");
}
