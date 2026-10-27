// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! A SQL scan opens at most its cap of files at once, and the cap changes
//! neither what the scan returns nor what it reads.

#![deny(clippy::unwrap_used)]

use std::{
    collections::HashMap,
    ops::Range,
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use arrow::{
    array::{Int64Array, RecordBatch},
    compute::concat_batches,
};
use async_trait::async_trait;
use bytes::Bytes;
use infino::{
    supertable::{
        Supertable,
        storage::{LocalFsStorageProvider, ObjectMeta, StorageError, StorageProvider},
    },
    test_helpers::{build_title_batch, default_supertable_options, lazy_foreground_disk_cache},
};
use tempfile::TempDir;
use tokio::time::sleep;

/// Files in the table, one per commit; more than the smaller caps below.
const FILES: usize = 12;
/// Rows per file: enough that a cold scan reads each file's data from
/// storage rather than from the open bytes the manifest carries inline.
const ROWS_PER_FILE: usize = 8_000;
/// Hex words in each title.
const WORDS_PER_TITLE: usize = 6;
/// Multiplier and increment of the generator behind the titles.
const LCG_MULT: u64 = 6_364_136_223_846_793_005;
const LCG_INC: u64 = 1_442_695_040_888_963_407;
/// Caps on the opens at once: one at a time, one that does not divide the
/// file count, and the default.
const CAPS: [Option<usize>; 3] = [Some(1), Some(7), None];
/// A cap under [`FILES`], for the test that counts files read at once.
const SMALL_CAP: usize = 3;
/// Rows per file where only the index reads matter.
const FEW_ROWS: usize = 1_000;
/// How long a read of a superfile takes, so reads in flight overlap.
const READ_LATENCY: Duration = Duration::from_millis(20);
/// The key suffix of a superfile's object, to tell its reads from the
/// manifest's.
const SUPERFILE_SUFFIX: &str = ".sf.parquet";
/// Words every file holds, one on even rows and one on odd rows: no hex word
/// contains either, and no row holds both.
const EVEN_WORD: &str = "even";
const ODD_WORD: &str = "odd";

/// Storage that counts every range read it serves, and delays each read of a
/// superfile while tracking how many superfiles have a read in flight.
#[derive(Debug)]
struct ObservedStorage {
    inner: Arc<dyn StorageProvider>,
    range_reads: AtomicUsize,
    /// Reads in flight per superfile key.
    reading: Mutex<HashMap<String, usize>>,
    /// The most superfiles ever read at once.
    peak_files_reading: AtomicUsize,
}

impl ObservedStorage {
    fn new(inner: &Arc<dyn StorageProvider>) -> Arc<Self> {
        Arc::new(Self {
            inner: Arc::clone(inner),
            range_reads: AtomicUsize::new(0),
            reading: Mutex::new(HashMap::new()),
            peak_files_reading: AtomicUsize::new(0),
        })
    }

    fn start_read(&self, uri: &str) -> Reading<'_> {
        let mut reading = self.reading.lock().expect("reading");
        *reading.entry(uri.to_string()).or_default() += 1;
        self.peak_files_reading
            .fetch_max(reading.len(), Ordering::SeqCst);
        Reading(self, uri.to_string())
    }
}

/// Marks one superfile read finished when dropped.
struct Reading<'a>(&'a ObservedStorage, String);

impl Drop for Reading<'_> {
    fn drop(&mut self) {
        let mut reading = self.0.reading.lock().expect("reading");
        if let Some(n) = reading.get_mut(&self.1) {
            *n -= 1;
            if *n == 0 {
                reading.remove(&self.1);
            }
        }
    }
}

#[async_trait]
impl StorageProvider for ObservedStorage {
    async fn head(&self, uri: &str) -> Result<ObjectMeta, StorageError> {
        self.inner.head(uri).await
    }
    async fn get(&self, uri: &str) -> Result<(Bytes, ObjectMeta), StorageError> {
        self.inner.get(uri).await
    }
    async fn get_range(&self, uri: &str, range: Range<u64>) -> Result<Bytes, StorageError> {
        self.range_reads.fetch_add(1, Ordering::SeqCst);
        if !uri.ends_with(SUPERFILE_SUFFIX) {
            return self.inner.get_range(uri, range).await;
        }
        let _reading = self.start_read(uri);
        sleep(READ_LATENCY).await;
        self.inner.get_range(uri, range).await
    }
    async fn put_atomic(&self, uri: &str, bytes: Bytes) -> Result<Option<String>, StorageError> {
        self.inner.put_atomic(uri, bytes).await
    }
    async fn put_overwrite(&self, uri: &str, bytes: Bytes) -> Result<(), StorageError> {
        self.inner.put_overwrite(uri, bytes).await
    }
    async fn put_if_match(
        &self,
        uri: &str,
        bytes: Bytes,
        etag: Option<&str>,
    ) -> Result<Option<String>, StorageError> {
        self.inner.put_if_match(uri, bytes, etag).await
    }
    async fn put_multipart(
        &self,
        uri: &str,
    ) -> Result<Box<dyn object_store::MultipartUpload>, StorageError> {
        self.inner.put_multipart(uri).await
    }
    async fn delete(&self, uri: &str) -> Result<(), StorageError> {
        self.inner.delete(uri).await
    }
}

/// Titles that do not compress, so a file's bytes grow with its rows: each
/// is a few hex words drawn from a generator seeded by file and row, then
/// [`EVEN_WORD`] or [`ODD_WORD`] by the row's parity.
fn titles(file: usize, rows: usize) -> Vec<String> {
    (0..rows)
        .map(|row| {
            let mut state = (file as u64) << 32 | row as u64;
            let mut words: Vec<String> = (0..WORDS_PER_TITLE)
                .map(|_| {
                    state = state.wrapping_mul(LCG_MULT).wrapping_add(LCG_INC);
                    format!("{state:016x}")
                })
                .collect();
            let parity = if row % 2 == 0 { EVEN_WORD } else { ODD_WORD };
            words.push(parity.to_string());
            words.join(" ")
        })
        .collect()
}

/// A table of [`FILES`] files of `rows` rows each, on storage under `dir`.
fn table_of(dir: &Path, rows: usize) -> Arc<dyn StorageProvider> {
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir).expect("provider"));
    let table = Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
        .expect("create");
    let mut writer = table.writer().expect("writer");
    for file in 0..FILES {
        let titles = titles(file, rows);
        let refs: Vec<&str> = titles.iter().map(String::as_str).collect();
        writer.append(&build_title_batch(&refs)).expect("append");
        writer.commit().expect("commit");
    }
    storage
}

/// One query's rows as a single batch, so results split into batches
/// differently still compare equal.
fn rows(batches: &[RecordBatch]) -> RecordBatch {
    let schema = batches.first().expect("a query returns a batch").schema();
    concat_batches(&schema, batches).expect("concat")
}

/// Every query's rows, and the range reads the first one made, from a cold
/// reader on `storage` that opens at most `cap` files at once.
fn cold_scan(
    storage: &Arc<dyn StorageProvider>,
    cap: Option<usize>,
    queries: &[String],
) -> (Vec<RecordBatch>, usize) {
    let observed = ObservedStorage::new(storage);
    let counted: Arc<dyn StorageProvider> = observed.clone();
    let cache_dir = TempDir::new().expect("cache tempdir");
    let cache = lazy_foreground_disk_cache(Arc::clone(&counted), cache_dir.path());
    let mut options = default_supertable_options()
        .with_storage(counted)
        .with_disk_cache(cache);
    if let Some(cap) = cap {
        options = options.with_scan_open_concurrency(cap);
    }
    let table = Supertable::open(options).expect("open");
    let reader = table.reader().expect("reader");
    assert_eq!(reader.n_superfiles(), FILES);

    let before = observed.range_reads.load(Ordering::SeqCst);
    let mut out = Vec::with_capacity(queries.len());
    let mut first_reads = 0;
    for (i, sql) in queries.iter().enumerate() {
        out.push(rows(&reader.query_sql(sql).expect("query")));
        if i == 0 {
            first_reads = observed.range_reads.load(Ordering::SeqCst) - before;
        }
    }
    (out, first_reads)
}

#[test]
fn a_cold_scan_returns_and_reads_the_same_at_any_open_cap() {
    let storage_dir = TempDir::new().expect("storage tempdir");
    let storage = table_of(storage_dir.path(), ROWS_PER_FILE);

    // A word from one file only, so the dictionary decides which rows match
    // and the scan resolves candidate rows per file before reading any.
    let word = titles(FILES / 2, ROWS_PER_FILE)[ROWS_PER_FILE / 2]
        .split(' ')
        .next()
        .expect("a word")
        .to_string();
    let queries = [
        "SELECT SUM(length(title)) AS n FROM supertable".to_string(),
        "SELECT COUNT(*) AS n FROM supertable WHERE substr(title, 1, 1) = 'a'".to_string(),
        "SELECT title FROM supertable WHERE substr(title, 2, 1) = 'b' ORDER BY title LIMIT 5"
            .to_string(),
        format!("SELECT title FROM supertable WHERE title ILIKE '%{word}%'"),
    ];

    let runs: Vec<(Option<usize>, Vec<RecordBatch>, usize)> = CAPS
        .iter()
        .map(|&cap| {
            let (results, reads) = cold_scan(&storage, cap, &queries);
            (cap, results, reads)
        })
        .collect();
    let (_, default_results, default_reads) = runs.last().expect("the default run");
    assert!(
        *default_reads > 0,
        "the first query reads data from storage, so the read counts below compare something"
    );
    assert_eq!(
        default_results[3].num_rows(),
        1,
        "the word matches the one row it was taken from"
    );
    for (cap, results, reads) in &runs {
        assert_eq!(results, default_results, "results at open cap {cap:?}");
        assert_eq!(
            reads, default_reads,
            "range reads of the first cold query at open cap {cap:?}: the cap limits opens only"
        );
    }
}

/// The most superfiles a cold scan read at once, from a reader on `storage`
/// behind a cold disk cache that opens at most `cap` files at once, and the
/// query's count.
fn peak_files_read(storage: &Arc<dyn StorageProvider>, cap: Option<usize>) -> (usize, i64) {
    let observed = ObservedStorage::new(storage);
    let observed_storage: Arc<dyn StorageProvider> = observed.clone();
    let cache_dir = TempDir::new().expect("cache tempdir");
    let cache = lazy_foreground_disk_cache(Arc::clone(&observed_storage), cache_dir.path());
    let mut options = default_supertable_options()
        .with_storage(observed_storage)
        .with_disk_cache(cache);
    if let Some(cap) = cap {
        options = options.with_scan_open_concurrency(cap);
    }
    let table = Supertable::open(options).expect("open");
    let reader = table.reader().expect("reader");
    // Every file holds both words, so none is pruned and each one's
    // dictionary and postings are read to find its candidate rows; no row
    // holds both, so there are none and no data is read after.
    let out = reader
        .query_sql(&format!(
            "SELECT COUNT(*) AS n FROM supertable \
             WHERE title ILIKE '%{EVEN_WORD}%' AND title ILIKE '%{ODD_WORD}%'"
        ))
        .expect("query");
    let count = out[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("count column")
        .value(0);
    (observed.peak_files_reading.load(Ordering::SeqCst), count)
}

#[test]
fn a_cold_scan_reads_at_most_the_cap_of_files_at_once() {
    let storage_dir = TempDir::new().expect("storage tempdir");
    let storage = table_of(storage_dir.path(), FEW_ROWS);

    // Under the default cap every file is read at once, so the fan-out the
    // cap limits is real.
    assert_eq!(peak_files_read(&storage, None), (FILES, 0));
    // Under a small cap no more files than it are read at once.
    assert_eq!(peak_files_read(&storage, Some(SMALL_CAP)), (SMALL_CAP, 0));
}
