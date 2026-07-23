// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Streaming sorted-run inputs for the clustered compaction merge.
//!
//! Every clustered superfile is internally key-ordered — the writer
//! sorts each commit and the clustered merge sorts each output — so a
//! compaction job's inputs are sorted runs. This module turns one such
//! input into a [`SendableRecordBatchStream`]: the live (tombstone-
//! filtered) rows decoded a bounded batch at a time, in file row order,
//! so a k-way merge can consume the run without ever materializing the
//! whole superfile.
//!
//! Vector payloads ride the stream row-aligned with their scalars: the
//! stream schema is the scalar read schema plus one
//! `FixedSizeList<Float32, dim>` column per configured vector column,
//! rebuilt from the file's decoded vector rows batch by batch. (sq8-
//! family vector merges byte-splice their blobs in row order and never
//! reach this module.)
//!
//! On top of the run sources sits [`streaming_clustered_merge`], the
//! bounded-memory merge itself: DataFusion's streaming merge combines
//! the runs on the clustering key (ascending, nulls last — the writer
//! sort's exact semantics via the shared [`CLUSTER_KEY_SORT`]), a
//! memory pool sized to the compaction ceiling accounts every in-
//! flight batch, and the merged stream is cut into target-sized
//! superfiles as it flows. Consecutive outputs are consecutive slices
//! of one globally ordered stream, so their key ranges chain without
//! overlap and the scan's ordering declaration stays provable across
//! all of them.
//!
//! # Fan-in admission and the cascade
//!
//! Each run charges a conservative reserve against the ceiling
//! (double its worst-case decoded batch, plus its largest per-file
//! decoded vector payload — the sources materialize vectors one file
//! at a time). When every run fits at once, one pass merges them all.
//! Otherwise the merge cascades: fold the admitted prefix (never
//! fewer than [`STREAMING_MERGE_MIN_FAN_IN`], so a fold always makes
//! progress) into one intermediate run — a chain of target-sized,
//! key-ordered superfiles streamed back-to-back as a single sorted
//! input — append it behind the remaining runs, and repeat. Every
//! fold shrinks the run count by at least one, so the loop is bounded
//! by the input count; the final pass produces the published outputs.
//!
//! The reserves are metadata-derived estimates (parquet's encoded-
//! but-uncompressed page sizes), and decoded Arrow batches can
//! outgrow them — dictionary- or RLE-heavy pages decode into far more
//! memory than they occupy encoded. A pass whose actual working set
//! exhausts its pool is therefore *retried*, not failed: first at
//! half the fan-in (remembered for the rest of the job — reserves
//! that under-admitted once would under-admit again), and once the
//! fan-in has narrowed all the way to the minimum, with a doubled
//! pool per attempt, bounded by
//! [`STREAMING_MERGE_MAX_POOL_DOUBLINGS`]. Every retry either shrinks
//! the fan-in or is bounded by the doubling cap, so the job still
//! terminates; without the retry a single under-estimate would abort
//! the whole compaction and leave the table unconverged.
//!
//! # What the ceiling does and does not bound
//!
//! The memory pool bounds the merge's *decoded* working set — the
//! Arrow batches in flight plus per-run stream buffers — which is
//! what grew with job size on the in-memory route. Compressed
//! superfile bytes are outside the pool: inputs are opened resident
//! exactly like every other merge route (mmap-backed when a disk
//! cache is attached), and intermediate cascade superfiles are held
//! as in-process bytes of the same compressed magnitude as the inputs
//! they replace. A pass whose admitted runs' reserve exceeds the
//! ceiling still gets a pool large enough for its minimum working set
//! (see [`pass_pool_bytes`]), so a pathological pair of runs degrades
//! to an over-ceiling pool (further grown by the bounded retry
//! doublings above when even that estimate falls short) rather than
//! an unmergeable table.
//!
//! # Identity and re-runs
//!
//! Output ids and URIs are UUIDv5 digests of the compaction id and
//! the output index, and each output uploads as soon as it is cut. A
//! crashed job's uploads are unreferenced by any manifest, so the GC
//! sweep reclaims them; a retake by another compactor mints a new
//! compaction id and cannot collide. Within one job identity the
//! upload is idempotent: the bytes are deterministic for the sealed
//! inputs, so an already-present object (atomic create losing to a
//! prior attempt of the same identity) is the same content.

use std::{mem, sync::Arc};

use arrow_array::{Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion::{
    error::DataFusionError,
    execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool},
    physical_expr::{LexOrdering, PhysicalExpr, PhysicalSortExpr, expressions::Column},
    physical_plan::{
        SendableRecordBatchStream,
        metrics::{BaselineMetrics, ExecutionPlanMetricsSet},
        sorts::streaming_merge::StreamingMergeBuilder,
        stream::RecordBatchStreamAdapter,
    },
};
use futures::{StreamExt, executor::block_on, stream};
use parquet::arrow::arrow_reader::ParquetRecordBatchReader;
use roaring::RoaringBitmap;
use tokio::sync::mpsc;
use tracing::warn;
use uuid::Uuid;

use crate::{
    storage::StorageProvider,
    superfile::{
        SuperfileReader,
        builder::{BuilderOptions, check_merge_input},
        error::BuildError as SuperfileBuildError,
    },
    supertable::{
        BuildError, SuperfileEntry, SuperfileUri, SupertableOptions,
        handle::SupertableInner,
        query::provider::CLUSTER_KEY_SORT,
        writer::{
            BufferedBatch, PreparedSuperfile, ShardOutput, build_one_shard_with_layout,
            prepare_superfile_with_uri, put_new_superfile_bytes,
        },
    },
};

/// Rows per batch on both sides of the merge: the bounded parquet
/// decode of each input run and the merged output batches the consumer
/// sees. Sized like a typical execution batch — small enough that a
/// couple of in-flight batches per run stay far below any realistic
/// memory ceiling, large enough to amortize per-batch overhead.
pub(super) const STREAMING_MERGE_BATCH_ROWS: usize = 8 * 1024;

/// Name and dimensionality of one vector column riding the merge.
#[derive(Clone)]
pub(super) struct VectorShape {
    name: String,
    dim: usize,
}

/// The configured vector columns as merge-stream shapes, in
/// declaration order (the order the writer's buffered batches and the
/// superfile builder expect).
pub(super) fn vector_shapes(options: &SupertableOptions) -> Arc<[VectorShape]> {
    options
        .vector_columns
        .iter()
        .map(|vc| VectorShape {
            name: vc.column.clone(),
            dim: vc.dim,
        })
        .collect()
}

/// The merged stream's schema: the scalar read schema with one
/// non-null `FixedSizeList<Float32, dim>` field appended per vector
/// column, so vector payloads ride the merge row-aligned with their
/// scalars and split back out on the far side.
pub(super) fn merged_stream_schema(
    scalar: &SchemaRef,
    shapes: &[VectorShape],
) -> Result<SchemaRef, BuildError> {
    if shapes.is_empty() {
        return Ok(scalar.clone());
    }
    let mut fields: Vec<Field> = scalar.fields().iter().map(|f| f.as_ref().clone()).collect();
    for shape in shapes {
        if scalar.field_with_name(&shape.name).is_ok() {
            return Err(BuildError::Store(format!(
                "streaming clustered merge: vector column '{}' collides with a scalar column",
                shape.name
            )));
        }
        fields.push(Field::new(
            &shape.name,
            DataType::FixedSizeList(vector_item_field(), shape.dim as i32),
            false,
        ));
    }
    Ok(Arc::new(Schema::new(fields)))
}

/// The `FixedSizeList` element field vector columns use on the merged
/// stream.
fn vector_item_field() -> Arc<Field> {
    Arc::new(Field::new("item", DataType::Float32, false))
}

/// Tombstone-filtered bounded-batch source for one sorted run, with the
/// file's vector rows re-attached as `FixedSizeList` columns.
///
/// The parquet reader yields the surviving rows in ascending file row
/// order; `survivors` mirrors that order, so batch `k`'s vector rows
/// are `survivors[cursor .. cursor + k.num_rows()]` indexed into the
/// file's decoded vector columns.
struct LiveRunBatches {
    parquet: ParquetRecordBatchReader,
    stream_schema: SchemaRef,
    shapes: Arc<[VectorShape]>,
    /// Per vector column: every row of the file (tombstoned included),
    /// decoded once for the run's lifetime and dropped with it.
    vectors: Vec<Vec<Vec<f32>>>,
    /// Ascending file row ids surviving the tombstone filter. Empty
    /// when the table has no vector columns (nothing to align).
    survivors: Vec<u32>,
    cursor: usize,
}

impl LiveRunBatches {
    fn new(
        reader: &SuperfileReader,
        tombstones: Option<Arc<RoaringBitmap>>,
        stream_schema: SchemaRef,
        shapes: Arc<[VectorShape]>,
    ) -> Result<Self, BuildError> {
        let parquet = reader
            .live_batch_reader(tombstones.clone(), Some(STREAMING_MERGE_BATCH_ROWS))
            .map_err(|e| BuildError::Store(format!("streaming clustered merge: {e}")))?;
        let mut vectors = Vec::with_capacity(shapes.len());
        let mut survivors = Vec::new();
        if !shapes.is_empty() {
            let vec_reader = reader
                .vec()
                .ok_or(BuildError::Superfile(SuperfileBuildError::VectorReadError))?;
            for shape in shapes.iter() {
                vectors.push(
                    vec_reader
                        .get_vectors_for_merge(&shape.name)
                        .map_err(|_| BuildError::Superfile(SuperfileBuildError::VectorReadError))?,
                );
            }
            let n_docs = u32::try_from(reader.n_docs()).map_err(|_| {
                BuildError::Store("streaming clustered merge: row count exceeds u32".to_string())
            })?;
            survivors = match &tombstones {
                Some(bitmap) => (0..n_docs).filter(|row| !bitmap.contains(*row)).collect(),
                None => (0..n_docs).collect(),
            };
        }
        Ok(Self {
            parquet,
            stream_schema,
            shapes,
            vectors,
            survivors,
            cursor: 0,
        })
    }

    /// Rebuild `scalar` on the stream schema with this batch's vector
    /// rows appended as `FixedSizeList` columns.
    fn attach_vectors(&mut self, scalar: RecordBatch) -> Result<RecordBatch, BuildError> {
        let n_rows = scalar.num_rows();
        let mut columns = scalar.columns().to_vec();
        if !self.shapes.is_empty() {
            let rows = self
                .survivors
                .get(self.cursor..self.cursor + n_rows)
                .ok_or_else(|| {
                    BuildError::Store(
                        "streaming clustered merge: batch rows drifted from the tombstone filter"
                            .to_string(),
                    )
                })?;
            for (col_idx, shape) in self.shapes.iter().enumerate() {
                let all_rows = &self.vectors[col_idx];
                let mut values = Vec::with_capacity(n_rows * shape.dim);
                for &row in rows {
                    let vector = all_rows
                        .get(row as usize)
                        .ok_or(BuildError::Superfile(SuperfileBuildError::VectorReadError))?;
                    if vector.len() != shape.dim {
                        return Err(BuildError::Superfile(SuperfileBuildError::VectorReadError));
                    }
                    values.extend_from_slice(vector);
                }
                let list = FixedSizeListArray::new(
                    vector_item_field(),
                    shape.dim as i32,
                    Arc::new(Float32Array::from(values)) as ArrayRef,
                    None,
                );
                columns.push(Arc::new(list));
            }
        }
        self.cursor += n_rows;
        RecordBatch::try_new(self.stream_schema.clone(), columns)
            .map_err(|e| BuildError::Store(format!("streaming clustered merge: {e}")))
    }
}

impl Iterator for LiveRunBatches {
    type Item = Result<RecordBatch, DataFusionError>;

    fn next(&mut self) -> Option<Self::Item> {
        let scalar = match self.parquet.next()? {
            Ok(batch) => batch,
            Err(e) => return Some(Err(DataFusionError::External(Box::new(e)))),
        };
        Some(
            self.attach_vectors(scalar)
                .map_err(|e| DataFusionError::External(Box::new(e))),
        )
    }
}

/// One merge input superfile as a sorted-run stream: its live rows on
/// the merge's row schema ([`merged_stream_schema`]), decoded
/// [`STREAMING_MERGE_BATCH_ROWS`] rows at a time in file row order.
/// Construction validates the reader up front (parquet metadata,
/// vector decode); per-batch decode errors surface through the stream.
pub(super) fn sorted_run_stream(
    reader: &SuperfileReader,
    tombstones: Option<Arc<RoaringBitmap>>,
    stream_schema: SchemaRef,
    shapes: Arc<[VectorShape]>,
) -> Result<SendableRecordBatchStream, BuildError> {
    let batches = LiveRunBatches::new(reader, tombstones, stream_schema.clone(), shapes)?;
    Ok(Box::pin(RecordBatchStreamAdapter::new(
        stream_schema,
        stream::iter(batches),
    )))
}

/// Bytes of one `f32` vector component.
const F32_BYTES: usize = size_of::<f32>();

/// Conservative multiple of a run's worst-case decoded batch charged at
/// admission: the merge holds roughly one batch in its cursor and one
/// in flight per input stream, and for each it also holds the
/// row-format encoding of the key columns — which can weigh as much as
/// the decoded batch itself when the key dominates the row (e.g. a wide
/// string key). Two batches plus two row-format copies → four.
const STREAMING_MERGE_PER_RUN_RESERVE_FACTOR: u64 = 4;

/// Fewest runs a pass may merge. Two is the minimum that makes
/// progress (a fold of one run would loop forever), so admission never
/// returns less even when a single run's reserve exceeds the ceiling —
/// the pass pool grows instead (see [`pass_pool_bytes`]).
const STREAMING_MERGE_MIN_FAN_IN: usize = 2;

/// Name the merge's memory reservation registers under; surfaces in
/// pool-exhaustion errors so they attribute to this path.
const STREAMING_MERGE_CONSUMER: &str = "clustered-streaming-merge";

/// Bounded pool growth for a minimum-fan-in pass that still exhausts
/// its pool: each retry doubles the pool, up to this many doublings
/// (2^8 = 256x — enough headroom for dictionary-heavy inputs whose
/// decoded batches dwarf parquet's encoded-uncompressed estimate).
/// Growth only engages once the fan-in has already narrowed to
/// [`STREAMING_MERGE_MIN_FAN_IN`], extending the documented
/// over-ceiling degradation for pathological pairs; a pass that still
/// exhausts after the last doubling surfaces the resources error.
const STREAMING_MERGE_MAX_POOL_DOUBLINGS: u32 = 8;

/// Absolute minimum for a pass's memory pool. The per-run reserves are
/// derived from parquet's `total_byte_size`, which meters
/// encoded-but-uncompressed pages: small files whose whole payload is
/// one dictionary-encoded row group decode into noticeably more Arrow
/// memory than that (plus the merge cursors' row-encoded keys and
/// allocator rounding), so a ceiling tuned to the byte scale of such
/// files could starve even the minimum two-run pass. A few MiB of
/// absolute floor covers that small-file regime outright; production
/// ceilings are orders of magnitude above it, so there the configured
/// ceiling alone governs the pool.
const STREAMING_MERGE_POOL_FLOOR_BYTES: u64 = 8 * 1024 * 1024;

/// Capacity of the shard hand-off channel between the pool-side merge
/// driver and the async publisher: at most one completed output waits
/// while the previous one uploads, so finished superfiles never pile
/// up in memory.
const SHARD_CHANNEL_CAPACITY: usize = 1;

/// What a finished streaming merge hands back to `merge_superfiles`.
pub(super) struct StreamingMergeReport {
    /// Publish-ready outputs, in key order (ranges chain without
    /// overlap). Bytes are already durable; only entries remain.
    pub(super) prepared: Vec<PreparedSuperfile>,
    /// Cascade folds performed before the final pass (0 = every run
    /// was admitted at once).
    pub(super) cascade_folds: usize,
}

/// One key-ordered superfile inside a run. Cloning is cheap (two
/// refcounts), which is what lets a pool-exhausted pass retry from
/// the same inputs.
#[derive(Clone)]
struct RunFile {
    reader: Arc<SuperfileReader>,
    tombstones: Option<Arc<RoaringBitmap>>,
}

/// Why one merge pass failed. Pool exhaustion is retryable — the
/// admission reserves are metadata-derived estimates, so a pass whose
/// actual decoded working set outgrows them narrows its fan-in (or, at
/// the minimum fan-in, grows its pool) and tries again. Anything else
/// aborts the job.
enum PassError {
    PoolExhausted(BuildError),
    Fatal(BuildError),
}

/// Classify one merged-stream error: the memory pool's refusal
/// surfaces as `ResourcesExhausted` (possibly wrapped) and is the
/// retryable case; everything else is fatal.
fn classify_merge_error(e: DataFusionError) -> PassError {
    let retryable = matches!(e.find_root(), DataFusionError::ResourcesExhausted(_));
    let wrapped = BuildError::Store(format!("streaming clustered merge: {e}"));
    if retryable {
        PassError::PoolExhausted(wrapped)
    } else {
        PassError::Fatal(wrapped)
    }
}

/// One sorted run: a chain of key-ordered superfiles whose ranges
/// chain without overlap, streamed back-to-back as a single sorted
/// merge input. Original job inputs are one file each; cascade folds
/// produce multi-file chains.
type SortedRun = Vec<RunFile>;

/// Everything one merge pass needs, owned so the driver can move onto
/// a writer-pool thread.
struct MergePassContext {
    options: Arc<SupertableOptions>,
    /// The inputs' scalar read schema (no vector columns).
    scalar_schema: SchemaRef,
    /// [`scalar_schema`](Self::scalar_schema) plus one trailing
    /// `FixedSizeList<Float32, dim>` field per vector column.
    stream_schema: SchemaRef,
    shapes: Arc<[VectorShape]>,
    /// Rows per emitted superfile — one target size worth of the
    /// merged stream.
    rows_per_cut: usize,
    /// Byte bound handed to the merge's memory pool.
    pool_bytes: usize,
}

/// Merge `readers` — every live row of a clustered job's inputs — into
/// globally key-ordered, range-disjoint superfiles with the merge's
/// decoded working set bounded by `merge_memory_bytes`. `n_outputs`
/// sets the cut granularity exactly like the in-memory route. Outputs
/// upload to storage as they are cut (deterministic per
/// `compaction_id` + output index) and come back `storage_prewritten`,
/// so the commit skips its own write. An all-tombstoned job returns an
/// empty report.
pub(super) async fn streaming_clustered_merge(
    inner: &Arc<SupertableInner>,
    readers: Vec<(Arc<SuperfileReader>, Option<Arc<RoaringBitmap>>)>,
    n_outputs: usize,
    compaction_id: Uuid,
    merge_memory_bytes: u64,
) -> Result<StreamingMergeReport, BuildError> {
    let options = Arc::clone(&inner.options);
    let storage = options.storage.clone().ok_or_else(|| {
        BuildError::Store("streaming clustered merge requires durable storage".to_string())
    })?;

    // Validate every input against the first one's shape — the same
    // gate the materializing merge routes apply.
    let (first, _) = readers
        .first()
        .ok_or(BuildError::Superfile(SuperfileBuildError::BatchReadError))?;
    let merge_opts = BuilderOptions::new_from_reader(first);
    for (reader, _) in &readers {
        check_merge_input(&merge_opts, reader)?;
    }

    let scalar_schema = first.schema().clone();
    let shapes = vector_shapes(&options);
    let stream_schema = merged_stream_schema(&scalar_schema, &shapes)?;

    let total_live_rows: u64 = readers
        .iter()
        .map(|(reader, tombstones)| {
            reader
                .n_docs()
                .saturating_sub(tombstones.as_ref().map_or(0, |b| b.len()))
        })
        .sum();
    if total_live_rows == 0 {
        return Ok(StreamingMergeReport {
            prepared: Vec::new(),
            cascade_folds: 0,
        });
    }
    let rows_per_cut = total_live_rows
        .div_ceil(n_outputs.max(1) as u64)
        .clamp(1, usize::MAX as u64) as usize;

    let mut runs: Vec<SortedRun> = readers
        .into_iter()
        .map(|(reader, tombstones)| vec![RunFile { reader, tombstones }])
        .collect();
    let mut cascade_folds = 0usize;
    let mut output_idx = 0usize;
    // Narrowed when a pass exhausts its pool, and remembered for the
    // rest of the job: reserves that under-admitted once would
    // under-admit again on the very next pass.
    let mut fan_in_cap = usize::MAX;
    loop {
        let mut fan_in = admitted_fan_in(&runs, &shapes, merge_memory_bytes)?.min(fan_in_cap);
        let mut pool_doublings = 0u32;
        // One pass, retried on pool exhaustion: narrower fan-in first,
        // then a doubled pool once the fan-in is already minimal. The
        // runs stay untouched until an attempt succeeds, so a retry
        // rebuilds its streams from the same (cheaply cloned) inputs.
        loop {
            let is_final = runs.len() <= fan_in;
            let take = if is_final { runs.len() } else { fan_in };
            let pass_runs: Vec<SortedRun> = runs[..take].to_vec();
            let pool_bytes = pass_pool_bytes(&pass_runs, &shapes, merge_memory_bytes)?
                .saturating_mul(1usize << pool_doublings);
            let ctx = MergePassContext {
                options: Arc::clone(&options),
                scalar_schema: scalar_schema.clone(),
                stream_schema: stream_schema.clone(),
                shapes: Arc::clone(&shapes),
                rows_per_cut,
                pool_bytes,
            };

            let (tx, rx) = mpsc::channel(SHARD_CHANNEL_CAPACITY);
            let pool = Arc::clone(&options.writer_pool);
            pool.spawn(move || drive_merge_pass(ctx, pass_runs, &tx));

            let outcome = if is_final {
                consume_final_pass(
                    inner,
                    &storage,
                    &options,
                    rx,
                    compaction_id,
                    &mut output_idx,
                )
                .await
                .map(PassOutcome::Outputs)
            } else {
                consume_fold_pass(&options, rx)
                    .await
                    .map(PassOutcome::Chain)
            };
            match outcome {
                Ok(PassOutcome::Outputs(prepared)) => {
                    return Ok(StreamingMergeReport {
                        prepared,
                        cascade_folds,
                    });
                }
                Ok(PassOutcome::Chain(chain)) => {
                    runs.drain(..take);
                    runs.push(chain);
                    cascade_folds += 1;
                    break;
                }
                Err(PassError::PoolExhausted(error)) => {
                    if fan_in > STREAMING_MERGE_MIN_FAN_IN {
                        fan_in = (fan_in / 2).max(STREAMING_MERGE_MIN_FAN_IN);
                        fan_in_cap = fan_in;
                        warn!(
                            fan_in,
                            "compact: streaming merge pass exhausted its pool; retrying narrower"
                        );
                    } else if pool_doublings < STREAMING_MERGE_MAX_POOL_DOUBLINGS {
                        pool_doublings += 1;
                        warn!(
                            pool_doublings,
                            "compact: minimum-fan-in streaming merge pass exhausted its pool; \
                             retrying with a doubled pool"
                        );
                    } else {
                        return Err(error);
                    }
                }
                Err(PassError::Fatal(error)) => return Err(error),
            }
        }
    }
}

/// What one successful merge pass produced.
enum PassOutcome {
    /// Final pass: the publish-ready outputs, already durable.
    Outputs(Vec<PreparedSuperfile>),
    /// Cascade fold: the intermediate chain, one new sorted run.
    Chain(SortedRun),
}

/// Final-pass consumer: publish every cut as it arrives (upload +
/// release). `output_idx` advances across attempts, never per attempt:
/// a retried pass must not reuse a deterministic output identity whose
/// bytes may already exist with the failed attempt's content (the
/// upload treats an already-present object as success). Burned indices
/// from a failed attempt are unreferenced uploads the gc sweep
/// reclaims.
async fn consume_final_pass(
    inner: &Arc<SupertableInner>,
    storage: &Arc<dyn StorageProvider>,
    options: &Arc<SupertableOptions>,
    mut rx: mpsc::Receiver<Result<ShardOutput, PassError>>,
    compaction_id: Uuid,
    output_idx: &mut usize,
) -> Result<Vec<PreparedSuperfile>, PassError> {
    let mut prepared = Vec::new();
    while let Some(item) = rx.recv().await {
        let shard = item?;
        let idx = *output_idx;
        *output_idx += 1;
        if let Some(output) =
            prepare_streaming_output(inner, storage, options, shard, compaction_id, idx)
                .await
                .map_err(PassError::Fatal)?
        {
            prepared.push(output);
        }
    }
    Ok(prepared)
}

/// Cascade-fold consumer: reopen each intermediate output as a run
/// file. The chain is a consecutive slice of one sorted pass, so it is
/// itself a sorted run for the next pass.
async fn consume_fold_pass(
    options: &Arc<SupertableOptions>,
    mut rx: mpsc::Receiver<Result<ShardOutput, PassError>>,
) -> Result<SortedRun, PassError> {
    let mut chain: SortedRun = Vec::new();
    while let Some(item) = rx.recv().await {
        let shard = item?;
        let reader =
            SuperfileReader::open_with(shard.bytes().clone(), options.superfile_open_options())
                .map_err(|e| {
                    PassError::Fatal(BuildError::Store(format!(
                        "streaming clustered merge: reopen intermediate: {e}"
                    )))
                })?;
        chain.push(RunFile {
            reader: Arc::new(reader),
            tombstones: None,
        });
    }
    if chain.is_empty() {
        return Err(PassError::Fatal(BuildError::Store(
            "streaming clustered merge: cascade fold produced no rows".to_string(),
        )));
    }
    Ok(chain)
}

/// Conservative ceiling charge for streaming one run: double its
/// worst-case decoded batch (scalar bound: the largest row group's
/// uncompressed size — a batch never spans row groups — plus one
/// batch's worth of vector rows), plus the run's largest per-file
/// decoded vector payload — the chained source materializes vectors
/// one file at a time.
fn run_stream_reserve_bytes(run: &SortedRun, shapes: &[VectorShape]) -> Result<u64, BuildError> {
    let vector_row_bytes: u64 = shapes.iter().map(|s| (s.dim * F32_BYTES) as u64).sum();
    let batch_vector_bytes = (STREAMING_MERGE_BATCH_ROWS as u64).saturating_mul(vector_row_bytes);
    let mut max_batch_bytes = 0u64;
    let mut max_file_vector_bytes = 0u64;
    for file in run {
        let scalar_bytes = file
            .reader
            .max_row_group_uncompressed_bytes()
            .map_err(|e| BuildError::Store(format!("streaming clustered merge: {e}")))?;
        max_batch_bytes = max_batch_bytes.max(scalar_bytes.saturating_add(batch_vector_bytes));
        max_file_vector_bytes =
            max_file_vector_bytes.max(file.reader.n_docs().saturating_mul(vector_row_bytes));
    }
    Ok(STREAMING_MERGE_PER_RUN_RESERVE_FACTOR
        .saturating_mul(max_batch_bytes)
        .saturating_add(max_file_vector_bytes))
}

/// How many of `runs` (in order) one pass admits: the longest prefix
/// whose summed reserves fit `budget`, floored at
/// [`STREAMING_MERGE_MIN_FAN_IN`] so a fold always makes progress and
/// capped at the run count.
fn admitted_fan_in(
    runs: &[SortedRun],
    shapes: &[VectorShape],
    budget: u64,
) -> Result<usize, BuildError> {
    let mut admitted = 0usize;
    let mut total = 0u64;
    for run in runs {
        let reserve = run_stream_reserve_bytes(run, shapes)?;
        if admitted >= STREAMING_MERGE_MIN_FAN_IN && total.saturating_add(reserve) > budget {
            break;
        }
        total = total.saturating_add(reserve);
        admitted += 1;
    }
    Ok(admitted.clamp(STREAMING_MERGE_MIN_FAN_IN.min(runs.len()), runs.len()))
}

/// The merge pool for one pass: the configured compaction ceiling,
/// floored at the pass's own conservative reserve (so an admitted pass
/// can always hold its per-run working set of a couple of decoded
/// batches plus cursor state — the reserve-based floor only exceeds
/// the ceiling when the admitted minimum of
/// [`STREAMING_MERGE_MIN_FAN_IN`] runs does) and at
/// [`STREAMING_MERGE_POOL_FLOOR_BYTES`] (the small-file decode-slack
/// floor). A pass that still exhausts the pool fails with a resources
/// error instead of silently breaching the ceiling.
fn pass_pool_bytes(
    runs: &[SortedRun],
    shapes: &[VectorShape],
    budget: u64,
) -> Result<usize, BuildError> {
    let mut total = 0u64;
    for run in runs {
        total = total.saturating_add(run_stream_reserve_bytes(run, shapes)?);
    }
    Ok(budget
        .max(total)
        .max(STREAMING_MERGE_POOL_FLOOR_BYTES)
        .min(usize::MAX as u64) as usize)
}

/// The clustering key as a physical sort declaration over the merged
/// stream schema — ascending, nulls last per key column, in key order:
/// byte-identical semantics to the writer's commit sort
/// ([`CLUSTER_KEY_SORT`] is the shared constant).
fn cluster_key_lex_ordering(key: &[String], schema: &SchemaRef) -> Result<LexOrdering, BuildError> {
    let exprs = key
        .iter()
        .map(|name| {
            let idx = schema
                .index_of(name)
                .map_err(|_| BuildError::ClusterKeyColumnMissing {
                    column: name.clone(),
                })?;
            let expr: Arc<dyn PhysicalExpr> = Arc::new(Column::new(name, idx));
            Ok(PhysicalSortExpr::new(expr, CLUSTER_KEY_SORT))
        })
        .collect::<Result<Vec<_>, BuildError>>()?;
    LexOrdering::new(exprs).ok_or_else(|| {
        BuildError::Store("streaming clustered merge: empty clustering key".to_string())
    })
}

/// Merge already-sorted run streams into one globally key-ordered
/// stream: DataFusion's k-way streaming merge over the clustering-key
/// ordering ([`cluster_key_lex_ordering`]), output batches capped at
/// [`STREAMING_MERGE_BATCH_ROWS`], every in-flight batch accounted
/// against a fresh [`GreedyMemoryPool`] of `pool_bytes`. Round-robin
/// tie-breaking is disabled so equal keys drain lower-index inputs
/// first — deterministic output for a deterministic job identity,
/// mirroring the in-memory sort's stable ties.
fn merge_sorted_runs(
    streams: Vec<SendableRecordBatchStream>,
    schema: &SchemaRef,
    cluster_by: &[String],
    pool_bytes: usize,
) -> Result<SendableRecordBatchStream, BuildError> {
    let ordering = cluster_key_lex_ordering(cluster_by, schema)?;
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(pool_bytes));
    let reservation = MemoryConsumer::new(STREAMING_MERGE_CONSUMER).register(&pool);
    let metrics = ExecutionPlanMetricsSet::new();
    StreamingMergeBuilder::new()
        .with_streams(streams)
        .with_schema(schema.clone())
        .with_expressions(&ordering)
        .with_metrics(BaselineMetrics::new(&metrics, 0))
        .with_batch_size(STREAMING_MERGE_BATCH_ROWS)
        .with_reservation(reservation)
        .with_round_robin_tie_breaker(false)
        .build()
        .map_err(|e| BuildError::Store(format!("streaming clustered merge: {e}")))
}

/// One run as a single sorted merge input: its files' live batches
/// streamed back-to-back. File sources construct lazily inside the
/// chain, so at most one file's decoded vector columns are resident
/// per run and each file's buffers drop as soon as its batches are
/// exhausted.
fn chained_run_stream(
    run: SortedRun,
    stream_schema: SchemaRef,
    shapes: Arc<[VectorShape]>,
) -> SendableRecordBatchStream {
    let schema = stream_schema.clone();
    let chained = stream::iter(run).flat_map(move |file| {
        match sorted_run_stream(
            &file.reader,
            file.tombstones,
            schema.clone(),
            Arc::clone(&shapes),
        ) {
            Ok(batches) => batches,
            Err(e) => {
                let failed: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
                    schema.clone(),
                    stream::once(async move { Err(DataFusionError::External(Box::new(e))) }),
                ));
                failed
            }
        }
    });
    Box::pin(RecordBatchStreamAdapter::new(stream_schema, chained))
}

/// Split one merged stream batch back into the writer's buffered
/// shape: the scalar prefix on the scalar schema plus one flat
/// `Float32Array` per vector column.
fn merged_batch_to_buffered(
    batch: &RecordBatch,
    scalar_schema: &SchemaRef,
    shapes: &[VectorShape],
) -> Result<BufferedBatch, BuildError> {
    let n_scalar = scalar_schema.fields().len();
    let scalar = RecordBatch::try_new(scalar_schema.clone(), batch.columns()[..n_scalar].to_vec())
        .map_err(|e| BuildError::Store(format!("streaming clustered merge: {e}")))?;
    let mut vectors = Vec::with_capacity(shapes.len());
    for (idx, shape) in shapes.iter().enumerate() {
        let column = batch.column(n_scalar + idx);
        let list = column
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .ok_or_else(|| {
                BuildError::Store(format!(
                    "streaming clustered merge: vector column '{}' lost its list shape",
                    shape.name
                ))
            })?;
        let flat = list
            .values()
            .slice(list.offset() * shape.dim, list.len() * shape.dim);
        let flat = flat
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| {
                BuildError::Store(format!(
                    "streaming clustered merge: vector column '{}' is not f32",
                    shape.name
                ))
            })?
            .clone();
        vectors.push(Arc::new(flat));
    }
    Ok(BufferedBatch { scalar, vectors })
}

/// Zero-copy row split of one buffered batch at `at`: `(head, tail)`
/// where `head` holds the first `at` rows. `tail` is `None` when the
/// batch has exactly `at` rows.
fn split_buffered_at(
    batch: &BufferedBatch,
    at: usize,
    shapes: &[VectorShape],
) -> (BufferedBatch, Option<BufferedBatch>) {
    let n_rows = batch.scalar.num_rows();
    debug_assert!(at > 0 && at <= n_rows);
    let slice_rows = |from: usize, len: usize| BufferedBatch {
        scalar: batch.scalar.slice(from, len),
        vectors: batch
            .vectors
            .iter()
            .zip(shapes)
            .map(|(v, shape)| Arc::new(v.slice(from * shape.dim, len * shape.dim)))
            .collect(),
    };
    let head = slice_rows(0, at);
    let tail = (at < n_rows).then(|| slice_rows(at, n_rows - at));
    (head, tail)
}

/// Pool-thread half of one merge pass: build the per-run streams,
/// drive DataFusion's streaming merge, cut the merged stream at
/// `rows_per_cut`, build one superfile per cut, and hand each shard to
/// the async consumer. Errors travel through the same channel.
fn drive_merge_pass(
    ctx: MergePassContext,
    runs: Vec<SortedRun>,
    tx: &mpsc::Sender<Result<ShardOutput, PassError>>,
) {
    if let Err(e) = drive_merge_pass_inner(&ctx, runs, tx) {
        // Best effort: the consumer may already be gone.
        let _ = tx.blocking_send(Err(e));
    }
}

fn drive_merge_pass_inner(
    ctx: &MergePassContext,
    runs: Vec<SortedRun>,
    tx: &mpsc::Sender<Result<ShardOutput, PassError>>,
) -> Result<(), PassError> {
    let streams: Vec<SendableRecordBatchStream> = runs
        .into_iter()
        .map(|run| chained_run_stream(run, ctx.stream_schema.clone(), Arc::clone(&ctx.shapes)))
        .collect();
    let mut merged = merge_sorted_runs(
        streams,
        &ctx.stream_schema,
        &ctx.options.cluster_by,
        ctx.pool_bytes,
    )
    .map_err(PassError::Fatal)?;

    let mut pending: Vec<BufferedBatch> = Vec::new();
    let mut pending_rows = 0usize;
    // The leaf streams are synchronous iterators, so polling here never
    // parks; `block_on` just adapts the stream interface on this
    // dedicated pool worker.
    while let Some(item) = block_on(merged.next()) {
        let batch = item.map_err(classify_merge_error)?;
        if batch.num_rows() == 0 {
            continue;
        }
        let mut buffered = merged_batch_to_buffered(&batch, &ctx.scalar_schema, &ctx.shapes)
            .map_err(PassError::Fatal)?;
        drop(batch);
        loop {
            let rows = buffered.scalar.num_rows();
            let room = ctx.rows_per_cut - pending_rows;
            if rows < room {
                pending_rows += rows;
                pending.push(buffered);
                break;
            }
            let (head, tail) = split_buffered_at(&buffered, room, &ctx.shapes);
            pending.push(head);
            emit_shard(ctx, &mut pending, tx)?;
            pending_rows = 0;
            match tail {
                Some(rest) => buffered = rest,
                None => break,
            }
        }
    }
    if pending_rows > 0 {
        emit_shard(ctx, &mut pending, tx)?;
    }
    Ok(())
}

/// Build one superfile from the pending cut and hand it off. Fails
/// when the consumer has gone away (its error is authoritative then).
fn emit_shard(
    ctx: &MergePassContext,
    pending: &mut Vec<BufferedBatch>,
    tx: &mpsc::Sender<Result<ShardOutput, PassError>>,
) -> Result<(), PassError> {
    let chunk = mem::take(pending);
    let shard = build_one_shard_with_layout(&chunk, &ctx.options, ctx.options.vector_layout, None)
        .map_err(PassError::Fatal)?;
    drop(chunk);
    tx.blocking_send(Ok(shard)).map_err(|_| {
        PassError::Fatal(BuildError::Store(
            "streaming clustered merge: output consumer dropped".to_string(),
        ))
    })
}

/// Deterministic identity of streaming output `idx` under
/// `compaction_id`: UUIDv5 digests, so a re-run of the same job writes
/// the same keys instead of accumulating orphans.
fn streaming_output_uri(compaction_id: Uuid, idx: usize) -> SuperfileUri {
    SuperfileUri(Uuid::new_v5(
        &compaction_id,
        format!("clustered-streaming-output-uri-{idx}").as_bytes(),
    ))
}

/// See [`streaming_output_uri`]; the entry's `superfile_id` half.
fn streaming_output_id(compaction_id: Uuid, idx: usize) -> Uuid {
    Uuid::new_v5(
        &compaction_id,
        format!("clustered-streaming-output-id-{idx}").as_bytes(),
    )
}

/// Derive the publish artifacts for one final-pass shard, upload its
/// bytes immediately, and release them: the returned entry is
/// `storage_prewritten`, so the commit only swaps the manifest.
async fn prepare_streaming_output(
    inner: &Arc<SupertableInner>,
    storage: &Arc<dyn StorageProvider>,
    options: &Arc<SupertableOptions>,
    shard: ShardOutput,
    compaction_id: Uuid,
    idx: usize,
) -> Result<Option<PreparedSuperfile>, BuildError> {
    let uri = streaming_output_uri(compaction_id, idx);
    let Some(mut prepared) = prepare_superfile_with_uri(inner.as_ref(), shard, Some(uri))? else {
        return Ok(None);
    };
    let entry = Arc::new(SuperfileEntry {
        superfile_id: streaming_output_id(compaction_id, idx),
        ..(*prepared.entry).clone()
    });
    let (upload_uri, bytes) = prepared.bytes_for_storage.take().ok_or_else(|| {
        BuildError::Store("streaming clustered merge: prepared output lost its bytes".to_string())
    })?;
    put_new_superfile_bytes(
        storage,
        options.put_multipart_threshold_bytes,
        upload_uri,
        bytes,
    )
    .await
    .map_err(|e| BuildError::Store(format!("streaming clustered merge: upload: {e}")))?;
    Ok(Some(PreparedSuperfile {
        entry,
        // All byte dispositions are dropped: the upload above is the
        // durable copy, queries hydrate lazily (disk cache or storage
        // GET), and pinning target-size outputs here would defeat the
        // eager release the streaming path exists for.
        bytes_for_store: None,
        bytes_for_storage: None,
        bytes_for_cache: None,
        storage_prewritten: true,
    }))
}

#[cfg(test)]
mod tests {
    use std::iter;

    use arrow_array::{
        Array, Decimal128Array, Int64Array, StringArray, cast::AsArray, types::Decimal128Type,
    };
    use bytes::Bytes;
    use futures::{StreamExt, executor::block_on};

    use super::*;
    use crate::{
        superfile::builder::{BuilderOptions, SuperfileBuilder},
        test_helpers::{decimal128_id_field, decimal128_ids, default_vector_config},
    };

    /// Rows in the multi-batch fixture — enough that a bounded decode
    /// must span several batches.
    const MULTI_BATCH_ROWS: u64 = 20_000;

    /// Generous pool for the merge unit tests — pool sizing is
    /// exercised by the compaction-level tests, not here.
    const TEST_POOL_BYTES: usize = 1 << 20;

    /// `doc_id`-only schema for the scalar stream fixtures.
    fn id_only_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![decimal128_id_field("doc_id")]))
    }

    /// One superfile holding `ids` (in the given order) and nothing else.
    fn id_only_superfile(ids: &[u64]) -> SuperfileReader {
        let schema = id_only_schema();
        let opts = BuilderOptions::new(schema.clone(), "doc_id", vec![], vec![], None);
        let mut b = SuperfileBuilder::new(opts).expect("builder");
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(decimal128_ids(ids.iter().copied())) as ArrayRef],
        )
        .expect("batch matches schema");
        b.add_batch(&batch, &[]).expect("add batch");
        let bytes = Bytes::from(b.finish().expect("finish"));
        SuperfileReader::open(bytes).expect("open superfile")
    }

    /// Collect a stream into its batches, panicking on any item error.
    fn collect_batches(stream: SendableRecordBatchStream) -> Vec<RecordBatch> {
        block_on(stream.collect::<Vec<_>>())
            .into_iter()
            .map(|item| item.expect("stream item decodes"))
            .collect()
    }

    /// The `doc_id` values of `batches`, concatenated in stream order.
    fn ids_of(batches: &[RecordBatch]) -> Vec<i128> {
        batches
            .iter()
            .flat_map(|batch| {
                let ids = batch
                    .column_by_name("doc_id")
                    .expect("doc_id column")
                    .as_primitive::<Decimal128Type>()
                    .clone();
                (0..ids.len()).map(move |i| ids.value(i))
            })
            .collect()
    }

    /// A run stream must yield every live row in file row order, in
    /// batches bounded by [`STREAMING_MERGE_BATCH_ROWS`], and a file
    /// larger than one batch must span several.
    #[test]
    fn stream_preserves_order_and_bounds_batches() {
        let ids: Vec<u64> = (0..MULTI_BATCH_ROWS).collect();
        let reader = id_only_superfile(&ids);
        let schema = reader.schema().clone();
        let stream =
            sorted_run_stream(&reader, None, schema, Arc::from([])).expect("run stream builds");
        let batches = collect_batches(stream);

        assert!(
            batches.len() >= 2,
            "a {MULTI_BATCH_ROWS}-row file must stream as several bounded batches"
        );
        assert!(
            batches
                .iter()
                .all(|b| b.num_rows() <= STREAMING_MERGE_BATCH_ROWS),
            "every batch must respect the row bound"
        );
        let expected: Vec<i128> = (0..MULTI_BATCH_ROWS as i128).collect();
        assert_eq!(ids_of(&batches), expected, "file row order must survive");
    }

    /// Tombstoned rows are filtered out of the stream; the survivors
    /// keep their relative order.
    #[test]
    fn stream_drops_tombstoned_rows() {
        let reader = id_only_superfile(&[10, 11, 12, 13, 14, 15]);
        let mut deleted = RoaringBitmap::new();
        deleted.insert(1); // id 11
        deleted.insert(4); // id 14
        let schema = reader.schema().clone();
        let stream = sorted_run_stream(&reader, Some(Arc::new(deleted)), schema, Arc::from([]))
            .expect("run stream builds");
        let batches = collect_batches(stream);
        assert_eq!(ids_of(&batches), vec![10, 12, 13, 15]);
    }

    /// Vector rows ride the stream row-aligned with their scalars:
    /// after a tombstone filter, each surviving row's `FixedSizeList`
    /// entry still holds that row's own vector.
    #[test]
    fn stream_attaches_vectors_row_aligned() {
        const DIM: usize = 16;
        const ROWS: usize = 8;
        let schema = id_only_schema();
        let vector_config = default_vector_config("emb", 42);
        let opts = BuilderOptions::new(
            schema.clone(),
            "doc_id",
            vec![],
            vec![vector_config.clone()],
            None,
        );
        let mut b = SuperfileBuilder::new(opts).expect("builder");
        let ids: Vec<u64> = (0..ROWS as u64).collect();
        // Row r's vector is [r+1.0; DIM] — distinguishable per row.
        let flat: Vec<f32> = (0..ROWS)
            .flat_map(|r| iter::repeat_n(r as f32 + 1.0, DIM))
            .collect();
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(decimal128_ids(ids.iter().copied())) as ArrayRef],
        )
        .expect("batch matches schema");
        b.add_batch(&batch, &[&flat]).expect("add batch");
        let bytes = Bytes::from(b.finish().expect("finish"));
        let reader = SuperfileReader::open(bytes).expect("open superfile");

        let shapes: Arc<[VectorShape]> = Arc::from([VectorShape {
            name: "emb".to_string(),
            dim: DIM,
        }]);
        let stream_schema =
            merged_stream_schema(reader.schema(), &shapes).expect("stream schema builds");
        let mut deleted = RoaringBitmap::new();
        deleted.insert(2);
        let stream = sorted_run_stream(
            &reader,
            Some(Arc::new(deleted)),
            stream_schema,
            Arc::clone(&shapes),
        )
        .expect("run stream builds");
        let batches = collect_batches(stream);

        let survivors: Vec<i128> = vec![0, 1, 3, 4, 5, 6, 7];
        assert_eq!(ids_of(&batches), survivors);
        let batch = &batches[0];
        let ids = batch
            .column_by_name("doc_id")
            .expect("doc_id column")
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("doc_id is Decimal128")
            .clone();
        let lists = batch
            .column_by_name("emb")
            .expect("emb column")
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .expect("emb is FixedSizeList")
            .clone();
        for row in 0..batch.num_rows() {
            let values = lists.value(row);
            let values = values
                .as_any()
                .downcast_ref::<Float32Array>()
                .expect("f32 items");
            let expected = ids.value(row) as f32 + 1.0;
            assert_eq!(values.len(), DIM);
            assert!(
                (0..DIM).all(|i| values.value(i) == expected),
                "row {row} lost its own vector"
            );
        }
    }

    /// Nullable `[category: Utf8?, val: Int64]` schema for the merge
    /// fixtures.
    fn category_val_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("category", DataType::Utf8, true),
            Field::new("val", DataType::Int64, false),
        ]))
    }

    /// One handmade sorted-run stream over [`category_val_schema`] rows
    /// (already ascending, nulls last — the writer's key order).
    fn handmade_sorted_run(rows: &[(Option<&str>, i64)]) -> SendableRecordBatchStream {
        let schema = category_val_schema();
        let categories = StringArray::from(rows.iter().map(|(c, _)| *c).collect::<Vec<_>>());
        let vals = Int64Array::from(rows.iter().map(|(_, v)| *v).collect::<Vec<_>>());
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(categories), Arc::new(vals)],
        )
        .expect("batch matches schema");
        Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::iter(vec![Ok(batch)]),
        ))
    }

    /// Three overlapping sorted runs — duplicate keys across runs, nulls
    /// in the key — merge into ONE globally sorted stream: ascending,
    /// nulls last, every input row present exactly once.
    #[test]
    fn merge_overlapping_runs_into_one_sorted_stream() {
        let runs = vec![
            handmade_sorted_run(&[
                (Some("alpha"), 1),
                (Some("bravo"), 2),
                (Some("bravo"), 3),
                (Some("delta"), 4),
                (None, 5),
            ]),
            handmade_sorted_run(&[(Some("alpha"), 6), (Some("charlie"), 7), (None, 8)]),
            handmade_sorted_run(&[(Some("bravo"), 9), (Some("delta"), 10), (Some("echo"), 11)]),
        ];
        let merged = merge_sorted_runs(
            runs,
            &category_val_schema(),
            &["category".to_string()],
            TEST_POOL_BYTES,
        )
        .expect("merge builds");
        let batches = collect_batches(merged);

        let mut categories: Vec<Option<String>> = Vec::new();
        let mut vals: Vec<i64> = Vec::new();
        for batch in &batches {
            assert!(batch.num_rows() <= STREAMING_MERGE_BATCH_ROWS);
            let cats = batch.column(0).as_string::<i32>();
            let v = batch
                .column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("val is Int64");
            for row in 0..batch.num_rows() {
                categories.push((!cats.is_null(row)).then(|| cats.value(row).to_string()));
                vals.push(v.value(row));
            }
        }
        let expected = [
            Some("alpha"),
            Some("alpha"),
            Some("bravo"),
            Some("bravo"),
            Some("bravo"),
            Some("charlie"),
            Some("delta"),
            Some("delta"),
            Some("echo"),
            None,
            None,
        ]
        .map(|c| c.map(str::to_string))
        .to_vec();
        assert_eq!(
            categories, expected,
            "merged keys must be globally ascending with nulls last"
        );
        let mut seen = vals.clone();
        seen.sort_unstable();
        assert_eq!(
            seen,
            (1..=11).collect::<Vec<i64>>(),
            "every input row must appear exactly once"
        );
    }

    /// The retry trigger: only the memory pool's refusal is retryable
    /// (directly or wrapped), everything else is fatal — a retry on a
    /// fatal error would loop on a pass that can never succeed.
    #[test]
    fn classify_merge_error_retries_only_pool_exhaustion() {
        let exhausted = DataFusionError::ResourcesExhausted("pool".to_string());
        assert!(matches!(
            classify_merge_error(exhausted),
            PassError::PoolExhausted(_)
        ));

        // Wrapped exhaustion (context frames) still classifies by root.
        let wrapped = DataFusionError::ResourcesExhausted("pool".to_string())
            .context("while merging sorted runs");
        assert!(matches!(
            classify_merge_error(wrapped),
            PassError::PoolExhausted(_)
        ));

        let fatal = DataFusionError::External(Box::new(BuildError::Store("io".to_string())));
        assert!(matches!(classify_merge_error(fatal), PassError::Fatal(_)));
    }
}
