// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Picks which superfiles to merge.
//!
//! no I/O. `supertable::compact` gathers the
//! stats, calls [`select`], then merges each [`CompactionJob`].
//! Compaction is single-level — a target-sized superfile is never
//! re-compacted.

mod streaming;

use std::{
    collections::{BTreeMap, HashMap},
    mem,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

use arrow_array::Float32Array;
use bytes::Bytes;
use chrono::Utc;
use futures::{
    future::join_all,
    stream::{self, StreamExt},
};
use roaring::RoaringBitmap;
use tokio::{sync::oneshot, time};
use tracing::{debug, warn};
use uuid::Uuid;

use crate::{
    Supertable,
    config::CompactionSettings,
    runtime_bridge::bridge_on_runtime,
    superfile::{
        SuperfileReader,
        builder::{BuilderOptions, SuperfileBuilder, merge_rows_from_reader},
        error::BuildError as SuperfileBuildError,
        vector::layout::VectorLayout,
    },
    supertable::{
        BuildError, CommitError, ManifestSnapshot, SuperfileEntry, SuperfileUri, SupertableOptions,
        build::fanout_shards_in_pool_scope,
        compaction::streaming::streaming_clustered_merge,
        error::CompactionError,
        handle::{hidden_vector_index_compaction_settings, is_hidden_vector_index_table},
        manifest::list::{DrainedVersionRanges, PartitionStrategy},
        query::dispatch::open_compaction_input,
        wal::{
            Etag, SealRecord, TombstonesSidecar, WalStore,
            tombstones_admin::{self, TombstonesAdminError},
        },
        writer::{
            BufferedBatch, NewEntryBirthVersions, PreparedSuperfile, ShardOutput, backoff_delay,
            build_one_shard_with_layout, finalize_compaction_commit, prepare_superfile,
            refresh_slow_vector_state, sort_buffer_by_cluster_key, split_buffer_into_row_shards,
            split_overflow_cells, try_commit_attempt,
        },
    },
};

struct CompactionSlot<'a>(&'a AtomicBool);

impl Drop for CompactionSlot<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

const MIB: u64 = 1024 * 1024;

/// How many multiples of the raw input size an unordered merge reserves from
/// the connection budget: the merge still materializes every input at once,
/// and the builder holds roughly another copy until `finish`.
const MERGE_MEMORY_RESERVE_FACTOR: u64 = 2;

/// A clustered merge row-encodes the key columns and gathers a sorted copy
/// of the payload in bounded chunks while the materialized inputs are still
/// alive, so its transient peak is roughly the unordered merge plus that
/// sorted copy and the encoded keys. One factor above the unordered merge
/// is deliberately conservative headroom: these factors meter compressed
/// input bytes against decompressed in-memory transients.
const CLUSTERED_MERGE_MEMORY_RESERVE_FACTOR: u64 = 3;

/// The engagement threshold for the streaming clustered merge: the
/// in-memory route reserves [`CLUSTERED_MERGE_MEMORY_RESERVE_FACTOR`]
/// times the raw input bytes, so once that reserve would cross the
/// compaction memory ceiling the job routes through the streaming
/// k-way merge, which bounds its decoded working set to the ceiling
/// itself. Below the threshold the in-memory sort stays — it is one
/// pass with no per-batch bookkeeping and its transients already fit.
fn clustered_merge_needs_streaming(input_bytes: u64, max_memory_bytes: u64) -> bool {
    input_bytes.saturating_mul(CLUSTERED_MERGE_MEMORY_RESERVE_FACTOR) > max_memory_bytes
}

/// Whether a clustered table can take the streaming merge at all:
/// durable storage must be attached (outputs upload as they are cut),
/// and no vector column may use an sq8-family rerank codec — those
/// merges byte-splice their vector blobs in row order and cannot
/// re-sort, so they keep today's in-memory fallback. Cell-partitioned
/// vector tables are excluded separately at both call sites (their
/// manifests carry the `VectorCell` strategy), matching the reader-set
/// multi-cell fallback in [`Supertable::merge_superfiles`].
fn clustered_streaming_available(options: &SupertableOptions) -> bool {
    options.storage.is_some()
        && !options
            .vector_columns
            .iter()
            .any(|vc| vc.rerank_codec.is_sq8_residual_family())
}

/// Stats for one superfile. The caller fills these in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SuperfileStats {
    pub superfile_id: Uuid,
    /// Partition it belongs to.
    /// never merge across partitions.
    pub partition_key: Vec<u8>,
    pub size_bytes: u64,
    pub n_docs: u64,
    pub tombstoned_docs: u64,
    /// Already owned by another compaction so skip it.
    pub sealed_by_other: bool,
    /// Commit version the superfile was born at. A merged superfile carries
    /// the OLDEST input's `birth_version`, so user-table merge jobs must
    /// never mix inputs from opposite sides of the hidden drain watermark
    /// (see [`split_stats_at_drain_watermark`]).
    pub birth_version: u64,
}

impl SuperfileStats {
    fn live_docs(&self) -> u64 {
        self.n_docs.saturating_sub(self.tombstoned_docs)
    }

    /// Bytes left after dropping deleted rows.
    fn live_bytes(&self) -> u64 {
        if self.n_docs == 0 {
            return 0;
        }
        (self.size_bytes as u128 * self.live_docs() as u128 / self.n_docs as u128) as u64
    }
}

/// Split merge candidates at the hidden drain watermark: inputs whose
/// `birth_version` the hidden index has already drained versus inputs it has
/// not. A merged superfile is stamped with the OLDEST input `birth_version`
/// (see `run_compaction_job`), so a job mixing the two sides would inherit a
/// drained version and the drain's `!drained.contains(birth_version)` filter
/// would skip it — the undrained inputs' vectors would silently never enter
/// the hidden index (a permanent recall hole). Merging within either side is
/// safe: all-drained stays drained, all-undrained keeps an undrained version
/// and is drained as one source.
fn split_stats_at_drain_watermark(
    stats: Vec<SuperfileStats>,
    drained: &DrainedVersionRanges,
) -> (Vec<SuperfileStats>, Vec<SuperfileStats>) {
    stats
        .into_iter()
        .partition(|s| drained.contains(s.birth_version))
}

/// A set of superfiles to merge into one new superfile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactionJob {
    pub partition_key: Vec<u8>,
    pub inputs: Vec<Uuid>,
    /// Estimated size of the merged superfile.
    pub estimated_output_bytes: u64,
}

/// Plan compaction: pack each partition's small superfiles into
/// as many target-sized jobs as they fill. Leftovers that can't
/// reach the floor are left for next time.
pub fn select(superfiles: &[SuperfileStats], cfg: &CompactionSettings) -> Vec<CompactionJob> {
    let target_bytes = cfg.target_superfile_size_mb.saturating_mul(MIB);
    // `0%` disables the size leg: a merge then fires on the fragment-count
    // floor alone (>= 2 inputs), which is how the hidden index consolidates
    // drain generations that are far below any byte threshold.
    let min_output_bytes =
        (target_bytes as u128 * cfg.min_fill_percent.clamp(0, 100) as u128 / 100) as u64;
    let max_memory_bytes = cfg.max_memory_mb.saturating_mul(MIB);

    let mut by_partition: BTreeMap<&[u8], Vec<&SuperfileStats>> = BTreeMap::new();
    for s in superfiles {
        by_partition.entry(&s.partition_key).or_default().push(s);
    }

    let mut jobs = Vec::new();
    for (key, segs) in by_partition {
        pack_partition(
            key,
            segs,
            target_bytes,
            min_output_bytes,
            max_memory_bytes,
            &mut jobs,
        );
    }
    jobs
}

fn pack_partition(
    key: &[u8],
    segs: Vec<&SuperfileStats>,
    target_bytes: u64,
    min_output_bytes: u64,
    max_memory_bytes: u64,
    jobs: &mut Vec<CompactionJob>,
) {
    // Exclude superfiles already at target size — they are done and
    // re-compacting them gains nothing.
    let mut candidates: Vec<&SuperfileStats> = segs
        .into_iter()
        .filter(|s| !s.sealed_by_other && s.size_bytes < target_bytes)
        .collect();

    // Most-deleted first (reclaim space soonest), then smallest, then ID.
    candidates.sort_by(|a, b| {
        let lhs = a.tombstoned_docs as u128 * b.n_docs.max(1) as u128;
        let rhs = b.tombstoned_docs as u128 * a.n_docs.max(1) as u128;
        rhs.cmp(&lhs)
            .then(a.size_bytes.cmp(&b.size_bytes))
            .then(a.superfile_id.cmp(&b.superfile_id))
    });

    let mut pending = PendingJob::default();
    for s in candidates {
        if !pending.fits(s, target_bytes, max_memory_bytes) {
            pending.emit(key, min_output_bytes, jobs);
        }
        pending.push(s);
    }
    pending.emit(key, min_output_bytes, jobs);
}

#[derive(Default)]
struct PendingJob {
    inputs: Vec<Uuid>,
    live_bytes: u64,
    raw_bytes: u64,
}

impl PendingJob {
    fn fits(&self, s: &SuperfileStats, target_bytes: u64, max_memory_bytes: u64) -> bool {
        self.live_bytes + s.live_bytes() <= target_bytes
            && self.raw_bytes + s.size_bytes <= max_memory_bytes
    }

    fn push(&mut self, s: &SuperfileStats) {
        self.raw_bytes += s.size_bytes;
        self.inputs.push(s.superfile_id);
        self.live_bytes += s.live_bytes();
    }

    /// Emit a CompactionJob if ≥ 2 inputs and live bytes reach `min_output_bytes`.
    fn emit(&mut self, key: &[u8], min_output_bytes: u64, jobs: &mut Vec<CompactionJob>) {
        if self.inputs.len() >= 2 && self.live_bytes >= min_output_bytes {
            jobs.push(CompactionJob {
                partition_key: key.to_vec(),
                inputs: mem::take(&mut self.inputs),
                estimated_output_bytes: self.live_bytes,
            });
        }
        *self = PendingJob::default();
    }
}

impl Supertable {
    /// Compaction entry point.
    /// Gathers per-superfile stats from the current manifest snapshot,
    /// selects compaction jobs, then for each job seals every input
    /// superfile's tombstone sidecar so no concurrent deletes can land
    /// during the merge window.
    pub(crate) fn compact(&self, cfg: &CompactionSettings) -> Result<(), CompactionError> {
        bridge_on_runtime(self.compact_async(cfg), &self.inner().query_runtime())
    }

    pub(crate) async fn compact_async(
        &self,
        cfg: &CompactionSettings,
    ) -> Result<(), CompactionError> {
        Self::compact_one_table(self, cfg).await?;
        if matches!(
            self.inner().manifest.load().get_partition_strategy(),
            PartitionStrategy::VectorCell { .. }
        ) {
            refresh_slow_vector_state(self.inner())
                .await
                .map_err(|error| CompactionError::Refresh(error.to_string()))?;
        } else if let Some(hidden) = self.inner().vector_index_table.as_ref() {
            Self::compact_one_table(hidden, &hidden_vector_index_compaction_settings()).await?;
            // The hidden pass settled vector membership (merges + finalize +
            // any cell splits); its `update`s cleared the slow-CAS ref, so
            // republish the entry blob and restamp. Hidden tables have no
            // manifest parts, so publication is required for reopen and a
            // failure must be visible to the caller.
            refresh_slow_vector_state(hidden.inner())
                .await
                .map_err(|error| CompactionError::Refresh(error.to_string()))?;
        }
        Ok(())
    }

    pub(crate) async fn compact_one_table(
        table: &Supertable,
        cfg: &CompactionSettings,
    ) -> Result<(), CompactionError> {
        let inner = table.inner();

        match inner.compaction_outstanding.compare_exchange(
            false,
            true,
            Ordering::Acquire,
            Ordering::Relaxed,
        ) {
            Ok(_) => {}
            Err(_) => return Err(CompactionError::AlreadyCompacting),
        }
        let _slot = CompactionSlot(&inner.compaction_outstanding);

        // Phase 1 (split-then-merge): split every over-cap cell first, from the
        // live grid, before merge-job selection. An over-cap cell is thus never
        // merged just to be re-split (the merge output would be discarded), and
        // the split runs as its own snapshot-consistent phase, so it can't remove
        // a superfile a later merge job in this pass planned to use.
        if is_hidden_vector_index_table(&inner.options) {
            split_overflow_cells(Arc::clone(inner))
                .await
                .map_err(|e| CompactionError::Build(e.to_string()))?;
        }

        let manifest = inner.manifest.load_full();

        // Prefetch sidecars using the cache to batch storage GETs.
        // This populates both bitmap and seal information for all superfiles.
        // The cache returns empty bitmaps for superfiles without tombstones.
        let superfile_ids: Vec<Uuid> = manifest
            .get_all_superfiles()
            .iter()
            .map(|e| e.superfile_id)
            .collect();

        let sidecar_map: HashMap<Uuid, (Arc<RoaringBitmap>, Option<SealRecord>)> =
            if let Some(cache) = &inner.tombstone_cache {
                let now = Instant::now();
                cache.prefetch(&superfile_ids, now).await;

                // Build a map of superfile_id → (bitmap, seal) by checking the cache.
                // Cache hits are O(1); any misses are already prefetched above.
                superfile_ids
                    .iter()
                    .filter_map(|id| match cache.sidecar_for(*id, now) {
                        Ok((bitmap, seal)) => Some((*id, (bitmap, seal))),
                        Err(_) => None,
                    })
                    .collect()
            } else {
                // Fallback for in-memory-only tables (no storage, no tombstone cache).
                HashMap::new()
            };

        // Build SuperfileStats for every superfile in the snapshot.
        let now = Utc::now();
        let stale_seal_timeout = std::time::Duration::from_millis(cfg.stale_seal_timeout_ms);
        let stats: Vec<SuperfileStats> = manifest
            .get_all_superfiles()
            .iter()
            .map(|entry| {
                let (bitmap, seal) = sidecar_map
                    .get(&entry.superfile_id)
                    .cloned()
                    .unwrap_or_else(|| (Arc::new(RoaringBitmap::new()), None));
                let tombstoned_docs = bitmap.len();
                let sealed_by_other = seal.as_ref().is_some_and(|s| {
                    !tombstones_admin::is_seal_stale(s.sealed_at, now, stale_seal_timeout)
                });
                SuperfileStats {
                    superfile_id: entry.superfile_id,
                    partition_key: entry.partition_key.clone(),
                    size_bytes: entry
                        .subsection_offsets
                        .as_ref()
                        .map(|o| o.total_size)
                        .unwrap_or(0),
                    n_docs: entry.n_docs,
                    tombstoned_docs,
                    sealed_by_other,
                    birth_version: entry.birth_version,
                }
            })
            .collect();

        // A user table with a hidden vector index selects jobs per side of
        // the drain watermark, never across it (see
        // [`split_stats_at_drain_watermark`] for why a mixed merge loses
        // vectors). Tables without a hidden sibling select over everything.
        let stat_groups: Vec<Vec<SuperfileStats>> = match inner.vector_index_table.as_ref() {
            Some(hidden) => {
                let drained = hidden.inner().manifest.load_full().get_drained_ranges();
                let (drained_stats, undrained_stats) =
                    split_stats_at_drain_watermark(stats, &drained);
                vec![drained_stats, undrained_stats]
            }
            None => vec![stats],
        };
        // Clustered tables fuse same-partition sibling jobs so one global
        // order can slice them into range-disjoint outputs; each fused job
        // then asks the merge for enough outputs to stay near the target
        // size. When the streaming merge is available a fused job may
        // exceed the in-memory ceiling — the merge streams it under that
        // ceiling instead of degrading to overlapping per-job sorts.
        // Without it (no storage, sq8 vectors) fusion keeps the ceiling as
        // its cap, exactly the pre-streaming behavior.
        let clustered = !inner.options.cluster_by.is_empty();
        let target_bytes = cfg.target_superfile_size_mb.saturating_mul(MIB);
        let max_memory_bytes = cfg.max_memory_mb.saturating_mul(MIB);
        let cell_partitioned = matches!(
            manifest.get_partition_strategy(),
            PartitionStrategy::VectorCell { .. }
        );
        let fusion_cap_bytes = if !cell_partitioned && clustered_streaming_available(&inner.options)
        {
            u64::MAX
        } else {
            max_memory_bytes
        };
        for stats in &stat_groups {
            let mut jobs = select(stats, cfg);
            if clustered {
                jobs = coalesce_clustered_jobs(jobs, stats, fusion_cap_bytes);
            }
            for job in jobs {
                let n_outputs = if clustered {
                    clustered_output_count(job.estimated_output_bytes, target_bytes)
                } else {
                    1
                };
                table
                    .run_compaction_job(job, n_outputs, stale_seal_timeout, max_memory_bytes)
                    .await?;
                table
                    .refresh()
                    .await
                    .map_err(|e| CompactionError::Refresh(e.to_string()))?;
            }
        }

        Ok(())
    }

    /// Merges the given superfiles.
    ///
    /// Unclustered tables (and the sq8 IVF vector routes, whose row order is
    /// dictated by the vector layout) produce exactly one merged superfile
    /// with rows in input-concatenation order — `n_outputs` is ignored.
    ///
    /// When the table declares a clustering key, the merge instead re-sorts
    /// the combined live rows by the key (the same lexicographic ascending,
    /// nulls-last order every commit is written in) and splits the one
    /// sorted run into up to `n_outputs` contiguous chunks, so consecutive
    /// outputs carry non-overlapping key ranges and the ordered scan path
    /// can fire across them. A clustered job whose in-memory sort would
    /// exceed `max_memory_bytes` (see [`clustered_merge_needs_streaming`])
    /// takes the streaming k-way merge instead — same order, same output
    /// shape, decoded working set bounded by the ceiling. `compaction_id`
    /// seeds the streaming outputs' deterministic ids; the in-memory
    /// routes ignore it.
    pub(crate) async fn merge_superfiles(
        &self,
        superfiles: &[Arc<SuperfileEntry>],
        n_outputs: usize,
        compaction_id: Uuid,
        max_memory_bytes: u64,
    ) -> Result<Vec<PreparedSuperfile>, BuildError> {
        let manifest = { self.inner().manifest.load().clone() };
        let store = manifest.options.store.clone();
        let disk_cache = manifest.options.disk_cache.clone();
        let storage = manifest.options.storage.clone();
        let tombstone_cache = self.inner().tombstone_cache.clone();
        let clustered = !self.inner().options.cluster_by.is_empty();

        // The in-memory routes reserve budget for the whole input size
        // since they load it all at once; the streaming clustered route
        // reserves the compaction ceiling instead — that is the bound its
        // merge pool enforces on decoded transients.
        let input_bytes: u64 = superfiles
            .iter()
            .map(|e| e.subsection_offsets.as_ref().map_or(0, |o| o.total_size))
            .sum();
        // Cell-partitioned vector tables never reach the row-materializing
        // clustered branch below (their reader sets are multi-cell), so
        // they must not route to streaming either — the two decisions are
        // kept in lockstep here so the reservation always matches the
        // route actually taken.
        let cell_partitioned = matches!(
            manifest.get_partition_strategy(),
            PartitionStrategy::VectorCell { .. }
        );
        let streaming = clustered
            && !cell_partitioned
            && clustered_streaming_available(&self.inner().options)
            && clustered_merge_needs_streaming(input_bytes, max_memory_bytes);
        // Multiply the input bytes to cover the merge buffer and overhead; a
        // clustered merge additionally holds the sorted chunk copy and the
        // row-encoded key columns while the inputs are still alive.
        let reserve_factor = if clustered {
            CLUSTERED_MERGE_MEMORY_RESERVE_FACTOR
        } else {
            MERGE_MEMORY_RESERVE_FACTOR
        };
        let estimated_bytes = if streaming {
            max_memory_bytes
        } else {
            input_bytes.saturating_mul(reserve_factor)
        } as usize;
        let _memory_reservation = manifest
            .options
            .connection_memory_budget
            .try_reserve(estimated_bytes)
            .map_err(|e| BuildError::MemoryBudgetExceeded(e.to_string()))?;

        let mut superfile_readers_fut = Vec::with_capacity(superfiles.len());
        for entry in superfiles {
            let open_fut = async {
                let r = open_compaction_input(&store, disk_cache.as_ref(), storage.as_ref(), entry)
                    .await;
                (entry.superfile_id, r)
            };
            superfile_readers_fut.push(open_fut);
        }
        let readers = join_all(superfile_readers_fut).await;

        let now = Instant::now();
        if let Some(tombstone_cache) = &tombstone_cache {
            let superfile_ids = superfiles
                .iter()
                .map(|entry| entry.superfile_id)
                .collect::<Vec<_>>();

            tombstone_cache.prefetch(&superfile_ids, now).await;
        }

        let mut readers_with_tombstones = Vec::with_capacity(readers.len());
        for (superfile_id, reader) in readers {
            let bitmap = tombstone_cache
                .as_ref()
                .map(|t| t.bitmap_for(superfile_id, now))
                .transpose()
                .map_err(|e| BuildError::Store(e.to_string()))?;

            let reader = reader.map_err(|e| BuildError::Store(e.to_string()))?;
            readers_with_tombstones.push((reader.clone(), bitmap));
        }

        let first_vec = readers_with_tombstones
            .first()
            .and_then(|(reader, _)| reader.vec());
        let multi_cell = first_vec.is_some_and(|v| v.is_multi_cell());
        let sq8_merge = first_vec.and_then(|v| {
            v.vector_columns_config()
                .next()
                .map(|c| c.rerank_codec.is_sq8_residual_family())
        });

        // Clustered merge (row-materializing route only): sort the combined
        // live rows by the key and build one superfile per contiguous chunk.
        // The sq8 IVF routes byte-splice their vector blobs and pin the row
        // order to the vector layout, so they cannot re-sort; their merged
        // ranges simply keep overlapping and the scan stays on its unordered
        // fallback for those tables.
        if clustered && !multi_cell && sq8_merge != Some(true) {
            if streaming {
                let report = streaming_clustered_merge(
                    self.inner(),
                    readers_with_tombstones,
                    n_outputs,
                    compaction_id,
                    max_memory_bytes,
                )
                .await?;
                debug!(
                    inputs = superfiles.len(),
                    outputs = report.prepared.len(),
                    cascade_folds = report.cascade_folds,
                    "compact: streaming clustered merge"
                );
                if report.prepared.is_empty() {
                    return Err(BuildError::NoDocsToBuild);
                }
                return Ok(report.prepared);
            }
            let shard_outputs = clustered_merge_shards(
                Arc::clone(&self.inner().options),
                readers_with_tombstones,
                n_outputs,
            )
            .await?;
            let mut prepared = Vec::with_capacity(shard_outputs.len());
            for shard in shard_outputs {
                if let Some(superfile) = prepare_superfile(self.inner().as_ref(), shard)? {
                    prepared.push(superfile);
                }
            }
            if prepared.is_empty() {
                return Err(BuildError::NoDocsToBuild);
            }
            return Ok(prepared);
        }

        let (merged_bytes, superfile_stats) = {
            if multi_cell && sq8_merge == Some(true) {
                SuperfileBuilder::build_from_multi_cell_sq8_ivf_readers(&readers_with_tombstones)?
            } else if sq8_merge == Some(true) {
                SuperfileBuilder::build_from_sq8_ivf_readers(&readers_with_tombstones)?
            } else {
                SuperfileBuilder::build_from_readers(&readers_with_tombstones)?
            }
        };
        let merged_bytes = Bytes::from(merged_bytes);

        let shard = ShardOutput::new_with_params(
            merged_bytes,
            superfile_stats.n_docs,
            superfile_stats.id_min,
            superfile_stats.id_max,
            superfile_stats.scalar_stats,
        );

        let prepared_superfile = prepare_superfile(self.inner().as_ref(), shard)?;

        prepared_superfile
            .map(|superfile| vec![superfile])
            .ok_or(BuildError::NoDocsToBuild)
    }

    /// Executes one compaction job: seal the inputs, merge them, and swap
    /// the merged output(s) into the manifest. `n_outputs` only matters for
    /// clustered tables, where it caps how many contiguous key-range chunks
    /// the merged rows split into (see [`Supertable::merge_superfiles`]);
    /// pass 1 everywhere else. `max_memory_bytes` is the compaction memory
    /// ceiling that decides between the in-memory and streaming clustered
    /// merges.
    pub(crate) async fn run_compaction_job(
        &self,
        job: CompactionJob,
        n_outputs: usize,
        stale_seal_timeout: std::time::Duration,
        max_memory_bytes: u64,
    ) -> Result<(), CompactionError> {
        let inner = self.inner();
        let manifest = inner.manifest.load_full();
        let storage = manifest
            .options
            .storage
            .as_ref()
            .ok_or(CompactionError::NoStorage)?
            .clone();
        let wal_store = WalStore::new(storage.clone());

        // Resolve input Arc<SuperfileEntry> from the snapshot.
        let inputs: Vec<Arc<SuperfileEntry>> = job
            .inputs
            .iter()
            .map(|id| {
                manifest
                    .get_all_superfiles()
                    .iter()
                    .find(|e| e.superfile_id == *id)
                    .cloned()
                    .ok_or(CompactionError::SuperfileNotFound(*id))
            })
            .collect::<Result<_, _>>()?;

        let opts = Arc::clone(&inner.options);
        let max_retries = opts.max_commit_retries.max(1);

        // Seal every input sidecar so no writer can land a tombstone
        // on a file that's about to disappear, and so another
        // compactor doesn't pick up the same inputs. If we die
        // before unsealing (crash, not a caught error), `seal`
        // itself lets a later compactor take over once the seal
        // goes stale.
        let compaction_id = Uuid::new_v4();
        let sealed_at = Utc::now();
        let mut sealed: Vec<SealedInput> = Vec::with_capacity(inputs.len());
        for entry in &inputs {
            let (sidecar, etag) = match seal_with_bounded_retry(
                &wal_store,
                entry.superfile_id,
                compaction_id,
                sealed_at,
                stale_seal_timeout,
                max_retries,
            )
            .await
            {
                Ok(v) => v,
                Err(e) => {
                    unseal_all(&wal_store, sealed).await;
                    return Err(e);
                }
            };
            sealed.push(SealedInput {
                superfile_id: entry.superfile_id,
                bitmap: sidecar.bitmap,
                etag,
            });
        }

        let merged_segments = match self
            .merge_superfiles(&inputs, n_outputs, compaction_id, max_memory_bytes)
            .await
        {
            Ok(segments) => segments,
            Err(e) => {
                unseal_all(&wal_store, sealed).await;
                return Err(CompactionError::Build(e.to_string()));
            }
        };

        // Carry the OLDEST input's birth_version so a merge of
        // already-drained inputs stays <= the drain watermark (skipped, not
        // re-drained). See the hidden-index `drained_ranges` design. Every
        // output of a multi-output clustered merge holds rows from any
        // input, so they all share the same conservative stamp.
        let birth_version = inputs.iter().map(|e| e.birth_version).min().unwrap_or(0);
        let partition_hint = inputs.first().and_then(|e| e.partition_hint);
        let vector_layout = inputs
            .first()
            .map(|e| e.vector_layout)
            .unwrap_or(VectorLayout::Ivf);

        let mut new_entries = Vec::with_capacity(merged_segments.len());
        let mut pending_storage_writes = Vec::with_capacity(merged_segments.len());
        let mut reader_cache_warms = Vec::new();
        let mut pending_cache_inserts = Vec::new();
        for segment in merged_segments {
            let PreparedSuperfile {
                entry: merged_prepared,
                bytes_for_store,
                bytes_for_storage,
                bytes_for_cache,
                storage_prewritten,
            } = segment;
            let merged_entry = Arc::new(SuperfileEntry {
                birth_version,
                // Left empty: the manifest's `update()` stamps the
                // partition key at commit time from `partition_hint`.
                partition_key: Vec::new(),
                partition_hint,
                vector_layout,
                ..(*merged_prepared).clone()
            });
            if let Some(warm) = bytes_for_store {
                reader_cache_warms.push((merged_entry.superfile_id, warm));
            }
            pending_cache_inserts.extend(bytes_for_cache);
            new_entries.push(merged_entry);
            if storage_prewritten {
                // Streaming clustered outputs are already durable at their
                // deterministic keys; the commit only swaps the manifest.
                debug_assert!(bytes_for_storage.is_none());
            } else {
                pending_storage_writes
                    .push(bytes_for_storage.ok_or(CompactionError::EmptyMergedSuperfile)?);
            }
        }

        for attempt in 0..max_retries {
            let current = inner.manifest.load_full();

            // Another compactor already merged our inputs — nothing left to commit.
            let entries_to_remove = match resolve_entries_to_remove(&current, &job.inputs) {
                Ok(entries) => entries,
                Err(_missing) => return Ok(()),
            };

            let mut pending_storage_replaces: Vec<(SuperfileUri, Bytes)> = Vec::new();

            match try_commit_attempt(
                storage.clone(),
                Arc::clone(&opts),
                current,
                &new_entries,
                &entries_to_remove,
                NewEntryBirthVersions::Preserve,
                &mut pending_storage_writes,
                &mut pending_storage_replaces,
            )
            .await
            {
                Ok(new_manifest) => {
                    inner.manifest.store(Arc::new(new_manifest));
                    // Warm the merged superfile(s) into the in-memory reader
                    // cache, same as a normal writer commit does. Without
                    // this every query against them misses and re-fetches +
                    // re-opens from storage every single time.
                    for (superfile_id, (uri, bytes)) in reader_cache_warms {
                        if let Err(e) = opts.store.insert(uri, bytes) {
                            warn!(
                                superfile_id = %superfile_id,
                                error = %e,
                                "compact: failed to warm reader cache for merged superfile"
                            );
                        }
                    }
                    // Drop the merged-away inputs so the in-memory cache
                    // doesn't grow forever across repeated compactions.
                    // The disk cache is already size-bounded (LRU), so its
                    // stale entries just age out on their own.
                    for entry in &entries_to_remove {
                        opts.store.remove(&entry.uri);
                    }
                    // Disk-cache warm + background storage reclaim ride the
                    // shared post-commit finalizer (the same path writer
                    // commits use), so the two paths can't drift.
                    finalize_compaction_commit(
                        Arc::clone(inner),
                        &storage,
                        &new_entries,
                        &entries_to_remove,
                        pending_cache_inserts,
                    )
                    .await;
                    return Ok(());
                }
                Err(CommitError::WriteContentionExhausted) if attempt + 1 < max_retries => {
                    if let Err(e) = self.refresh().await {
                        unseal_all(&wal_store, sealed).await;
                        return Err(CompactionError::Refresh(e.to_string()));
                    }
                    // Input vanished mid-retry (someone else merged it away).
                    // Our built output no longer matches reality, so abort
                    // instead of retrying the commit.
                    if let Err(missing) =
                        resolve_entries_to_remove(&inner.manifest.load_full(), &job.inputs)
                    {
                        unseal_all(&wal_store, sealed).await;
                        return Err(CompactionError::SuperfileNotFound(missing));
                    }
                    time::sleep(backoff_delay(attempt)).await;
                }
                Err(e) => {
                    unseal_all(&wal_store, sealed).await;
                    return Err(CompactionError::Commit(e.to_string()));
                }
            }
        }

        unseal_all(&wal_store, sealed).await;
        Err(CompactionError::Commit(
            "commit retries exhausted".to_string(),
        ))
    }
}

/// Ceiling count of target-sized outputs a clustered merge should split
/// into, from the job's live-byte estimate. A zero target (or estimate)
/// degrades to a single output.
fn clustered_output_count(estimated_live_bytes: u64, target_bytes: u64) -> usize {
    if target_bytes == 0 {
        return 1;
    }
    estimated_live_bytes.div_ceil(target_bytes).max(1) as usize
}

/// Fuse a clustered table's same-partition jobs into multi-output jobs.
///
/// `select` packs each partition's fragments into independent target-sized
/// jobs. Merging each of those on its own would sort each output
/// independently, leaving sibling outputs overlapping in key range, so the
/// scan's ordering declaration could never fire across them. Fusing the
/// siblings into one job makes the merge sort ALL their rows in a single
/// run and slice it into contiguous chunks — pairwise-disjoint outputs.
///
/// Fusion is capped at `max_memory_bytes` of raw input per fused job.
/// When the streaming clustered merge can serve the table the caller
/// lifts the cap (`u64::MAX`) — an over-ceiling fused job streams under
/// the ceiling instead of being split. Without the streaming route the
/// cap stays at the compaction memory ceiling, so a partition too big to
/// sort in one bounded pass degrades to per-job merges (each still
/// internally sorted, ranges possibly overlapping). Selection is
/// untouched: exactly the superfiles `select` picked get merged, only
/// the job boundaries move. Relies on `select` emitting same-partition
/// jobs consecutively (it packs partitions in `BTreeMap` order).
fn coalesce_clustered_jobs(
    jobs: Vec<CompactionJob>,
    superfiles: &[SuperfileStats],
    max_memory_bytes: u64,
) -> Vec<CompactionJob> {
    let size_by_id: HashMap<Uuid, u64> = superfiles
        .iter()
        .map(|s| (s.superfile_id, s.size_bytes))
        .collect();
    let raw_bytes = |job: &CompactionJob| -> u64 {
        job.inputs
            .iter()
            .map(|id| size_by_id.get(id).copied().unwrap_or(0))
            .sum()
    };
    let mut fused: Vec<(CompactionJob, u64)> = Vec::with_capacity(jobs.len());
    for job in jobs {
        let job_raw = raw_bytes(&job);
        match fused.last_mut() {
            Some((acc, acc_raw))
                if acc.partition_key == job.partition_key
                    && acc_raw.saturating_add(job_raw) <= max_memory_bytes =>
            {
                acc.inputs.extend(job.inputs);
                acc.estimated_output_bytes = acc
                    .estimated_output_bytes
                    .saturating_add(job.estimated_output_bytes);
                *acc_raw = acc_raw.saturating_add(job_raw);
            }
            _ => fused.push((job, job_raw)),
        }
    }
    fused.into_iter().map(|(job, _)| job).collect()
}

/// CPU half of a clustered merge, bridged onto the writer rayon pool via a
/// oneshot so the calling tokio worker keeps driving I/O while the sort and
/// the chunk builds run.
async fn clustered_merge_shards(
    options: Arc<SupertableOptions>,
    readers: Vec<(Arc<SuperfileReader>, Option<Arc<RoaringBitmap>>)>,
    n_outputs: usize,
) -> Result<Vec<ShardOutput>, BuildError> {
    let pool = Arc::clone(&options.writer_pool);
    let (tx, rx) = oneshot::channel();
    pool.spawn(move || {
        let result = clustered_merge_shards_in_pool(&options, &readers, n_outputs);
        let _ = tx.send(result);
    });
    rx.await
        .map_err(|_| BuildError::Store("clustered merge task dropped its result".to_string()))?
}

/// Body of [`clustered_merge_shards`]; runs on a writer-pool worker thread.
///
/// Materializes each input's live rows (the batch read already dropped
/// tombstoned rows, so the sort only ever sees survivors), runs them
/// through the same clustering sort every commit uses, splits the single
/// sorted run into `n_outputs` contiguous row chunks, and builds one
/// superfile per chunk on the pool. Because the chunks slice one globally
/// sorted run, consecutive outputs partition the key space in order instead
/// of being independently sorted, and each output's column statistics are
/// computed from exactly its own chunk.
fn clustered_merge_shards_in_pool(
    options: &SupertableOptions,
    readers: &[(Arc<SuperfileReader>, Option<Arc<RoaringBitmap>>)],
    n_outputs: usize,
) -> Result<Vec<ShardOutput>, BuildError> {
    let first = readers
        .first()
        .ok_or(BuildError::Superfile(SuperfileBuildError::BatchReadError))?;
    // Validate every input against the first one's shape, exactly like the
    // unordered merge (`SuperfileBuilder::build_from_readers`) does.
    let merge_opts = BuilderOptions::new_from_reader(&first.0);
    let mut buffered = Vec::with_capacity(readers.len());
    for (reader, bitmap) in readers {
        let (scalar, vectors) = merge_rows_from_reader(&merge_opts, reader, bitmap.clone())?;
        if scalar.num_rows() == 0 {
            // Fully tombstoned input: nothing live to carry forward.
            continue;
        }
        let vectors = vectors
            .into_iter()
            .map(|values| Arc::new(Float32Array::from(values)))
            .collect();
        buffered.push(BufferedBatch { scalar, vectors });
    }

    let sorted = sort_buffer_by_cluster_key(&buffered, options)?;
    drop(buffered);
    let total_rows: usize = sorted.iter().map(|b| b.scalar.num_rows()).sum();
    if total_rows == 0 {
        return Ok(Vec::new());
    }

    let vector_dims: Vec<usize> = options.vector_columns.iter().map(|vc| vc.dim).collect();
    let shards = split_buffer_into_row_shards(sorted, n_outputs.clamp(1, total_rows), &vector_dims);
    fanout_shards_in_pool_scope(&shards, |slice| {
        build_one_shard_with_layout(slice.as_slice(), options, options.vector_layout, None)
    })
}

/// One superfile this attempt sealed: enough to unseal it later with
/// no extra GET (`unseal` uses the etag + bitmap straight from `seal`).
struct SealedInput {
    superfile_id: Uuid,
    bitmap: RoaringBitmap,
    etag: Etag,
}

/// Cap on in-flight unseal calls. Single-writer model: one compactor
/// commits at a time, so there's no throughput reason to fire every
/// unseal at once.
const MAX_CONCURRENT_UNSEALS: usize = 8;

/// Best-effort: clear every seal this attempt placed. Each one is an
/// independent sidecar, so order doesn't matter, but they're bounded
/// to a small number in flight rather than all at once.
async fn unseal_all(wal_store: &WalStore, sealed: Vec<SealedInput>) {
    let results = stream::iter(sealed.into_iter().map(|s| {
        let wal_store = wal_store.clone();
        async move {
            let result =
                tombstones_admin::unseal(&wal_store, s.superfile_id, s.bitmap, &s.etag).await;
            (s.superfile_id, result)
        }
    }))
    .buffer_unordered(MAX_CONCURRENT_UNSEALS)
    .collect::<Vec<_>>()
    .await;
    for (superfile_id, result) in results {
        if let Err(e) = result {
            warn!(superfile_id = %superfile_id, error = %e, "compact: failed to unseal after aborting");
        }
    }
}

/// Look up `job_inputs` in `current`, in order. `Err` carries the first
/// missing id (removed by another compactor).
fn resolve_entries_to_remove(
    current: &ManifestSnapshot,
    job_inputs: &[Uuid],
) -> Result<Vec<Arc<SuperfileEntry>>, Uuid> {
    job_inputs
        .iter()
        .map(|id| {
            current
                .get_all_superfiles()
                .iter()
                .find(|e| e.superfile_id == *id)
                .cloned()
                .ok_or(*id)
        })
        .collect()
}

/// Seal one input, retrying a CAS race with a writer up to `max_retries`
/// times with backoff. `CasLost` just means a writer landed a tombstone
/// bit between our read and write — not an abandoned compaction.
async fn seal_with_bounded_retry(
    wal_store: &WalStore,
    superfile_id: Uuid,
    compaction_id: Uuid,
    sealed_at: chrono::DateTime<Utc>,
    stale_seal_timeout: std::time::Duration,
    max_retries: u32,
) -> Result<(TombstonesSidecar, Etag), CompactionError> {
    for attempt in 0..max_retries {
        match tombstones_admin::seal(
            wal_store,
            superfile_id,
            compaction_id,
            sealed_at,
            stale_seal_timeout,
        )
        .await
        {
            Ok(sealed) => return Ok(sealed),
            Err(TombstonesAdminError::CasLost { .. }) if attempt + 1 < max_retries => {
                time::sleep(backoff_delay(attempt)).await;
            }
            Err(TombstonesAdminError::CasLost { .. }) => {
                return Err(CompactionError::Seal("seal retries exhausted".to_string()));
            }
            Err(TombstonesAdminError::AlreadySealed {
                superfile_id,
                existing_compaction_id,
            }) => {
                return Err(CompactionError::SidecarConflict {
                    superfile_id,
                    existing_compaction_id,
                });
            }
            Err(TombstonesAdminError::WalStore(e)) => {
                return Err(CompactionError::Seal(e.to_string()));
            }
        }
    }
    Err(CompactionError::Seal("seal retries exhausted".to_string()))
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, mem, str, sync::Arc};

    use arrow_array::{ArrayRef, LargeStringArray, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::prelude::{col, lit};
    use tempfile::TempDir;
    use tokio::task;

    use super::*;
    use crate::{
        BoolMode, Supertable,
        config::DEFAULT_STALE_SEAL_TIMEOUT_MS,
        memory::ConnectionMemoryBudget,
        superfile::vector::rerank_codec::RerankCodec,
        supertable::{
            compaction::streaming::StreamingMergeReport,
            error::CompactionError,
            manifest::list::ScalarStatsAgg,
            storage::{LocalFsStorageProvider, StorageProvider},
        },
        test_helpers::{
            build_title_batch, decimal128_ids, default_supertable_options, default_vector_config,
        },
    };

    const DEFAULT_STALE_SEAL_TIMEOUT: std::time::Duration =
        std::time::Duration::from_millis(DEFAULT_STALE_SEAL_TIMEOUT_MS);

    /// A ceiling no test fixture ever exceeds: merges stay on the
    /// in-memory routes (`clustered_merge_needs_streaming` is false).
    const IN_MEMORY_MERGE_CEILING: u64 = u64::MAX;

    /// A ceiling every fixture exceeds: clustered merges take the
    /// streaming route with the minimum fan-in, so cascades engage
    /// whenever a job has more than two inputs.
    const STREAMING_MERGE_CEILING: u64 = 1;

    fn mib(n: u64) -> u64 {
        n * MIB
    }

    fn seg(id: u128, size_mib: u64, n_docs: u64, tombstoned: u64) -> SuperfileStats {
        SuperfileStats {
            superfile_id: Uuid::from_u128(id),
            partition_key: Vec::new(),
            size_bytes: mib(size_mib),
            n_docs,
            tombstoned_docs: tombstoned,
            sealed_by_other: false,
            birth_version: 0,
        }
    }

    /// Two mergeable fragments on opposite sides of the drain watermark must
    /// land in different selection groups: a single mixed job would stamp the
    /// merged superfile with the drained input's (older) `birth_version` and
    /// the drain would skip the undrained rows forever.
    #[test]
    fn drain_watermark_partition_never_mixes_drained_and_undrained() {
        // Watermark: versions 0..=10 drained.
        let drained = DrainedVersionRanges::from_intervals(vec![(0, 10)]).expect("valid intervals");
        let mut a = seg(1, 1, 1000, 0);
        a.birth_version = 5; // drained
        let mut b = seg(2, 1, 1000, 0);
        b.birth_version = 20; // undrained
        let mut c = seg(3, 1, 1000, 0);
        c.birth_version = 21; // undrained

        // Sanity: without the watermark split, selection would happily merge
        // all three into one job — the exact F1 hazard.
        let all = vec![a.clone(), b.clone(), c.clone()];
        let cfg = CompactionSettings {
            target_superfile_size_mb: 2048,
            min_fill_percent: 0,
            ..CompactionSettings::default()
        };
        let mixed = select(&all, &cfg);
        assert_eq!(mixed.len(), 1);
        assert_eq!(mixed[0].inputs.len(), 3, "guard: unsplit selection mixes");

        let (drained_side, undrained_side) = split_stats_at_drain_watermark(all, &drained);
        assert_eq!(
            drained_side
                .iter()
                .map(|s| s.superfile_id)
                .collect::<Vec<_>>(),
            vec![Uuid::from_u128(1)]
        );
        assert_eq!(undrained_side.len(), 2);
        // Group-wise selection: the drained side alone can't merge (one
        // input); the undrained side merges its two fragments.
        assert!(select(&drained_side, &cfg).is_empty());
        let jobs = select(&undrained_side, &cfg);
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].inputs.len(), 2);
        assert!(
            !jobs[0].inputs.contains(&Uuid::from_u128(1)),
            "undrained job must not contain the drained input"
        );
    }

    fn default_cfg() -> CompactionSettings {
        CompactionSettings::default() // 1 GiB target, 80% floor
    }

    #[test]
    fn empty_input_yields_no_jobs() {
        assert!(select(&[], &default_cfg()).is_empty());
    }

    #[test]
    fn below_fill_floor_skips() {
        // 400 MiB total < 80% of 1 GiB.
        let segs = vec![seg(1, 200, 1000, 0), seg(2, 200, 1000, 0)];
        assert!(select(&segs, &default_cfg()).is_empty());
    }

    #[test]
    fn packs_one_job_and_leaves_remainder() {
        // 6 × 200 MiB: one job of 5 (1000 MiB), 6th left over.
        let segs: Vec<_> = (0..6).map(|i| seg(i, 200, 1000, 0)).collect();
        let jobs = select(&segs, &default_cfg());
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].inputs.len(), 5);
        assert_eq!(jobs[0].estimated_output_bytes, mib(1000));
    }

    #[test]
    fn splits_many_superfiles_into_multiple_jobs() {
        // 12 × 200 MiB: two jobs of 5, last 2 left over.
        let segs: Vec<_> = (0..12).map(|i| seg(i, 200, 1000, 0)).collect();
        let jobs = select(&segs, &default_cfg());
        assert_eq!(jobs.len(), 2);
        assert!(jobs.iter().all(|j| j.inputs.len() == 5));
    }

    #[test]
    fn already_target_sized_superfile_is_never_re_compacted() {
        let big = seg(99, 1024, 1_000_000, 0);
        let mut segs = vec![big.clone()];
        segs.extend((0..5).map(|i| seg(i, 200, 1000, 0)));
        let jobs = select(&segs, &default_cfg());
        assert_eq!(jobs.len(), 1);
        assert!(!jobs[0].inputs.contains(&big.superfile_id));
    }

    #[test]
    fn output_estimate_uses_live_bytes() {
        // 5 × 400 MiB raw, half deleted → 200 MiB live each.
        let segs: Vec<_> = (0..5).map(|i| seg(i, 400, 1000, 500)).collect();
        let jobs = select(&segs, &default_cfg());
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].inputs.len(), 5);
        assert_eq!(jobs[0].estimated_output_bytes, mib(1000));
    }

    #[test]
    fn prefers_most_deleted_first() {
        let mut segs: Vec<_> = (0..9).map(|i| seg(i, 100, 1000, 0)).collect();
        let dead_heavy = seg(100, 100, 1000, 900);
        segs.push(dead_heavy.clone());
        let jobs = select(&segs, &default_cfg());
        assert_eq!(jobs[0].inputs[0], dead_heavy.superfile_id);
    }

    #[test]
    fn sealed_by_other_is_excluded() {
        let mut owned = seg(1, 200, 1000, 0);
        owned.sealed_by_other = true;
        let segs = vec![owned, seg(2, 200, 1000, 0), seg(3, 200, 1000, 0)];
        for job in select(&segs, &default_cfg()) {
            assert!(!job.inputs.contains(&Uuid::from_u128(1)));
        }
    }

    #[test]
    fn fewer_than_two_candidates_skips() {
        assert!(select(&[seg(1, 200, 1000, 0)], &default_cfg()).is_empty());
    }

    // ---- SuperfileStats live_docs / live_bytes -----------------------

    #[test]
    fn live_docs_subtracts_tombstones_and_saturates() {
        let s = seg(1, 100, 1000, 250);
        assert_eq!(s.live_docs(), 750);
        // More tombstones than docs saturates to zero rather than
        // underflowing.
        let over = seg(2, 100, 100, 200);
        assert_eq!(over.live_docs(), 0);
    }

    #[test]
    fn live_bytes_scales_by_live_fraction() {
        // 100 MiB, half the docs tombstoned → ~50 MiB live.
        let s = seg(1, 100, 1000, 500);
        assert_eq!(s.live_bytes(), mib(100) / 2);
    }

    #[test]
    fn live_bytes_zero_docs_is_zero() {
        // A 0-doc superfile must report 0 live bytes (guards the
        // division-by-zero branch).
        let s = seg(1, 100, 0, 0);
        assert_eq!(s.live_bytes(), 0);
    }

    // ---- PendingJob fits / push -------------------------------------

    #[test]
    fn pending_job_fits_until_target_exceeded() {
        let target = mib(100);
        let max_memory = mib(1000);
        let mut p = PendingJob::default();
        let a = seg(1, 60, 1000, 0); // 60 MiB live
        assert!(p.fits(&a, target, max_memory));
        p.push(&a);
        assert_eq!(p.live_bytes, mib(60));
        assert_eq!(p.inputs.len(), 1);
        // A second 60 MiB superfile would overflow the 100 MiB target.
        let b = seg(2, 60, 1000, 0);
        assert!(!p.fits(&b, target, max_memory));
        // A 40 MiB superfile fits exactly to the boundary.
        let c = seg(3, 40, 1000, 0);
        assert!(p.fits(&c, target, max_memory));
    }

    #[test]
    fn pending_job_fits_respects_max_memory_even_under_target() {
        // live_bytes fits comfortably under target, but raw size_bytes
        // (pre-tombstone) would blow past a tight memory ceiling.
        let target = mib(1000);
        let max_memory = mib(100);
        let mut p = PendingJob::default();
        let a = seg(1, 60, 1000, 0); // 60 MiB raw, 60 MiB live
        assert!(p.fits(&a, target, max_memory));
        p.push(&a);
        let b = seg(2, 60, 1000, 0); // would push raw to 120 MiB > 100 MiB cap
        assert!(!p.fits(&b, target, max_memory));
    }

    #[test]
    fn pending_job_emit_requires_two_inputs() {
        // A single-input pending job never emits even if it reaches
        // the fill floor.
        let mut jobs = Vec::new();
        let mut p = PendingJob::default();
        p.push(&seg(1, 200, 1000, 0));
        p.emit(&[], 0, &mut jobs);
        assert!(jobs.is_empty(), "single-input job must not emit");
        // Reset to default after emit attempt.
        assert_eq!(p.inputs.len(), 0);
        assert_eq!(p.live_bytes, 0);
    }

    // ---- run_compaction_job error arms ------------------------------

    #[tokio::test(flavor = "multi_thread")]
    async fn run_compaction_job_unknown_input_surfaces_not_found() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);
        commit_titles(&st, &["alpha first", "alpha second"]);
        // A job referencing a superfile id that isn't in the manifest
        // must surface SuperfileNotFound.
        let bogus = Uuid::from_u128(0xDEAD_BEEF);
        let job = CompactionJob {
            partition_key: Vec::new(),
            inputs: vec![bogus],
            estimated_output_bytes: 0,
        };
        let err = st
            .run_compaction_job(job, 1, DEFAULT_STALE_SEAL_TIMEOUT, IN_MEMORY_MERGE_CEILING)
            .await
            .expect_err("must error on unknown input");
        assert!(
            matches!(err, CompactionError::SuperfileNotFound(id) if id == bogus),
            "{err:?}"
        );
    }

    /// Resolves every present input in order; reports the missing one by id.
    #[tokio::test(flavor = "multi_thread")]
    async fn resolve_entries_to_remove_reports_the_missing_input() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);
        commit_titles(&st, &["alpha first", "alpha second"]);
        commit_titles(&st, &["bravo first", "bravo second"]);

        let manifest = st.inner().manifest.load_full();
        let ids: Vec<Uuid> = manifest
            .get_all_superfiles()
            .iter()
            .map(|e| e.superfile_id)
            .collect();
        assert_eq!(ids.len(), 2);

        // All present.
        let resolved = resolve_entries_to_remove(&manifest, &ids).expect("both inputs are present");
        assert_eq!(
            resolved.iter().map(|e| e.superfile_id).collect::<Vec<_>>(),
            ids
        );

        // One missing.
        let vanished = Uuid::from_u128(0xDEAD_BEEF);
        let mut job_inputs = ids.clone();
        job_inputs.push(vanished);
        let err = resolve_entries_to_remove(&manifest, &job_inputs)
            .expect_err("a missing input must be reported");
        assert_eq!(err, vanished);
    }

    /// If one input is already sealed by a different, still-live
    /// compaction, we abort -- but must unseal whatever we already
    /// sealed ourselves this attempt, not leave it stranded.
    #[tokio::test(flavor = "multi_thread")]
    async fn compact_unseals_its_own_inputs_when_a_later_one_conflicts() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);

        commit_titles(&st, &["alpha first", "alpha second"]);
        commit_titles(&st, &["bravo first", "bravo second"]);

        let entries = st.reader().manifest().superfiles.clone();
        assert_eq!(entries.len(), 2);
        let (entry_a, entry_b) = (&entries[0], &entries[1]);

        // entry_b is already held by a different, still-live compaction.
        let storage = st
            .inner()
            .manifest
            .load_full()
            .options
            .storage
            .clone()
            .expect("storage-backed table");
        let wal_store = WalStore::new(storage);
        let foreign_cid = Uuid::new_v4();
        tombstones_admin::seal(
            &wal_store,
            entry_b.superfile_id,
            foreign_cid,
            Utc::now(),
            DEFAULT_STALE_SEAL_TIMEOUT,
        )
        .await
        .expect("seal entry_b as foreign");

        let job = CompactionJob {
            partition_key: entry_a.partition_key.clone(),
            inputs: vec![entry_a.superfile_id, entry_b.superfile_id],
            estimated_output_bytes: 1,
        };
        let err = st
            .run_compaction_job(job, 1, DEFAULT_STALE_SEAL_TIMEOUT, IN_MEMORY_MERGE_CEILING)
            .await
            .expect_err("must conflict on entry_b");
        assert!(matches!(err, CompactionError::SidecarConflict { .. }));

        // entry_a got sealed by us first, then unsealed on the abort.
        let (sidecar_a, _) = wal_store
            .get_tombstones(entry_a.superfile_id)
            .await
            .expect("get")
            .expect("present");
        assert!(sidecar_a.seal.is_none());

        // entry_b's foreign seal is untouched -- it's not ours to clear.
        let (sidecar_b, _) = wal_store
            .get_tombstones(entry_b.superfile_id)
            .await
            .expect("get")
            .expect("present");
        assert_eq!(
            sidecar_b.seal.expect("still sealed").compaction_id,
            foreign_cid
        );
    }

    /// A stale seal (left behind by a crashed compactor, no error
    /// ever caught to clean it up) must not exclude its superfile
    /// from selection forever. Once it's older than
    /// `DEFAULT_STALE_SEAL_TIMEOUT`, a fresh `compact_async` call
    /// must pick it up and actually merge it.
    #[tokio::test(flavor = "multi_thread")]
    async fn compact_recovers_a_superfile_stuck_under_a_stale_seal() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);

        commit_titles(&st, &["alpha first", "alpha second"]);
        commit_titles(&st, &["bravo first", "bravo second"]);
        commit_titles(&st, &["charlie first", "charlie second"]);
        commit_titles(&st, &["delta first", "delta second"]);
        commit_titles(&st, &["echo first", "echo second"]);
        commit_titles(&st, &["foxtrot first", "foxtrot second"]);
        commit_titles(&st, &["golf first", "golf second"]);
        commit_titles(&st, &["hotel first", "hotel second"]);
        commit_titles(&st, &["india first", "india second"]);
        commit_titles(&st, &["juliet first", "juliet second"]);

        let entries = st.reader().manifest().superfiles.clone();
        let crashed_entry = &entries[0];

        // Simulate a compactor that sealed this file and then died
        // long enough ago that its seal is now stale.
        let storage = st
            .inner()
            .manifest
            .load_full()
            .options
            .storage
            .clone()
            .expect("storage-backed table");
        let wal_store = WalStore::new(storage);
        let old_time = Utc::now()
            - chrono::Duration::from_std(DEFAULT_STALE_SEAL_TIMEOUT).unwrap_or_default()
            - chrono::Duration::seconds(1);
        tombstones_admin::seal(
            &wal_store,
            crashed_entry.superfile_id,
            Uuid::new_v4(),
            old_time,
            DEFAULT_STALE_SEAL_TIMEOUT,
        )
        .await
        .expect("simulate a stale seal");

        st.compact_async(&small_compact_cfg())
            .await
            .expect("compact must succeed and recover the stale seal");

        // The stuck superfile must not still be sitting in the
        // manifest under its original id -- it has to have actually
        // been picked up and merged, not just left alone while its
        // 9 unsealed siblings merged around it.
        let still_stuck = st
            .reader()
            .manifest()
            .superfiles
            .iter()
            .any(|s| s.superfile_id == crashed_entry.superfile_id);
        assert!(
            !still_stuck,
            "the stale-sealed superfile must have been merged, not left behind"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn compact_sync_wrapper_runs_jobs() {
        // Exercise the sync `compact()` entry point (the
        // runtime-bridge wrapper around `compact_async`). Use
        // spawn_blocking so we're not inside a tokio runtime when
        // the bridge tries to block.
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);
        for titles in [
            ["alpha first", "alpha second"],
            ["bravo first", "bravo second"],
            ["charlie first", "charlie second"],
            ["delta first", "delta second"],
            ["echo first", "echo second"],
            ["foxtrot first", "foxtrot second"],
            ["golf first", "golf second"],
            ["hotel first", "hotel second"],
            ["india first", "india second"],
            ["juliet first", "juliet second"],
        ] {
            commit_titles(&st, &titles);
        }
        let before = st.manifest_id();
        let cfg = small_compact_cfg();
        task::spawn_blocking(move || st.compact(&cfg).map(|_| st.manifest_id()))
            .await
            .expect("join")
            .map(|after| {
                assert!(after > before, "sync compact must have run a job");
            })
            .expect("compact");
    }

    #[test]
    fn hidden_profile_select_merges_small_same_cell_files() {
        let mut segs = Vec::new();
        for i in 0..4 {
            let mut s = seg(i, 1, 1000, 0);
            s.partition_key = 3u32.to_le_bytes().to_vec();
            segs.push(s);
        }
        // Exercises same-cell selection grouping independent of the
        // production target; a small target keeps the 1 MiB fixtures under
        // the ceiling while their combined size clears the fill floor.
        let cfg = CompactionSettings {
            target_superfile_size_mb: 8,
            min_fill_percent: 40,
            ..CompactionSettings::default()
        };
        let jobs = select(&segs, &cfg);
        assert!(
            !jobs.is_empty(),
            "expected a merge job for 4×1MiB files in one cell partition"
        );
        assert_eq!(jobs[0].partition_key, 3u32.to_le_bytes().to_vec());
        assert!(jobs[0].inputs.len() >= 2);
    }

    #[test]
    fn zero_fill_floor_merges_tiny_fragments_on_count() {
        // Hidden-index policy: a 0% fill floor drives consolidation on the
        // >= 2 fragment count alone. Two sub-target fragments in one cell must
        // merge even though their combined bytes are a tiny fraction of the
        // target — each unmerged fragment is a drain generation that costs a
        // query a fine-run. Under a byte floor the same fragments never merge.
        let mut segs = Vec::new();
        for i in 0..2 {
            let mut s = seg(i, 1, 1000, 0); // 1 MiB each
            s.partition_key = 7u32.to_le_bytes().to_vec();
            segs.push(s);
        }
        let count_driven = CompactionSettings {
            target_superfile_size_mb: 2048,
            min_fill_percent: 0,
            ..CompactionSettings::default()
        };
        let jobs = select(&segs, &count_driven);
        assert_eq!(
            jobs.len(),
            1,
            "0% floor must merge 2 tiny fragments on count"
        );
        assert_eq!(jobs[0].inputs.len(), 2);

        // 2 MiB is far below 40% of a 2 GiB target → the byte floor blocks it.
        let byte_floored = CompactionSettings {
            min_fill_percent: 40,
            ..count_driven.clone()
        };
        assert!(
            select(&segs, &byte_floored).is_empty(),
            "a byte floor must block consolidation of tiny fragments"
        );
    }

    #[test]
    fn partitions_packed_independently() {
        let mut segs = Vec::new();
        for i in 0..5 {
            let mut s = seg(i, 200, 1000, 0);
            s.partition_key = vec![0xA];
            segs.push(s);
        }
        for i in 5..10 {
            let mut s = seg(i, 200, 1000, 0);
            s.partition_key = vec![0xB];
            segs.push(s);
        }
        let jobs = select(&segs, &default_cfg());
        assert_eq!(jobs.len(), 2);
        let a = jobs
            .iter()
            .find(|j| j.partition_key == vec![0xA])
            .expect("partition A job");
        assert!(a.inputs.iter().all(|id| id.as_u128() < 5));
    }

    // Tests for merge_superfiles function
    #[tokio::test(flavor = "multi_thread")]
    async fn merge_superfiles_merges_two_superfiles() {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let st =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("create supertable");

        // Create first superfile with 2 rows
        {
            let mut w = st.writer().expect("writer");
            let batch = build_title_batch(&["first doc", "second doc"]);
            w.append(&batch).expect("append");
            w.commit().expect("commit");
        }

        // Create second superfile with 2 rows
        {
            let mut w = st.writer().expect("writer");
            let batch = build_title_batch(&["third doc", "fourth doc"]);
            w.append(&batch).expect("append");
            w.commit().expect("commit");
        }

        // Get the superfiles to merge
        let reader = st.reader();
        let superfiles: Vec<Arc<SuperfileEntry>> = reader
            .manifest()
            .get_all_superfiles()
            .iter()
            .take(2)
            .cloned()
            .collect();

        assert_eq!(superfiles.len(), 2, "should have 2 superfiles");

        // Merge the superfiles - should succeed
        let _merged_superfile = st
            .merge_superfiles(&superfiles, 1, Uuid::new_v4(), IN_MEMORY_MERGE_CEILING)
            .await
            .expect("merge_superfiles should succeed")
            .pop()
            .expect("exactly one merged superfile");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn merge_superfiles_preserves_scalar_stats() {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let st =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("create supertable");

        // Create first superfile with apple/banana
        {
            let mut w = st.writer().expect("writer");
            let batch = build_title_batch(&["apple", "banana"]);
            w.append(&batch).expect("append");
            w.commit().expect("commit");
        }

        // Create second superfile with cherry/date
        {
            let mut w = st.writer().expect("writer");
            let batch = build_title_batch(&["cherry", "date"]);
            w.append(&batch).expect("append");
            w.commit().expect("commit");
        }

        let reader = st.reader();
        let superfiles: Vec<Arc<SuperfileEntry>> = reader
            .manifest()
            .get_all_superfiles()
            .iter()
            .take(2)
            .cloned()
            .collect();

        // Precompute expected stats from source superfiles
        let expected_n_docs: u64 = superfiles.iter().map(|sf| sf.n_docs).sum();
        let expected_id_min = superfiles
            .iter()
            .map(|sf| sf.id_min)
            .min()
            .unwrap_or(i128::MAX);
        let expected_id_max = superfiles
            .iter()
            .map(|sf| sf.id_max)
            .max()
            .unwrap_or(i128::MIN);

        // Merge should succeed and preserve scalar stats
        let merged_superfile = st
            .merge_superfiles(&superfiles, 1, Uuid::new_v4(), IN_MEMORY_MERGE_CEILING)
            .await
            .expect("merge_superfiles should succeed")
            .pop()
            .expect("exactly one merged superfile");

        // Verify merged superfile stats match expected values
        assert_eq!(
            merged_superfile.entry.n_docs, expected_n_docs,
            "n_docs should be sum of input superfiles"
        );
        assert_eq!(
            merged_superfile.entry.id_min, expected_id_min,
            "id_min should be minimum across all superfiles"
        );
        assert_eq!(
            merged_superfile.entry.id_max, expected_id_max,
            "id_max should be maximum across all superfiles"
        );

        // Verify scalar stats for title column (lexicographic ordering: apple < banana < cherry < date)
        let title_stats = merged_superfile
            .entry
            .scalar_stats
            .get("title")
            .expect("merged entry should have title column stats");

        // Extract min and max string values from the arrays
        let title_min_arr = title_stats
            .min
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("title column should be LargeStringArray");
        let title_max_arr = title_stats
            .max
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("title column should be LargeStringArray");

        // Verify exact min/max values (apple is min across all data, date is max)
        let min_value = title_min_arr.value(0);
        let max_value = title_max_arr.value(0);
        assert_eq!(min_value, "apple", "minimum title should be 'apple'");
        assert_eq!(max_value, "date", "maximum title should be 'date'");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn merge_superfiles_combines_multiple_superfiles() {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let st =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("create supertable");

        // Create three superfiles with 2 rows each. Each batch gets a
        // unique word that survives tokenization (no underscores/numbers).
        let batch_titles = [
            ["alpha first", "alpha second"],
            ["beta first", "beta second"],
            ["gamma first", "gamma second"],
        ];
        for titles in &batch_titles {
            let mut w = st.writer().expect("writer");
            let batch = build_title_batch(titles);
            w.append(&batch).expect("append");
            w.commit().expect("commit");
        }

        let reader = st.reader();
        let superfiles: Vec<Arc<SuperfileEntry>> = reader
            .manifest()
            .get_all_superfiles()
            .iter()
            .take(3)
            .cloned()
            .collect();

        assert_eq!(superfiles.len(), 3, "should have 3 superfiles");

        // Merging 3 superfiles should succeed
        let merged_superfile = st
            .merge_superfiles(&superfiles, 1, Uuid::new_v4(), IN_MEMORY_MERGE_CEILING)
            .await
            .expect("merge_superfiles should succeed")
            .pop()
            .expect("exactly one merged superfile");

        // Verify merged superfile stats
        assert_eq!(
            merged_superfile.entry.n_docs, 6,
            "merged superfile should have 6 documents (3 files × 2 docs each)"
        );

        let source_id_min = superfiles
            .iter()
            .map(|sf| sf.id_min)
            .min()
            .unwrap_or(i128::MAX);
        let source_id_max = superfiles
            .iter()
            .map(|sf| sf.id_max)
            .max()
            .unwrap_or(i128::MIN);
        assert_eq!(merged_superfile.entry.id_min, source_id_min);
        assert_eq!(merged_superfile.entry.id_max, source_id_max);

        // Verify no data loss by querying the merged reader
        let merged_reader = merged_superfile
            .open_reader()
            .expect("merged superfile should have bytes")
            .expect("open reader on merged superfile");

        assert_eq!(merged_reader.n_docs(), 6, "reader should report 6 docs");

        // Each batch has 2 docs sharing a unique word — search for each batch's unique term
        for term in &["alpha", "beta", "gamma"] {
            let hits = merged_reader
                .token_match("title", &[*term], BoolMode::And)
                .await
                .unwrap_or_else(|_| panic!("token_match for '{term}'"));
            assert_eq!(hits.len(), 2, "term '{term}' should match exactly 2 docs");
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn merge_superfiles_respects_connection_memory_budget() {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));

        // Write the data with a normal budget first — ingest draws from the
        // same connection budget, so a tight limit here would starve the
        // setup appends too.
        let st =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("create supertable");
        {
            let mut w = st.writer().expect("writer");
            let batch = build_title_batch(&["first doc", "second doc"]);
            w.append(&batch).expect("append");
            w.commit().expect("commit");
        }
        {
            let mut w = st.writer().expect("writer");
            let batch = build_title_batch(&["third doc", "fourth doc"]);
            w.append(&batch).expect("append");
            w.commit().expect("commit");
        }

        // Reopen the same committed data under a starved budget to exercise
        // the merge-time reservation.
        let mut opts = default_supertable_options().with_storage(Arc::clone(&storage));
        opts.connection_memory_budget = ConnectionMemoryBudget::with_limit(1);
        let st = Supertable::create(opts).expect("reopen supertable");

        let reader = st.reader();
        let superfiles: Vec<Arc<SuperfileEntry>> = reader.manifest().get_all_superfiles().to_vec();

        match st
            .merge_superfiles(&superfiles, 1, Uuid::new_v4(), IN_MEMORY_MERGE_CEILING)
            .await
        {
            Err(BuildError::MemoryBudgetExceeded(_)) => {}
            Err(other) => panic!("expected MemoryBudgetExceeded, got {other:?}"),
            Ok(_) => panic!("merge must be refused over budget"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn merge_superfiles_single_superfile() {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let st =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("create supertable");

        // Create a single superfile
        {
            let mut w = st.writer().expect("writer");
            let batch = build_title_batch(&["only doc", "second doc"]);
            w.append(&batch).expect("append");
            w.commit().expect("commit");
        }

        let reader = st.reader();
        let superfiles: Vec<Arc<SuperfileEntry>> = reader
            .manifest()
            .get_all_superfiles()
            .iter()
            .take(1)
            .cloned()
            .collect();

        assert_eq!(superfiles.len(), 1, "should have 1 superfile");

        // Merging a single superfile should succeed
        let merged_superfile = st
            .merge_superfiles(&superfiles, 1, Uuid::new_v4(), IN_MEMORY_MERGE_CEILING)
            .await
            .expect("merge_superfiles should succeed")
            .pop()
            .expect("exactly one merged superfile");

        // Verify merged superfile stats
        assert_eq!(
            merged_superfile.entry.n_docs, 2,
            "merged superfile should have 2 documents"
        );

        let source_id_min = superfiles
            .iter()
            .map(|sf| sf.id_min)
            .min()
            .unwrap_or(i128::MAX);
        let source_id_max = superfiles
            .iter()
            .map(|sf| sf.id_max)
            .max()
            .unwrap_or(i128::MIN);
        assert_eq!(merged_superfile.entry.id_min, source_id_min);
        assert_eq!(merged_superfile.entry.id_max, source_id_max);

        // Verify no data loss by querying the merged reader
        let merged_reader = merged_superfile
            .open_reader()
            .expect("merged superfile should have bytes")
            .expect("open reader on merged superfile");

        assert_eq!(merged_reader.n_docs(), 2, "reader should report 2 docs");

        let only_hits = merged_reader
            .token_match("title", &["only"], BoolMode::And)
            .await
            .expect("token_match for 'only'");
        assert_eq!(
            only_hits.len(),
            1,
            "should find exactly 1 doc matching 'only'"
        );

        let second_hits = merged_reader
            .token_match("title", &["second"], BoolMode::And)
            .await
            .expect("token_match for 'second'");
        assert_eq!(
            second_hits.len(),
            1,
            "should find exactly 1 doc matching 'second'"
        );
    }

    // ---- clustered merges --------------------------------------------

    /// Storage-backed table clustered by `title`, plus its storage so
    /// streaming tests can read back eagerly-uploaded outputs.
    fn make_clustered_st_with_storage(dir: &TempDir) -> (Supertable, Arc<dyn StorageProvider>) {
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let st = Supertable::create(
            default_supertable_options()
                .with_cluster_by(vec!["title".to_string()])
                .expect("title is a sortable clustering key")
                .with_storage(Arc::clone(&storage)),
        )
        .expect("create clustered supertable");
        (st, storage)
    }

    /// Storage-backed table clustered by `title`.
    fn make_clustered_st(dir: &TempDir) -> Supertable {
        make_clustered_st_with_storage(dir).0
    }

    /// The `title` values of an open superfile, in file row order.
    fn titles_from_reader(reader: &SuperfileReader) -> Vec<String> {
        let batch = reader.get_record_batch(None).expect("decode rows");
        let titles = batch
            .column_by_name("title")
            .expect("title column")
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("title is LargeUtf8");
        (0..batch.num_rows())
            .map(|i| titles.value(i).to_string())
            .collect()
    }

    /// The `title` values of one prepared merge output, in file row order.
    fn titles_of(prepared: &PreparedSuperfile) -> Vec<String> {
        let reader = prepared
            .open_reader()
            .expect("merged superfile has bytes")
            .expect("open merged superfile");
        titles_from_reader(&reader)
    }

    /// Open one streaming merge output from storage: the streaming route
    /// uploads eagerly and returns entries without bytes.
    async fn open_streamed_output(
        storage: &Arc<dyn StorageProvider>,
        prepared: &PreparedSuperfile,
    ) -> SuperfileReader {
        let (bytes, _) = storage
            .get(&prepared.entry.uri.storage_path())
            .await
            .expect("streaming output must be durable before commit");
        SuperfileReader::open(bytes).expect("open streamed output")
    }

    /// `(min, max)` of a title stats entry.
    fn title_stat_bounds(stats: &ScalarStatsAgg) -> (String, String) {
        let min = stats
            .min
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("min is LargeUtf8")
            .value(0)
            .to_string();
        let max = stats
            .max
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("max is LargeUtf8")
            .value(0)
            .to_string();
        (min, max)
    }

    /// A clustered merge must emit the combined rows in key order — the
    /// input files interleave (each commit is sorted, the commits overlap)
    /// so input-concatenation order would NOT be sorted — and the merged
    /// entry's column statistics must span exactly the sorted output.
    #[tokio::test(flavor = "multi_thread")]
    async fn clustered_merge_sorts_live_rows_by_key() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_clustered_st(&dir);
        commit_titles(&st, &["delta doc", "alpha doc"]);
        commit_titles(&st, &["charlie doc", "bravo doc"]);

        let superfiles = st.reader().manifest().get_all_superfiles().to_vec();
        let merged = st
            .merge_superfiles(&superfiles, 1, Uuid::new_v4(), IN_MEMORY_MERGE_CEILING)
            .await
            .expect("clustered merge")
            .pop()
            .expect("exactly one merged superfile");

        assert_eq!(
            titles_of(&merged),
            vec!["alpha doc", "bravo doc", "charlie doc", "delta doc"],
            "merged rows must be in clustering-key order"
        );
        let (min, max) =
            title_stat_bounds(merged.entry.scalar_stats.get("title").expect("title stats"));
        assert_eq!((min.as_str(), max.as_str()), ("alpha doc", "delta doc"));
    }

    /// Tombstone exclusion and the clustering sort compose: the merge
    /// output holds only live rows, still in key order, and the recomputed
    /// statistics cover only the survivors.
    #[tokio::test(flavor = "multi_thread")]
    async fn clustered_merge_excludes_tombstones_and_stays_sorted() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_clustered_st(&dir);
        commit_titles(&st, &["delta doc", "alpha doc"]);
        commit_titles(&st, &["charlie doc", "bravo doc"]);
        {
            let mut w = st.writer().expect("writer");
            let pending = w.delete(col("title").eq(lit("delta doc"))).expect("delete");
            assert_eq!(pending.matched, 1);
            w.commit().expect("commit delete");
        }

        let superfiles = st.reader().manifest().get_all_superfiles().to_vec();
        let merged = st
            .merge_superfiles(&superfiles, 1, Uuid::new_v4(), IN_MEMORY_MERGE_CEILING)
            .await
            .expect("clustered merge")
            .pop()
            .expect("exactly one merged superfile");

        assert_eq!(
            titles_of(&merged),
            vec!["alpha doc", "bravo doc", "charlie doc"],
            "only live rows, in key order"
        );
        let (min, max) =
            title_stat_bounds(merged.entry.scalar_stats.get("title").expect("title stats"));
        assert_eq!(
            (min.as_str(), max.as_str()),
            ("alpha doc", "charlie doc"),
            "stats must reflect the tombstone-filtered sorted output"
        );
    }

    /// `n_outputs > 1` slices ONE globally sorted run into contiguous
    /// chunks: every output is internally sorted and consecutive outputs'
    /// key ranges chain without overlap.
    #[tokio::test(flavor = "multi_thread")]
    async fn clustered_merge_splits_into_contiguous_disjoint_chunks() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_clustered_st(&dir);
        commit_titles(&st, &["golf doc", "echo doc", "alpha doc", "charlie doc"]);
        commit_titles(&st, &["hotel doc", "foxtrot doc", "bravo doc", "delta doc"]);

        let superfiles = st.reader().manifest().get_all_superfiles().to_vec();
        let merged = st
            .merge_superfiles(&superfiles, 2, Uuid::new_v4(), IN_MEMORY_MERGE_CEILING)
            .await
            .expect("clustered merge");
        assert_eq!(merged.len(), 2, "two target-sized outputs requested");

        assert_eq!(
            titles_of(&merged[0]),
            vec!["alpha doc", "bravo doc", "charlie doc", "delta doc"]
        );
        assert_eq!(
            titles_of(&merged[1]),
            vec!["echo doc", "foxtrot doc", "golf doc", "hotel doc"]
        );
        let (_, first_max) =
            title_stat_bounds(merged[0].entry.scalar_stats.get("title").expect("stats"));
        let (second_min, _) =
            title_stat_bounds(merged[1].entry.scalar_stats.get("title").expect("stats"));
        assert!(
            first_max <= second_min,
            "consecutive outputs must not overlap: {first_max:?} vs {second_min:?}"
        );
    }

    /// Byte-behavior guard for tables WITHOUT a clustering key: the merge
    /// keeps input-concatenation row order exactly (no sort sneaks in).
    #[tokio::test(flavor = "multi_thread")]
    async fn unclustered_merge_preserves_input_row_order() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);
        commit_titles(&st, &["zulu doc", "alpha doc"]);
        commit_titles(&st, &["mike doc", "bravo doc"]);

        let superfiles = st.reader().manifest().get_all_superfiles().to_vec();
        let merged = st
            .merge_superfiles(&superfiles, 1, Uuid::new_v4(), IN_MEMORY_MERGE_CEILING)
            .await
            .expect("merge")
            .pop()
            .expect("exactly one merged superfile");

        assert_eq!(
            titles_of(&merged),
            vec!["zulu doc", "alpha doc", "mike doc", "bravo doc"],
            "unclustered merge must preserve append/manifest order"
        );
    }

    /// End-to-end job on a clustered table: multiple outputs land in the
    /// manifest, replace all inputs, and carry chained (disjoint) key-range
    /// statistics for the scan's ordering declaration to read.
    #[tokio::test(flavor = "multi_thread")]
    async fn run_compaction_job_clustered_publishes_disjoint_outputs() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_clustered_st(&dir);
        commit_titles(&st, &["golf doc", "alpha doc"]);
        commit_titles(&st, &["hotel doc", "bravo doc"]);
        commit_titles(&st, &["echo doc", "charlie doc"]);
        commit_titles(&st, &["foxtrot doc", "delta doc"]);

        let inputs: Vec<Uuid> = st
            .reader()
            .manifest()
            .get_all_superfiles()
            .iter()
            .map(|e| e.superfile_id)
            .collect();
        let job = CompactionJob {
            partition_key: Vec::new(),
            inputs,
            estimated_output_bytes: 1,
        };
        st.run_compaction_job(job, 2, DEFAULT_STALE_SEAL_TIMEOUT, IN_MEMORY_MERGE_CEILING)
            .await
            .expect("clustered job");

        let entries = st.reader().manifest().get_all_superfiles().to_vec();
        assert_eq!(entries.len(), 2, "the job's two outputs replace all inputs");
        assert_eq!(entries.iter().map(|e| e.n_docs).sum::<u64>(), 8);
        let mut bounds: Vec<(String, String)> = entries
            .iter()
            .map(|e| title_stat_bounds(e.scalar_stats.get("title").expect("title stats")))
            .collect();
        bounds.sort();
        assert!(
            bounds[0].1 <= bounds[1].0,
            "published outputs must carry disjoint key ranges: {bounds:?}"
        );
    }

    // ---- streaming clustered merges -----------------------------------

    /// Vector fixture dimensionality (`default_vector_config` shape).
    const DIM: usize = 16;

    /// Open every manifest superfile the way a compaction job would, with
    /// no tombstones — direct-call input for `streaming_clustered_merge`.
    async fn open_job_readers(
        st: &Supertable,
    ) -> Vec<(Arc<SuperfileReader>, Option<Arc<RoaringBitmap>>)> {
        let manifest = st.inner().manifest.load_full();
        let store = manifest.options.store.clone();
        let disk_cache = manifest.options.disk_cache.clone();
        let storage = manifest.options.storage.clone();
        let mut readers = Vec::new();
        for entry in manifest.get_all_superfiles().iter() {
            let reader =
                open_compaction_input(&store, disk_cache.as_ref(), storage.as_ref(), entry)
                    .await
                    .expect("open compaction input");
            readers.push((reader, None));
        }
        readers
    }

    /// Past the ceiling the clustered merge streams: outputs come back
    /// `storage_prewritten` (already durable, no bytes attached), rows
    /// arrive in global key order, and the recomputed statistics span
    /// exactly the sorted output — the same contract as the in-memory
    /// route.
    #[tokio::test(flavor = "multi_thread")]
    async fn streaming_merge_engages_past_ceiling_and_sorts_by_key() {
        let dir = TempDir::new().expect("tempdir");
        let (st, storage) = make_clustered_st_with_storage(&dir);
        commit_titles(&st, &["delta doc", "alpha doc"]);
        commit_titles(&st, &["charlie doc", "bravo doc"]);

        let superfiles = st.reader().manifest().get_all_superfiles().to_vec();
        let merged = st
            .merge_superfiles(&superfiles, 1, Uuid::new_v4(), STREAMING_MERGE_CEILING)
            .await
            .expect("streaming clustered merge")
            .pop()
            .expect("exactly one merged superfile");

        assert!(
            merged.storage_prewritten,
            "past the ceiling the merge must take the streaming route"
        );
        assert!(
            merged.bytes_for_storage.is_none() && merged.bytes_for_store.is_none(),
            "streaming outputs release their bytes after the eager upload"
        );
        let reader = open_streamed_output(&storage, &merged).await;
        assert_eq!(
            titles_from_reader(&reader),
            vec!["alpha doc", "bravo doc", "charlie doc", "delta doc"],
            "merged rows must be in clustering-key order"
        );
        let (min, max) =
            title_stat_bounds(merged.entry.scalar_stats.get("title").expect("title stats"));
        assert_eq!((min.as_str(), max.as_str()), ("alpha doc", "delta doc"));
    }

    /// Below the ceiling the in-memory route is untouched: outputs carry
    /// their bytes for the commit to write, exactly as before.
    #[tokio::test(flavor = "multi_thread")]
    async fn clustered_merge_below_ceiling_keeps_the_in_memory_route() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_clustered_st(&dir);
        commit_titles(&st, &["delta doc", "alpha doc"]);
        commit_titles(&st, &["charlie doc", "bravo doc"]);

        let superfiles = st.reader().manifest().get_all_superfiles().to_vec();
        let merged = st
            .merge_superfiles(&superfiles, 1, Uuid::new_v4(), IN_MEMORY_MERGE_CEILING)
            .await
            .expect("in-memory clustered merge")
            .pop()
            .expect("exactly one merged superfile");
        assert!(!merged.storage_prewritten);
        assert!(merged.bytes_for_storage.is_some());
    }

    /// An unclustered table never streams, whatever the ceiling: same
    /// route, same concatenation row order, bytes attached for commit.
    #[tokio::test(flavor = "multi_thread")]
    async fn unclustered_merge_ignores_the_streaming_ceiling() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);
        commit_titles(&st, &["zulu doc", "alpha doc"]);
        commit_titles(&st, &["mike doc", "bravo doc"]);

        let superfiles = st.reader().manifest().get_all_superfiles().to_vec();
        let merged = st
            .merge_superfiles(&superfiles, 1, Uuid::new_v4(), STREAMING_MERGE_CEILING)
            .await
            .expect("unclustered merge")
            .pop()
            .expect("exactly one merged superfile");
        assert!(!merged.storage_prewritten);
        assert_eq!(
            titles_of(&merged),
            vec!["zulu doc", "alpha doc", "mike doc", "bravo doc"],
            "unclustered merge must preserve append/manifest order"
        );
    }

    /// Tombstones are excluded through the streaming merge and the
    /// survivors stay globally sorted, with stats spanning only them.
    #[tokio::test(flavor = "multi_thread")]
    async fn streaming_merge_excludes_tombstones_and_stays_sorted() {
        let dir = TempDir::new().expect("tempdir");
        let (st, storage) = make_clustered_st_with_storage(&dir);
        commit_titles(&st, &["delta doc", "alpha doc"]);
        commit_titles(&st, &["charlie doc", "bravo doc"]);
        {
            let mut w = st.writer().expect("writer");
            let pending = w.delete(col("title").eq(lit("delta doc"))).expect("delete");
            assert_eq!(pending.matched, 1);
            w.commit().expect("commit delete");
        }

        let superfiles = st.reader().manifest().get_all_superfiles().to_vec();
        let merged = st
            .merge_superfiles(&superfiles, 1, Uuid::new_v4(), STREAMING_MERGE_CEILING)
            .await
            .expect("streaming clustered merge")
            .pop()
            .expect("exactly one merged superfile");
        assert!(merged.storage_prewritten);
        let reader = open_streamed_output(&storage, &merged).await;
        assert_eq!(
            titles_from_reader(&reader),
            vec!["alpha doc", "bravo doc", "charlie doc"],
            "only live rows, in key order"
        );
        let (min, max) =
            title_stat_bounds(merged.entry.scalar_stats.get("title").expect("title stats"));
        assert_eq!(
            (min.as_str(), max.as_str()),
            ("alpha doc", "charlie doc"),
            "stats must reflect the tombstone-filtered sorted output"
        );
    }

    /// A four-run job under the minimum fan-in cascades — at least two
    /// folds before the final pass — and the outputs are still one
    /// globally sorted stream sliced into chained-disjoint chunks.
    #[tokio::test(flavor = "multi_thread")]
    async fn streaming_cascade_at_min_fan_in_stays_globally_ordered() {
        let dir = TempDir::new().expect("tempdir");
        let (st, storage) = make_clustered_st_with_storage(&dir);
        commit_titles(&st, &["golf doc", "alpha doc"]);
        commit_titles(&st, &["hotel doc", "bravo doc"]);
        commit_titles(&st, &["echo doc", "charlie doc"]);
        commit_titles(&st, &["foxtrot doc", "delta doc"]);

        let readers = open_job_readers(&st).await;
        assert_eq!(readers.len(), 4);
        let report = streaming_clustered_merge(
            st.inner(),
            readers,
            2,
            Uuid::new_v4(),
            STREAMING_MERGE_CEILING,
        )
        .await
        .expect("cascading streaming merge");

        assert!(
            report.cascade_folds >= 2,
            "4 runs at fan-in 2 need at least two folds, got {}",
            report.cascade_folds
        );
        assert_eq!(report.prepared.len(), 2, "two target-sized outputs");
        let first = open_streamed_output(&storage, &report.prepared[0]).await;
        let second = open_streamed_output(&storage, &report.prepared[1]).await;
        assert_eq!(
            titles_from_reader(&first),
            vec!["alpha doc", "bravo doc", "charlie doc", "delta doc"]
        );
        assert_eq!(
            titles_from_reader(&second),
            vec!["echo doc", "foxtrot doc", "golf doc", "hotel doc"]
        );
        let (_, first_max) = title_stat_bounds(
            report.prepared[0]
                .entry
                .scalar_stats
                .get("title")
                .expect("stats"),
        );
        let (second_min, _) = title_stat_bounds(
            report.prepared[1]
                .entry
                .scalar_stats
                .get("title")
                .expect("stats"),
        );
        assert!(
            first_max <= second_min,
            "consecutive outputs must not overlap: {first_max:?} vs {second_min:?}"
        );
    }

    /// Output identity is a pure function of the compaction id and the
    /// output index: the same job identity re-run lands the same
    /// superfile ids and URIs (its uploads overwrite-or-match instead of
    /// accumulating orphans), and distinct indexes never collide.
    #[tokio::test(flavor = "multi_thread")]
    async fn streaming_outputs_have_deterministic_idempotent_identity() {
        let dir = TempDir::new().expect("tempdir");
        let (st, _storage) = make_clustered_st_with_storage(&dir);
        commit_titles(&st, &["golf doc", "alpha doc"]);
        commit_titles(&st, &["hotel doc", "bravo doc"]);

        let compaction_id = Uuid::new_v4();
        let first_run = streaming_clustered_merge(
            st.inner(),
            open_job_readers(&st).await,
            2,
            compaction_id,
            STREAMING_MERGE_CEILING,
        )
        .await
        .expect("first streaming run");
        let second_run = streaming_clustered_merge(
            st.inner(),
            open_job_readers(&st).await,
            2,
            compaction_id,
            STREAMING_MERGE_CEILING,
        )
        .await
        .expect("same-identity re-run must overwrite partials safely");

        let identity = |run: &StreamingMergeReport| -> Vec<(Uuid, SuperfileUri)> {
            run.prepared
                .iter()
                .map(|p| (p.entry.superfile_id, p.entry.uri))
                .collect()
        };
        let first_ids = identity(&first_run);
        let second_ids = identity(&second_run);
        assert_eq!(first_ids, second_ids, "identity must be deterministic");
        let distinct: HashSet<_> = first_ids.iter().collect();
        assert_eq!(distinct.len(), first_ids.len(), "indexes must not collide");
        assert_eq!(
            first_run
                .prepared
                .iter()
                .map(|p| p.entry.n_docs)
                .sum::<u64>(),
            4
        );
    }

    /// Vector payloads ride the streaming merge row-aligned: after the
    /// k-way merge interleaves rows across sorted inputs (and cuts them
    /// back into superfiles), each row's one-hot embedding still sits
    /// next to its own scalar values — the property vector recall rests
    /// on. Inputs are plain-IVF superfiles built on the table's own
    /// builder options: cell-partitioned vector tables keep the
    /// multi-cell fallback on the table route, so this drives the
    /// streaming machinery directly.
    #[tokio::test(flavor = "multi_thread")]
    async fn streaming_merge_keeps_vectors_row_aligned() {
        // Global category list; category `i`'s embedding is one-hot at
        // dim `i`, so the expected post-merge vector for sorted row `i`
        // is exactly one-hot(i).
        let categories = [
            "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
        ];
        // The user schema declares the vector column as a FixedSizeList
        // field (creation-time validation requires it); the scalar
        // parquet schema drops it and the builder carries the payload.
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
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let st = Supertable::create(
            SupertableOptions::new(schema, vec![], vec![default_vector_config("emb", 0)], None)
                .expect("valid options")
                .with_cluster_by(vec!["category".into()])
                .expect("valid clustering key")
                .with_storage(Arc::clone(&storage)),
        )
        .expect("create clustered vector supertable");

        // One key-sorted input superfile per call, on the table's own
        // builder options; `rows` are global category indexes and double
        // as the one-hot dim and the row id.
        let build_input = |rows: &[usize]| -> Arc<SuperfileReader> {
            let options = &st.inner().options;
            let mut builder =
                SuperfileBuilder::new(options.builder_options()).expect("input builder");
            let cats: Vec<&str> = rows.iter().map(|&i| categories[i]).collect();
            let mut flat = vec![0.0f32; rows.len() * DIM];
            for (chunk, &i) in flat.chunks_mut(DIM).zip(rows) {
                chunk[i] = 1.0;
            }
            let batch = RecordBatch::try_new(
                options.scalar_schema(),
                vec![
                    Arc::new(decimal128_ids(rows.iter().map(|&i| i as u64))) as ArrayRef,
                    Arc::new(LargeStringArray::from(cats)),
                ],
            )
            .expect("input batch");
            builder.add_batch(&batch, &[&flat]).expect("add batch");
            let bytes = Bytes::from(builder.finish().expect("finish input"));
            Arc::new(SuperfileReader::open(bytes).expect("open input"))
        };
        // Sorted within each input, interleaved across them, so the
        // merge really permutes rows between the two runs.
        let readers = vec![
            (build_input(&[0, 2, 4, 6]), None),
            (build_input(&[1, 3, 5, 7]), None),
        ];

        let report = streaming_clustered_merge(
            st.inner(),
            readers,
            1,
            Uuid::new_v4(),
            STREAMING_MERGE_CEILING,
        )
        .await
        .expect("streaming clustered merge");
        let total_rows: u64 = report.prepared.iter().map(|p| p.entry.n_docs).sum();
        assert_eq!(total_rows, categories.len() as u64);
        let merged = report
            .prepared
            .first()
            .expect("at least one merged superfile");
        assert!(merged.storage_prewritten);

        let reader = open_streamed_output(&storage, merged).await;
        let batch = reader.get_record_batch(None).expect("decode rows");
        let cats = batch
            .column_by_name("category")
            .expect("category column")
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("category is LargeUtf8");
        let vectors = reader
            .vec()
            .expect("vector section survives the merge")
            .get_vectors_fp32("emb")
            .expect("decode merged vectors");
        assert_eq!(batch.num_rows(), categories.len());
        assert_eq!(vectors.len(), categories.len());
        for (row, expected) in categories.iter().enumerate() {
            assert_eq!(cats.value(row), *expected, "rows must be key-sorted");
            let mut want = vec![0.0f32; DIM];
            want[row] = 1.0;
            assert_eq!(
                vectors[row], want,
                "row {row}'s vector must stay aligned with its scalar row"
            );
        }
    }

    /// Route predicates: the threshold engages strictly past the ceiling,
    /// and availability requires storage plus a non-sq8 vector surface.
    #[test]
    fn streaming_route_predicates() {
        assert!(!clustered_merge_needs_streaming(10, 30));
        assert!(clustered_merge_needs_streaming(11, 30));
        assert!(
            !clustered_merge_needs_streaming(u64::MAX, u64::MAX),
            "saturating reserve never exceeds an unbounded ceiling"
        );

        let no_storage = default_supertable_options();
        assert!(!clustered_streaming_available(&no_storage));

        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let with_storage = default_supertable_options().with_storage(Arc::clone(&storage));
        assert!(clustered_streaming_available(&with_storage));

        // sq8-family rerank codecs byte-splice their merges: no streaming.
        let mut sq8 = default_vector_config("emb", 0);
        sq8.rerank_codec = RerankCodec::Sq8Residual;
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
        let sq8_options = SupertableOptions::new(schema, vec![], vec![sq8], None)
            .expect("valid sq8 options")
            .with_storage(storage);
        assert!(!clustered_streaming_available(&sq8_options));
    }

    // ---- coalesce_clustered_jobs / clustered_output_count ------------

    #[test]
    fn clustered_output_count_rounds_up_and_degrades_to_one() {
        assert_eq!(clustered_output_count(0, mib(1)), 1);
        assert_eq!(clustered_output_count(mib(1), mib(1)), 1);
        assert_eq!(clustered_output_count(mib(1) + 1, mib(1)), 2);
        assert_eq!(clustered_output_count(mib(5), mib(2)), 3);
        assert_eq!(clustered_output_count(mib(5), 0), 1, "zero target");
    }

    #[test]
    fn coalesce_fuses_same_partition_jobs_up_to_the_memory_ceiling() {
        let segs: Vec<SuperfileStats> = (0..6).map(|i| seg(i, 100, 1000, 0)).collect();
        let job = |ids: &[u128]| CompactionJob {
            partition_key: Vec::new(),
            inputs: ids.iter().map(|i| Uuid::from_u128(*i)).collect(),
            estimated_output_bytes: mib(100) * ids.len() as u64,
        };
        // Ceiling admits four inputs' raw bytes: jobs A+B fuse, C stays.
        let fused = coalesce_clustered_jobs(
            vec![job(&[0, 1]), job(&[2, 3]), job(&[4, 5])],
            &segs,
            mib(400),
        );
        assert_eq!(fused.len(), 2);
        assert_eq!(
            fused[0].inputs,
            [0u128, 1, 2, 3].map(Uuid::from_u128).to_vec(),
            "first two jobs fuse in order"
        );
        assert_eq!(fused[0].estimated_output_bytes, mib(400));
        assert_eq!(fused[1].inputs, [4u128, 5].map(Uuid::from_u128).to_vec());
    }

    #[test]
    fn coalesce_never_fuses_across_partitions() {
        let mut segs: Vec<SuperfileStats> = (0..4).map(|i| seg(i, 100, 1000, 0)).collect();
        segs[0].partition_key = vec![0xA];
        segs[1].partition_key = vec![0xA];
        segs[2].partition_key = vec![0xB];
        segs[3].partition_key = vec![0xB];
        let job = |key: u8, ids: &[u128]| CompactionJob {
            partition_key: vec![key],
            inputs: ids.iter().map(|i| Uuid::from_u128(*i)).collect(),
            estimated_output_bytes: mib(100),
        };
        let fused = coalesce_clustered_jobs(
            vec![job(0xA, &[0, 1]), job(0xB, &[2, 3])],
            &segs,
            mib(10_000),
        );
        assert_eq!(fused.len(), 2, "partition boundary blocks fusion");
        assert_eq!(fused[0].partition_key, vec![0xA]);
        assert_eq!(fused[1].partition_key, vec![0xB]);
    }

    /// An in-memory supertable (no storage, no tombstone cache) takes
    /// the empty-sidecar-map fallback arm in `compact_async`: it still
    /// builds per-superfile stats and runs `select`, and with a single
    /// committed superfile `select` finds nothing to do, so the call
    /// returns `Ok(())` without touching storage.
    #[tokio::test(flavor = "multi_thread")]
    async fn compact_in_memory_table_takes_empty_sidecar_fallback() {
        let st =
            Supertable::create(default_supertable_options()).expect("create in-memory supertable");
        {
            let mut w = st.writer().expect("writer");
            w.append(&build_title_batch(&["alpha first", "alpha second"]))
                .expect("append");
            w.commit().expect("commit");
        }
        let before = st.manifest_id();
        st.compact_async(&small_compact_cfg())
            .await
            .expect("in-memory compact is a no-op, not an error");
        assert_eq!(
            st.manifest_id(),
            before,
            "single superfile yields no compaction job"
        );
    }

    // ─── Helpers shared by the end-to-end compact() tests ─────────────────

    fn make_st(dir: &TempDir) -> Supertable {
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
            .expect("create supertable")
    }

    /// Compact config designed to trigger on tiny test superfiles.
    /// target = 1 MiB, fill floor = 1 % → min_output_bytes ≈ 10 KiB.
    /// Individual files must be < 10 KiB to be candidates; their
    /// combined live_bytes must reach 10 KiB for a job to be emitted.
    fn small_compact_cfg() -> CompactionSettings {
        CompactionSettings {
            target_superfile_size_mb: 1,
            min_fill_percent: 1,
            ..CompactionSettings::default()
        }
    }

    fn commit_titles(st: &Supertable, titles: &[&str]) {
        let mut w = st.writer().expect("writer");
        w.append(&build_title_batch(titles)).expect("append");
        w.commit().expect("commit");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn compact_rejects_concurrent_call_while_slot_held() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);

        // Manually set the slot as if a compaction is running.
        st.inner()
            .compaction_outstanding
            .store(true, Ordering::Release);

        let err = st
            .compact_async(&small_compact_cfg())
            .await
            .expect_err("must reject while slot held");

        assert!(
            matches!(err, CompactionError::AlreadyCompacting),
            "expected AlreadyCompacting, got {err:?}"
        );

        // Release so the supertable is clean for drop.
        st.inner()
            .compaction_outstanding
            .store(false, Ordering::Release);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn compact_slot_released_after_completion() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);

        commit_titles(&st, &["alpha first", "alpha second"]);

        st.compact_async(&small_compact_cfg())
            .await
            .expect("first compact");

        // Slot must be released so a second call succeeds.
        st.compact_async(&small_compact_cfg())
            .await
            .expect("second compact after slot release");
    }

    // OCC retry tests
    #[tokio::test(flavor = "multi_thread")]
    async fn compact_succeeds_when_concurrent_writer_commits_during_compaction() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);

        // Enough superfiles to trigger a compaction job.
        for title in &[
            ["alpha first", "alpha second"],
            ["bravo first", "bravo second"],
            ["charlie first", "charlie second"],
            ["delta first", "delta second"],
            ["echo first", "echo second"],
            ["foxtrot first", "foxtrot second"],
            ["golf first", "golf second"],
            ["hotel first", "hotel second"],
            ["india first", "india second"],
            ["juliet first", "juliet second"],
        ] {
            commit_titles(&st, title);
        }

        let before_docs = st.reader().n_docs_total();
        let st2 = st.clone();

        // Race a writer commit against compaction. The compactor will
        // hit WriteContentionExhausted on its first pointer CAS attempt
        // (or succeed before the writer — either way both must succeed).
        let writer_handle = task::spawn_blocking(move || {
            commit_titles(&st2, &["kilo first", "kilo second"]);
        });

        st.compact_async(&small_compact_cfg())
            .await
            .expect("compact must succeed despite concurrent writer");

        writer_handle.await.expect("writer task");

        // All docs from both paths must be visible after refresh.
        st.refresh().await.expect("refresh");
        let after_docs = st.reader().n_docs_total();
        assert_eq!(
            after_docs,
            before_docs + 2,
            "writer's 2 docs must survive alongside compacted data"
        );
    }

    // ─── End-to-end compact() tests ────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn compact_reduces_superfile_count() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);

        // Ten commits, each with a unique first word so the merged bloom is verifiable.
        // 10 × ~1217 bytes ≈ 12 170 bytes > min_output_bytes (~10 485) → job emitted.
        commit_titles(&st, &["alpha cherry", "alpha mango"]);
        commit_titles(&st, &["bravo cherry", "bravo mango"]);
        commit_titles(&st, &["charlie delta", "charlie echo"]);
        commit_titles(&st, &["foxtrot golf", "foxtrot hotel"]);
        commit_titles(&st, &["india first", "india second"]);
        commit_titles(&st, &["lima first", "lima second"]);
        commit_titles(&st, &["november first", "november second"]);
        commit_titles(&st, &["quebec first", "quebec second"]);
        commit_titles(&st, &["romeo first", "romeo second"]);
        commit_titles(&st, &["sierra first", "sierra second"]);

        let before = st.reader();
        let before_manifest_id = before.manifest_id();
        let before_n_superfiles = before.n_superfiles();
        let input_ids: HashSet<Uuid> = before
            .manifest()
            .superfiles
            .iter()
            .map(|s| s.superfile_id)
            .collect();
        let expected_birth_version = before
            .manifest()
            .superfiles
            .iter()
            .map(|s| s.birth_version)
            .min()
            .expect("at least one superfile before compaction");
        let expected_docs = before.n_docs_total();
        let expected_id_min = before
            .manifest()
            .superfiles
            .iter()
            .map(|s| s.id_min)
            .min()
            .expect("at least one superfile before compaction");
        let expected_id_max = before
            .manifest()
            .superfiles
            .iter()
            .map(|s| s.id_max)
            .max()
            .expect("at least one superfile before compaction");

        st.compact_async(&small_compact_cfg())
            .await
            .expect("compact");

        let after = st.reader();
        let sfs = &after.manifest().superfiles;

        assert!(
            after.manifest_id() == before_manifest_id + 1,
            "no compaction jobs ran; adjust small_compact_cfg() if superfiles exceed \
             min_output_bytes"
        );
        assert!(
            sfs.len() < before_n_superfiles,
            "superfile count should decrease after compaction"
        );
        assert!(
            !sfs.iter().any(|s| input_ids.contains(&s.superfile_id)),
            "original superfile IDs must not appear after compaction"
        );
        assert_eq!(
            sfs[0].birth_version, expected_birth_version,
            "compaction must preserve the oldest input birth version"
        );

        // Doc count preserved across the merge
        assert_eq!(after.n_docs_total(), expected_docs);

        // Merged entry ID range spans all original inputs
        let merged_min = sfs
            .iter()
            .map(|s| s.id_min)
            .min()
            .expect("at least one superfile after compaction");
        let merged_max = sfs
            .iter()
            .map(|s| s.id_max)
            .max()
            .expect("at least one superfile after compaction");
        assert!(merged_min == expected_id_min);
        assert!(merged_max == expected_id_max);

        // Partition key consistent across all remaining superfiles
        assert!(sfs.iter().all(|s| s.partition_key == sfs[0].partition_key));

        // FTS bloom covers the unique first word from each of the 10 input batches
        let fts = sfs[0]
            .fts_summary
            .get("title")
            .expect("fts summary present");
        for term in &[
            b"alpha" as &[u8],
            b"bravo",
            b"charlie",
            b"foxtrot",
            b"india",
            b"lima",
            b"november",
            b"quebec",
            b"romeo",
            b"sierra",
        ] {
            assert!(
                fts.may_contain(term),
                "bloom missing term '{}'",
                str::from_utf8(term).expect("term literal is valid utf-8")
            );
        }

        // Box::leak(dir);
        mem::forget(dir);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn compact_no_op_when_single_superfile() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);

        commit_titles(&st, &["only doc", "second doc"]);

        let before_manifest_id = st.manifest_id();
        let before_n = st.reader().n_superfiles();

        st.compact_async(&small_compact_cfg())
            .await
            .expect("compact");

        assert_eq!(
            st.manifest_id(),
            before_manifest_id,
            "manifest_id must not change: a single superfile cannot form a merge job"
        );
        assert_eq!(st.reader().n_superfiles(), before_n);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn compact_no_op_when_below_fill_floor() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);

        commit_titles(&st, &["alpha first", "alpha second"]);
        commit_titles(&st, &["beta first", "beta second"]);

        let before_manifest_id = st.manifest_id();

        // fill floor = 100% of 1 GiB → min_output_bytes = 1 GiB.
        // Both tiny superfiles are candidates (each < 1 GiB) but their
        // combined live_bytes is far below 1 GiB, so no job is emitted.
        let cfg = CompactionSettings {
            target_superfile_size_mb: 1024,
            min_fill_percent: 100,
            ..CompactionSettings::default()
        };
        st.compact_async(&cfg).await.expect("compact");

        assert_eq!(
            st.manifest_id(),
            before_manifest_id,
            "manifest must not change when combined size is below the fill floor"
        );
        assert_eq!(st.reader().n_superfiles(), 2);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn reader_pinned_before_compact_sees_old_state() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);

        commit_titles(&st, &["alpha first", "alpha second"]);
        commit_titles(&st, &["bravo first", "bravo second"]);
        commit_titles(&st, &["charlie first", "charlie second"]);
        commit_titles(&st, &["delta first", "delta second"]);
        commit_titles(&st, &["echo first", "echo second"]);
        commit_titles(&st, &["foxtrot first", "foxtrot second"]);
        commit_titles(&st, &["golf first", "golf second"]);
        commit_titles(&st, &["hotel first", "hotel second"]);
        commit_titles(&st, &["india first", "india second"]);
        commit_titles(&st, &["juliet first", "juliet second"]);

        // Pin a snapshot before compaction.
        let reader_before = st.reader();
        let before_n = reader_before.n_superfiles();
        let before_manifest_id = reader_before.manifest_id();

        st.compact_async(&small_compact_cfg())
            .await
            .expect("compact");

        let reader_after = st.reader();

        // The pinned snapshot must be frozen — it still sees the original superfiles.
        assert_eq!(reader_before.n_superfiles(), before_n);
        assert_eq!(reader_before.manifest_id(), before_manifest_id);

        // A freshly-opened reader must reflect the post-compact manifest.
        assert!(
            reader_after.manifest_id() > before_manifest_id,
            "compact must have run for snapshot isolation to be observable; \
             adjust small_compact_cfg() if needed"
        );
        assert!(reader_after.n_superfiles() < before_n);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn fts_search_returns_correct_results_after_compact() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);

        // Ten commits so combined size exceeds min_output_bytes.
        commit_titles(&st, &["alpha first", "alpha second"]);
        commit_titles(&st, &["bravo first", "bravo second"]);
        commit_titles(&st, &["charlie first", "charlie second"]);
        commit_titles(&st, &["delta first", "delta second"]);
        commit_titles(&st, &["echo first", "echo second"]);
        commit_titles(&st, &["foxtrot first", "foxtrot second"]);
        commit_titles(&st, &["golf first", "golf second"]);
        commit_titles(&st, &["hotel first", "hotel second"]);
        commit_titles(&st, &["india first", "india second"]);
        commit_titles(&st, &["juliet first", "juliet second"]);

        let before_manifest_id = st.manifest_id();
        st.compact_async(&small_compact_cfg())
            .await
            .expect("compact");

        assert!(
            st.manifest_id() == before_manifest_id + 1,
            "compact must have run; adjust small_compact_cfg() if needed"
        );

        // Each batch-unique term should match exactly 2 docs.
        for term in &["alpha", "bravo", "charlie"] {
            let n: usize = st
                .token_match("title", term, BoolMode::And, None)
                .unwrap_or_else(|e| panic!("token_match for '{term}': {e}"))
                .iter()
                .map(|b| b.num_rows())
                .sum();
            assert_eq!(n, 2, "term '{term}' should match 2 docs after compact");
        }

        // The shared token 'first' appears once per batch: 10 batches → 10 docs.
        let n_first: usize = st
            .token_match("title", "first", BoolMode::And, None)
            .expect("token_match for 'first'")
            .iter()
            .map(|b| b.num_rows())
            .sum();
        assert_eq!(n_first, 10, "'first' should match 10 docs");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn fts_bloom_filter_covers_all_terms_after_compact() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);

        // Ten commits (2 docs each) so combined size exceeds min_output_bytes.
        // Each commit has a unique first word; all must survive in the merged bloom.
        commit_titles(&st, &["alpha first", "alpha second"]);
        commit_titles(&st, &["bravo first", "bravo second"]);
        commit_titles(&st, &["charlie first", "charlie second"]);
        commit_titles(&st, &["delta first", "delta second"]);
        commit_titles(&st, &["echo first", "echo second"]);
        commit_titles(&st, &["foxtrot first", "foxtrot second"]);
        commit_titles(&st, &["golf first", "golf second"]);
        commit_titles(&st, &["hotel first", "hotel second"]);
        commit_titles(&st, &["india first", "india second"]);
        commit_titles(&st, &["juliet first", "juliet second"]);

        let before_manifest_id = st.manifest_id();
        st.compact_async(&small_compact_cfg())
            .await
            .expect("compact");

        assert!(
            st.manifest_id() == before_manifest_id + 1,
            "compact must have run; adjust small_compact_cfg() if needed"
        );

        let r = st.reader();
        let sfs = &r.manifest().superfiles;
        assert!(sfs.len() < 10, "superfile count should have decreased");

        let fts = sfs[0]
            .fts_summary
            .get("title")
            .expect("fts summary present");
        for term in &[
            b"alpha" as &[u8],
            b"bravo",
            b"charlie",
            b"delta",
            b"echo",
            b"foxtrot",
            b"golf",
            b"hotel",
            b"india",
            b"juliet",
        ] {
            assert!(
                fts.may_contain(term),
                "bloom missing term '{}'",
                str::from_utf8(term).expect("term literal is valid utf-8")
            );
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn second_compact_is_no_op_after_full_merge() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);

        commit_titles(&st, &["alpha first", "alpha second"]);
        commit_titles(&st, &["bravo first", "bravo second"]);
        commit_titles(&st, &["charlie first", "charlie second"]);
        commit_titles(&st, &["delta first", "delta second"]);
        commit_titles(&st, &["echo first", "echo second"]);
        commit_titles(&st, &["foxtrot first", "foxtrot second"]);
        commit_titles(&st, &["golf first", "golf second"]);
        commit_titles(&st, &["hotel first", "hotel second"]);
        commit_titles(&st, &["india first", "india second"]);
        commit_titles(&st, &["juliet first", "juliet second"]);

        // First compact: merges all 10 tiny superfiles into one.
        let before_first_compact = st.manifest_id();
        st.compact_async(&small_compact_cfg())
            .await
            .expect("first compact");
        assert!(
            st.manifest_id() == before_first_compact + 1,
            "first compact must have run; adjust small_compact_cfg() if needed"
        );
        assert_eq!(st.inner().manifest.load_full().superfiles.len(), 1);

        let after_first_manifest_id = st.manifest_id();
        let after_first_n = st.reader().n_superfiles();

        // Second compact on the same data: the merged superfile is the only
        // file in its partition, so pack_partition emits no job (needs ≥ 2 inputs).
        st.compact_async(&small_compact_cfg())
            .await
            .expect("second compact");

        assert_eq!(
            st.manifest_id(),
            after_first_manifest_id,
            "second compact should produce no jobs"
        );
        assert_eq!(st.reader().n_superfiles(), after_first_n);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn compact_runs_multiple_compactions_on_separate_file_sets() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);

        // Batch A: ten superfiles with group-A terms (2 docs each = 20 docs total).
        // 10 × ~1217 bytes ≈ 12 170 bytes > min_output_bytes → job emitted.
        commit_titles(&st, &["alpha first", "alpha second"]);
        commit_titles(&st, &["bravo first", "bravo second"]);
        commit_titles(&st, &["charlie first", "charlie second"]);
        commit_titles(&st, &["delta first", "delta second"]);
        commit_titles(&st, &["echo first", "echo second"]);
        commit_titles(&st, &["foxtrot first", "foxtrot second"]);
        commit_titles(&st, &["golf first", "golf second"]);
        commit_titles(&st, &["hotel first", "hotel second"]);
        commit_titles(&st, &["india first", "india second"]);
        commit_titles(&st, &["juliet first", "juliet second"]);

        // First compact: merges the ten batch-A superfiles into one.
        let before_first_compact = st.manifest_id();
        st.compact_async(&small_compact_cfg())
            .await
            .expect("first compact");

        let manifest_id_after_first_compact = st.manifest_id();
        assert_eq!(manifest_id_after_first_compact, before_first_compact + 1);
        assert_eq!(
            st.reader().n_docs_total(),
            20,
            "batch A should have 20 docs"
        );

        // Batch B: ten more superfiles with group-B terms (2 docs each = 20 docs).
        commit_titles(&st, &["kilo first", "kilo second"]);
        commit_titles(&st, &["lima first", "lima second"]);
        commit_titles(&st, &["mike first", "mike second"]);
        commit_titles(&st, &["november first", "november second"]);
        commit_titles(&st, &["oscar first", "oscar second"]);
        commit_titles(&st, &["papa first", "papa second"]);
        commit_titles(&st, &["quebec first", "quebec second"]);
        commit_titles(&st, &["romeo first", "romeo second"]);
        commit_titles(&st, &["sierra first", "sierra second"]);
        commit_titles(&st, &["tango first", "tango second"]);

        // Second compact: runs a job on the new batch-B superfiles.
        // The merged-A superfile is above min_output_bytes so it is not a
        // candidate; the ten batch-B files combine to exceed the floor.
        st.compact_async(&small_compact_cfg())
            .await
            .expect("second compact");

        // The manifest must have advanced past the ten batch-B commits.
        assert!(
            st.manifest_id() == manifest_id_after_first_compact + 10 + 1,
            "second compact must have run a job on the batch-B superfiles"
        );

        // All 40 docs must be visible after both compaction rounds.
        let r = st.reader();
        assert_eq!(r.n_docs_total(), 40, "all docs must be preserved");
        assert!(
            r.n_superfiles() < 8,
            "overall superfile count must have decreased from original 20"
        );

        // ManifestSnapshot consistency: per-entry doc counts sum to 40.
        let sfs = &r.manifest().superfiles;
        let total_from_manifest: u64 = sfs.iter().map(|s| s.n_docs).sum();
        assert_eq!(total_from_manifest, 40);

        // ID range is monotonically ordered within each remaining superfile.
        for sf in sfs.iter() {
            assert!(sf.id_min <= sf.id_max);
        }

        drop(r);

        // FTS: every batch-unique term must be searchable and return exactly 2 docs.
        for term in &[
            "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india",
            "juliet", "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo",
            "sierra", "tango",
        ] {
            let n: usize = st
                .token_match("title", term, BoolMode::And, None)
                .unwrap_or_else(|e| panic!("token_match for '{term}': {e}"))
                .iter()
                .map(|b| b.num_rows())
                .sum();
            assert_eq!(n, 2, "term '{term}' should match exactly 2 docs");
        }
    }

    /// The merged superfile from compaction must be warmed into the
    /// reader cache, and the merged-away inputs must be evicted from it.
    #[tokio::test(flavor = "multi_thread")]
    async fn compact_warms_merged_superfile_and_evicts_merged_away_ones_from_cache() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);

        // Combined size must clear small_compact_cfg()'s ~10KB floor,
        // or select() emits no job at all.
        commit_titles(&st, &["alpha first", "alpha second"]);
        commit_titles(&st, &["bravo first", "bravo second"]);
        commit_titles(&st, &["charlie first", "charlie second"]);
        commit_titles(&st, &["delta first", "delta second"]);
        commit_titles(&st, &["echo first", "echo second"]);
        commit_titles(&st, &["foxtrot first", "foxtrot second"]);
        commit_titles(&st, &["golf first", "golf second"]);
        commit_titles(&st, &["hotel first", "hotel second"]);
        commit_titles(&st, &["india first", "india second"]);
        commit_titles(&st, &["juliet first", "juliet second"]);

        let old_uris: Vec<_> = st
            .reader()
            .manifest()
            .superfiles
            .iter()
            .map(|s| s.uri)
            .collect();
        assert_eq!(old_uris.len(), 10);
        // Each commit already warmed the cache on its own.
        for uri in &old_uris {
            assert!(
                st.inner().options.store.reader(uri).is_ok(),
                "pre-merge superfile {uri:?} should already be warm from its own commit"
            );
        }

        st.compact_async(&small_compact_cfg())
            .await
            .expect("compact");

        let merged_uri = st.reader().manifest().superfiles[0].uri;
        assert!(
            st.inner().options.store.reader(&merged_uri).is_ok(),
            "merged superfile must be warmed into the in-memory cache right after compact"
        );
        for uri in &old_uris {
            assert!(
                st.inner().options.store.reader(uri).is_err(),
                "merged-away superfile {uri:?} must be evicted from the in-memory cache"
            );
        }
    }

    /// Same as the in-memory case, but for a disk-cache-attached table:
    /// the merged superfile should already be resident in the disk
    /// cache right after compact, with no cold fetch needed.
    #[tokio::test(flavor = "multi_thread")]
    async fn compact_warms_merged_superfile_into_disk_cache() {
        use crate::supertable::reader_cache::{DiskCacheConfig, DiskCacheStore, LruPolicy};

        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let cache = DiskCacheStore::new_unpinned(
            Arc::clone(&storage),
            DiskCacheConfig {
                cache_root: dir.path().join("disk-cache"),
                mmap_cold_threshold_secs: 0,
                eviction: Box::new(LruPolicy::new()),
                ..Default::default()
            },
        )
        .expect("disk cache");
        let st = Supertable::create(
            default_supertable_options()
                .with_storage(Arc::clone(&storage))
                .with_disk_cache(Arc::clone(&cache)),
        )
        .expect("create supertable");

        commit_titles(&st, &["alpha first", "alpha second"]);
        commit_titles(&st, &["bravo first", "bravo second"]);
        commit_titles(&st, &["charlie first", "charlie second"]);
        commit_titles(&st, &["delta first", "delta second"]);
        commit_titles(&st, &["echo first", "echo second"]);
        commit_titles(&st, &["foxtrot first", "foxtrot second"]);
        commit_titles(&st, &["golf first", "golf second"]);
        commit_titles(&st, &["hotel first", "hotel second"]);
        commit_titles(&st, &["india first", "india second"]);
        commit_titles(&st, &["juliet first", "juliet second"]);

        st.compact_async(&small_compact_cfg())
            .await
            .expect("compact");

        let cold_fetches_after_compact = cache.stats().n_cold_fetches;

        // A query against the merged file must not trigger a cold
        // fetch -- it should already be resident from compaction's
        // own warm-up.
        let n: usize = st
            .token_match("title", "alpha", BoolMode::And, None)
            .expect("token_match")
            .iter()
            .map(|b| b.num_rows())
            .sum();
        assert_eq!(n, 2);
        assert_eq!(
            cache.stats().n_cold_fetches,
            cold_fetches_after_compact,
            "querying the merged superfile should not cold-fetch -- it \
             should already be warm in the disk cache from compaction"
        );
    }

    /// Vocabulary for realistic term-frequency spread (no `rand` dep).
    const LATENCY_BENCH_WORDS: &[&str] = &[
        "system",
        "storage",
        "query",
        "index",
        "engine",
        "object",
        "table",
        "column",
        "vector",
        "search",
        "cluster",
        "replica",
        "cache",
        "buffer",
        "stream",
        "batch",
        "record",
        "field",
        "schema",
        "partition",
    ];

    fn env_usize(key: &str, default: usize) -> usize {
        std::env::var(key)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    }

    /// Builds one superfile's worth of rows. `shard_tag` is this
    /// superfile's unique narrow term; `broad_term` shows up in 1/3
    /// rows.
    fn latency_bench_shard_batch(
        shard_tag: &str,
        broad_term: &str,
        row_offset: usize,
        n_rows: usize,
    ) -> arrow_array::RecordBatch {
        let titles: Vec<String> = (0..n_rows)
            .map(|local_i| {
                let i = row_offset + local_i;
                // Cheap multiplicative hash, spreads word choice without a rand dep.
                let words: Vec<&str> = (0..5)
                    .map(|k| {
                        let h = (i as u64)
                            .wrapping_mul(2_654_435_761)
                            .wrapping_add(k as u64 * 40_503);
                        LATENCY_BENCH_WORDS[(h % LATENCY_BENCH_WORDS.len() as u64) as usize]
                    })
                    .collect();
                let common = if i.is_multiple_of(3) {
                    format!(" {broad_term}")
                } else {
                    String::new()
                };
                format!("{shard_tag}{common} {} row{i}", words.join(" "))
            })
            .collect();
        let refs: Vec<&str> = titles.iter().map(String::as_str).collect();
        build_title_batch(&refs)
    }

    fn latency_bench_warm_median(
        st: &Supertable,
        query: &str,
        warmup_iters: usize,
        measured_iters: usize,
    ) -> u128 {
        for _ in 0..warmup_iters {
            st.bm25_search("title", query, 10, BoolMode::Or, None)
                .expect("bm25_search warmup");
        }
        let mut samples = Vec::with_capacity(measured_iters);
        for _ in 0..measured_iters {
            let start = Instant::now();
            st.bm25_search("title", query, 10, BoolMode::Or, None)
                .expect("bm25_search measured");
            samples.push(start.elapsed().as_micros());
        }
        samples.sort_unstable();
        samples[samples.len() / 2]
    }

    /// Exact match count (unlike `bm25_search`'s top-k), so it catches
    /// old pre-compact files leaking back into results.
    fn latency_bench_count_hits(st: &Supertable, query: &str) -> u64 {
        st.count("title", query, BoolMode::Or).expect("count")
    }

    /// Warm `bm25_search` latency after merging many small superfiles
    /// into one, on a real local-filesystem corpus (no cloud needed).
    /// Scale via env vars: `INFINO_COMPACT_BENCH_TOTAL_MB` (default 500),
    /// `INFINO_COMPACT_BENCH_N_SUPERFILES` (default 40),
    /// `INFINO_COMPACT_BENCH_TARGET_MB` (default = total).
    #[ignore = "perf diagnostic for issue #372/#378; run with --ignored --nocapture"]
    #[tokio::test(flavor = "multi_thread")]
    async fn compact_latency_at_scale() {
        const APPROX_BYTES_PER_DOC: u64 = 90;
        const BROAD_TERM: &str = "broadterm";
        const WARMUP_ITERS: usize = 20;
        const MEASURED_ITERS: usize = 50;

        let total_mb = env_usize("INFINO_COMPACT_BENCH_TOTAL_MB", 500);
        let n_superfiles = env_usize("INFINO_COMPACT_BENCH_N_SUPERFILES", 40);
        let compact_target_mb = env_usize("INFINO_COMPACT_BENCH_TARGET_MB", total_mb.max(1)) as u64;

        let total_docs = (total_mb as u64 * 1_000_000) / APPROX_BYTES_PER_DOC;
        let docs_per_superfile = (total_docs as usize / n_superfiles).max(1);

        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("local fs provider"));
        let st =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("create supertable");

        let narrow_term = format!("shard{}", n_superfiles / 2);

        for i in 0..n_superfiles {
            let shard_tag = format!("shard{i}");
            let mut w = st.writer().expect("writer");
            w.append(&latency_bench_shard_batch(
                &shard_tag,
                BROAD_TERM,
                i * docs_per_superfile,
                docs_per_superfile,
            ))
            .expect("append");
            w.commit().expect("commit");
        }

        let n_before = st.reader().n_superfiles();
        let docs_before = st.reader().n_docs_total();
        let narrow_hits_before = latency_bench_count_hits(&st, &narrow_term);
        let broad_hits_before = latency_bench_count_hits(&st, BROAD_TERM);
        let narrow_before =
            latency_bench_warm_median(&st, &narrow_term, WARMUP_ITERS, MEASURED_ITERS);
        let broad_before = latency_bench_warm_median(&st, BROAD_TERM, WARMUP_ITERS, MEASURED_ITERS);

        st.compact_async(&CompactionSettings {
            target_superfile_size_mb: compact_target_mb,
            min_fill_percent: 1,
            ..CompactionSettings::default()
        })
        .await
        .expect("compact");

        let n_after = st.reader().n_superfiles();
        assert!(n_after < n_before, "compact should reduce superfile count");

        // No old-file double-counting: doc/hit counts must be identical.
        assert_eq!(st.reader().n_docs_total(), docs_before);
        assert_eq!(
            latency_bench_count_hits(&st, &narrow_term),
            narrow_hits_before
        );
        assert_eq!(latency_bench_count_hits(&st, BROAD_TERM), broad_hits_before);

        let narrow_after =
            latency_bench_warm_median(&st, &narrow_term, WARMUP_ITERS, MEASURED_ITERS);
        let broad_after = latency_bench_warm_median(&st, BROAD_TERM, WARMUP_ITERS, MEASURED_ITERS);

        eprintln!(
            "superfiles: {n_before} -> {n_after}, narrow: {narrow_before}us -> {narrow_after}us, \
             broad: {broad_before}us -> {broad_after}us"
        );

        // Narrow only ever touches one relevant superfile (bloom-skips
        // the rest either way), so it must stay flat regardless of
        // merge count.
        assert!(
            narrow_after <= narrow_before * 2,
            "narrow query regressed: {narrow_before}us -> {narrow_after}us"
        );

        mem::forget(dir);
    }

    /// compact() drops the manifest's superfile count right away, but
    /// the merged-away files stay on disk until a gc() sweep past the
    /// safety gap deletes them.
    #[tokio::test(flavor = "multi_thread")]
    async fn compact_reduces_manifest_count_but_gc_safety_gap_leaves_old_files_on_disk() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);
        let storage = st
            .inner()
            .manifest
            .load_full()
            .options
            .storage
            .clone()
            .expect("storage-backed table");

        for titles in [
            ["alpha first", "alpha second"],
            ["bravo first", "bravo second"],
            ["charlie first", "charlie second"],
            ["delta first", "delta second"],
            ["echo first", "echo second"],
            ["foxtrot first", "foxtrot second"],
            ["golf first", "golf second"],
            ["hotel first", "hotel second"],
            ["india first", "india second"],
            ["juliet first", "juliet second"],
        ] {
            commit_titles(&st, &titles);
        }

        let before_n_superfiles = st.reader().n_superfiles();
        let before_data_objects = storage
            .list_with_prefix_metadata("data")
            .await
            .expect("list data/ before compact")
            .len();
        assert_eq!(before_data_objects, before_n_superfiles);

        st.compact_async(&small_compact_cfg())
            .await
            .expect("compact");

        let after_n_superfiles = st.reader().n_superfiles();
        assert!(
            after_n_superfiles < before_n_superfiles,
            "manifest superfile count must drop right after compact"
        );

        // Old inputs are orphaned, not deleted, until gc() runs.
        let after_data_objects = storage
            .list_with_prefix_metadata("data")
            .await
            .expect("list data/ after compact")
            .len();
        assert_eq!(after_data_objects, before_data_objects + 1);

        // Default 1-day safety gap: everything here is brand new, so
        // gc() deletes nothing yet.
        let default_gap_report = st
            .gc(crate::config::DEFAULT_GC_SAFETY_GAP)
            .expect("gc with default safety gap");
        assert_eq!(default_gap_report.objects_deleted, 0);
        let after_default_gc_objects = storage
            .list_with_prefix_metadata("data")
            .await
            .expect("list data/ after default-gap gc")
            .len();
        assert_eq!(after_default_gc_objects, before_data_objects + 1);

        // A shrunk safety gap reclaims the orphaned inputs, and disk
        // count catches up with the manifest.
        let zero_gap_report = st
            .gc(std::time::Duration::ZERO)
            .expect("gc with zero safety gap");
        assert!(
            zero_gap_report.objects_deleted > 0,
            "a gc() past the safety gap must reclaim the orphaned pre-merge inputs"
        );
        let after_zero_gap_objects = storage
            .list_with_prefix_metadata("data")
            .await
            .expect("list data/ after zero-gap gc")
            .len();
        assert_eq!(after_zero_gap_objects, after_n_superfiles);

        mem::forget(dir);
    }

    /// A superfile sealed by an abandoned compaction attempt (a merge
    /// that started but never finished) is never unsealed, so
    /// `pack_partition`'s `!sealed_by_other` filter excludes it from
    /// every future compaction pass, forever.
    #[tokio::test(flavor = "multi_thread")]
    async fn superfiles_sealed_by_an_abandoned_compaction_are_stranded_forever() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);

        commit_titles(&st, &["alpha first", "alpha second"]);
        commit_titles(&st, &["bravo first", "bravo second"]);

        let stranded_ids: Vec<Uuid> = st
            .reader()
            .manifest()
            .superfiles
            .iter()
            .map(|s| s.superfile_id)
            .collect();
        assert_eq!(stranded_ids.len(), 2);

        // Simulate a compaction that sealed its inputs then died
        // before committing the merge.
        let storage = st
            .inner()
            .manifest
            .load_full()
            .options
            .storage
            .clone()
            .expect("storage-backed table");
        let wal_store = WalStore::new(storage);
        let abandoned_compaction_id = Uuid::new_v4();
        let sealed_at = Utc::now();
        for id in &stranded_ids {
            tombstones_admin::seal(
                &wal_store,
                *id,
                abandoned_compaction_id,
                sealed_at,
                DEFAULT_STALE_SEAL_TIMEOUT,
            )
            .await
            .expect("seal");
        }

        // New data arrives and a generous compaction config runs.
        commit_titles(&st, &["charlie first", "charlie second"]);
        commit_titles(&st, &["delta first", "delta second"]);

        let cfg = CompactionSettings {
            target_superfile_size_mb: 1024,
            min_fill_percent: 1,
            ..CompactionSettings::default()
        };
        st.compact_async(&cfg)
            .await
            .expect("compact must not error");

        // The two stranded superfiles are still sitting untouched —
        // they can never be merged, so they leak permanently.
        let remaining_ids: HashSet<Uuid> = st
            .reader()
            .manifest()
            .superfiles
            .iter()
            .map(|s| s.superfile_id)
            .collect();
        for id in &stranded_ids {
            assert!(remaining_ids.contains(id));
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn compact_runs_multiple_compactions_on_separate_file_sets_in_same_job() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);

        // Each superfile must be large enough that 30 combined overflow the 1 MiB
        // target, forcing the selector to emit two jobs. Write 4096 batches per
        // commit so each superfile holds 4096 × 2 = 8192 docs.
        let commit_bulk = |titles: &[&str]| {
            let mut w = st.writer().expect("writer");
            for _ in 0..4096 {
                w.append(&build_title_batch(titles)).expect("append");
            }
            w.commit().expect("commit");
        };

        // Batch A: ten superfiles; 10 × 8192 = 81920 docs total.
        commit_bulk(&["alpha first", "alpha second"]);
        commit_bulk(&["bravo first", "bravo second"]);
        commit_bulk(&["charlie first", "charlie second"]);
        commit_bulk(&["delta first", "delta second"]);
        commit_bulk(&["echo first", "echo second"]);
        commit_bulk(&["foxtrot first", "foxtrot second"]);
        commit_bulk(&["golf first", "golf second"]);
        commit_bulk(&["hotel first", "hotel second"]);
        commit_bulk(&["india first", "india second"]);
        commit_bulk(&["juliet first", "juliet second"]);

        // Batch B: twenty superfiles (2 iterations × 10 terms); 20 × 8192 = 163840 docs total.
        for _ in 0..2 {
            commit_bulk(&["kilo first", "kilo second"]);
            commit_bulk(&["lima first", "lima second"]);
            commit_bulk(&["mike first", "mike second"]);
            commit_bulk(&["november first", "november second"]);
            commit_bulk(&["oscar first", "oscar second"]);
            commit_bulk(&["papa first", "papa second"]);
            commit_bulk(&["quebec first", "quebec second"]);
            commit_bulk(&["romeo first", "romeo second"]);
            commit_bulk(&["sierra first", "sierra second"]);
            commit_bulk(&["tango first", "tango second"]);
        }

        // 30 superfiles total; 81920 + 163840 = 245760 docs.
        let manifest_id_before_first_compact = st.manifest_id();
        st.compact_async(&small_compact_cfg())
            .await
            .expect("second compact");

        // compact() must have run two jobs (one per file set → manifest +2).
        assert!(
            st.manifest_id() == manifest_id_before_first_compact + 2,
            "compact must have run two jobs, one per file set"
        );

        // All 245760 docs must be visible after compaction.
        let r = st.reader();
        assert_eq!(r.n_docs_total(), 245760, "all docs must be preserved");
        assert!(
            r.n_superfiles() == 2,
            "overall superfile count must have decreased from original 30"
        );

        // ManifestSnapshot consistency: per-entry doc counts sum to 245760.
        let sfs = &r.manifest().superfiles;
        let total_from_manifest: u64 = sfs.iter().map(|s| s.n_docs).sum();
        assert_eq!(total_from_manifest, 245760);

        // ID range is monotonically ordered within each remaining superfile.
        for sf in sfs.iter() {
            assert!(sf.id_min <= sf.id_max);
        }

        drop(r);

        // FTS: batch-A terms committed once → 1 × 8192 = 8192 hits each.
        for term in &[
            "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india",
            "juliet",
        ] {
            let n: usize = st
                .token_match("title", term, BoolMode::And, None)
                .unwrap_or_else(|e| panic!("token_match for '{term}': {e}"))
                .iter()
                .map(|b| b.num_rows())
                .sum();
            assert_eq!(n, 8192, "term '{term}' should match exactly 8192 docs");
        }

        // FTS: batch-B terms committed twice → 2 × 8192 = 16384 hits each.
        for term in &[
            "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo", "sierra",
            "tango",
        ] {
            let n: usize = st
                .token_match("title", term, BoolMode::And, None)
                .unwrap_or_else(|e| panic!("token_match for '{term}': {e}"))
                .iter()
                .map(|b| b.num_rows())
                .sum();
            assert_eq!(n, 16384, "term '{term}' should match exactly 16384 docs");
        }
    }
}
