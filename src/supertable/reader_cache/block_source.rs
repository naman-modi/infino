// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! [`BlockCachedSource`] — block-granular NVMe retention for lazy
//! (range-GET-backed) superfile reads.
//!
//! The disk cache historically had exactly two states per superfile: a lazy
//! reader whose every `range()` was a fresh object-store GET (nothing
//! retained), or a fully-promoted mmap of the whole object. Between "first
//! touch" and "full promotion" the same byte ranges were re-fetched on every
//! query — and full promotion of everything cannot work once the table
//! outgrows local disk (a 1B-row index is TBs).
//!
//! This source is the missing middle state: reads through it land in a
//! sparse local file at fixed block granularity. A miss fetches one
//! block-aligned GET per run of missing blocks and returns those bytes. Every
//! later read of them, from any query on the shared reader, is local, with no
//! GET. Disk (and budget) use grows with the bytes queries touch, not the object
//! size, which is what lets the cache serve tables far larger than local disk.
//!
//! # A read never waits on the cache disk, and fetches a run once
//!
//! A query gets its bytes as soon as the GET returns. Writing them to the block
//! file happens afterwards on a blocking thread, because a background download
//! can keep the same disk busy for minutes. Reads that miss the same blocks at
//! the same time share one GET:
//!
//! ```text
//!   read ─► missing run ─► register it in `runs` ─► GET ─► bytes ──► back to the query
//!     │                                                │
//!     └─► run another read registered ─► same GET ─────┤ (whichever read polls it drives it)
//!                                                      ├─► in memory: later reads get them from here
//!                                                      ▼
//!          wait for a block writer (a task, no thread)
//!                                                      ▼
//!          blocking thread: write ─► mark filled ─► out of memory ─► fsync + index
//! ```
//!
//! - A run is in memory from its GET until its blocks are on the block file. A
//!   read looks there first, then in the filled blocks; anything in neither it
//!   registers and fetches.
//! - The GET is a shared future: the read that registered it and every read
//!   that needs its blocks meanwhile await the same one, so no read depends on
//!   another being polled. It is cancelled only when all of them are dropped.
//!   A prefetch and the sync read path do not register their runs.
//! - The index on disk lists only blocks that were fsynced, so a crash loses at
//!   most the blocks not synced yet, and later reads fetch them again.
//! - Each store runs at most [`BLOCK_WRITERS`] writes at once, so a slow disk
//!   can't take the blocking threads a query needs.
//! - Each store holds up to [`WRITE_BEHIND_CAP_BYTES`] of fetched runs not
//!   written yet. Past that, a read still gets its bytes, but they are not
//!   cached. A run bigger than the cap is still let in when nothing else is
//!   waiting, and a prefetch, which waits for its own write, is never refused.
//!
//! [`BLOCK_WRITERS`]: super::disk::BLOCK_WRITERS
//! [`WRITE_BEHIND_CAP_BYTES`]: super::disk::WRITE_BEHIND_CAP_BYTES
//!
//! Budget integration: each newly filled run reserves its bytes against the
//! owning [`DiskCacheStore`]'s budget (with LRU eviction pressure) *before*
//! fetching, and the entry's shared `size_bytes` counter grows as blocks
//! land — so eviction sees a lazy entry's true footprint. On budget
//! exhaustion, or once this source's cache entry has been replaced (eviction
//! / mmap promotion), reads stop filling blocks instead of failing: they come
//! from a whole-file local copy when the cache holds one, else uncached from
//! object storage. The source gives its reserved bytes back on `Drop`, when the
//! last reader or the last queued write lets go of it; the sparse file and its
//! index stay on disk so a later generation can adopt them.

#[cfg(test)]
use std::sync::Condvar;
use std::{
    collections::{BTreeMap, btree_map::Entry},
    fs,
    os::unix::fs::FileExt,
    path::PathBuf,
    process,
    sync::{
        Arc, Mutex, OnceLock, Weak,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::Duration,
};

use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use futures::{
    StreamExt,
    future::{BoxFuture, FutureExt, Shared, WeakShared, join_all, try_join_all},
    stream,
};
use memmap2::Mmap;
use roaring::RoaringBitmap;
use tokio::{
    select,
    task::{JoinHandle, spawn_blocking},
    time::timeout,
};

#[cfg(test)]
use super::disk::BlockWriteTicket;
use super::disk::{ArcMmapOwner, DiskCacheStore, Reservation};
use crate::{
    runtime_bridge::shared_io_runtime,
    superfile::{LazyByteSource, LazyByteSourceError},
    supertable::manifest::SuperfileUri,
};

/// Cache block size. Misses fetch block-aligned runs, so this bounds both
/// the read amplification of a small scan (a request pays at most one
/// leading + one trailing partial block of overhead) and the bitmap size
/// (a 32 MiB cell superfile is 64 blocks; a 27 GiB one at 1B scale is
/// ~55K). Post-drain vector queries read ~0.25–2 MiB scan ranges, so
/// 512 KiB keeps first-touch overshoot well under 2× while still
/// coalescing a multi-MiB scan into a handful of blocks.
pub(super) const CACHE_BLOCK_BYTES: u64 = 512 * 1024;

const IDX_HEADER: [u8; 8] = CACHE_BLOCK_BYTES.to_le_bytes();

/// How long a read waits on another read's GET before it GETs the same blocks
/// itself and keeps whichever lands first. Far above a normal GET (on GCS about
/// 45 ms at p50 and under 140 ms at p90), far below a stalled one (one attempt
/// can run to the store's 60 s request timeout): it costs nothing normally, and
/// bounds how long one stuck GET holds the reads sharing it.
const HEDGE_AFTER: Duration = Duration::from_secs(2);

static PERSIST_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Lazily-initialized sparse backing file. Created on the first cached read
/// (the object size may only be known after the open-time `tail()` on
/// unknown-size sources). `None` means creation failed once — the source
/// then serves plain passthrough reads forever (cache disabled, not broken).
struct BlockFile {
    file: fs::File,
    size: u64,
    /// Lazily-created read-only mapping for zero-copy hit service.
    /// Inner `None` = mapping failed once; keep serving via pread.
    mmap: OnceLock<Option<Arc<Mmap>>>,
}

/// Block-caching wrapper around a network-backed [`LazyByteSource`].
/// See the module docs for semantics.
pub(crate) struct BlockCachedSource {
    inner: Arc<dyn LazyByteSource>,
    store: Weak<DiskCacheStore>,
    uri: SuperfileUri,
    path: PathBuf,
    /// Distinguishes this source from a replacement entry for the same URI.
    entry_token: Arc<()>,
    /// Whether this source reserves and releases touched-block bytes itself.
    owns_accounting: bool,
    /// Virtual hole: `(offset, len)` of a subsection whose reads bypass the
    /// block cache and fetch exact ranges from the inner source. Set to the
    /// FTS subsection: posting reads are ~KiB-sized and scattered, so
    /// rounding each to a 512 KiB block over-fetches ~200× per read on a
    /// wide OR (the block size is tuned for vector's 0.25–2 MiB scans).
    /// Their warm locality comes from the background-fill mmap promotion,
    /// not from this cache. So a read inside the hole never fills blocks on a
    /// miss, but is served from disk when its blocks are there; only
    /// [`Self::prefetch`] fills them. Reads only partially overlapping the
    /// hole keep block semantics.
    passthrough: Option<(u64, u64)>,
    state: OnceLock<Option<BlockFile>>,
    /// Filled-block set. Guarded by a sync mutex; never held across await.
    filled: Mutex<RoaringBitmap>,
    /// Bytes of filled blocks — shared with the owning `CachedEntry`'s
    /// `size_bytes`, so eviction candidates report a lazy entry's real
    /// footprint as it grows.
    filled_bytes: Arc<AtomicU64>,
    /// A handle to this source, so a queued write can keep it alive until the
    /// write is done.
    me: Weak<BlockCachedSource>,
    /// Runs in memory, keyed by first block; see [`MemoryRun`]. Never held
    /// across await, and never while a run's last fetch could drop: that drops
    /// its [`Claim`], which takes this lock.
    runs: Mutex<BTreeMap<u32, MemoryRun>>,
    /// Source of [`MemoryRun::id`].
    next_run_id: AtomicU64,
    /// Set when blocks are marked filled after the last index write. A failed
    /// fsync sets it again, so the next flush retries.
    dirty: AtomicBool,
    /// Set while an index flush runs; see [`Self::flush_index`].
    flushing: AtomicBool,
    /// Tests set this to hold block-file writes; see [`Self::stall_writes`].
    #[cfg(test)]
    write_stall: (Mutex<WriteStallState>, Condvar),
}

/// How a missing run is fetched and cached.
#[derive(Clone, Copy)]
enum Fill {
    /// For a read: may evict colder entries to make room, is not cached once
    /// the cap on unwritten runs is full, and doesn't wait for its write.
    Read,
    /// Ahead of reads: uses only free budget and waits for its own write, so
    /// the cap never refuses it; it holds the bytes until then either way.
    Prefetch,
}

/// A fetched run of blocks waiting to be written to the block file, made by
/// [`BlockCachedSource::admit_run`]. While it exists it holds:
/// - the disk budget reserved for its blocks,
/// - its share of the store's cap on unwritten runs,
/// - its entry in `runs`, so reads get the bytes from memory.
///
/// [`Self::spawn_write`] writes it. Dropping it gives back whatever it still holds,
/// whether the write worked, failed or never ran, and writes the index if the
/// blocks were written.
struct FetchedRun {
    source: Arc<BlockCachedSource>,
    store: Arc<DiskCacheStore>,
    /// Its entry in `runs`.
    id: u64,
    first: u32,
    last: u32,
    bytes: Bytes,
    /// Reserved bytes to give back on drop: all of them until the write is
    /// done, then only those of blocks another fill wrote first.
    unlanded: u64,
    landed: bool,
    /// Declared after `source` so it drops after it: a test that waits for the
    /// writes to finish also sees the source dropped.
    #[cfg(test)]
    _ticket: BlockWriteTicket,
}

impl FetchedRun {
    /// Writes the run once one of the store's block writers is free; `false` if
    /// the write failed. It runs on the shared I/O runtime, never the caller's: a
    /// sync read from a rayon thread runs on a throwaway runtime, and dropping it
    /// would wait for the write or cancel it.
    fn spawn_write(self) -> JoinHandle<bool> {
        shared_io_runtime().spawn(async move {
            let writer = self.store.block_writer().await;
            spawn_blocking(move || {
                let _writer = writer;
                self.land()
            })
            .await
            .unwrap_or(false)
        })
    }

    /// Writes the run to the block file and marks its blocks filled; `false` if
    /// the write failed. Blocking.
    fn land(mut self) -> bool {
        #[cfg(test)]
        self.source.wait_while_stalled();

        let Some(bf) = self.source.created_block_file() else {
            return false;
        };
        let at = u64::from(self.first) * CACHE_BLOCK_BYTES;
        if bf.file.write_all_at(&self.bytes, at).is_err() {
            return false;
        }

        let newly = self.source.mark_filled(bf.size, self.first, self.last);
        self.source.filled_bytes.fetch_add(newly, Ordering::AcqRel);
        self.unlanded = self.unlanded.saturating_sub(newly);
        self.landed = true;
        true
    }
}

impl Drop for FetchedRun {
    fn drop(&mut self) {
        // Out of memory only now, after `land` marked the blocks filled, so a
        // read never finds them in neither place.
        self.source.remove_run(self.first, self.id);
        self.store.release_write_behind(self.bytes.len() as u64);

        if self.unlanded > 0 {
            self.store.release_block_bytes(self.unlanded);
        }

        if self.landed {
            self.source.flush_index();
        }
    }
}

/// A run's fetch, shared by every read that needs its blocks: the read that
/// registered it and every read that finds it in `runs` await clones of one
/// future, and whichever polls it drives the GET. Dropping the last clone
/// cancels the GET.
type RunFetch = Shared<BoxFuture<'static, Fetched>>;

/// How a shared run fetch ends.
#[derive(Clone)]
enum Fetched {
    /// The run's bytes, starting at its first block.
    Bytes(Bytes),
    /// Read elsewhere (the whole-file copy or object storage): the store is
    /// gone, this source is no longer the live entry, the budget is full, or
    /// the GET came back short.
    Elsewhere,
    /// The GET failed.
    Failed(FetchError),
}

/// A failed GET, shared by every read of the run. The first read to take the
/// error returns it; the others read elsewhere, which tries the GET once more
/// each.
#[derive(Clone)]
struct FetchError(Arc<Mutex<Option<LazyByteSourceError>>>);

impl FetchError {
    fn new(error: LazyByteSourceError) -> Self {
        Self(Arc::new(Mutex::new(Some(error))))
    }

    fn take(&self) -> Option<LazyByteSourceError> {
        self.0.lock().expect("fetch error mutex poisoned").take()
    }
}

/// A run of blocks in memory, in `runs` under its first block: from its
/// registration until its blocks are on the block file.
struct MemoryRun {
    /// Tells this run from a later one under the same first block, so a guard
    /// only ever removes its own.
    id: u64,
    /// Last block of the run (inclusive).
    last: u32,
    bytes: RunBytes,
}

/// Where a [`MemoryRun`]'s bytes are.
enum RunBytes {
    /// Its GET is in flight. Weak, so `runs` never keeps a fetch alive: once
    /// every read awaiting it is dropped, the GET is cancelled and its
    /// [`Claim`] takes the entry out.
    Fetching(WeakShared<BoxFuture<'static, Fetched>>),
    /// Fetched, waiting for its block-file write. These are the run's bytes,
    /// starting at its first block.
    Fetched(Bytes),
}

/// The entry a read registered in `runs`, owned by the run's fetch. Unless
/// [`BlockCachedSource::admit_run`] hands it to the run's [`FetchedRun`], the
/// entry leaves `runs` when the claim drops: when the fetch ends without the
/// run being cached, or is cancelled.
struct Claim {
    source: Arc<BlockCachedSource>,
    first: u32,
    id: u64,
    handed_on: bool,
}

impl Drop for Claim {
    fn drop(&mut self) {
        if !self.handed_on {
            self.source.remove_run(self.first, self.id);
        }
    }
}

/// Where one stretch `[first, last]` of a read's blocks is right now.
enum Segment {
    /// On the block file.
    Filled { first: u32, last: u32 },
    /// In memory, fetched. `bytes` start at block `run_first`.
    InMemory {
        first: u32,
        last: u32,
        run_first: u32,
        bytes: Bytes,
    },
    /// Being fetched in the run that starts at block `run_first`; `fetch` ends
    /// with its bytes. `own` when this read registered the run, rather than
    /// found another read's.
    Fetching {
        first: u32,
        last: u32,
        run_first: u32,
        fetch: RunFetch,
        own: bool,
    },
    /// None of the above, and not registered: only for callers that don't
    /// register runs (a prefetch, the sync path), or a source being dropped.
    Missing { first: u32, last: u32 },
}

impl Segment {
    fn blocks(&self) -> (u32, u32) {
        match *self {
            Self::Filled { first, last }
            | Self::InMemory { first, last, .. }
            | Self::Fetching { first, last, .. }
            | Self::Missing { first, last } => (first, last),
        }
    }
}

/// Where one block is, while [`BlockCachedSource::segments`] groups blocks.
#[derive(Clone, Copy, PartialEq)]
enum BlockAt {
    Filled,
    Missing,
    /// In the fetched run at this index of the snapshot.
    InMemory(usize),
    /// In the run being fetched at this index of the snapshot.
    Fetching(usize),
}

impl BlockAt {
    /// Not readable yet: the read waits for it or fetches it.
    fn is_unavailable(self) -> bool {
        matches!(self, Self::Fetching(_) | Self::Missing)
    }
}

/// A run in `runs` overlapping a read, as [`BlockCachedSource::segments`] found it.
enum RunState {
    Fetched(Bytes),
    Fetching(RunFetch),
}

impl BlockCachedSource {
    #[cfg(test)]
    pub(crate) fn new(
        inner: Arc<dyn LazyByteSource>,
        store: Weak<DiskCacheStore>,
        uri: SuperfileUri,
        path: PathBuf,
    ) -> Arc<Self> {
        Self::new_with_accounting(inner, store, uri, path, true, None)
    }

    /// Construct a sparse source whose owning entry has already reserved the
    /// complete object size. `passthrough` is the optional exact-read hole
    /// (see the field docs) — the FTS subsection on the cold-open path.
    pub(crate) fn new_pre_reserved(
        inner: Arc<dyn LazyByteSource>,
        store: Weak<DiskCacheStore>,
        uri: SuperfileUri,
        path: PathBuf,
        passthrough: Option<(u64, u64)>,
    ) -> Arc<Self> {
        Self::new_with_accounting(inner, store, uri, path, false, passthrough)
    }

    pub(crate) fn new_with_accounting(
        inner: Arc<dyn LazyByteSource>,
        store: Weak<DiskCacheStore>,
        uri: SuperfileUri,
        path: PathBuf,
        owns_accounting: bool,
        passthrough: Option<(u64, u64)>,
    ) -> Arc<Self> {
        Arc::new_cyclic(|me| Self {
            inner,
            store,
            uri,
            path,
            entry_token: Arc::new(()),
            owns_accounting,
            passthrough,
            state: OnceLock::new(),
            filled: Mutex::new(RoaringBitmap::new()),
            filled_bytes: Arc::new(AtomicU64::new(0)),
            me: me.clone(),
            runs: Mutex::new(BTreeMap::new()),
            next_run_id: AtomicU64::new(0),
            dirty: AtomicBool::new(false),
            flushing: AtomicBool::new(false),
            #[cfg(test)]
            write_stall: (Mutex::new(WriteStallState::default()), Condvar::new()),
        })
    }

    /// Whether `[start, start + len)` lies fully inside the passthrough hole.
    fn in_passthrough(&self, start: u64, len: u64) -> bool {
        match self.passthrough {
            Some((off, hole_len)) => start >= off && start + len <= off + hole_len,
            None => false,
        }
    }

    /// The blocks file for a read. A hole read only uses one that exists, since
    /// reads never fill the hole.
    fn block_file_for(&self, in_hole: bool) -> Option<&BlockFile> {
        if in_hole {
            self.created_block_file()
        } else {
            self.block_file()
        }
    }

    /// Fill every block of `[start, start + len)` ahead of reads, with up to
    /// `streams` GETs of about `chunk_bytes` each in flight. Unlike a read, this
    /// also fills the passthrough hole, so a scan of the whole hole is served
    /// from disk. Uses only free space, never evicting, and stops quietly when
    /// it runs out; later reads then pass through as before.
    pub(crate) async fn prefetch(
        &self,
        start: u64,
        len: u64,
        chunk_bytes: u64,
        streams: usize,
    ) -> Result<(), LazyByteSourceError> {
        let Some(bf) = self.block_file() else {
            return Ok(());
        };
        let end = start.saturating_add(len).min(bf.size);
        if start >= end {
            return Ok(());
        }
        let (b0, b1) = Self::block_span(start, end - start);
        let per_chunk = (chunk_bytes / CACHE_BLOCK_BYTES).max(1) as u32;
        let chunks = (b0..=b1)
            .step_by(per_chunk as usize)
            .map(|c0| (c0, c0.saturating_add(per_chunk - 1).min(b1)));
        // On a stop, chunks not started yet are skipped. Chunks already running
        // finish, so the prefetch returns only after every write it started.
        let stop = AtomicBool::new(false);
        let mut fills = stream::iter(chunks)
            .map(|(c0, c1)| {
                let stop = &stop;
                async move {
                    if stop.load(Ordering::Acquire) {
                        return Ok(true);
                    }
                    self.prefetch_chunk(bf, c0, c1).await
                }
            })
            .buffer_unordered(streams.max(1));
        let mut result = Ok(());
        while let Some(filled) = fills.next().await {
            match filled {
                Ok(true) => {}
                Ok(false) => stop.store(true, Ordering::Release),
                Err(e) => {
                    stop.store(true, Ordering::Release);
                    if result.is_ok() {
                        result = Err(e);
                    }
                }
            }
        }
        result
    }

    /// Whether `token` is this source's own identity token, compared by pointer. Lets the owning
    /// cache entry check "is this source still current" without cloning the `Arc`.
    pub(crate) fn owns_token(&self, token: &Arc<()>) -> bool {
        Arc::ptr_eq(&self.entry_token, token)
    }

    /// Whether this source charges the budget for the blocks it fills, rather than riding on its
    /// owning entry's reservation.
    pub(crate) fn owns_accounting(&self) -> bool {
        self.owns_accounting
    }

    /// Shared filled-bytes counter, installed as the cache entry's
    /// `size_bytes` so accounting and eviction see live growth.
    pub(crate) fn filled_bytes_handle(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.filled_bytes)
    }

    /// The sparse file, created on first use once the object size is known.
    fn block_file(&self) -> Option<&BlockFile> {
        let size = self.inner.size();
        if size == 0 {
            // Size not discovered yet (pre-`tail()` on an unknown-size
            // source) — don't latch the OnceLock; try again next read.
            return None;
        }
        self.state
            .get_or_init(|| {
                if let Some(adopted) = self.try_adopt(size) {
                    return Some(adopted);
                }
                let _ = fs::remove_file(&self.path);
                let _ = fs::remove_file(self.idx_path());
                let file = fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create_new(true)
                    .open(&self.path)
                    .ok()?;
                file.set_len(size).ok()?;
                Some(BlockFile {
                    file,
                    size,
                    mmap: OnceLock::new(),
                })
            })
            .as_ref()
    }

    /// The sparse file, if it was created. Unlike [`Self::block_file`], never
    /// creates it.
    fn created_block_file(&self) -> Option<&BlockFile> {
        self.state.get()?.as_ref()
    }

    fn idx_path(&self) -> PathBuf {
        let mut p = self.path.clone().into_os_string();
        p.push(".idx");
        PathBuf::from(p)
    }

    /// Adopt a prior generation's sparse file when its persisted index proves
    /// which ranges are valid (immutable superfile, so size-match is enough).
    fn try_adopt(&self, size: u64) -> Option<BlockFile> {
        let meta = fs::metadata(&self.path).ok()?;
        if meta.len() != size {
            return None;
        }
        let idx_bytes = fs::read(self.idx_path()).ok()?;
        let (bitmap, adopted_bytes) = indexed_filled_bytes(size, &idx_bytes)?;
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.path)
            .ok()?;
        if let Some(store) = self.store.upgrade() {
            if self.owns_accounting {
                store.account_adopted_bytes(&self.uri, adopted_bytes);
            } else {
                store.release_scanned_block_file(&self.uri);
            }
        }
        self.filled_bytes.store(adopted_bytes, Ordering::Release);
        *self.filled.lock().expect("filled bitmap mutex poisoned") = bitmap;
        Some(BlockFile {
            file,
            size,
            mmap: OnceLock::new(),
        })
    }

    /// Serialize the filled-block bitmap for persistence. Taken before the
    /// `sync_data()` barrier so the persisted index only ever names blocks whose
    /// writes the fsync already flushed — a concurrent fill that marks a new
    /// block after this snapshot is simply absent (a safe undercount), never a
    /// durably-claimed hole.
    fn snapshot_index(&self) -> Vec<u8> {
        let filled = self.filled.lock().expect("filled bitmap mutex poisoned");
        serialize_index(&filled)
    }

    /// Write a pre-serialized index snapshot, only after its blocks are durable
    /// AND only while `self.path` still names THIS source's own data inode.
    ///
    /// The shared, generation-less `.blocks` / `.blocks.idx` paths mean a later
    /// generation can unlink our data file and publish a fresh inode (all holes)
    /// at the same path while we are still alive. A plain `path.exists()` guard
    /// then lets a dropped or superseded generation rename its own filled-block
    /// bitmap onto the successor's inode — over-claiming that inode's holes. A
    /// subsequent `try_adopt` trusts the stale bitmap and preads zeros where a
    /// Parquet page-index (ColumnIndex) tag belongs, so the decode fails with
    /// `Required field null_pages is missing`. Comparing inodes keeps the
    /// persisted index faithful to whatever data file the path currently names.
    fn persist_idx(&self, bf: &BlockFile, snapshot: &[u8]) {
        use std::os::unix::fs::MetadataExt;
        let own = match bf.file.metadata() {
            Ok(meta) => meta.ino(),
            Err(_) => return,
        };
        match fs::metadata(&self.path) {
            Ok(meta) if meta.ino() == own => {}
            // Path is gone or now names another generation's inode — never
            // stamp our bitmap onto it.
            _ => return,
        }
        let idx = self.idx_path();
        let seq = PERSIST_COUNTER.fetch_add(1, Ordering::Relaxed);
        let mut tmp = idx.clone().into_os_string();
        tmp.push(format!(".tmp.{}.{seq}", process::id()));
        let tmp = PathBuf::from(tmp);
        if fs::write(&tmp, snapshot).is_ok() {
            let _ = fs::rename(&tmp, &idx);
        }
    }

    /// Byte length of block `b` (the trailing block may be partial).
    fn block_len(size: u64, b: u32) -> u64 {
        let start = u64::from(b) * CACHE_BLOCK_BYTES;
        (size - start).min(CACHE_BLOCK_BYTES)
    }

    /// Inclusive block index range covering `[start, start + len)`.
    fn block_span(start: u64, len: u64) -> (u32, u32) {
        let b0 = start / CACHE_BLOCK_BYTES;
        let b1 = (start + len - 1) / CACHE_BLOCK_BYTES;
        (b0 as u32, b1 as u32)
    }

    /// Byte start and length of blocks `[first, last]` in a file of `size` bytes.
    fn run_bytes(size: u64, first: u32, last: u32) -> (u64, u64) {
        let start = u64::from(first) * CACHE_BLOCK_BYTES;
        let end = ((u64::from(last) + 1) * CACHE_BLOCK_BYTES).min(size);
        (start, end - start)
    }

    /// The block file that can serve `start..start + len`, or `None` when the read bypasses the
    /// blocks: no block file (in the hole, none created yet), or a range past the end (which the
    /// inner source reports as an error).
    fn blocks_for(&self, start: u64, len: u64, in_hole: bool) -> Option<&BlockFile> {
        let bf = self.block_file_for(in_hole)?;
        (start.saturating_add(len) <= bf.size).then_some(bf)
    }

    /// The cache's fully resident copy of this superfile, when it holds one. Every async read the
    /// blocks cannot serve goes through it before object storage, so a reader a query still holds
    /// after a promotion stays off object storage. The copy is plain bytes, never another block
    /// source, so the read cannot come back here.
    fn whole_file(&self) -> Option<Arc<dyn LazyByteSource>> {
        self.store.upgrade()?.whole_file_source(&self.uri)
    }

    fn all_filled(&self, b0: u32, b1: u32) -> bool {
        let filled = self.filled.lock().expect("filled bitmap mutex poisoned");
        (b0..=b1).all(|b| filled.contains(b))
    }

    /// The store, if this source is still the live cache entry. Once eviction or
    /// promotion replaced the entry, this source still serves the blocks it has
    /// but fills no new ones.
    fn store_if_current(&self) -> Option<Arc<DiskCacheStore>> {
        let store = self.store.upgrade()?;
        store
            .lazy_block_entry_is_current(&self.uri, &self.entry_token)
            .then_some(store)
    }

    /// Whether a run is in memory: being fetched, or waiting to be written. The
    /// idle sweep keeps such an entry, because `filled_bytes` does not count the
    /// run until it is written.
    pub(crate) fn has_runs_in_memory(&self) -> bool {
        !self.runs.lock().expect("runs mutex poisoned").is_empty()
    }

    /// Takes run `id` out of `runs`, if it is still there under block `first`.
    fn remove_run(&self, first: u32, id: u64) {
        let mut runs = self.runs.lock().expect("runs mutex poisoned");
        if runs.get(&first).is_some_and(|run| run.id == id) {
            runs.remove(&first);
        }
    }

    /// Marks `[b0, b1]` filled and flags the index for the next
    /// [`Self::flush_index`]. Returns the bytes of blocks newly marked (another
    /// fill may have marked some first). Their bytes must already be written.
    fn mark_filled(&self, size: u64, b0: u32, b1: u32) -> u64 {
        let mut filled = self.filled.lock().expect("filled bitmap mutex poisoned");
        let mut newly = 0u64;
        for b in b0..=b1 {
            if filled.insert(b) {
                newly += Self::block_len(size, b);
            }
        }
        if newly > 0 {
            self.dirty.store(true, Ordering::SeqCst);
        }
        newly
    }

    /// Fsyncs the block file, then writes the index of the blocks marked filled.
    /// Blocking: runs at the end of a block write, or in `Drop`.
    /// - One flush runs per source. A call that finds one running returns at
    ///   once; the running one covers its blocks.
    /// - If blocks were marked during the fsync, the running flush goes again,
    ///   so a burst of writes costs one or two fsyncs, not one each.
    /// - A failed fsync leaves the blocks for the next flush, or for Drop.
    fn flush_index(&self) {
        let Some(bf) = self.created_block_file() else {
            return;
        };

        while self.dirty.load(Ordering::SeqCst) {
            if self.flushing.swap(true, Ordering::SeqCst) {
                return;
            }

            // Clear the flag before reading the bitmap: a block marked after this
            // sets it again, and the loop runs once more.
            self.dirty.store(false, Ordering::SeqCst);
            let snapshot = self.snapshot_index();
            let synced = bf.file.sync_data().is_ok();

            if synced {
                self.persist_idx(bf, &snapshot);
            } else {
                // Keep the flag set so the next flush retries, instead of looping
                // on a failing disk.
                self.dirty.store(true, Ordering::SeqCst);
            }

            self.flushing.store(false, Ordering::SeqCst);

            if !synced {
                return;
            }
        }
    }

    /// Splits blocks `[b0, b1]` into stretches by where their bytes are. Each
    /// block is, in this order: in a fetched run, on the block file, in a run
    /// being fetched, or missing. A write marks its blocks filled before its run
    /// leaves `runs`, so a block found in neither really is missing.
    ///
    /// With `register`, each missing stretch is registered in `runs`, under the
    /// same lock as the check so two reads never both fetch it, and comes back
    /// as `Fetching` with its new fetch.
    ///
    /// Waiting never costs more GETs than fetching: where runs being fetched
    /// split the missing blocks of one unavailable stretch into several pieces,
    /// the blocks from the first missing one to the last are fetched as one run,
    /// with one GET, as they would be if nothing were being fetched:
    ///
    /// ```text
    ///   found:    [fetching][missing][fetching][missing][fetching]
    ///   fetched:  [fetching][        one new run        ][fetching]
    /// ```
    fn segments(&self, b0: u32, b1: u32, register: bool) -> Vec<Segment> {
        let mut runs = self.runs.lock().expect("runs mutex poisoned");
        // Fetched runs first, so a block also in a run being fetched is read
        // from the one already back. A fetch every read dropped is no run.
        let mut snapshot: Vec<(u32, u32, RunState)> = runs
            .range(..=b1)
            .filter(|(_, run)| run.last >= b0)
            .filter_map(|(&first, run)| {
                let state = match &run.bytes {
                    RunBytes::Fetched(bytes) => RunState::Fetched(bytes.clone()),
                    RunBytes::Fetching(fetch) => RunState::Fetching(fetch.upgrade()?),
                };
                Some((first, run.last, state))
            })
            .collect();
        snapshot.sort_by_key(|(_, _, state)| matches!(state, RunState::Fetching(_)));

        let filled = self.filled.lock().expect("filled bitmap mutex poisoned");
        let at = |b: u32| {
            let run = snapshot
                .iter()
                .position(|&(first, last, _)| first <= b && b <= last);
            match run {
                Some(i) if matches!(snapshot[i].2, RunState::Fetched(_)) => BlockAt::InMemory(i),
                _ if filled.contains(b) => BlockAt::Filled,
                Some(i) => BlockAt::Fetching(i),
                None => BlockAt::Missing,
            }
        };
        let mut pieces: Vec<(u32, u32, BlockAt)> = Vec::new();
        for b in b0..=b1 {
            let here = at(b);
            match pieces.last_mut() {
                Some((_, last, current)) if *current == here => *last = b,
                _ => pieces.push((b, b, here)),
            }
        }
        drop(filled);

        let mut merged = Vec::with_capacity(pieces.len());
        for stretch in pieces.chunk_by(|a, b| a.2.is_unavailable() == b.2.is_unavailable()) {
            let missing = |piece: &(u32, u32, BlockAt)| piece.2 == BlockAt::Missing;
            match (
                stretch.iter().position(missing),
                stretch.iter().rposition(missing),
            ) {
                (Some(a), Some(z)) if a < z => {
                    merged.extend_from_slice(&stretch[..a]);
                    merged.push((stretch[a].0, stretch[z].1, BlockAt::Missing));
                    merged.extend_from_slice(&stretch[z + 1..]);
                }
                _ => merged.extend_from_slice(stretch),
            }
        }

        let segments = merged
            .into_iter()
            .map(|(first, last, at)| match at {
                BlockAt::Filled => Segment::Filled { first, last },
                BlockAt::InMemory(i) | BlockAt::Fetching(i) => {
                    let run_first = snapshot[i].0;
                    match &snapshot[i].2 {
                        RunState::Fetched(bytes) => Segment::InMemory {
                            first,
                            last,
                            run_first,
                            bytes: bytes.clone(),
                        },
                        RunState::Fetching(fetch) => Segment::Fetching {
                            first,
                            last,
                            run_first,
                            fetch: fetch.clone(),
                            own: false,
                        },
                    }
                }
                BlockAt::Missing => match register.then(|| self.register(&mut runs, first, last)) {
                    Some(Some(fetch)) => Segment::Fetching {
                        first,
                        last,
                        run_first: first,
                        fetch,
                        own: true,
                    },
                    _ => Segment::Missing { first, last },
                },
            })
            .collect();

        // The snapshot's fetches may be the last of some run's; drop them only
        // once `runs` is unlocked.
        drop(runs);
        drop(snapshot);
        segments
    }

    /// Registers missing run `[first, last]` in `runs` and returns its new
    /// fetch, not started: the caller's first poll starts the GET. `None` while
    /// this source is being dropped.
    fn register(
        &self,
        runs: &mut BTreeMap<u32, MemoryRun>,
        first: u32,
        last: u32,
    ) -> Option<RunFetch> {
        let source = self.me.upgrade()?;
        let id = self.next_run_id.fetch_add(1, Ordering::Relaxed);
        let claim = Claim {
            source: Arc::clone(&source),
            first,
            id,
            handed_on: false,
        };
        let fetch = async move {
            match source.block_file() {
                Some(bf) => source.fetch_now(bf, first, last, Some(claim)).await,
                None => Fetched::Elsewhere,
            }
        }
        .boxed()
        .shared();
        // Replaces at most a run every read dropped (`segments` saw no run here),
        // whose claim then finds its id gone.
        runs.insert(
            first,
            MemoryRun {
                id,
                last,
                bytes: RunBytes::Fetching(fetch.downgrade().expect("a fetch not polled yet")),
            },
        );
        Some(fetch)
    }

    /// Builds `[start, start + len)` from `segments`. `None` if a segment is
    /// missing or a local read fails.
    fn assemble(
        &self,
        bf: &BlockFile,
        start: u64,
        len: u64,
        segments: &[Segment],
    ) -> Option<Bytes> {
        let end = start + len;
        let mut pieces = Vec::with_capacity(segments.len());

        for segment in segments {
            let (first, last) = segment.blocks();
            let s = (u64::from(first) * CACHE_BLOCK_BYTES).max(start);
            let e = ((u64::from(last) + 1) * CACHE_BLOCK_BYTES).min(end);
            pieces.push(match segment {
                Segment::Filled { .. } => self.read_local(bf, s, e - s)?,
                Segment::InMemory {
                    run_first, bytes, ..
                } => {
                    let base = u64::from(*run_first) * CACHE_BLOCK_BYTES;
                    bytes.slice((s - base) as usize..(e - base) as usize)
                }
                Segment::Fetching { .. } | Segment::Missing { .. } => return None,
            });
        }

        if let [piece] = pieces.as_slice() {
            // A small read gets a copy: a caller can keep it as long as its reader
            // lives (a vector reader keeps its headers), and a slice would keep
            // the whole fetched run in memory.
            return Some(if len < CACHE_BLOCK_BYTES {
                Bytes::copy_from_slice(piece)
            } else {
                piece.clone()
            });
        }

        let mut out = BytesMut::with_capacity(len as usize);

        for piece in &pieces {
            out.extend_from_slice(piece);
        }

        Some(out.freeze())
    }

    /// Serves a read that is not fully on the block file: filled blocks from the
    /// file, fetched runs from memory, and every other run through its shared
    /// fetch, its own or another read's (hedged past [`HEDGE_AFTER`]). It awaits
    /// all of them at once, so it takes as long as the slowest, not their sum. `None` sends the read to the
    /// whole-file copy or to object storage, when one of those fetches did (see
    /// [`Fetched::Elsewhere`]), another read already returned the GET error, or a
    /// local read failed.
    async fn read_through(
        &self,
        bf: &BlockFile,
        start: u64,
        len: u64,
    ) -> Result<Option<Bytes>, LazyByteSourceError> {
        let (b0, b1) = Self::block_span(start, len);
        let segments = self.segments(b0, b1, true);

        // The first run that ends without bytes stops the read, and drops its
        // other fetches: a fetch this read alone was awaiting is cancelled,
        // while one that already got its bytes has queued them for writing.
        let segments = try_join_all(segments.into_iter().map(|segment| async move {
            let Segment::Fetching {
                first,
                last,
                run_first,
                fetch,
                own,
            } = segment
            else {
                return Ok(segment);
            };
            let (run_first, fetched) = if own {
                (run_first, fetch.await)
            } else {
                self.wait_or_hedge(bf, first, last, run_first, fetch).await
            };
            match fetched {
                Fetched::Bytes(bytes) => Ok(Segment::InMemory {
                    first,
                    last,
                    run_first,
                    bytes,
                }),
                stopped => Err(stopped),
            }
        }))
        .await;

        match segments {
            Ok(segments) => Ok(self.assemble(bf, start, len, &segments)),
            Err(Fetched::Failed(error)) => error.take().map_or(Ok(None), Err),
            Err(_) => Ok(None),
        }
    }

    /// Waits for another read's fetch of blocks `[first, last]`, in the run that
    /// starts at block `run_first`. Past [`HEDGE_AFTER`] it also GETs those
    /// blocks itself and keeps whichever lands first; a hedge that ends without
    /// bytes is dropped and the wait goes on. Returns the block the bytes start
    /// at, and how the fetch ended.
    async fn wait_or_hedge(
        &self,
        bf: &BlockFile,
        first: u32,
        last: u32,
        run_first: u32,
        mut fetch: RunFetch,
    ) -> (u32, Fetched) {
        if let Ok(fetched) = timeout(HEDGE_AFTER, &mut fetch).await {
            return (run_first, fetched);
        }
        select! {
            fetched = &mut fetch => (run_first, fetched),
            hedged = self.fetch_now(bf, first, last, None) => match hedged {
                Fetched::Bytes(_) => (first, hedged),
                _ => (run_first, fetch.await),
            },
        }
    }

    /// Fetches blocks `[first, last]` and queues them for writing without
    /// waiting for the write; see [`Self::fetch_missing`].
    async fn fetch_now(
        &self,
        bf: &BlockFile,
        first: u32,
        last: u32,
        claim: Option<Claim>,
    ) -> Fetched {
        match self.fetch_missing(bf, first, last, Fill::Read, claim).await {
            Ok(Some((bytes, run))) => {
                if let Some(run) = run {
                    // Don't wait for the write: the reads already have their bytes.
                    drop(run.spawn_write());
                }
                Fetched::Bytes(bytes)
            }
            Ok(None) => Fetched::Elsewhere,
            Err(error) => Fetched::Failed(FetchError::new(error)),
        }
    }

    /// Fetches blocks `[first, last]` with one GET, reserving their budget
    /// first, and queues them for writing. Returns:
    /// - `None` when the read has to go elsewhere: the store is gone or this
    ///   source is no longer the live entry, no budget is left, or the GET
    ///   returned another length (the object is not the one this reader opened);
    /// - the bytes and no run when they are fetched but not queued: the cap on
    ///   unwritten runs is full, or a run starting at the same block is queued.
    ///
    /// If the caller is dropped mid-GET, the reservation gives its bytes back.
    /// `claim` is the run's entry in `runs` when a read registered it; see
    /// [`Self::admit_run`].
    async fn fetch_missing(
        &self,
        bf: &BlockFile,
        first: u32,
        last: u32,
        fill: Fill,
        claim: Option<Claim>,
    ) -> Result<Option<(Bytes, Option<FetchedRun>)>, LazyByteSourceError> {
        let Some(store) = self.store_if_current() else {
            return Ok(None);
        };
        let (run_start, run_len) = Self::run_bytes(bf.size, first, last);

        let reservation = if self.owns_accounting {
            let reserved = match fill {
                Fill::Read => store.reserve(run_len).await.ok(),
                Fill::Prefetch => store.try_reserve(run_len),
            };
            let Some(reserved) = reserved else {
                return Ok(None);
            };
            Some(reserved)
        } else {
            None
        };

        let bytes = self.inner.range(run_start, run_len).await?;
        if bytes.len() as u64 != run_len {
            return Ok(None);
        }

        let run = self.admit_run(&store, first, last, &bytes, reservation, fill, claim);
        Ok(Some((bytes, run)))
    }

    /// Queues fetched run `[first, last]` for writing: takes its share of the
    /// store's cap on unwritten runs, keeps its bytes in `runs`, and hands it the
    /// reserved budget. A registered run keeps the `claim`'s entry; an
    /// unregistered one (a prefetch's) gets a new entry. `None` (the reservation
    /// is given back, and a registered run leaves `runs`) if a read finds the
    /// cap full, or an unregistered run finds another run starting at `first`.
    fn admit_run(
        &self,
        store: &Arc<DiskCacheStore>,
        first: u32,
        last: u32,
        bytes: &Bytes,
        reservation: Option<Reservation<'_>>,
        fill: Fill,
        claim: Option<Claim>,
    ) -> Option<FetchedRun> {
        let source = self.me.upgrade()?;

        let run_len = bytes.len() as u64;
        match fill {
            Fill::Read => {
                if !store.try_admit_write_behind(run_len) {
                    return None;
                }
            }
            Fill::Prefetch => store.admit_write_behind(run_len),
        }

        let mut runs = self.runs.lock().expect("runs mutex poisoned");
        let id = match claim {
            Some(mut claim) => {
                if let Some(run) = runs.get_mut(&first).filter(|run| run.id == claim.id) {
                    run.bytes = RunBytes::Fetched(bytes.clone());
                }
                claim.handed_on = true;
                claim.id
            }
            None => match runs.entry(first) {
                // That run owns this key until it is written, so this one is
                // served but not cached.
                Entry::Occupied(_) => {
                    store.release_write_behind(run_len);
                    return None;
                }
                Entry::Vacant(slot) => {
                    let id = self.next_run_id.fetch_add(1, Ordering::Relaxed);
                    slot.insert(MemoryRun {
                        id,
                        last,
                        bytes: RunBytes::Fetched(bytes.clone()),
                    });
                    id
                }
            },
        };
        drop(runs);

        let unlanded = reservation.map_or(0, |reserved| {
            reserved.commit();
            run_len
        });

        Some(FetchedRun {
            source,
            #[cfg(test)]
            _ticket: store.block_write_started(),
            store: Arc::clone(store),
            id,
            first,
            last,
            bytes: bytes.clone(),
            unlanded,
            landed: false,
        })
    }

    /// Holds this source's block-file writes until the returned guard drops, so a
    /// test can look at a run before it is written. The guard lets them go even
    /// if an assert fails, so a failing test can't hang.
    #[cfg(test)]
    pub(crate) fn stall_writes(&self) -> WriteStall<'_> {
        self.write_stall.0.lock().expect("write stall poisoned").on = true;
        WriteStall(self)
    }

    /// How many block writes are held by [`Self::stall_writes`] right now.
    #[cfg(test)]
    fn stalled_writes(&self) -> usize {
        self.write_stall
            .0
            .lock()
            .expect("write stall poisoned")
            .waiting
    }

    #[cfg(test)]
    fn wait_while_stalled(&self) {
        let (state, resume) = &self.write_stall;
        let mut state = state.lock().expect("write stall poisoned");
        state.waiting += 1;
        while state.on {
            state = resume.wait(state).expect("write stall poisoned");
        }
        state.waiting -= 1;
    }

    /// Serve `[start, start+len)` from the sparse file. `None` on a read
    /// error (caller degrades to passthrough).
    fn read_local(&self, bf: &BlockFile, start: u64, len: u64) -> Option<Bytes> {
        // Zero-copy hit service: filled ranges are handed out as slices of
        // one shared read-only mapping instead of alloc+pread per range. At
        // law width a warm vector query reads ~20 MB through here; the
        // per-range alloc+zero+pread copies were the measured ~6-7 ms
        // prefix-service floor of phase A (and the residual copy cost in
        // the deferred rerank's exact-range gather). Slices keep the
        // mapping alive after eviction, exactly like the promoted
        // parquet/FTS mmaps.
        let mapped = bf.mmap.get_or_init(|| {
            // SAFETY: the blocks file is pre-sized at creation
            // (`file.set_len(size)`) and only ever written through
            // `write_all_at` within that size, so a mapping of the full
            // file never outruns it (no SIGBUS); the read-only shared
            // mapping stays page-cache-coherent with those writes, and
            // only blocks found filled (`all_filled` or a `Filled`
            // segment) are read through it.
            unsafe { Mmap::map(&bf.file) }.ok().map(Arc::new)
        });
        if let Some(m) = mapped.as_ref() {
            let s = usize::try_from(start).ok()?;
            let e = s.checked_add(usize::try_from(len).ok()?)?;
            if e <= m.len() {
                return Some(Bytes::from_owner(ArcMmapOwner(Arc::clone(m))).slice(s..e));
            }
        }
        let mut out = vec![0u8; len as usize];
        bf.file.read_exact_at(&mut out, start).ok()?;
        Some(Bytes::from(out))
    }

    /// One chunk of [`Self::prefetch`]: fetches every block of `[b0, b1]` that is
    /// neither filled nor in memory, and waits for it to be written. It reserves
    /// only free budget, so a prefetch never evicts what queries use. It first
    /// waits for runs reads are fetching (driving their GETs), then fetches what
    /// is still missing, so a run a read did not cache is filled too. A run
    /// another fill queued, or that a read registered after that, is left to
    /// it. `false` stops the prefetch, when:
    /// - the budget is full,
    /// - this source is no longer the live entry, or the store is gone,
    /// - the GET came back short, or the write failed.
    async fn prefetch_chunk(
        &self,
        bf: &BlockFile,
        b0: u32,
        b1: u32,
    ) -> Result<bool, LazyByteSourceError> {
        let fetching =
            self.segments(b0, b1, false)
                .into_iter()
                .filter_map(|segment| match segment {
                    Segment::Fetching { fetch, .. } => Some(fetch),
                    _ => None,
                });
        join_all(fetching).await;

        for segment in self.segments(b0, b1, false) {
            let Segment::Missing { first, last } = segment else {
                continue;
            };

            let Some((_, run)) = self
                .fetch_missing(bf, first, last, Fill::Prefetch, None)
                .await?
            else {
                return Ok(false);
            };

            if let Some(run) = run
                && !run.spawn_write().await.unwrap_or(false)
            {
                return Ok(false);
            }
        }

        Ok(true)
    }
}

/// What [`BlockCachedSource::stall_writes`] holds: whether writes wait, and how
/// many are waiting.
#[cfg(test)]
#[derive(Default)]
struct WriteStallState {
    on: bool,
    waiting: usize,
}

/// Holds a source's block-file writes; see [`BlockCachedSource::stall_writes`].
#[cfg(test)]
pub(crate) struct WriteStall<'a>(&'a BlockCachedSource);

#[cfg(test)]
impl Drop for WriteStall<'_> {
    fn drop(&mut self) {
        let (state, resume) = &self.0.write_stall;
        state.lock().expect("write stall poisoned").on = false;
        resume.notify_all();
    }
}

pub(super) fn serialize_index(bitmap: &RoaringBitmap) -> Vec<u8> {
    let mut buf = Vec::with_capacity(IDX_HEADER.len() + bitmap.serialized_size());
    buf.extend_from_slice(&IDX_HEADER);
    let _ = bitmap.serialize_into(&mut buf);
    buf
}

pub(super) fn indexed_filled_bytes(size: u64, idx_bytes: &[u8]) -> Option<(RoaringBitmap, u64)> {
    let body = idx_bytes.strip_prefix(&IDX_HEADER)?;
    let bitmap = RoaringBitmap::deserialize_from(body).ok()?;
    let n_blocks = size.div_ceil(CACHE_BLOCK_BYTES);
    if bitmap.max().is_some_and(|m| u64::from(m) >= n_blocks) {
        return None;
    }
    let filled = bitmap
        .iter()
        .map(|b| BlockCachedSource::block_len(size, b))
        .sum();
    Some((bitmap, filled))
}

impl Drop for BlockCachedSource {
    fn drop(&mut self) {
        // Every queued write holds the source, so none is running now. This flush
        // only retries one that failed.
        self.flush_index();
        // Release accounted bytes only; the file and index persist for adoption.
        if self.owns_accounting
            && let Some(store) = self.store.upgrade()
        {
            let filled = self.filled_bytes.load(Ordering::Acquire);
            if filled > 0 {
                store.release_block_bytes(filled);
            }
        }
    }
}

#[async_trait]
impl LazyByteSource for BlockCachedSource {
    fn size(&self) -> u64 {
        self.inner.size()
    }

    async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
        if len == 0 {
            return Ok(Bytes::new());
        }
        let in_hole = self.in_passthrough(start, len);
        if let Some(bf) = self.blocks_for(start, len, in_hole) {
            let (b0, b1) = Self::block_span(start, len);
            if self.all_filled(b0, b1) {
                if let Some(bytes) = self.read_local(bf, start, len) {
                    return Ok(bytes);
                }
            } else if !in_hole
                // Missing blocks are fetched and kept, except in the hole,
                // which reads never fill.
                && let Some(bytes) = self.read_through(bf, start, len).await?
            {
                return Ok(bytes);
            }
        }

        match self.whole_file() {
            Some(file) => file.range(start, len).await,
            None => self.inner.range(start, len).await,
        }
    }

    // No whole-file fallback here: a caller that gets `None` takes the async `range`, which has it.
    fn try_get_range_sync(&self, start: u64, len: u64) -> Option<Bytes> {
        if len == 0 {
            return Some(Bytes::new());
        }
        let Some(bf) = self.block_file_for(self.in_passthrough(start, len)) else {
            return self.inner.try_get_range_sync(start, len);
        };
        if start.saturating_add(len) > bf.size {
            return None;
        }
        let (b0, b1) = Self::block_span(start, len);
        if self.all_filled(b0, b1) {
            return self.read_local(bf, start, len);
        }
        // Blocks in memory come from there. A block missing or still being
        // fetched (this read can't wait), or a failed local read, sends the read
        // to the inner source.
        self.assemble(bf, start, len, &self.segments(b0, b1, false))
            .or_else(|| self.inner.try_get_range_sync(start, len))
    }

    async fn tail(&self, len: u64) -> Result<(Bytes, u64), LazyByteSourceError> {
        // Pass through: `tail` both fetches and (on unknown-size sources)
        // discovers the object size, which the inner source caches. Tail
        // bytes are open-time metadata already retained by the open-blob /
        // prefetch overlay above this source, so caching them here as
        // (mostly partial) blocks buys nothing.
        self.inner.tail(len).await
    }
}

#[cfg(test)]
mod tests {
    use std::{
        future::pending,
        path::Path,
        pin::pin,
        sync::{atomic::AtomicUsize, mpsc},
        task::Poll,
        thread,
        time::Duration,
    };

    use futures::{FutureExt, poll};
    use tempfile::tempdir;
    use tokio::{
        sync::Notify,
        time::{Instant, advance, sleep, timeout},
    };

    use super::*;
    use crate::{
        runtime_bridge::bridge_sync_to_async_send,
        supertable::reader_cache::{
            ColdFetchMode, DiskCacheConfig, LruPolicy, disk::BLOCK_WRITERS,
        },
    };

    /// How long a test waits for something that should happen right away, before
    /// failing instead of hanging.
    const NO_WAIT_DEADLINE: Duration = Duration::from_secs(10);

    /// How often [`wait_until`] checks its condition.
    const POLL_INTERVAL: Duration = Duration::from_millis(1);

    /// How long a test waits to see that something does not happen.
    const STAYS_PUT: Duration = Duration::from_millis(50);

    /// Waits until `cond` holds, failing with `what` after [`NO_WAIT_DEADLINE`].
    async fn wait_until(mut cond: impl FnMut() -> bool, what: &str) {
        timeout(NO_WAIT_DEADLINE, async {
            while !cond() {
                sleep(POLL_INTERVAL).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("timed out waiting until {what}"));
    }

    /// In-memory fake source that counts `range` calls.
    struct CountingSource {
        blob: Bytes,
        calls: AtomicUsize,
    }

    impl CountingSource {
        fn new(n: usize) -> Self {
            let blob: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();
            Self {
                blob: Bytes::from(blob),
                calls: AtomicUsize::new(0),
            }
        }

        fn calls(&self) -> usize {
            self.calls.load(Ordering::Acquire)
        }
    }

    #[async_trait]
    impl LazyByteSource for CountingSource {
        fn size(&self) -> u64 {
            self.blob.len() as u64
        }

        async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
            self.calls.fetch_add(1, Ordering::AcqRel);
            let (s, e) = (start as usize, (start + len) as usize);
            if e > self.blob.len() {
                return Err(LazyByteSourceError::OutOfBounds {
                    start,
                    len,
                    size: self.blob.len() as u64,
                });
            }
            Ok(self.blob.slice(s..e))
        }
    }

    /// A store whose budget admits everything; `noop_storage` is never hit
    /// because the block source's inner fake serves all reads.
    fn test_store(dir: &Path, budget: u64) -> Arc<DiskCacheStore> {
        use std::{ops::Range, time::SystemTime};

        use object_store::MultipartUpload;

        use crate::storage::{ObjectMeta, StorageError, StorageProvider};

        #[derive(Debug)]
        struct NoopStorage;

        fn unimplemented_err(uri: &str) -> StorageError {
            StorageError::Permanent {
                uri: uri.into(),
                source: "noop storage".into(),
            }
        }

        #[async_trait]
        impl StorageProvider for NoopStorage {
            async fn head(&self, uri: &str) -> Result<ObjectMeta, StorageError> {
                let _ = uri;
                Ok(ObjectMeta {
                    size: 0,
                    etag: None,
                    last_modified: SystemTime::UNIX_EPOCH,
                })
            }
            async fn get(&self, uri: &str) -> Result<(Bytes, ObjectMeta), StorageError> {
                Err(unimplemented_err(uri))
            }
            async fn get_range(&self, uri: &str, _r: Range<u64>) -> Result<Bytes, StorageError> {
                Err(unimplemented_err(uri))
            }
            async fn put_atomic(
                &self,
                uri: &str,
                _b: Bytes,
            ) -> Result<Option<String>, StorageError> {
                Err(unimplemented_err(uri))
            }
            async fn put_overwrite(&self, uri: &str, _b: Bytes) -> Result<(), StorageError> {
                Err(unimplemented_err(uri))
            }
            async fn put_if_match(
                &self,
                uri: &str,
                _b: Bytes,
                _etag: Option<&str>,
            ) -> Result<Option<String>, StorageError> {
                Err(unimplemented_err(uri))
            }
            async fn put_multipart(
                &self,
                uri: &str,
            ) -> Result<Box<dyn MultipartUpload>, StorageError> {
                Err(unimplemented_err(uri))
            }
            async fn delete(&self, _uri: &str) -> Result<(), StorageError> {
                Ok(())
            }
        }

        let cfg = DiskCacheConfig {
            cache_root: dir.to_path_buf(),
            disk_budget_bytes: budget,
            cold_fetch_mode: ColdFetchMode::LazyForegroundWithBackgroundFill,
            eviction: Box::new(LruPolicy::new()),
            ..DiskCacheConfig::default()
        };
        DiskCacheStore::new_unpinned(Arc::new(NoopStorage), cfg).expect("test store")
    }

    /// Second read of the same bytes never touches the inner source, the
    /// returned bytes are identical, and accounting matches the touched
    /// block footprint (not the object size).
    #[tokio::test]
    async fn repeat_reads_are_served_locally_and_accounted_by_blocks() {
        const OBJ: usize = 3 * CACHE_BLOCK_BYTES as usize + 1000;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let uri = SuperfileUri::new_v4();
        let inner = Arc::new(CountingSource::new(OBJ));
        let src = BlockCachedSource::new(
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
            Arc::downgrade(&store),
            uri,
            dir.path().join("t.blocks"),
        );
        // Install the source as current for its (synthetic) entry.
        store.install_block_entry_for_test(uri, Arc::clone(&src));

        // Read spanning blocks 0..=2 (one contiguous missing run → 1 GET).
        let start = 100u64;
        let len = 2 * CACHE_BLOCK_BYTES + 500;
        let first = src.range(start, len).await.expect("first read");
        assert_eq!(first, inner.blob.slice(100..(start + len) as usize));
        assert_eq!(inner.calls(), 1, "one block-run GET for the miss");

        // Identical read → zero inner calls.
        let second = src.range(start, len).await.expect("second read");
        assert_eq!(second, first);
        assert_eq!(inner.calls(), 1, "repeat read must not touch the source");

        // Sub-range and sync reads also come from local blocks.
        let sub = src.range(start + 10, 100).await.expect("sub read");
        assert_eq!(sub, inner.blob.slice(110..210));
        let sync = src
            .try_get_range_sync(start + 10, 100)
            .expect("sync read of filled blocks");
        assert_eq!(sync, sub);
        assert_eq!(inner.calls(), 1);

        // Accounting = 3 whole blocks (0..=2), not the object size.
        store.block_writes_settled().await;
        let expected = 3 * CACHE_BLOCK_BYTES;
        assert_eq!(src.filled_bytes_handle().load(Ordering::Acquire), expected);
        assert_eq!(store.stats().current_bytes, expected);

        // Touch the trailing partial block: its length is size - 3*B.
        let tail_start = 3 * CACHE_BLOCK_BYTES + 10;
        let t = src.range(tail_start, 50).await.expect("tail block read");
        store.block_writes_settled().await;
        assert_eq!(
            t,
            inner
                .blob
                .slice(tail_start as usize..tail_start as usize + 50)
        );
        assert_eq!(inner.calls(), 2);
        assert_eq!(
            src.filled_bytes_handle().load(Ordering::Acquire),
            expected + 1000,
            "trailing partial block accounts its real length"
        );

        // Drop the source: accounting released, but the sparse file and its
        // index persist for a later generation to adopt.
        let path = dir.path().join("t.blocks");
        assert!(path.exists());
        store.remove_block_entry_for_test(&uri);
        drop(src);
        assert_eq!(store.stats().current_bytes, 0);
        assert!(path.exists(), "blocks file survives drop for adoption");
        let mut idx = path.clone().into_os_string();
        idx.push(".idx");
        assert!(
            std::path::PathBuf::from(idx).exists(),
            "index sidecar persisted alongside the blocks file"
        );
    }

    /// Reopen/reconnect warm survival: a fresh source adopts a prior
    /// generation's persisted blocks and serves filled ranges with zero refetch.
    #[tokio::test]
    async fn adopt_reuses_blocks_across_source_generations() {
        const OBJ: usize = 3 * CACHE_BLOCK_BYTES as usize + 1000;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let uri = SuperfileUri::new_v4();
        let path = dir.path().join("t.blocks");

        let inner1 = Arc::new(CountingSource::new(OBJ));
        let src1 = BlockCachedSource::new(
            Arc::clone(&inner1) as Arc<dyn LazyByteSource>,
            Arc::downgrade(&store),
            uri,
            path.clone(),
        );
        store.install_block_entry_for_test(uri, Arc::clone(&src1));
        let start = 100u64;
        let len = 2 * CACHE_BLOCK_BYTES + 500;
        let first = src1.range(start, len).await.expect("gen1 read");
        store.block_writes_settled().await;
        assert_eq!(inner1.calls(), 1);
        let filled = src1.filled_bytes_handle().load(Ordering::Acquire);

        store.remove_block_entry_for_test(&uri);
        drop(src1);
        assert_eq!(store.stats().current_bytes, 0);
        assert!(path.exists());

        let inner2 = Arc::new(CountingSource::new(OBJ));
        let src2 = BlockCachedSource::new(
            Arc::clone(&inner2) as Arc<dyn LazyByteSource>,
            Arc::downgrade(&store),
            uri,
            path.clone(),
        );
        store.install_block_entry_for_test(uri, Arc::clone(&src2));

        let again = src2.range(start, len).await.expect("gen2 read");
        assert_eq!(again, first);
        assert_eq!(inner2.calls(), 0, "adopted blocks serve without refetch");
        assert_eq!(
            src2.filled_bytes_handle().load(Ordering::Acquire),
            filled,
            "adopted footprint re-accounted"
        );
        assert_eq!(store.stats().current_bytes, filled);

        let tail_start = 3 * CACHE_BLOCK_BYTES + 10;
        let _ = src2.range(tail_start, 50).await.expect("gen2 tail");
        assert_eq!(inner2.calls(), 1, "an unfilled range still fetches");
    }

    /// An absent index sidecar makes adoption fall back to a fresh file rather
    /// than trust the sparse bytes blindly.
    #[tokio::test]
    async fn adopt_falls_back_to_fresh_when_index_is_missing() {
        const OBJ: usize = 3 * CACHE_BLOCK_BYTES as usize + 1000;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let uri = SuperfileUri::new_v4();
        let path = dir.path().join("t.blocks");

        let inner1 = Arc::new(CountingSource::new(OBJ));
        let src1 = BlockCachedSource::new(
            Arc::clone(&inner1) as Arc<dyn LazyByteSource>,
            Arc::downgrade(&store),
            uri,
            path.clone(),
        );
        store.install_block_entry_for_test(uri, Arc::clone(&src1));
        let _ = src1
            .range(100, 2 * CACHE_BLOCK_BYTES + 500)
            .await
            .expect("gen1");
        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
        drop(src1);

        let mut idx = path.clone().into_os_string();
        idx.push(".idx");
        std::fs::remove_file(PathBuf::from(idx)).expect("drop the index");

        let inner2 = Arc::new(CountingSource::new(OBJ));
        let src2 = BlockCachedSource::new(
            Arc::clone(&inner2) as Arc<dyn LazyByteSource>,
            Arc::downgrade(&store),
            uri,
            path.clone(),
        );
        store.install_block_entry_for_test(uri, Arc::clone(&src2));
        let _ = src2
            .range(100, 2 * CACHE_BLOCK_BYTES + 500)
            .await
            .expect("gen2");
        assert_eq!(
            inner2.calls(),
            1,
            "no index means fresh fetch, not stale reuse"
        );
    }

    #[test]
    fn indexed_filled_bytes_rejects_a_block_id_past_the_file() {
        let size = 2 * CACHE_BLOCK_BYTES;
        let mut bitmap = RoaringBitmap::new();
        bitmap.insert(2);
        assert!(indexed_filled_bytes(size, &serialize_index(&bitmap)).is_none());
    }

    #[test]
    fn indexed_filled_bytes_sums_in_range_blocks() {
        let size = 2 * CACHE_BLOCK_BYTES + 100;
        let mut bitmap = RoaringBitmap::new();
        bitmap.insert(0);
        bitmap.insert(2);
        let (_, filled) =
            indexed_filled_bytes(size, &serialize_index(&bitmap)).expect("valid index");
        assert_eq!(filled, CACHE_BLOCK_BYTES + 100);
    }

    #[test]
    fn indexed_filled_bytes_rejects_a_mismatched_header() {
        let size = 2 * CACHE_BLOCK_BYTES;
        let mut bitmap = RoaringBitmap::new();
        bitmap.insert(0);
        let mut bytes = serialize_index(&bitmap);
        bytes[0] ^= 0xFF;
        assert!(indexed_filled_bytes(size, &bytes).is_none());
    }

    /// Regression for the transient `Required field type_ is missing` decode
    /// crash: two sources for the same superfile share a deterministic
    /// `.blocks` path. The second source's `block_file()` opens with
    /// `truncate(true)`, zeroing the sparse file the first still holds open —
    /// so the first, still trusting its "filled" bitmap, reads zeros where it
    /// had cached real bytes. (In production the second source is a
    /// post-eviction cold refetch of a superfile a long scan still holds; the
    /// zeros land where a Parquet Thrift tag belongs and the decode panics.)
    ///
    /// The second source reads a DISJOINT range so its truncate does not
    /// coincidentally refill the first source's blocks. Fails today (the first
    /// reader reads zeros); passes once eviction unlinks the `.blocks` file so
    /// the refetch gets a fresh inode and the live reader keeps its own.
    #[tokio::test]
    async fn stale_shared_blocks_file_must_not_corrupt_live_reader() {
        const OBJ: usize = 4 * CACHE_BLOCK_BYTES as usize + 1000;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let uri = SuperfileUri::new_v4();
        // Both sources map to the same per-URI scratch path (the bug).
        let path = dir.path().join("shared.blocks");

        // Reader A becomes the current entry, fills blocks 0..=2 onto the
        // shared file, and stays alive (a long scan holding it).
        let inner_a = Arc::new(CountingSource::new(OBJ));
        let a = BlockCachedSource::new(
            Arc::clone(&inner_a) as Arc<dyn LazyByteSource>,
            Arc::downgrade(&store),
            uri,
            path.clone(),
        );
        store.install_block_entry_for_test(uri, Arc::clone(&a));
        let start = 100u64;
        let len = 2 * CACHE_BLOCK_BYTES + 500;
        let want = inner_a.blob.slice(start as usize..(start + len) as usize);
        let a_first = a.range(start, len).await.expect("A first read");
        store.block_writes_settled().await;
        assert_eq!(a_first, want, "A reads correct bytes before the collision");
        assert_eq!(inner_a.calls(), 1, "A's blocks are on the shared file");

        // Eviction removes the catalog entry but leaves the `.blocks` file on
        // disk (today's bug); reader A is still held by its long scan.
        store.remove_block_entry_for_test(&uri);

        // Reader B is the post-eviction refetch: same uri, same path, becomes
        // current, and fills only the trailing block — but its `block_file()`
        // truncates the shared inode A still holds open first.
        let inner_b = Arc::new(CountingSource::new(OBJ));
        let b = BlockCachedSource::new(
            Arc::clone(&inner_b) as Arc<dyn LazyByteSource>,
            Arc::downgrade(&store),
            uri,
            path.clone(),
        );
        store.install_block_entry_for_test(uri, Arc::clone(&b));
        let tail = 3 * CACHE_BLOCK_BYTES + 10;
        let _ = b
            .range(tail, 50)
            .await
            .expect("B read (truncates shared file)");
        store.block_writes_settled().await;

        // A re-reads the range it already cached. Its bitmap still says filled,
        // so it serves from the now-truncated shared file.
        let a_again = a.range(start, len).await.expect("A re-read");
        assert_eq!(
            a_again, want,
            "live reader A must still serve its cached bytes, not zeros"
        );
    }

    /// A superseded generation's `.blocks.idx` must never over-claim a
    /// SUCCESSOR generation's data inode. This is the sibling of the
    /// truncate race above: `#489` stopped a refetch from truncating a held
    /// inode by giving each generation a fresh inode; but a dropped
    /// generation still stamps its own filled-block bitmap onto the shared
    /// `.blocks.idx` path — which, after eviction + refetch, now names a
    /// DIFFERENT inode whose corresponding blocks are holes. A later
    /// generation then `try_adopt`s that data file on a size-match alone,
    /// trusts the stale bitmap, and preads zeros where a Parquet page-index
    /// (ColumnIndex) tag belongs → `Required field null_pages is missing`.
    ///
    /// Interleaving: A (gen1) fetches block b onto inode X, its write queued.
    /// Eviction unlinks the files (X stays alive under A). B (gen2)
    /// cold-opens → fresh inode Y (holes), fills a DISJOINT block, persists
    /// idx {other}. A's write runs and persists {b}, over Y's path. D (gen3)
    /// adopts Y (size matches) with bitmap {b} and reads block b → zeros.
    /// Fails today; passes once `persist_idx` fences its write to its own
    /// data inode.
    #[tokio::test]
    async fn dropped_generation_idx_must_not_overclaim_successor_inode() {
        const OBJ: usize = 4 * CACHE_BLOCK_BYTES as usize + 1000;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let uri = SuperfileUri::new_v4();
        let path = dir.path().join("shared.blocks");

        // Gen1 A fetches block 0 (covering `start`) onto inode X. Its write is
        // held until gen2 owns the path.
        let inner_a = Arc::new(CountingSource::new(OBJ));
        let a = BlockCachedSource::new(
            Arc::clone(&inner_a) as Arc<dyn LazyByteSource>,
            Arc::downgrade(&store),
            uri,
            path.clone(),
        );
        store.install_block_entry_for_test(uri, Arc::clone(&a));
        let start = 100u64;
        let len = CACHE_BLOCK_BYTES; // block 0 (and a sliver of 1)
        let want = inner_a.blob.slice(start as usize..(start + len) as usize);
        let a_stall = a.stall_writes();
        assert_eq!(a.range(start, len).await.expect("A fill"), want);

        // Real eviction: drop the catalog entry AND unlink the on-disk files,
        // so the next generation cannot adopt them and gets a FRESH inode. A
        // survives (held by its queued write) with inode X open.
        store.remove_block_entry_for_test(&uri);
        let idx_path = a.idx_path();
        let _ = fs::remove_file(&path);
        let _ = fs::remove_file(&idx_path);

        // Gen2 B cold-opens: try_adopt fails (files gone) → fresh inode Y, all
        // holes. It fills a DISJOINT trailing block and persists idx {3} for Y.
        let inner_b = Arc::new(CountingSource::new(OBJ));
        let b = BlockCachedSource::new(
            Arc::clone(&inner_b) as Arc<dyn LazyByteSource>,
            Arc::downgrade(&store),
            uri,
            path.clone(),
        );
        store.install_block_entry_for_test(uri, Arc::clone(&b));
        let tail = 3 * CACHE_BLOCK_BYTES + 10;
        let _ = b.range(tail, 50).await.expect("B fill disjoint block");
        wait_until(|| idx_path.exists(), "B writes its index").await;

        // A's write runs now and writes its {0} bitmap. WITHOUT the inode fence
        // this renames {0} onto `.blocks.idx`, which now names Y (block 0 = hole).
        drop(a_stall);
        store.block_writes_settled().await;
        drop(a);

        // Gen3 D cold-opens → try_adopt(size) accepts Y (len matches) with the
        // over-claimed bitmap {0}, then serves block 0 from a hole = zeros.
        let inner_d = Arc::new(CountingSource::new(OBJ));
        let d = BlockCachedSource::new(
            Arc::clone(&inner_d) as Arc<dyn LazyByteSource>,
            Arc::downgrade(&store),
            uri,
            path.clone(),
        );
        store.install_block_entry_for_test(uri, Arc::clone(&d));
        let d_read = d.range(start, len).await.expect("D read");
        assert_eq!(
            d_read, want,
            "D must serve real bytes, not zeros from a successor inode's hole \
             claimed by a dropped generation's stale bitmap"
        );
    }

    /// Two disjoint missing runs in one request → one GET per run.
    #[tokio::test]
    async fn disjoint_missing_runs_fetch_separately() {
        const OBJ: usize = 6 * CACHE_BLOCK_BYTES as usize;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let uri = SuperfileUri::new_v4();
        let inner = Arc::new(CountingSource::new(OBJ));
        let src = BlockCachedSource::new(
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
            Arc::downgrade(&store),
            uri,
            dir.path().join("runs.blocks"),
        );
        store.install_block_entry_for_test(uri, Arc::clone(&src));

        // Fill block 2 first.
        let b = CACHE_BLOCK_BYTES;
        let _ = src.range(2 * b, 10).await.expect("fill middle block");
        assert_eq!(inner.calls(), 1);

        // Read blocks 1..=3: blocks 1 and 3 are missing → two run GETs.
        let got = src.range(b, 3 * b).await.expect("spanning read");
        assert_eq!(got, inner.blob.slice(b as usize..(4 * b) as usize));
        assert_eq!(inner.calls(), 3, "two missing runs around the filled block");

        store.remove_block_entry_for_test(&uri);
    }

    /// When the entry is no longer current (evicted/promoted), reads still
    /// succeed as passthrough and accounting stops growing.
    #[tokio::test]
    async fn stale_entry_degrades_to_passthrough() {
        const OBJ: usize = 2 * CACHE_BLOCK_BYTES as usize;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let uri = SuperfileUri::new_v4();
        let inner = Arc::new(CountingSource::new(OBJ));
        let src = BlockCachedSource::new(
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
            Arc::downgrade(&store),
            uri,
            dir.path().join("stale.blocks"),
        );
        // Never installed as current → every miss is passthrough.
        let a = src.range(0, 64).await.expect("passthrough read");
        let bb = src.range(0, 64).await.expect("passthrough read again");
        assert_eq!(a, bb);
        assert_eq!(inner.calls(), 2, "uncached passthrough on both reads");
        assert_eq!(src.filled_bytes_handle().load(Ordering::Acquire), 0);
        assert_eq!(store.stats().current_bytes, 0);
    }

    /// A source with a passthrough hole of `hole` over an `obj`-byte blob,
    /// installed as current for a fresh uri.
    fn holed_source(
        dir: &Path,
        store: &Arc<DiskCacheStore>,
        obj: usize,
        hole: (u64, u64),
    ) -> (SuperfileUri, Arc<CountingSource>, Arc<BlockCachedSource>) {
        let uri = SuperfileUri::new_v4();
        let inner = Arc::new(CountingSource::new(obj));
        let src = BlockCachedSource::new_with_accounting(
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
            Arc::downgrade(store),
            uri,
            dir.join("hole.blocks"),
            true,
            Some(hole),
        );
        store.install_block_entry_for_test(uri, Arc::clone(&src));
        (uri, inner, src)
    }

    /// Hole reads stay exact passthrough until a prefetch fills the hole; then
    /// they, and sync reads, come from disk with no more GETs.
    #[tokio::test]
    async fn prefetched_hole_is_served_from_disk() {
        let b = CACHE_BLOCK_BYTES;
        let obj = 16 * b as usize;
        let hole = (2 * b + 100, 12 * b);
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) = holed_source(dir.path(), &store, obj, hole);

        // Before: every hole read is its own exact GET.
        let read_start = 3 * b + 5;
        let _ = src.range(read_start, 1000).await.expect("hole read");
        let _ = src.range(read_start, 1000).await.expect("hole read again");
        assert_eq!(inner.calls(), 2, "hole reads pass through");
        assert_eq!(src.filled_bytes_handle().load(Ordering::Acquire), 0);

        // Blocks 2..=14 in 4-block chunks: 4 GETs.
        src.prefetch(hole.0, hole.1, 4 * b, 2)
            .await
            .expect("prefetch");
        assert_eq!(inner.calls(), 6, "one GET per prefetch chunk");
        assert_eq!(src.filled_bytes_handle().load(Ordering::Acquire), 13 * b);

        let got = src.range(read_start, 1000).await.expect("hole read");
        assert_eq!(
            got,
            inner
                .blob
                .slice(read_start as usize..read_start as usize + 1000)
        );
        let sync = src
            .try_get_range_sync(read_start, 1000)
            .expect("sync hole read from disk");
        assert_eq!(sync, got);
        assert_eq!(inner.calls(), 6, "prefetched hole reads need no GETs");

        store.remove_block_entry_for_test(&uri);
    }

    /// `prefetch_range` fills a lazy entry's hole through the store.
    #[tokio::test]
    async fn prefetch_range_fills_a_lazy_entry() {
        let b = CACHE_BLOCK_BYTES;
        let hole = (b, 4 * b);
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) = holed_source(dir.path(), &store, 8 * b as usize, hole);

        store
            .prefetch_range(&uri, hole.0, hole.1)
            .await
            .expect("prefetch");
        assert_eq!(src.filled_bytes_handle().load(Ordering::Acquire), 4 * b);
        let calls = inner.calls();
        let _ = src.range(2 * b, 100).await.expect("hole read");
        assert_eq!(inner.calls(), calls, "served from the prefetched blocks");

        store.remove_block_entry_for_test(&uri);
    }

    /// A prefetch that runs out of budget stops without an error, leaves no
    /// budget reserved for blocks it did not fill, and reads it did not cover
    /// still pass through correctly.
    #[tokio::test]
    async fn prefetch_stops_when_the_budget_runs_out() {
        let b = CACHE_BLOCK_BYTES;
        let obj = 10 * b as usize;
        let hole = (0, 10 * b);
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), 2 * b);
        let (uri, inner, src) = holed_source(dir.path(), &store, obj, hole);

        src.prefetch(hole.0, hole.1, b, 4)
            .await
            .expect("a budget stop is not an error");
        let filled = src.filled_bytes_handle().load(Ordering::Acquire);
        assert!(filled <= 2 * b);
        assert_eq!(
            store.stats().current_bytes,
            filled,
            "only filled blocks stay charged"
        );

        let late = 8 * b + 7;
        let got = src.range(late, 100).await.expect("uncovered hole read");
        assert_eq!(got, inner.blob.slice(late as usize..late as usize + 100));

        store.remove_block_entry_for_test(&uri);
    }

    /// A prefetch that fills nothing does not rewrite the block index.
    #[tokio::test]
    async fn prefetch_of_filled_blocks_skips_the_index_write() {
        let b = CACHE_BLOCK_BYTES;
        let hole = (0, 4 * b);
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) = holed_source(dir.path(), &store, 4 * b as usize, hole);

        src.prefetch(hole.0, hole.1, b, 2)
            .await
            .expect("first prefetch");
        let idx = src.idx_path();
        assert!(idx.exists(), "first prefetch writes the index");
        fs::remove_file(&idx).expect("remove index");

        let calls = inner.calls();
        src.prefetch(hole.0, hole.1, b, 2)
            .await
            .expect("second prefetch");
        assert_eq!(inner.calls(), calls, "nothing left to fetch");
        assert!(
            !idx.exists(),
            "nothing filled, so the index is not rewritten"
        );

        store.remove_block_entry_for_test(&uri);
    }

    /// Prefetching a uri with no lazy entry does nothing.
    #[tokio::test]
    async fn prefetch_range_skips_a_uri_that_is_not_open_lazily() {
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        store
            .prefetch_range(&SuperfileUri::new_v4(), 0, CACHE_BLOCK_BYTES)
            .await
            .expect("no entry is not an error");
    }

    /// A source whose GETs never finish.
    struct HangingSource {
        size: u64,
    }

    #[async_trait]
    impl LazyByteSource for HangingSource {
        fn size(&self) -> u64 {
            self.size
        }

        async fn range(&self, _start: u64, _len: u64) -> Result<Bytes, LazyByteSourceError> {
            pending().await
        }
    }

    /// A read dropped while its GET is in flight gives back the budget it
    /// reserved for the fill.
    #[tokio::test]
    async fn dropped_fill_releases_its_reservation() {
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let uri = SuperfileUri::new_v4();
        let src = BlockCachedSource::new(
            Arc::new(HangingSource {
                size: 4 * CACHE_BLOCK_BYTES,
            }) as Arc<dyn LazyByteSource>,
            Arc::downgrade(&store),
            uri,
            dir.path().join("hang.blocks"),
        );
        store.install_block_entry_for_test(uri, Arc::clone(&src));

        // Polled once: the fill reserves, then waits on the GET and is dropped.
        assert!(src.range(0, 100).now_or_never().is_none(), "the GET hangs");
        assert_eq!(
            store.stats().current_bytes,
            0,
            "the reservation is given back"
        );

        store.remove_block_entry_for_test(&uri);
    }

    /// A prefetch on a full cache fills nothing rather than evict blocks a
    /// query filled.
    #[tokio::test]
    async fn prefetch_never_evicts() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), 2 * b);
        let uri = SuperfileUri::new_v4();
        let inner = Arc::new(CountingSource::new(2 * b as usize));
        let queried = BlockCachedSource::new(
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
            Arc::downgrade(&store),
            uri,
            dir.path().join("queried.blocks"),
        );
        store.install_block_entry_for_test(uri, Arc::clone(&queried));
        let _ = queried.range(0, 2 * b).await.expect("query read");
        assert_eq!(store.stats().current_bytes, 2 * b, "the cache is full");

        let (other_uri, _, other) = holed_source(dir.path(), &store, 4 * b as usize, (0, 4 * b));
        other.prefetch(0, 4 * b, b, 2).await.expect("prefetch");
        assert_eq!(other.filled_bytes_handle().load(Ordering::Acquire), 0);
        assert!(store.is_cached(&uri), "the query's entry stays");
        assert_eq!(store.stats().current_bytes, 2 * b);
        let calls = inner.calls();
        let _ = queried.range(0, 2 * b).await.expect("query read again");
        assert_eq!(inner.calls(), calls, "still served from its blocks");

        store.remove_block_entry_for_test(&other_uri);
        store.remove_block_entry_for_test(&uri);
    }

    /// A source over `inner`, installed as current for a fresh uri.
    fn installed_source(
        dir: &Path,
        store: &Arc<DiskCacheStore>,
        inner: Arc<dyn LazyByteSource>,
    ) -> (SuperfileUri, Arc<BlockCachedSource>) {
        let uri = SuperfileUri::new_v4();
        let src = BlockCachedSource::new(inner, Arc::downgrade(store), uri, dir.join("wb.blocks"));
        store.install_block_entry_for_test(uri, Arc::clone(&src));
        (uri, src)
    }

    /// A [`CountingSource`] over an `obj`-byte blob, behind an installed source.
    fn counting_source(
        dir: &Path,
        store: &Arc<DiskCacheStore>,
        obj: usize,
    ) -> (SuperfileUri, Arc<CountingSource>, Arc<BlockCachedSource>) {
        let inner = Arc::new(CountingSource::new(obj));
        let (uri, src) =
            installed_source(dir, store, Arc::clone(&inner) as Arc<dyn LazyByteSource>);
        (uri, inner, src)
    }

    /// A read whose block write is held:
    ///  - returns right away, with the right bytes,
    ///  - repeat reads, async and sync, come from memory with no GET,
    ///  - no index lists the blocks yet.
    /// Once the write runs, the blocks are filled, reserved once, and indexed.
    #[tokio::test]
    async fn a_read_never_waits_for_its_block_write() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) = counting_source(dir.path(), &store, 4 * b as usize);

        let stall = src.stall_writes();
        let start = 100u64;
        let len = 2 * b;
        let want = inner.blob.slice(start as usize..(start + len) as usize);
        let first = timeout(NO_WAIT_DEADLINE, src.range(start, len))
            .await
            .expect("the read does not wait for its write");
        assert_eq!(first.expect("first read"), want);
        assert_eq!(inner.calls(), 1);

        assert_eq!(src.range(start, len).await.expect("repeat read"), want);
        let sync = src
            .try_get_range_sync(start + 10, 100)
            .expect("sync read of a pending run");
        assert_eq!(sync, inner.blob.slice(110..210));
        assert_eq!(inner.calls(), 1, "pending blocks are served from memory");
        assert_eq!(src.filled_bytes_handle().load(Ordering::Acquire), 0);
        assert!(
            !src.idx_path().exists(),
            "no index names an unwritten block"
        );
        assert_eq!(
            store.stats().current_bytes,
            3 * b,
            "the run is charged while it waits"
        );

        drop(stall);
        store.block_writes_settled().await;
        assert_eq!(src.filled_bytes_handle().load(Ordering::Acquire), 3 * b);
        assert_eq!(store.stats().current_bytes, 3 * b);
        assert_eq!(store.write_behind_in_use(), 0);
        let idx = fs::read(src.idx_path()).expect("index written after the write");
        let (bitmap, _) = indexed_filled_bytes(4 * b, &idx).expect("valid index");
        assert_eq!(bitmap.iter().collect::<Vec<_>>(), vec![0, 1, 2]);
        assert_eq!(src.range(start, len).await.expect("read from disk"), want);
        assert_eq!(inner.calls(), 1);

        store.remove_block_entry_for_test(&uri);
    }

    /// A run that is never written, like one a shutting-down runtime drops,
    /// gives everything back: its budget, its share of the cap, its pending
    /// entry. No index lists its blocks.
    #[tokio::test]
    async fn a_run_dropped_unrun_gives_everything_back() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, _inner, src) = counting_source(dir.path(), &store, 2 * b as usize);
        let bf = src.block_file().expect("block file");

        let (_, run) = src
            .fetch_missing(bf, 0, 0, Fill::Read, None)
            .await
            .expect("GET")
            .expect("fetched");
        let run = run.expect("queued");
        assert_eq!(store.stats().current_bytes, b, "the run holds its charge");
        assert!(src.has_runs_in_memory());

        drop(run);
        assert_eq!(store.stats().current_bytes, 0);
        assert_eq!(store.write_behind_in_use(), 0);
        assert!(!src.has_runs_in_memory());
        assert_eq!(src.filled_bytes_handle().load(Ordering::Acquire), 0);
        assert!(!src.idx_path().exists());
        store.remove_block_entry_for_test(&uri);
    }

    /// A sync read from a thread with no runtime (a rayon reader thread) runs on
    /// a throwaway runtime.
    ///  - the read does not wait for its write,
    ///  - the write still happens after that runtime is gone.
    #[tokio::test]
    async fn a_bridged_read_neither_waits_for_nor_loses_its_write() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) = counting_source(dir.path(), &store, 2 * b as usize);

        let stall = src.stall_writes();
        let (tx, rx) = mpsc::channel();
        let reader = Arc::clone(&src);
        thread::spawn(move || {
            let got = bridge_sync_to_async_send(async move { reader.range(100, 64).await });
            let _ = tx.send(got);
        });
        let got = rx
            .recv_timeout(NO_WAIT_DEADLINE)
            .expect("the bridged read does not wait for its write");
        assert_eq!(got.expect("bridged read"), inner.blob.slice(100..164));

        drop(stall);
        store.block_writes_settled().await;
        assert_eq!(src.filled_bytes_handle().load(Ordering::Acquire), b);
        assert_eq!(store.stats().current_bytes, b);
        assert_eq!(
            store.write_behind_in_use(),
            0,
            "its share of the cap came back"
        );
        store.remove_block_entry_for_test(&uri);
    }

    /// A read that starts inside a pending run gets the right bytes from it,
    /// async and sync, with no GET.
    #[tokio::test]
    async fn a_read_inside_a_pending_run_slices_it_at_its_offset() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) = counting_source(dir.path(), &store, 3 * b as usize);

        let stall = src.stall_writes();
        let _ = src.range(0, 3 * b).await.expect("pend blocks 0..=2");
        let at = b + 5;
        let got = src.range(at, 100).await.expect("read inside the run");
        assert_eq!(got, inner.blob.slice(at as usize..at as usize + 100));
        let sync = src
            .try_get_range_sync(at, b)
            .expect("sync read inside the run");
        assert_eq!(sync, inner.blob.slice(at as usize..(at + b) as usize));
        assert_eq!(inner.calls(), 1);

        drop(stall);
        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
    }

    /// A read that spans filled, pending and missing blocks returns them in
    /// order, and fetches only the missing ones.
    #[tokio::test]
    async fn a_read_stitches_filled_pending_and_missing_blocks() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) = counting_source(dir.path(), &store, 6 * b as usize);

        // Block 1 filled, then block 3 pending behind a stalled writer.
        let _ = src.range(b, 10).await.expect("fill block 1");
        store.block_writes_settled().await;
        let stall = src.stall_writes();
        let _ = src.range(3 * b, 10).await.expect("pend block 3");
        assert_eq!(inner.calls(), 2);

        // Blocks 0..=4: 0, 2 and 4 are missing, so three GETs.
        let start = 100;
        let end = 4 * b + 200;
        let got = src.range(start, end - start).await.expect("mixed read");
        assert_eq!(got, inner.blob.slice(start as usize..end as usize));
        assert_eq!(inner.calls(), 5);

        drop(stall);
        store.block_writes_settled().await;
        assert_eq!(src.filled_bytes_handle().load(Ordering::Acquire), 5 * b);
        store.remove_block_entry_for_test(&uri);
    }

    /// Two reads that miss the same run at once fetch it once: the second waits
    /// for the first's GET. The run is reserved once, written once, and the cap
    /// is given back in full.
    #[tokio::test]
    async fn racing_reads_of_one_run_fetch_it_once() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = held_source(2 * b as usize, false);
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );

        let stall = src.stall_writes();
        let mut one = pin!(src.range(0, 64));
        let mut two = pin!(src.range(10, 64));
        assert!(poll!(one.as_mut()).is_pending(), "the first read GETs");
        assert!(poll!(two.as_mut()).is_pending(), "the second read waits");
        inner.release();
        let (one, two) = timeout(NO_WAIT_DEADLINE, async { tokio::join!(one, two) })
            .await
            .expect("both reads finish");
        assert_eq!(one.expect("first read"), inner.inner.blob.slice(0..64));
        assert_eq!(two.expect("second read"), inner.inner.blob.slice(10..74));
        assert_eq!(inner.inner.calls(), 1, "one GET for both reads");
        assert_eq!(store.write_behind_in_use(), b, "only one run is queued");
        assert_eq!(store.stats().current_bytes, b, "only one is charged");

        drop(stall);
        store.block_writes_settled().await;
        assert_eq!(src.filled_bytes_handle().load(Ordering::Acquire), b);
        assert_eq!(store.stats().current_bytes, b, "charged once");
        assert_eq!(
            store.write_behind_in_use(),
            0,
            "its share of the cap came back"
        );
        store.remove_block_entry_for_test(&uri);
    }

    /// Many small reads of one block, issued together, cost one GET.
    #[tokio::test]
    async fn parallel_small_reads_of_one_block_fetch_it_once() {
        const READS: u64 = 8;
        const READ_LEN: u64 = 50;
        const READ_STRIDE: u64 = 100;
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = held_source(b as usize, false);
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );

        let mut reads: Vec<_> = (0..READS)
            .map(|i| Box::pin(src.range(i * READ_STRIDE, READ_LEN)))
            .collect();
        for read in &mut reads {
            assert!(poll!(read.as_mut()).is_pending(), "one GET, the rest wait");
        }
        inner.release();
        let got = timeout(NO_WAIT_DEADLINE, join_all(reads))
            .await
            .expect("every read finishes");
        for (i, bytes) in (0..READS).zip(got) {
            let s = (i * READ_STRIDE) as usize;
            assert_eq!(
                bytes.expect("read"),
                inner.inner.blob.slice(s..s + READ_LEN as usize)
            );
        }
        assert_eq!(inner.inner.calls(), 1, "one GET for all of them");

        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
    }

    /// A read dropped mid-GET leaves the GET to the read still waiting on it,
    /// which drives it to the end: one GET, and the run is cached.
    #[tokio::test]
    async fn a_dropped_fetcher_leaves_its_get_to_a_waiting_read() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = held_source(2 * b as usize, false);
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );

        let mut one = Box::pin(src.range(0, 64));
        let mut two = pin!(src.range(10, 64));
        assert!(poll!(one.as_mut()).is_pending(), "the first read GETs");
        assert!(poll!(two.as_mut()).is_pending(), "the second read waits");
        drop(one);
        inner.release();
        let two = timeout(NO_WAIT_DEADLINE, two)
            .await
            .expect("the second read finishes");
        assert_eq!(two.expect("second read"), inner.inner.blob.slice(10..74));
        assert_eq!(inner.inner.calls(), 1, "the first read's GET, finished");

        store.block_writes_settled().await;
        assert_eq!(src.filled_bytes_handle().load(Ordering::Acquire), b);
        assert_eq!(store.stats().current_bytes, b, "cached once");
        assert_eq!(store.write_behind_in_use(), 0);
        assert!(!src.has_runs_in_memory());
        store.remove_block_entry_for_test(&uri);
    }

    /// A read nobody polls any more can't hold up a read waiting on its GET:
    /// the waiting read drives it.
    #[tokio::test]
    async fn a_waiting_read_does_not_need_its_fetcher_polled() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = held_source(2 * b as usize, false);
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );

        let mut parked = pin!(src.range(0, 64));
        let mut waiting = pin!(src.range(10, 64));
        assert!(poll!(parked.as_mut()).is_pending(), "the first read GETs");
        assert!(
            poll!(waiting.as_mut()).is_pending(),
            "the second read waits"
        );
        inner.release();
        let waiting = timeout(NO_WAIT_DEADLINE, waiting)
            .await
            .expect("finishes with the first read never polled again");
        assert_eq!(
            waiting.expect("second read"),
            inner.inner.blob.slice(10..74)
        );
        assert_eq!(inner.inner.calls(), 1);
        assert_eq!(
            poll!(parked.as_mut()).map(|bytes| bytes.expect("first read")),
            Poll::Ready(inner.inner.blob.slice(0..64)),
            "the parked read finds its bytes when polled",
        );

        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
    }

    /// Every read of a run dropped mid-GET cancels it: nothing stays charged or
    /// in memory.
    #[tokio::test]
    async fn dropping_every_read_of_a_run_cancels_its_get() {
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::new(HangingSource {
                size: 2 * CACHE_BLOCK_BYTES,
            }) as Arc<dyn LazyByteSource>,
        );

        let mut one = Box::pin(src.range(0, 64));
        let mut two = Box::pin(src.range(10, 64));
        assert!(poll!(one.as_mut()).is_pending());
        assert!(poll!(two.as_mut()).is_pending());
        assert!(src.has_runs_in_memory(), "the run is registered");
        drop((one, two));
        assert!(!src.has_runs_in_memory(), "and gone with its last read");
        assert_eq!(store.stats().current_bytes, 0, "its budget is given back");
        store.remove_block_entry_for_test(&uri);
    }

    /// A failed GET reaches one of the reads waiting on it as its error. The
    /// other reads straight from the source instead, uncached.
    #[tokio::test]
    async fn a_failed_get_errors_one_read_and_the_other_reads_elsewhere() {
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = held_source(2 * CACHE_BLOCK_BYTES as usize, true);
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );

        let mut one = pin!(src.range(0, 64));
        let mut two = pin!(src.range(10, 64));
        assert!(poll!(one.as_mut()).is_pending(), "the first read GETs");
        assert!(poll!(two.as_mut()).is_pending(), "the second read waits");
        inner.release();
        let (one, two) = timeout(NO_WAIT_DEADLINE, async { tokio::join!(one, two) })
            .await
            .expect("both reads finish");
        let (err, ok, want) = match (one, two) {
            (Err(e), Ok(bytes)) => (e, bytes, inner.inner.blob.slice(10..74)),
            (Ok(bytes), Err(e)) => (e, bytes, inner.inner.blob.slice(0..64)),
            other => panic!("one error and one read expected, got {other:?}"),
        };
        assert!(matches!(err, LazyByteSourceError::OutOfBounds { .. }));
        assert_eq!(ok, want);
        assert_eq!(inner.inner.calls(), 1, "the other read's own GET");

        store.block_writes_settled().await;
        assert_eq!(store.stats().current_bytes, 0, "nothing was cached");
        assert_eq!(store.write_behind_in_use(), 0);
        assert!(!src.has_runs_in_memory());
        store.remove_block_entry_for_test(&uri);
    }

    /// Waiting never costs more GETs than fetching: a read whose missing blocks
    /// sit on both sides of a run in flight GETs them with the run, in one GET,
    /// as it would if nothing were in flight.
    #[tokio::test]
    async fn a_run_in_flight_never_splits_a_read_into_more_gets() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = held_source(3 * b as usize, false);
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );

        let mut middle = pin!(src.range(b + 10, 10));
        assert!(poll!(middle.as_mut()).is_pending(), "block 1 is in flight");
        let all = timeout(NO_WAIT_DEADLINE, src.range(0, 3 * b))
            .await
            .expect("the wide read doesn't wait")
            .expect("wide read");
        assert_eq!(all, inner.inner.blob.slice(0..(3 * b) as usize));
        assert_eq!(inner.inner.calls(), 1, "one GET for blocks 0 to 2");

        inner.release();
        let middle = timeout(NO_WAIT_DEADLINE, middle)
            .await
            .expect("the held read finishes")
            .expect("middle read");
        assert_eq!(
            middle,
            inner.inner.blob.slice((b + 10) as usize..(b + 20) as usize)
        );
        assert_eq!(inner.inner.calls(), 2, "as many GETs as without waiting");

        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
    }

    /// A read whose run sits at the edge of a run in flight waits for that run
    /// and GETs only the rest.
    #[tokio::test]
    async fn a_read_fetches_only_what_is_not_in_flight() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = held_source(2 * b as usize, false);
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );

        let mut first = pin!(src.range(10, 10));
        let mut both = pin!(src.range(0, 2 * b));
        assert!(poll!(first.as_mut()).is_pending(), "block 0 is in flight");
        assert!(
            poll!(both.as_mut()).is_pending(),
            "the wide read waits for it"
        );
        assert_eq!(inner.inner.calls(), 1, "and GETs block 1 alone");
        inner.release();
        let (first, both) = timeout(NO_WAIT_DEADLINE, async { tokio::join!(first, both) })
            .await
            .expect("both reads finish");
        assert_eq!(first.expect("first read"), inner.inner.blob.slice(10..20));
        assert_eq!(
            both.expect("wide read"),
            inner.inner.blob.slice(0..(2 * b) as usize)
        );
        assert_eq!(inner.inner.calls(), 2, "one GET per block");

        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
    }

    /// A prefetch waits for a run a read is fetching, then fetches only the rest.
    #[tokio::test]
    async fn prefetch_waits_for_a_run_a_read_is_fetching() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = held_source(2 * b as usize, false);
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );
        let bf = src.block_file().expect("block file");

        let mut read = pin!(src.range(0, 10));
        assert!(
            poll!(read.as_mut()).is_pending(),
            "block 0 is being fetched"
        );
        let mut chunk = pin!(src.prefetch_chunk(bf, 0, 1));
        assert!(
            poll!(chunk.as_mut()).is_pending(),
            "the prefetch waits for it"
        );
        assert_eq!(inner.inner.calls(), 0, "without a GET of its own yet");

        inner.release();
        let chunk = timeout(NO_WAIT_DEADLINE, chunk)
            .await
            .expect("the prefetch finishes")
            .expect("prefetch chunk");
        assert!(chunk, "the prefetch goes on");
        assert_eq!(inner.inner.calls(), 2, "block 0 by the read, block 1 by it");
        let read = timeout(NO_WAIT_DEADLINE, read)
            .await
            .expect("the read finishes")
            .expect("read");
        assert_eq!(read, inner.inner.blob.slice(0..10));

        store.block_writes_settled().await;
        assert!(src.all_filled(0, 1));
        assert_eq!(store.stats().current_bytes, 2 * b);
        store.remove_block_entry_for_test(&uri);
    }

    /// A run a read fetched but could not cache (the cap was full) is filled by
    /// a prefetch waiting on it.
    #[tokio::test]
    async fn prefetch_fills_a_run_a_read_left_uncached() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = held_source(b as usize, false);
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );
        let bf = src.block_file().expect("block file");
        let mut taken = 0;
        while store.try_admit_write_behind(b) {
            taken += b;
        }

        let mut read = pin!(src.range(0, 10));
        assert!(
            poll!(read.as_mut()).is_pending(),
            "block 0 is being fetched"
        );
        let mut chunk = pin!(src.prefetch_chunk(bf, 0, 0));
        assert!(
            poll!(chunk.as_mut()).is_pending(),
            "the prefetch waits for it"
        );
        inner.release();
        let (read, chunk) = timeout(NO_WAIT_DEADLINE, async { tokio::join!(read, chunk) })
            .await
            .expect("both finish");
        assert_eq!(read.expect("read"), inner.inner.blob.slice(0..10));
        assert!(chunk.expect("prefetch chunk"));
        assert_eq!(
            inner.inner.calls(),
            2,
            "the read's GET, then the prefetch's"
        );

        store.block_writes_settled().await;
        assert!(src.all_filled(0, 0), "the prefetch filled it");
        store.release_write_behind(taken);
        store.remove_block_entry_for_test(&uri);
    }

    /// A GET that returns fewer bytes than asked (the object changed) is passed
    /// through but never cached, by a read or by a prefetch.
    #[tokio::test]
    async fn a_short_get_is_never_cached() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = Arc::new(ShortSource(CountingSource::new(2 * b as usize)));
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );

        let got = src.range(0, 64).await.expect("short read passes through");
        assert_eq!(got, inner.0.blob.slice(0..63));
        src.prefetch(0, 2 * b, b, 1)
            .await
            .expect("a short prefetch stops quietly");

        store.block_writes_settled().await;
        assert_eq!(src.filled_bytes_handle().load(Ordering::Acquire), 0);
        assert_eq!(store.stats().current_bytes, 0, "nothing stays charged");
        assert_eq!(store.write_behind_in_use(), 0);
        assert!(!src.has_runs_in_memory());
        store.remove_block_entry_for_test(&uri);
    }

    /// A prefetch skips a run a read already fetched and has not written yet:
    /// no second GET, and its blocks are reserved once.
    #[tokio::test]
    async fn prefetch_skips_a_run_waiting_for_its_write() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) = counting_source(dir.path(), &store, 3 * b as usize);

        let stall = src.stall_writes();
        let _ = src.range(0, 10).await.expect("pend block 0");
        // One block per chunk, all three at once, so a prefetch that ignored
        // pending runs would fetch block 0 again. Its own writes are held too,
        // so let them go once blocks 1 and 2 are fetched.
        let release = async {
            wait_until(|| inner.calls() >= 3, "the prefetch fetches blocks 1 and 2").await;
            drop(stall);
        };
        let (prefetched, ()) = tokio::join!(src.prefetch(0, 3 * b, b, 3), release);
        prefetched.expect("prefetch around the pending run");
        assert_eq!(inner.calls(), 3, "only blocks 1 and 2 are fetched again");

        store.block_writes_settled().await;
        assert_eq!(src.filled_bytes_handle().load(Ordering::Acquire), 3 * b);
        assert_eq!(store.stats().current_bytes, 3 * b);
        store.remove_block_entry_for_test(&uri);
    }

    /// With the store's cap on unwritten runs full, a miss still gets its bytes
    /// but is not cached, and its reserved budget is given back.
    #[tokio::test]
    async fn a_full_write_behind_cap_serves_uncached() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) = counting_source(dir.path(), &store, 2 * b as usize);
        let mut taken = 0;
        while store.try_admit_write_behind(b) {
            taken += b;
        }

        let want = inner.blob.slice(0..64);
        assert_eq!(src.range(0, 64).await.expect("read"), want);
        assert_eq!(src.range(0, 64).await.expect("read again"), want);
        assert_eq!(inner.calls(), 2, "nothing was cached, so both reads GET");
        assert_eq!(src.filled_bytes_handle().load(Ordering::Acquire), 0);
        assert_eq!(
            store.stats().current_bytes,
            0,
            "the reservation is given back"
        );

        store.release_write_behind(taken);
        assert_eq!(src.range(0, 64).await.expect("read with room"), want);
        store.block_writes_settled().await;
        assert_eq!(src.filled_bytes_handle().load(Ordering::Acquire), b);
        store.remove_block_entry_for_test(&uri);
    }

    /// However many runs wait, at most `BLOCK_WRITERS` writes run at once, so a
    /// slow disk can't take every blocking thread. The rest wait as tasks.
    #[tokio::test]
    async fn block_writes_run_at_most_block_writers_at_once() {
        let b = CACHE_BLOCK_BYTES;
        let runs = BLOCK_WRITERS + 2;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) = counting_source(dir.path(), &store, runs * b as usize);

        let stall = src.stall_writes();
        for run in 0..runs as u64 {
            let _ = src.range(run * b, 10).await.expect("pend one block");
        }
        assert_eq!(inner.calls(), runs);
        wait_until(
            || src.stalled_writes() == BLOCK_WRITERS,
            "every block writer is busy",
        )
        .await;
        sleep(STAYS_PUT).await;
        assert_eq!(src.stalled_writes(), BLOCK_WRITERS, "the others wait");
        assert_eq!(store.write_behind_in_use(), runs as u64 * b);

        drop(stall);
        store.block_writes_settled().await;
        assert_eq!(
            src.filled_bytes_handle().load(Ordering::Acquire),
            runs as u64 * b
        );
        store.remove_block_entry_for_test(&uri);
    }

    /// A full cap on unwritten runs doesn't stop a prefetch: it waits for its own
    /// writes, so every chunk is still filled.
    #[tokio::test]
    async fn prefetch_fills_every_chunk_with_the_cap_full() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) = counting_source(dir.path(), &store, 3 * b as usize);
        let mut taken = 0;
        while store.try_admit_write_behind(b) {
            taken += b;
        }

        src.prefetch(0, 3 * b, b, 1).await.expect("prefetch");
        assert_eq!(inner.calls(), 3);
        assert_eq!(src.filled_bytes_handle().load(Ordering::Acquire), 3 * b);
        assert_eq!(store.write_behind_in_use(), taken, "its share came back");

        store.release_write_behind(taken);
        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
    }

    /// A prefetch that finds its run queued by a read while its own GET was in
    /// flight leaves the run to that read and goes on.
    #[tokio::test]
    async fn prefetch_goes_on_past_a_run_queued_meanwhile() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = held_source(2 * b as usize, false);
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );
        let bf = src.block_file().expect("block file");

        let stall = src.stall_writes();
        let read_meanwhile = async {
            wait_until(
                || inner.held.load(Ordering::Acquire) > 0,
                "the prefetch GETs",
            )
            .await;
            let _ = src.range(0, 10).await.expect("the read queues block 0");
            inner.release();
        };
        let (chunk, ()) = timeout(NO_WAIT_DEADLINE, async {
            tokio::join!(src.prefetch_chunk(bf, 0, 0), read_meanwhile)
        })
        .await
        .expect("the prefetch doesn't wait for the read's write");
        assert!(chunk.expect("prefetch chunk"), "the prefetch goes on");
        assert_eq!(
            store.write_behind_in_use(),
            b,
            "only the read's run is queued"
        );

        drop(stall);
        store.block_writes_settled().await;
        assert_eq!(src.filled_bytes_handle().load(Ordering::Acquire), b);
        assert_eq!(store.stats().current_bytes, b, "charged once");
        store.remove_block_entry_for_test(&uri);
    }

    /// Where runs being fetched split a read's missing blocks, the blocks from
    /// the first missing one to the last are one new run, with one GET, and it
    /// is cached, even when the read starts on a run being fetched.
    #[tokio::test]
    async fn a_merged_run_next_to_runs_being_fetched_is_cached() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = held_sources(4 * b as usize, 2, false);
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );

        let mut zero = pin!(src.range(10, 10));
        let mut two = pin!(src.range(2 * b + 10, 10));
        assert!(
            poll!(zero.as_mut()).is_pending(),
            "block 0 is being fetched"
        );
        assert!(poll!(two.as_mut()).is_pending(), "block 2 is being fetched");
        let mut all = pin!(src.range(0, 4 * b));
        assert!(
            poll!(all.as_mut()).is_pending(),
            "the wide read waits for block 0"
        );
        assert_eq!(inner.inner.calls(), 1, "and GETs blocks 1 to 3 in one go");

        inner.release();
        let (zero, two, all) = timeout(NO_WAIT_DEADLINE, async { tokio::join!(zero, two, all) })
            .await
            .expect("every read finishes");
        assert_eq!(zero.expect("block 0"), inner.inner.blob.slice(10..20));
        let two_at = (2 * b + 10) as usize;
        assert_eq!(
            two.expect("block 2"),
            inner.inner.blob.slice(two_at..two_at + 10)
        );
        assert_eq!(all.expect("wide read"), inner.inner.blob.slice(..));
        assert_eq!(inner.inner.calls(), 3, "one GET per run");

        store.block_writes_settled().await;
        assert!(src.all_filled(0, 3), "every block is cached");
        assert_eq!(src.filled_bytes_handle().load(Ordering::Acquire), 4 * b);
        assert_eq!(
            store.stats().current_bytes,
            4 * b,
            "each block charged once"
        );
        assert_eq!(store.write_behind_in_use(), 0);
        assert!(!src.has_runs_in_memory());
        store.remove_block_entry_for_test(&uri);
    }

    /// A read waiting on a run that starts before its first block slices its
    /// bytes at the right offset.
    #[tokio::test]
    async fn a_waiting_read_slices_a_longer_run_at_its_offset() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = held_source(3 * b as usize, false);
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );

        let mut wide = pin!(src.range(10, 2 * b));
        let (start, len) = (b + 5, b + 100);
        let mut inside = pin!(src.range(start, len));
        assert!(
            poll!(wide.as_mut()).is_pending(),
            "blocks 0 to 2 are being fetched"
        );
        assert!(poll!(inside.as_mut()).is_pending(), "the read inside waits");
        inner.release();
        let inside = timeout(NO_WAIT_DEADLINE, inside)
            .await
            .expect("the read inside finishes")
            .expect("read inside");
        assert_eq!(
            inside,
            inner
                .inner
                .blob
                .slice(start as usize..(start + len) as usize)
        );
        assert_eq!(inner.inner.calls(), 1);

        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
    }

    /// A run the cap on unwritten runs refuses still reaches the reads waiting
    /// on it; only later reads GET it again.
    #[tokio::test]
    async fn a_run_the_cap_refuses_still_reaches_its_waiting_reads() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = held_source(b as usize, false);
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );
        let mut taken = 0;
        while store.try_admit_write_behind(b) {
            taken += b;
        }

        let mut one = pin!(src.range(0, 64));
        let mut two = pin!(src.range(10, 64));
        assert!(poll!(one.as_mut()).is_pending());
        assert!(poll!(two.as_mut()).is_pending());
        inner.release();
        let (one, two) = timeout(NO_WAIT_DEADLINE, async { tokio::join!(one, two) })
            .await
            .expect("both reads finish");
        assert_eq!(one.expect("first read"), inner.inner.blob.slice(0..64));
        assert_eq!(two.expect("second read"), inner.inner.blob.slice(10..74));
        assert_eq!(inner.inner.calls(), 1, "one GET for both");
        assert!(!src.has_runs_in_memory(), "the run was not kept");
        assert_eq!(store.stats().current_bytes, 0);
        assert_eq!(store.write_behind_in_use(), taken);

        let _ = src.range(0, 64).await.expect("a later read");
        assert_eq!(inner.inner.calls(), 2, "GETs it again");
        store.release_write_behind(taken);
        store.remove_block_entry_for_test(&uri);
    }

    /// The sync path never waits for a run being fetched: it passes the read on.
    #[tokio::test]
    async fn the_sync_path_passes_on_a_run_being_fetched() {
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = held_source(2 * CACHE_BLOCK_BYTES as usize, false);
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );

        let mut read = pin!(src.range(0, 64));
        assert!(
            poll!(read.as_mut()).is_pending(),
            "block 0 is being fetched"
        );
        assert!(src.try_get_range_sync(10, 64).is_none(), "passed on");
        assert!(src.has_runs_in_memory(), "the run is still being fetched");
        inner.release();
        let read = timeout(NO_WAIT_DEADLINE, read)
            .await
            .expect("the read finishes")
            .expect("read");
        assert_eq!(read, inner.inner.blob.slice(0..64));
        assert_eq!(
            src.try_get_range_sync(10, 64),
            Some(inner.inner.blob.slice(10..74)),
            "now from memory or the block file",
        );
        assert_eq!(inner.inner.calls(), 1);
        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
    }

    /// A run that comes back short stops the read before its other runs GET, so
    /// it costs what reading one run at a time would: that GET, then the read
    /// from the source.
    #[tokio::test]
    async fn a_short_run_stops_the_read_before_its_other_runs() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = Arc::new(ShortAt {
            inner: CountingSource::new(3 * b as usize),
            start: 0,
        });
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );
        let _ = src.range(b + 10, 10).await.expect("fill block 1");
        store.block_writes_settled().await;

        let calls = inner.inner.calls();
        let _ = src.range(0, 3 * b).await;
        assert_eq!(inner.inner.calls() - calls, 2, "run 0, then the source");
        assert!(!src.has_runs_in_memory(), "run 2 never started");
        store.block_writes_settled().await;
        assert_eq!(store.stats().current_bytes, b, "only block 1 is cached");
        store.remove_block_entry_for_test(&uri);
    }

    /// Returns one byte less than asked for a range starting at `start`, as if
    /// that part of the object were replaced by a shorter one.
    struct ShortAt {
        inner: CountingSource,
        start: u64,
    }

    #[async_trait]
    impl LazyByteSource for ShortAt {
        fn size(&self) -> u64 {
            self.inner.size()
        }

        async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
            let bytes = self.inner.range(start, len).await?;
            Ok(if start == self.start {
                bytes.slice(..bytes.len().saturating_sub(1))
            } else {
                bytes
            })
        }
    }

    /// A read stuck behind another read's GET for longer than `HEDGE_AFTER` GETs
    /// the blocks itself, so one stuck GET can't hold it.
    #[tokio::test(start_paused = true)]
    async fn a_waiting_read_hedges_a_stuck_get() {
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = held_source(2 * CACHE_BLOCK_BYTES as usize, false);
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );

        let mut stuck = pin!(src.range(0, 64));
        assert!(
            poll!(stuck.as_mut()).is_pending(),
            "the first read's GET is stuck"
        );
        let started = Instant::now();
        let hedged = src.range(10, 64).await.expect("the waiting read");
        assert_eq!(hedged, inner.inner.blob.slice(10..74));
        assert!(started.elapsed() >= HEDGE_AFTER, "it waited first");
        assert_eq!(inner.inner.calls(), 1, "then GET the block itself");

        inner.release();
        assert_eq!(
            stuck.await.expect("the first read"),
            inner.inner.blob.slice(0..64)
        );
        store.remove_block_entry_for_test(&uri);
    }

    /// A read whose wait ends under `HEDGE_AFTER` sends no GET of its own.
    #[tokio::test(start_paused = true)]
    async fn a_waiting_read_does_not_hedge_a_get_that_lands_in_time() {
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = held_source(2 * CACHE_BLOCK_BYTES as usize, false);
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );

        let mut one = pin!(src.range(0, 64));
        let mut two = pin!(src.range(10, 64));
        assert!(poll!(one.as_mut()).is_pending());
        assert!(poll!(two.as_mut()).is_pending());
        advance(HEDGE_AFTER - Duration::from_millis(1)).await;
        assert!(poll!(two.as_mut()).is_pending(), "still waiting");
        inner.release();
        let (one, two) = tokio::join!(one, two);
        assert_eq!(one.expect("first read"), inner.inner.blob.slice(0..64));
        assert_eq!(two.expect("second read"), inner.inner.blob.slice(10..74));
        assert_eq!(inner.inner.calls(), 1, "one GET for both");
        store.remove_block_entry_for_test(&uri);
    }

    /// Holds its first `hold` ranges until [`Self::release`]; `held` counts the
    /// ones holding. With `fails`, a held range fails once let go instead of
    /// reading.
    struct HeldSource {
        inner: CountingSource,
        hold: AtomicUsize,
        held: AtomicUsize,
        released: AtomicBool,
        let_go: Notify,
        fails: bool,
    }

    impl HeldSource {
        fn release(&self) {
            self.released.store(true, Ordering::Release);
            self.let_go.notify_waiters();
        }
    }

    /// A [`HeldSource`] over an `obj`-byte blob that holds its first range.
    fn held_source(obj: usize, fails: bool) -> Arc<HeldSource> {
        held_sources(obj, 1, fails)
    }

    /// A [`HeldSource`] over an `obj`-byte blob that holds its first `hold` ranges.
    fn held_sources(obj: usize, hold: usize, fails: bool) -> Arc<HeldSource> {
        Arc::new(HeldSource {
            inner: CountingSource::new(obj),
            hold: AtomicUsize::new(hold),
            held: AtomicUsize::new(0),
            released: AtomicBool::new(false),
            let_go: Notify::new(),
            fails,
        })
    }

    #[async_trait]
    impl LazyByteSource for HeldSource {
        fn size(&self) -> u64 {
            self.inner.size()
        }

        async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
            let held = self
                .hold
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_sub(1))
                .is_ok();
            if held {
                self.held.fetch_add(1, Ordering::AcqRel);
                let mut let_go = pin!(self.let_go.notified());
                let_go.as_mut().enable();
                if !self.released.load(Ordering::Acquire) {
                    timeout(NO_WAIT_DEADLINE, let_go)
                        .await
                        .expect("the held GET is let go");
                }
                if self.fails {
                    return Err(LazyByteSourceError::OutOfBounds {
                        start,
                        len,
                        size: 0,
                    });
                }
            }
            self.inner.range(start, len).await
        }
    }

    /// Returns one byte less than asked, like an object replaced by a shorter one.
    struct ShortSource(CountingSource);

    #[async_trait]
    impl LazyByteSource for ShortSource {
        fn size(&self) -> u64 {
            self.0.size()
        }

        async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
            let bytes = self.0.range(start, len).await?;
            Ok(bytes.slice(..bytes.len().saturating_sub(1)))
        }
    }
}
