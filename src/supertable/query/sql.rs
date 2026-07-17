// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! `SupertableReader::query_sql` — DataFusion SQL over a pinned supertable snapshot.
//!
//! ## Public API
//!
//! ```ignore
//! let reader = supertable.reader();
//! let batches: Vec<RecordBatch> =
//!     reader.query_sql("SELECT category, COUNT(*) FROM supertable GROUP BY category")?;
//! ```
//!
//! Sync return type: callers don't need a tokio runtime.
//! Internally the reader drives the async DataFusion plan through the same
//! sync→async bridge used by BM25 and vector search.
//!
//! ## Strategy
//!
//! At `query_sql` time we:
//!
//!   1. Use the reader's already-pinned `Arc<Manifest>`.
//!   2. Register a [`SupertableProvider`] as `supertable` in a
//!      fresh `SessionContext`.
//!   3. `ctx.sql(sql).await.collect().await`.
//!
//! The provider's `scan` does the real work — see
//! [`crate::supertable::query::provider`]. In short, it applies
//! **two tiers of pruning**: infino's [`scalar_skip`] drops
//! definitely-irrelevant *superfiles* from the pushed-down `WHERE`
//! predicates, then DataFusion's `ParquetSource` prunes *row
//! groups / pages* and pushes projection + limit into the Parquet
//! reader over the surviving superfiles. This replaces the v1
//! `MemTable` path, which eagerly decoded every row group of every
//! superfile regardless of the query.
//!
//! [`scalar_skip`]: crate::supertable::query::skip::scalar_skip
//! [`SupertableProvider`]: crate::supertable::query::provider::SupertableProvider
//!
//! ## Schema
//!
//! The supertable's *user-visible* schema (`options.scalar_schema`)
//! contains id + scalar columns + FTS columns; vector columns are
//! stored in the embedded vector blob and never exposed via SQL
//! (callers reach them through `vector_search`). The parquet body
//! of each superfile was written with this same scalar schema, so
//! round-trip shape matches without projection or rewrite.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use arrow::{compute::cast, record_batch::RecordBatch};
use arrow_array::{Array, Decimal128Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion::{error::DataFusionError, execution::context::SessionContext, prelude::Expr};

use crate::{
    memory::budgeted_session_context,
    supertable::{
        error::QueryError,
        handle::{Supertable, SupertableReader},
        options::SupertableOptions,
        query::{
            covered_agg::CoveredAggregateRewrite,
            exec::{
                fts_exec::register_bm25, hybrid_exec::register_hybrid_search,
                match_exec::register_match, vector_exec::register_vector_search,
            },
            provider::{SupertableProvider, TABLE_NAME, view_string_schema},
        },
    },
};

/// Per-table SQL schemas, built once (`build_sql_schemas`) and cached on the
/// handle instead of recomputed per query. Cheap to clone (fields are `Arc`s).
///
/// - `scalar`: id + scalar + FTS columns, no vectors. What the search TVFs bind to.
/// - `scan`: `scalar` with non-FTS strings viewed as `Utf8View`
///   (`view_string_schema`). What the provider plans against.
/// - `declared`: name -> declared string type, for `cast_back_views` at the
///   result boundary.
#[derive(Clone)]
pub(crate) struct SqlSchemas {
    scalar: SchemaRef,
    scan: SchemaRef,
    declared: Arc<HashMap<String, DataType>>,
}

impl SqlSchemas {
    /// Plain scalar schema (id + scalar + FTS, no vectors) the TVFs bind to.
    pub(crate) fn scalar(&self) -> &SchemaRef {
        &self.scalar
    }

    /// String-viewed schema the provider plans against.
    pub(crate) fn scan(&self) -> &SchemaRef {
        &self.scan
    }

    /// Declared string types, for restoring result columns at the boundary.
    pub(crate) fn declared(&self) -> &Arc<HashMap<String, DataType>> {
        &self.declared
    }
}

/// Build the [`SqlSchemas`] for `options`. Called once per table; the result is
/// cached on the handle. This is the one place that walks the full column set,
/// so a wide (thousands of columns) table pays it once, not per query.
pub(crate) fn build_sql_schemas(options: &SupertableOptions) -> SqlSchemas {
    let scalar = options.scalar_schema();
    let fts: HashSet<&str> = options
        .fts_columns
        .iter()
        .map(|c| c.column.as_str())
        .collect();
    let scan = view_string_schema(&scalar, &fts);
    let declared = Arc::new(declared_string_types(&scalar));
    SqlSchemas {
        scalar,
        scan,
        declared,
    }
}

/// Classify a SQL execution error: budget exhaustion -> [`QueryError::OverBudget`]
/// (the catalog surfaces it as `InfinoError::OverBudget`), else an execute error.
fn exec_query_error(e: DataFusionError) -> QueryError {
    match e {
        DataFusionError::ResourcesExhausted(msg) => QueryError::OverBudget(msg),
        other => QueryError::Execute(other.to_string()),
    }
}

/// Name -> declared type for the string columns of the queried table(s). These
/// are the only columns cast-back can restore to a user-declared type; non-string
/// columns are irrelevant and skipped.
pub(crate) fn declared_string_types(schema: &Schema) -> HashMap<String, DataType> {
    schema
        .fields()
        .iter()
        .filter(|f| {
            matches!(
                f.data_type(),
                DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
            )
        })
        .map(|f| (f.name().clone(), f.data_type().clone()))
        .collect()
}

/// The type a result field should carry after cast-back, or `None` to leave it.
///
/// Only `Utf8View` result columns are touched (the scan views strings for
/// speed). A view column is matched by name to `declared`: a column the user
/// declared `Utf8View` is kept, one declared `Utf8`/`LargeUtf8` is restored to
/// exactly that type. A view column with no declared match (an aggregate,
/// alias, or other computed value) has no user-declared type, so it defaults to
/// `LargeUtf8`.
fn cast_back_type(field: &Field, declared: &HashMap<String, DataType>) -> Option<DataType> {
    if field.data_type() != &DataType::Utf8View {
        return None;
    }
    let target = declared
        .get(field.name())
        .cloned()
        .unwrap_or(DataType::LargeUtf8);

    (&target != field.data_type()).then_some(target)
}

/// Result schema after cast-back: each `Utf8View` field rewritten to its
/// [`cast_back_type`], others unchanged. Types a zero-row result to match what
/// a populated one returns.
pub(crate) fn cast_back_schema(result: &Schema, declared: &HashMap<String, DataType>) -> SchemaRef {
    let fields = result
        .fields()
        .iter()
        .map(|f| match cast_back_type(f, declared) {
            // clone + retype: keeps nullability and metadata (`Field::new` drops it).
            Some(t) => Arc::new(f.as_ref().clone().with_data_type(t)),
            None => Arc::clone(f),
        })
        .collect::<Vec<_>>();

    Arc::new(Schema::new_with_metadata(fields, result.metadata().clone()))
}

/// Undo the scan's string viewing at the result boundary: cast each `Utf8View`
/// column to its [`cast_back_type`], leave the rest. Keeps a user-declared
/// `Utf8View` intact, restores columns we viewed to their declared type, and
/// defaults computed view columns to `LargeUtf8`. Results are bounded, so the
/// cast is cheap; no view column means no work.
pub(crate) fn cast_back_views(
    batches: Vec<RecordBatch>,
    declared: &HashMap<String, DataType>,
) -> Result<Vec<RecordBatch>, QueryError> {
    let Some(first) = batches.first() else {
        return Ok(batches);
    };

    if !first
        .schema()
        .fields()
        .iter()
        .any(|f| f.data_type() == &DataType::Utf8View)
    {
        return Ok(batches);
    }

    let target = cast_back_schema(first.schema().as_ref(), declared);
    batches
        .into_iter()
        .map(|b| {
            let cols = b
                .columns()
                .iter()
                .zip(target.fields())
                .map(|(c, tf)| {
                    if c.data_type() == tf.data_type() {
                        Ok(Arc::clone(c))
                    } else {
                        cast(c, tf.data_type())
                            .map_err(|e| QueryError::Execute(format!("utf8view cast-back: {e}")))
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;

            RecordBatch::try_new(Arc::clone(&target), cols)
                .map_err(|e| QueryError::Execute(format!("utf8view cast-back rebuild: {e}")))
        })
        .collect()
}

impl SupertableReader {
    /// Run a SQL query against this reader's pinned snapshot.
    ///
    /// The snapshot is captured at `query_sql` entry — concurrent
    /// commits don't affect the in-flight query. Returns the
    /// concatenated `Vec<RecordBatch>` from
    /// `DataFrame::collect`.
    ///
    /// The SQL must reference the table as `supertable`. The
    /// available columns are id + scalar + FTS columns; vector
    /// columns are not exposed (use `vector_search` instead).
    ///
    /// Sync API. The first call allocates a tokio Runtime
    /// (single worker thread) cached on the `SupertableInner`;
    /// subsequent calls reuse it.
    // Single-table SQL — off the public surface; catalog-level SQL is the
    // public entry point. Reachable from tests/benches via `test-helpers`.
    #[cfg(any(test, feature = "test-helpers"))]
    #[cfg_attr(
        feature = "detailed-tracing",
        tracing::instrument(skip_all, fields(sql = sql))
    )]
    pub fn query_sql(&self, sql: &str) -> Result<Vec<RecordBatch>, QueryError> {
        // Read-consistency was applied when `Supertable::reader()` created
        // this pinned reader. SQL therefore observes the same snapshot as
        // `bm25_search` and `vector_search` on this handle.

        // Build (or reuse the cached) SessionContext for the pinned
        // snapshot — the pushdown-aware SupertableProvider plus the
        // search TVFs. See [`SupertableReader::sql_session_context`].
        let ctx = self.sql_session_context()?;

        // Cached declared string types: cast-back restores each result column
        // to what the user declared (and keeps a declared `Utf8View`).
        let schemas = self.sql_schemas();

        let sql = sql.to_owned();
        let drive = async move {
            let df = ctx
                .sql(&sql)
                .await
                .map_err(|e| QueryError::Plan(e.to_string()))?;

            let batches = df.collect().await.map_err(exec_query_error)?;
            cast_back_views(batches, schemas.declared())
        };

        // Drive through the shared sync→async bridge: ambient
        // runtime → block_in_place on the ambient handle; otherwise
        // the lazily-built owned query_runtime. See
        // [`SupertableReader::block_on`].
        self.block_on(drive)
    }

    /// Build (or reuse the cached) [`SessionContext`] for the
    /// current pinned manifest snapshot: the pushdown-aware
    /// [`SupertableProvider`] registered as `supertable`, plus the
    /// vector / BM25 / hybrid search TVFs.
    ///
    /// The cache keys on the manifest `Arc` — commits publish a new
    /// `Arc`, so any committed state since the last call forces a
    /// rebuild. A hit skips the ~1.5 ms `SessionContext::new()` +
    /// `register_*` setup. Shared by [`query_sql`](Self::query_sql)
    /// (SQL string) and [`scan_ids_matching`](Self::scan_ids_matching)
    /// (programmatic `Expr`), so mutation id-capture gets the same
    /// superfile-skip + row-group/page pruning + lazy tombstone
    /// filtering the read path uses.
    ///
    /// Freshness policy is applied when the reader is created by
    /// [`Supertable::reader`](crate::supertable::handle::Supertable::reader).
    #[cfg_attr(feature = "detailed-tracing", tracing::instrument(skip_all))]
    fn sql_session_context(&self) -> Result<SessionContext, QueryError> {
        // This reader already pins the snapshot; clone is a handful of
        // Arc refcount bumps.
        let reader = Arc::new(self.clone());
        let manifest = Arc::clone(reader.manifest());

        let mut guard = self
            .sql_session_cache()
            .lock()
            .expect("sql_session_cache mutex poisoned");
        if let Some((cached, ctx)) = &*guard
            && Arc::ptr_eq(cached, &manifest)
        {
            return Ok(ctx.clone());
        }

        let store = Arc::clone(&self.options().store);
        let disk_cache = self.options().disk_cache.as_ref().map(Arc::clone);
        // Cached per-table schemas: the provider scans the string-viewed `scan`
        // schema; the TVFs bind to the plain `scalar` schema.
        let schemas = self.sql_schemas();
        let provider = SupertableProvider::new(
            schemas.scan().clone(),
            Arc::clone(&manifest),
            store,
            disk_cache,
            reader.tombstone_cache.clone(),
        );

        // Gate SQL heap on the connection budget (shared across contexts, so
        // this reader's SQL counts against the same ceiling as the rest).
        let ctx = budgeted_session_context(&self.options().connection_memory_budget)
            .map_err(|e| QueryError::Plan(e.to_string()))?;

        // Covered/residual aggregate rewrite: filter-aligned range
        // aggregates answer covered segments from manifest statistics
        // and scan only the boundary segments. Appended after the
        // built-in rules so it sees pushed-down, normalized plans.
        ctx.add_optimizer_rule(Arc::new(CoveredAggregateRewrite));
        ctx.register_table(TABLE_NAME, Arc::new(provider))
            .map_err(|e| QueryError::Plan(e.to_string()))?;

        // Search TVFs (vector kNN, BM25 FTS, hybrid RRF) bound to
        // the pinned snapshot. They lower to custom `ExecutionPlan`
        // nodes that call the async kernels inside `execute()`.
        register_vector_search(&ctx, Arc::clone(&reader), schemas.scalar().clone());
        register_bm25(&ctx, Arc::clone(&reader), schemas.scalar().clone());
        // Unranked token / exact match TVFs (siblings of bm25_search).
        register_match(&ctx, Arc::clone(&reader), schemas.scalar().clone());
        register_hybrid_search(&ctx, Arc::clone(&reader), schemas.scalar().clone());

        *guard = Some((Arc::clone(&manifest), ctx.clone()));

        Ok(ctx)
    }

    /// Resolve a predicate to the matching `_id` values. Used by
    /// the writer's `delete()` / `update()` entry points to
    /// capture the target-id set at call time (step 0a in the
    /// update / delete pipeline).
    ///
    /// Runs through the same pushdown-aware [`SupertableProvider`]
    /// as `query_sql` (via [`sql_session_context`](Self::sql_session_context)):
    /// `expr` is applied as a `DataFrame::filter` and the result
    /// projected to just `_id`. Superfile skip, row-group / page
    /// pruning, and lazy tombstone filtering all apply, so a
    /// large-table delete/update predicate never materializes every
    /// superfile into memory.
    ///
    /// Note: the resolution is against the **current** manifest
    /// snapshot, exactly like a contemporaneous `query_sql` would
    /// see. Rows that newly match `expr` between this call and
    /// the eventual `commit()` are NOT in the returned set —
    /// captured-at-call semantics match SQL `UPDATE WHERE` /
    /// `DELETE WHERE`.
    pub(crate) fn scan_ids_matching(&self, expr: Expr) -> Result<Vec<i128>, QueryError> {
        // Resolve against this reader's pinned snapshot. Callers that need
        // current-state semantics create a fresh reader immediately before
        // invoking this helper.
        let ctx = self.sql_session_context()?;
        let id_column = self.options().id_column.clone();

        let drive = async move {
            let df = ctx
                .table(TABLE_NAME)
                .await
                .map_err(|e| QueryError::Plan(e.to_string()))?
                .filter(expr)
                .map_err(|e| QueryError::Plan(e.to_string()))?
                .select_columns(&[id_column.as_str()])
                .map_err(|e| QueryError::Plan(e.to_string()))?;
            let batches = df.collect().await.map_err(exec_query_error)?;
            extract_id_column(&batches)
        };

        self.block_on(drive)
    }
}

impl Supertable {
    /// Register this supertable's pushdown-aware provider into `ctx`
    /// under `name`, applying the read-consistency policy first. The
    /// catalog's multi-table [`Connection::query_sql`] calls this once
    /// per referenced table. Returns the pinned reader so the caller can
    /// later wire the same snapshot into search TVFs.
    ///
    /// [`Connection::query_sql`]: crate::Connection::query_sql
    pub(crate) fn register_into(
        &self,
        ctx: &SessionContext,
        name: &str,
    ) -> Result<Arc<SupertableReader>, QueryError> {
        self.ensure_fresh();
        let reader = Arc::new(self.reader());
        let manifest = Arc::clone(reader.manifest());
        let store = Arc::clone(&self.options().store);
        let disk_cache = self.options().disk_cache.as_ref().map(Arc::clone);
        // Provider scans the cached string-viewed schema.
        let provider = SupertableProvider::new(
            self.sql_schemas().scan().clone(),
            manifest,
            store,
            disk_cache,
            reader.tombstone_cache.clone(),
        );
        ctx.register_table(name, Arc::new(provider))
            .map_err(|e| QueryError::Plan(e.to_string()))?;
        Ok(reader)
    }
}

/// Drain `_id`-only batches into a `Vec<i128>`. The supertable's
/// `_id` is a Decimal128(38, 0) column; we read the raw 128-bit
/// integer value directly.
fn extract_id_column(batches: &[RecordBatch]) -> Result<Vec<i128>, QueryError> {
    let mut out: Vec<i128> = Vec::new();
    for batch in batches {
        if batch.num_columns() != 1 {
            return Err(QueryError::Plan(format!(
                "scan_ids_matching: expected 1-column batch, got {}",
                batch.num_columns()
            )));
        }
        let col = batch.column(0);
        let arr = col
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .ok_or_else(|| {
                QueryError::Plan("scan_ids_matching: _id column not Decimal128".into())
            })?;
        for i in 0..arr.len() {
            if arr.is_null(i) {
                continue;
            }
            out.push(arr.value(i));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use arrow_array::{
        Array, Decimal128Array, FixedSizeListArray, Float32Array, Int64Array, LargeStringArray,
        RecordBatch, StringArray, StringViewArray,
    };
    use arrow_schema::{DataType, Field, Schema};

    use crate::{
        memory::ConnectionMemoryBudget,
        storage::{LocalFsStorageProvider, StorageProvider},
        superfile::{
            builder::{FtsConfig, VectorConfig},
            vector::{distance::Metric, rerank_codec::RerankCodec},
        },
        supertable::{
            Supertable, SupertableOptions,
            error::QueryError,
            query::sql::{
                build_sql_schemas, cast_back_schema, cast_back_views, declared_string_types,
            },
        },
        test_helpers::default_tokenizer as tok,
    };

    /// Schema with id + scalar + FTS column. No vector; query_sql
    /// is scalar-only by design.
    fn schema_id_cat_title() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("category", DataType::LargeUtf8, false),
            Field::new("title", DataType::LargeUtf8, false),
        ]))
    }

    fn options_id_cat_title() -> SupertableOptions {
        // Single-threaded writer pool so each commit produces
        // exactly one superfile — keeps assertions on per-superfile
        // counts deterministic.
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("rayon pool"),
        );
        SupertableOptions::new(
            schema_id_cat_title(),
            vec![FtsConfig {
                column: "title".into(),
            }],
            vec![],
            Some(tok()),
        )
        .expect("valid options")
        .with_writer_pool(pool)
    }

    // Ingest `batch` on a measured supertable, then return a second handle over
    // the same durable storage under a 0-byte gate. Ingest is gated by the
    // budget too, so a query-gating test can't reuse one tiny-budget handle for
    // both; this does the setup on a measured handle and hands back the gated
    // reader. The returned `TempDir` guard must be held: dropping it deletes the
    // store the reader is still reading through.
    fn zero_gate_reader_after_ingest(batch: &RecordBatch) -> (tempfile::TempDir, Supertable) {
        let dir = tempfile::tempdir().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("localfs"));

        let ingest = Supertable::create(options_id_cat_title().with_storage(Arc::clone(&storage)))
            .expect("create");
        let mut w = ingest.writer().expect("writer");
        w.append(batch).expect("append");
        w.commit().expect("commit");

        let mut qopts = options_id_cat_title().with_storage(storage);
        qopts.connection_memory_budget = ConnectionMemoryBudget::with_limit(1);
        (dir, Supertable::open(qopts).expect("open"))
    }

    /// Build a small categorical batch — start id sequence at
    /// `start`, plant `cats[i] / titles[i]` per row.
    fn build_cat_batch(_start: u64, cats: &[&str], titles: &[&str]) -> RecordBatch {
        assert_eq!(cats.len(), titles.len());
        let cat_arr = LargeStringArray::from(cats.to_vec());
        let title_arr = LargeStringArray::from(titles.to_vec());
        RecordBatch::try_new(
            schema_id_cat_title(),
            vec![Arc::new(cat_arr), Arc::new(title_arr)],
        )
        .expect("build batch")
    }

    /// A single-superfile table seeded with one committed batch of
    /// `cats`/`titles`. Collapses the create + append + commit boilerplate the
    /// string-view tests share.
    fn seeded(cats: &[&str], titles: &[&str]) -> Supertable {
        let st = Supertable::create(options_id_cat_title()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_cat_batch(0, cats, titles)).expect("append");
        w.commit().expect("commit");
        st
    }

    /// Convenience: run a query and pull a single `Int64` aggregate
    /// value from cell (0,0).
    fn run_count(st: &Supertable, sql: &str) -> i64 {
        let batches = st.reader().query_sql(sql).expect("query_sql ok");
        assert!(!batches.is_empty(), "expected at least one result batch");
        let n = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("count column is Int64");
        n.value(0)
    }

    #[test]
    fn query_sql_count_star_returns_zero_on_empty_supertable() {
        let st = Supertable::create(options_id_cat_title()).expect("create");
        let n = run_count(&st, "SELECT COUNT(*) FROM supertable");
        assert_eq!(n, 0);
    }

    #[test]
    fn query_sql_count_star_returns_total_doc_count() {
        let st = Supertable::create(options_id_cat_title()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_cat_batch(
            0,
            &["rust", "rust", "python"],
            &["a", "b", "c"],
        ))
        .expect("append");
        w.commit().expect("commit");

        let n = run_count(&st, "SELECT COUNT(*) FROM supertable");
        assert_eq!(n, 3);
    }

    /// Regression test for the cold-reopen consumer leak. Running
    /// `query_sql` builds and caches a `SessionContext` on the
    /// `SupertableInner`, and that context registers the search TVFs.
    /// When the TVFs held a strong `Arc<SupertableReader>` (which holds
    /// the `Arc<SupertableInner>`), the chain
    /// `inner -> cached SessionContext -> TVF -> reader -> inner` formed a
    /// reference cycle that pinned the whole consumer — every fresh
    /// consumer reopen (the cold query path) leaked one, OOMing at scale.
    /// With the TVFs holding a `WeakReader`, dropping the last external
    /// handle releases the inner; a `Weak` to it must fail to upgrade.
    #[test]
    fn query_sql_session_cache_does_not_leak_consumer() {
        let weak = {
            let st = Supertable::create(options_id_cat_title()).expect("create");
            let mut w = st.writer().expect("writer");
            w.append(&build_cat_batch(0, &["rust"], &["a"]))
                .expect("append");
            w.commit().expect("commit");

            // Populate the cached SessionContext (registers the TVFs).
            assert_eq!(run_count(&st, "SELECT COUNT(*) FROM supertable"), 1);

            let weak = Arc::downgrade(st.inner());
            drop(w);
            drop(st);
            weak
        };

        assert!(
            weak.upgrade().is_none(),
            "SQL session cache leaked the consumer — the \
             inner -> SessionContext -> TVF -> reader -> inner cycle was not broken",
        );
    }

    #[test]
    fn query_sql_filter_predicate_applied_above_mem_table() {
        let st = Supertable::create(options_id_cat_title()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_cat_batch(
            0,
            &["rust", "rust", "python", "rust", "go"],
            &["a", "b", "c", "d", "e"],
        ))
        .expect("append");
        w.commit().expect("commit");

        let n = run_count(
            &st,
            "SELECT COUNT(*) FROM supertable WHERE category = 'rust'",
        );
        assert_eq!(n, 3);
    }

    #[test]
    fn query_sql_group_by_over_budget_is_refused() {
        // The reader path (second production ctx site) is gated too: a 0-byte
        // gate refuses the aggregate and surfaces as QueryError::OverBudget.
        let (_dir, st) = zero_gate_reader_after_ingest(&build_cat_batch(
            0,
            &["rust", "python", "rust"],
            &["a", "b", "c"],
        ));

        let err = st
            .reader()
            .query_sql("SELECT category, COUNT(*) FROM supertable GROUP BY category")
            .expect_err("0-byte gate refuses the aggregate");

        assert!(matches!(err, QueryError::OverBudget(_)), "got {err:?}");
    }

    #[test]
    fn query_sql_streaming_scan_is_not_refused_under_a_zero_gate() {
        // A projection streams (no buffering), so it runs even at a 0-byte gate:
        // the budget bounds sort/aggregate/join, not scans.
        let (_dir, st) =
            zero_gate_reader_after_ingest(&build_cat_batch(0, &["rust", "python"], &["a", "b"]));

        let rows: usize = st
            .reader()
            .query_sql("SELECT title FROM supertable")
            .expect("a streaming scan is not gated")
            .iter()
            .map(|b| b.num_rows())
            .sum();

        assert_eq!(rows, 2);
    }

    #[test]
    fn query_sql_group_by_returns_correct_per_category_counts() {
        let st = Supertable::create(options_id_cat_title()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_cat_batch(
            0,
            &["rust", "rust", "python", "rust", "python", "go"],
            &["a", "b", "c", "d", "e", "f"],
        ))
        .expect("append");
        w.commit().expect("commit");

        let batches = st
            .reader()
            .query_sql(
                "SELECT category, COUNT(*) AS n FROM supertable \
                 GROUP BY category ORDER BY category",
            )
            .expect("group-by query");
        assert_eq!(batches.len(), 1);

        let cat_col = batches[0].column(0);
        let counts = batches[0]
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("count is Int64");
        // DataFusion may materialize the GROUP BY key as Utf8,
        // LargeUtf8, or StringView depending on hash-aggregate
        // type promotion; accept all three.
        let extract = |i: usize| -> String {
            if let Some(a) = cat_col.as_any().downcast_ref::<LargeStringArray>() {
                a.value(i).to_string()
            } else if let Some(a) = cat_col.as_any().downcast_ref::<StringArray>() {
                a.value(i).to_string()
            } else if let Some(a) = cat_col.as_any().downcast_ref::<StringViewArray>() {
                a.value(i).to_string()
            } else {
                panic!("unexpected category column type: {:?}", cat_col.data_type())
            }
        };
        let mut got: Vec<(String, i64)> = (0..cat_col.len())
            .map(|i| (extract(i), counts.value(i)))
            .collect();
        got.sort();
        assert_eq!(
            got,
            vec![
                ("go".to_string(), 1),
                ("python".to_string(), 2),
                ("rust".to_string(), 3),
            ]
        );
    }

    // ---- Utf8View scan / cast-back ----------------------------------------

    /// The scan runs strings as `Utf8View`; the result is cast back to the
    /// declared type so no view leaks to a caller. Gate for the common case: a
    /// GROUP BY key on a `LargeUtf8` column comes back `LargeUtf8`, not a view.
    #[test]
    fn query_sql_string_group_by_key_is_large_utf8_not_view() {
        let st = seeded(&["rust", "go", "rust"], &["a", "b", "c"]);

        let batches = st
            .reader()
            .query_sql("SELECT category FROM supertable GROUP BY category")
            .expect("group-by");
        let col = batches[0].column(0);
        assert_eq!(
            col.data_type(),
            &DataType::LargeUtf8,
            "public result must be LargeUtf8, not Utf8View"
        );
        assert!(
            col.as_any().downcast_ref::<LargeStringArray>().is_some(),
            "category should downcast to LargeStringArray"
        );
        assert!(
            col.as_any().downcast_ref::<StringViewArray>().is_none(),
            "Utf8View must not leak to the caller"
        );
    }

    /// A projected + `ORDER BY` string column returns `LargeUtf8` and the
    /// values are correctly sorted (the view compare ran during the sort).
    #[test]
    fn query_sql_ordered_string_projection_is_large_utf8_and_sorted() {
        let st = seeded(&["rust", "go", "python"], &["a", "b", "c"]);
        let batches = st
            .reader()
            .query_sql("SELECT category FROM supertable ORDER BY category")
            .expect("order-by");
        let col = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("category is LargeUtf8 after cast-back");
        let got: Vec<&str> = (0..col.len()).map(|i| col.value(i)).collect();
        assert_eq!(got, vec!["go", "python", "rust"]);
    }

    /// Grouped `MIN(string)` aggregates on the view and returns `LargeUtf8`
    /// after cast-back, with correct per-group minima.
    #[test]
    fn query_sql_grouped_min_string_is_large_utf8() {
        let st = seeded(&["rust", "rust", "go", "go"], &["b", "a", "d", "c"]);
        let batches = st
            .reader()
            .query_sql(
                "SELECT category, MIN(title) AS m FROM supertable \
                 GROUP BY category ORDER BY category",
            )
            .expect("grouped min");
        let cat = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("category is LargeUtf8 after cast-back");
        let m = batches[0]
            .column(1)
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("MIN(title) is LargeUtf8 after cast-back");
        let got: Vec<(&str, &str)> = (0..cat.len()).map(|i| (cat.value(i), m.value(i))).collect();
        assert_eq!(got, vec![("go", "c"), ("rust", "a")]);
    }

    /// KNOWN LIMITATION: an *ungrouped* `MIN`/`MAX(string)` over a `Utf8View`
    /// column trips a DataFusion 53.1 `ProjectionPushdown` bug (`Schema
    /// mismatch: Utf8View vs LargeUtf8`) before the cast-back can run. Grouped
    /// MIN/MAX (above) is fine. Re-check on a DataFusion upgrade; if fixed
    /// there, un-ignore this.
    #[test]
    #[ignore = "DataFusion 53.1 ProjectionPushdown bug on ungrouped MIN/MAX over Utf8View"]
    fn query_sql_ungrouped_min_string_datafusion53_projection_pushdown_bug() {
        let st = seeded(&["rust", "go", "python"], &["a", "b", "c"]);
        let batches = st
            .reader()
            .query_sql("SELECT MIN(category) AS m FROM supertable")
            .expect("ungrouped min (fails on DF 53.1 with the view)");
        let col = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("MIN(string) is LargeUtf8 after cast-back");
        assert_eq!(col.value(0), "go");
    }

    /// A declared map covering the three string cases the cast-back must honor.
    fn declared(pairs: &[(&str, DataType)]) -> HashMap<String, DataType> {
        pairs
            .iter()
            .map(|(n, t)| (n.to_string(), t.clone()))
            .collect()
    }

    /// Unit: cast-back restores each `Utf8View` result column to its declared
    /// type. A declared `Utf8View` is kept, a declared `Utf8`/`LargeUtf8` is
    /// restored to exactly that (no widening), a column with no declared match
    /// (computed) defaults to `LargeUtf8`, and non-string columns are untouched.
    #[test]
    fn cast_back_views_restores_each_column_to_its_declared_type() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("large", DataType::Utf8View, false), // declared LargeUtf8, viewed
            Field::new("small", DataType::Utf8View, false), // declared Utf8, viewed
            Field::new("keep", DataType::Utf8View, false),  // declared Utf8View, keep
            Field::new("agg", DataType::Utf8View, false),   // computed, no declared type
            Field::new("n", DataType::Int64, false),        // non-string
        ]));
        let str_col = || Arc::new(StringViewArray::from(vec!["a", "b"]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                str_col(),
                str_col(),
                str_col(),
                str_col(),
                Arc::new(Int64Array::from(vec![1, 2])),
            ],
        )
        .expect("batch");

        let declared = declared(&[
            ("large", DataType::LargeUtf8),
            ("small", DataType::Utf8),
            ("keep", DataType::Utf8View),
        ]);
        let out = cast_back_views(vec![batch], &declared).expect("cast ok");
        let s = out[0].schema();
        assert_eq!(
            s.field(0).data_type(),
            &DataType::LargeUtf8,
            "declared LargeUtf8 restored"
        );
        assert_eq!(
            s.field(1).data_type(),
            &DataType::Utf8,
            "declared Utf8 restored, no widening"
        );
        assert_eq!(
            s.field(2).data_type(),
            &DataType::Utf8View,
            "user-declared view kept"
        );
        assert_eq!(
            s.field(3).data_type(),
            &DataType::LargeUtf8,
            "computed view defaults to LargeUtf8"
        );
        assert_eq!(
            s.field(4).data_type(),
            &DataType::Int64,
            "non-string untouched"
        );
    }

    /// Unit: nulls survive the cast-back intact.
    #[test]
    fn cast_back_views_preserves_nulls() {
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8View, true)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringViewArray::from(vec![
                Some("a"),
                None,
                Some("c"),
            ]))],
        )
        .expect("batch");
        let out =
            cast_back_views(vec![batch], &declared(&[("s", DataType::LargeUtf8)])).expect("cast");
        let s = out[0]
            .column(0)
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("view -> LargeUtf8");
        assert!(!s.is_null(0));
        assert!(s.is_null(1), "null preserved through cast-back");
        assert_eq!(s.value(2), "c");
    }

    /// Unit: a result with no view column takes the fast path unchanged.
    #[test]
    fn cast_back_views_without_views_is_unchanged() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "c",
            DataType::LargeUtf8,
            false,
        )]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(LargeStringArray::from(vec!["x"]))])
            .expect("batch");
        let out =
            cast_back_views(vec![batch], &declared(&[("c", DataType::LargeUtf8)])).expect("ok");
        assert_eq!(out[0].column(0).data_type(), &DataType::LargeUtf8);
    }

    /// Unit: `cast_back_schema` yields the same per-column types the data cast
    /// produces, and preserves nullability + metadata.
    #[test]
    fn cast_back_schema_matches_the_data_cast() {
        let mut md = HashMap::new();
        md.insert("k".to_string(), "v".to_string());
        let result = Schema::new_with_metadata(
            vec![
                Field::new("keep", DataType::Utf8View, true),
                Field::new("large", DataType::Utf8View, false),
                Field::new("n", DataType::Int64, false),
            ],
            md,
        );
        let out = cast_back_schema(
            &result,
            &declared(&[("keep", DataType::Utf8View), ("large", DataType::LargeUtf8)]),
        );
        assert_eq!(
            out.field(0).data_type(),
            &DataType::Utf8View,
            "declared view kept"
        );
        assert!(out.field(0).is_nullable(), "nullability preserved");
        assert_eq!(out.field(1).data_type(), &DataType::LargeUtf8);
        assert_eq!(out.field(2).data_type(), &DataType::Int64);
        assert_eq!(
            out.metadata().get("k").map(String::as_str),
            Some("v"),
            "metadata preserved"
        );
    }

    /// Unit: `build_sql_schemas` views the scan schema (non-FTS strings ->
    /// `Utf8View`, FTS kept), keeps the plain `scalar`, and records declared
    /// types. This is the walk done once per table.
    #[test]
    fn build_sql_schemas_views_scan_keeps_scalar_and_declares() {
        let s = build_sql_schemas(&options_id_cat_title());
        // scan: `category` (non-FTS string) viewed; `title` (FTS) kept.
        assert_eq!(
            s.scan()
                .field_with_name("category")
                .expect("category")
                .data_type(),
            &DataType::Utf8View,
        );
        assert_eq!(
            s.scan()
                .field_with_name("title")
                .expect("title")
                .data_type(),
            &DataType::LargeUtf8,
            "FTS column stays LargeUtf8 in the scan schema",
        );
        // scalar: no viewing.
        assert_eq!(
            s.scalar()
                .field_with_name("category")
                .expect("category")
                .data_type(),
            &DataType::LargeUtf8,
        );
        // declared: name -> declared type.
        assert_eq!(s.declared().get("category"), Some(&DataType::LargeUtf8));
        assert_eq!(s.declared().get("title"), Some(&DataType::LargeUtf8));
    }

    /// The per-table schemas are built once and memoized on the handle, not
    /// rebuilt per query (the whole point of the cache for wide tables).
    #[test]
    fn sql_schemas_is_memoized_across_calls() {
        let st = Supertable::create(options_id_cat_title()).expect("create");
        let a = st.sql_schemas();
        let b = st.sql_schemas();
        assert!(
            Arc::ptr_eq(&a, &b),
            "sql_schemas must be cached (same Arc), not recomputed per call",
        );
    }

    /// Unit: `declared_string_types` collects only string columns.
    #[test]
    fn declared_string_types_collects_string_columns_only() {
        let schema = Schema::new(vec![
            Field::new("a", DataType::LargeUtf8, false),
            Field::new("b", DataType::Utf8, false),
            Field::new("c", DataType::Utf8View, false),
            Field::new("n", DataType::Int64, false),
        ]);
        let m = declared_string_types(&schema);
        assert_eq!(m.len(), 3, "only the three string columns");
        assert_eq!(m.get("a"), Some(&DataType::LargeUtf8));
        assert_eq!(m.get("b"), Some(&DataType::Utf8));
        assert_eq!(m.get("c"), Some(&DataType::Utf8View));
        assert!(!m.contains_key("n"), "non-string skipped");
    }

    /// A column the user explicitly declared `Utf8View` must come back
    /// `Utf8View`: cast-back restores declared types, it does not force
    /// `LargeUtf8` on a type the user chose.
    #[test]
    fn query_sql_keeps_user_declared_utf8view_column() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("category", DataType::Utf8View, false), // user declares a view
            Field::new("title", DataType::LargeUtf8, false),   // FTS column must be LargeUtf8
        ]));
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("rayon pool"),
        );
        let opts = SupertableOptions::new(
            Arc::clone(&schema),
            vec![FtsConfig {
                column: "title".into(),
            }],
            vec![],
            Some(tok()),
        )
        .expect("valid options")
        .with_writer_pool(pool);

        let st = Supertable::create(opts).expect("create");
        let mut w = st.writer().expect("writer");
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringViewArray::from(vec!["rust", "go", "rust"])),
                Arc::new(LargeStringArray::from(vec!["a", "b", "c"])),
            ],
        )
        .expect("batch");
        w.append(&batch).expect("append");
        w.commit().expect("commit");

        let batches = st
            .reader()
            .query_sql("SELECT category FROM supertable GROUP BY category")
            .expect("group-by");
        assert_eq!(
            batches[0].column(0).data_type(),
            &DataType::Utf8View,
            "a user-declared Utf8View column must not be rewritten by cast-back"
        );
    }

    /// Alias on a viewed string column: the aliased output name has no declared
    /// type, so it defaults to `LargeUtf8`; values stay correct.
    #[test]
    fn query_sql_aliased_string_column_is_large_utf8() {
        let st = seeded(&["rust", "go", "rust"], &["a", "b", "c"]);

        let batches = st
            .reader()
            .query_sql("SELECT category AS c FROM supertable GROUP BY c ORDER BY c")
            .expect("alias");
        let col = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("aliased column is LargeUtf8");
        let got: Vec<&str> = (0..col.len()).map(|i| col.value(i)).collect();
        assert_eq!(got, vec!["go", "rust"]);
    }

    /// String column projected through a CTE: the name survives, so it is
    /// restored to its declared type.
    #[test]
    fn query_sql_cte_string_column_is_declared_type() {
        let st = seeded(&["rust", "go", "rust"], &["a", "b", "c"]);

        let batches = st
            .reader()
            .query_sql(
                "WITH t AS (SELECT category FROM supertable) \
                 SELECT category FROM t GROUP BY category ORDER BY category",
            )
            .expect("cte");
        let col = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("CTE column is LargeUtf8");
        let got: Vec<&str> = (0..col.len()).map(|i| col.value(i)).collect();
        assert_eq!(got, vec!["go", "rust"]);
    }

    /// String column projected through a FROM-subquery.
    #[test]
    fn query_sql_subquery_string_column_is_declared_type() {
        let st = seeded(&["rust", "go", "rust"], &["a", "b", "c"]);

        let batches = st
            .reader()
            .query_sql(
                "SELECT category FROM (SELECT category FROM supertable) sub \
                 GROUP BY category ORDER BY category",
            )
            .expect("subquery");
        let col = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("subquery column is LargeUtf8");
        let got: Vec<&str> = (0..col.len()).map(|i| col.value(i)).collect();
        assert_eq!(got, vec!["go", "rust"]);
    }

    /// No data loss through the view + cast-back for values that stress
    /// `Utf8View`'s layout: strings past the 12-byte inline limit (stored
    /// out-of-line), values sharing a 4-byte prefix (the view compares the
    /// prefix first, so it must fall through to the full bytes and keep them
    /// distinct), an empty string, and multi-byte unicode. GROUP BY exercises
    /// both the comparison (distinct groups) and the cast-back (exact values).
    #[test]
    fn query_sql_string_values_survive_view_and_cast_back() {
        let vals = [
            "",                        // empty
            "short",                   // inline (<= 12 bytes)
            "sixteen_byte_val",        // 16 bytes, out-of-line
            "prefabricated_alpha",     // shares "pref" 4-byte prefix ...
            "prefabricated_omega",     // ... differs later, must stay distinct
            "café_ünïcode_日本語_str", // multi-byte unicode, out-of-line
            "sixteen_byte_val",        // duplicate: must fold to one group
        ];
        let titles: Vec<&str> = (0..vals.len()).map(|_| "t").collect();
        let st = seeded(&vals, &titles);
        let batches = st
            .reader()
            .query_sql("SELECT category FROM supertable GROUP BY category ORDER BY category")
            .expect("group-by over layout-stressing values");
        let col = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("category is LargeUtf8 after cast-back");
        let mut got: Vec<&str> = (0..col.len()).map(|i| col.value(i)).collect();
        got.sort_unstable();

        // Every distinct value survives byte-for-byte; the duplicate folds to
        // one; the prefix-sharing pair stays as two.
        let mut want: Vec<&str> = vec![
            "",
            "café_ünïcode_日本語_str",
            "prefabricated_alpha",
            "prefabricated_omega",
            "short",
            "sixteen_byte_val",
        ];
        want.sort_unstable();
        assert_eq!(got, want);
    }

    /// `SELECT DISTINCT` on a viewed string column: dedup compares on the view,
    /// result comes back the declared type.
    #[test]
    fn query_sql_distinct_string_is_declared_type() {
        let st = seeded(&["rust", "go", "rust"], &["a", "b", "c"]);

        let batches = st
            .reader()
            .query_sql("SELECT DISTINCT category FROM supertable ORDER BY category")
            .expect("distinct");
        let col = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("distinct column is LargeUtf8");
        let got: Vec<&str> = (0..col.len()).map(|i| col.value(i)).collect();
        assert_eq!(got, vec!["go", "rust"]);
    }

    /// Self-join whose join key is a viewed string column: the equality runs on
    /// `Utf8View`, and the projected key comes back its declared type.
    #[test]
    fn query_sql_self_join_on_string_key() {
        let st = seeded(&["rust", "go", "rust"], &["a", "b", "c"]);

        let batches = st
            .reader()
            .query_sql(
                "SELECT a.category AS cat FROM supertable a \
                 JOIN supertable b ON a.category = b.category \
                 GROUP BY a.category ORDER BY a.category",
            )
            .expect("self-join on string key");
        let col = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("join key projects as LargeUtf8");
        let got: Vec<&str> = (0..col.len()).map(|i| col.value(i)).collect();
        assert_eq!(got, vec!["go", "rust"]);
    }

    #[test]
    fn query_sql_scans_across_multiple_superfiles() {
        // Three commits → three superfiles. SQL must aggregate across
        // all of them.
        let st = Supertable::create(options_id_cat_title()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_cat_batch(0, &["rust", "rust"], &["a", "b"]))
            .expect("a1");
        w.commit().expect("c1");
        w.append(&build_cat_batch(10, &["python"], &["c"]))
            .expect("a2");
        w.commit().expect("c2");
        w.append(&build_cat_batch(20, &["rust", "go"], &["d", "e"]))
            .expect("a3");
        w.commit().expect("c3");

        assert_eq!(st.reader().n_superfiles(), 3);

        let n_total = run_count(&st, "SELECT COUNT(*) FROM supertable");
        assert_eq!(n_total, 5);

        let n_rust = run_count(
            &st,
            "SELECT COUNT(*) FROM supertable WHERE category = 'rust'",
        );
        assert_eq!(n_rust, 3);
    }

    #[test]
    fn query_sql_equality_on_fts_column_across_superfiles_is_correct() {
        // Equality on the FTS-indexed `title` column drives the new
        // term-bloom prune leaf (plus the scalar min/max leaf). The two
        // superfiles whose bloom lacks "bravo" may be pruned, but the
        // result must still be exactly the one matching row — proving
        // the bloom prune never drops a match.
        let st = Supertable::create(options_id_cat_title()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_cat_batch(0, &["x"], &["alpha"]))
            .expect("a1");
        w.commit().expect("c1");
        w.append(&build_cat_batch(10, &["y"], &["bravo"]))
            .expect("a2");
        w.commit().expect("c2");
        w.append(&build_cat_batch(20, &["z"], &["charlie"]))
            .expect("a3");
        w.commit().expect("c3");
        assert_eq!(st.reader().n_superfiles(), 3);

        assert_eq!(
            run_count(&st, "SELECT COUNT(*) FROM supertable WHERE title = 'bravo'"),
            1
        );
        assert_eq!(
            run_count(
                &st,
                "SELECT COUNT(*) FROM supertable WHERE title = 'nonexistent'"
            ),
            0
        );
    }

    #[test]
    fn query_sql_multiword_equality_on_fts_column_is_correct() {
        // Multi-word literal: the equality lowers to a `TermPresence`
        // leaf over {rust, async, runtime} (AND). The second superfile's
        // bloom lacks those tokens and is pruned, yet results are exact
        // — DataFusion's FilterExec re-applies the full string equality.
        let st = Supertable::create(options_id_cat_title()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_cat_batch(0, &["lang"], &["rust async runtime"]))
            .expect("a1");
        w.commit().expect("c1");
        w.append(&build_cat_batch(10, &["lang"], &["python data science"]))
            .expect("a2");
        w.commit().expect("c2");
        assert_eq!(st.reader().n_superfiles(), 2);

        assert_eq!(
            run_count(
                &st,
                "SELECT COUNT(*) FROM supertable WHERE title = 'rust async runtime'"
            ),
            1
        );
        // Tokens present in superfile 1, but no row equals this exact
        // string — the prune is an optimization, correctness holds.
        assert_eq!(
            run_count(
                &st,
                "SELECT COUNT(*) FROM supertable WHERE title = 'rust async'"
            ),
            0
        );
    }

    #[test]
    fn query_sql_fts_equality_superset_is_narrowed_to_exact_match() {
        // Index-driven row selection: the candidate plan resolves
        // `WHERE title = 'rust async'` to the term-AND posting set, which
        // within one superfile is a *superset* — both rows below contain
        // {rust, async}. The FilterExec above the scan must narrow that
        // candidate superset to the single exact-equality row, proving
        // the row-level prune never over-returns.
        let st = Supertable::create(options_id_cat_title()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_cat_batch(
            0,
            &["x", "y"],
            &["rust async", "rust async runtime"],
        ))
        .expect("append");
        w.commit().expect("commit");

        assert_eq!(
            run_count(
                &st,
                "SELECT COUNT(*) FROM supertable WHERE title = 'rust async'",
            ),
            1,
        );
        let batches = st
            .reader()
            .query_sql("SELECT title FROM supertable WHERE title = 'rust async'")
            .expect("query");
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 1);
    }

    #[test]
    fn query_sql_fts_or_and_in_are_exact() {
        // OR of two FTS equalities, AND with a non-FTS conjunct, and IN —
        // all index-bounded except where a branch is un-boundable, and
        // all verified exact by FilterExec.
        let st = Supertable::create(options_id_cat_title()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_cat_batch(
            0,
            &["rust", "python", "rust", "go"],
            &["alpha", "beta", "gamma", "delta"],
        ))
        .expect("append");
        w.commit().expect("commit");

        // OR of two FTS equalities → union, exact.
        assert_eq!(
            run_count(
                &st,
                "SELECT COUNT(*) FROM supertable WHERE title = 'alpha' OR title = 'beta'",
            ),
            2,
        );
        // AND with a non-FTS conjunct: FTS branch bounds candidates, the
        // category check is verified in pass 2.
        assert_eq!(
            run_count(
                &st,
                "SELECT COUNT(*) FROM supertable \
                 WHERE title = 'alpha' AND category = 'rust'",
            ),
            1,
        );
        assert_eq!(
            run_count(
                &st,
                "SELECT COUNT(*) FROM supertable \
                 WHERE title = 'alpha' AND category = 'python'",
            ),
            0,
        );
        // IN on the FTS column → OR of equalities.
        assert_eq!(
            run_count(
                &st,
                "SELECT COUNT(*) FROM supertable WHERE title IN ('alpha', 'delta', 'zzz')",
            ),
            2,
        );
    }

    #[test]
    fn query_sql_not_predicates_are_exact() {
        // NOT / != aren't index-prefiltered (Unbounded → scan), but must
        // still be exact; and `= AND !=` prefilters on the `=` branch
        // while FilterExec applies the negation.
        let st = Supertable::create(options_id_cat_title()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_cat_batch(
            0,
            &["rust", "python", "rust", "go"],
            &["alpha", "beta", "alpha", "delta"],
        ))
        .expect("append");
        w.commit().expect("commit");

        // Standalone NOT (scan fallback): 4 rows, 2 are 'alpha' → 2 left.
        assert_eq!(
            run_count(
                &st,
                "SELECT COUNT(*) FROM supertable WHERE NOT (title = 'alpha')",
            ),
            2,
        );
        // `!=` (NotEq) likewise.
        assert_eq!(
            run_count(
                &st,
                "SELECT COUNT(*) FROM supertable WHERE title != 'alpha'"
            ),
            2,
        );
        // `= AND !=`: candidates from the `title='alpha'` branch (2 rows),
        // then FilterExec drops category='rust' → 1 remains.
        assert_eq!(
            run_count(
                &st,
                "SELECT COUNT(*) FROM supertable \
                 WHERE title = 'alpha' AND category != 'rust'",
            ),
            0,
        );
        assert_eq!(
            run_count(
                &st,
                "SELECT COUNT(*) FROM supertable \
                 WHERE title = 'alpha' AND category != 'python'",
            ),
            2,
        );
    }

    #[test]
    fn query_sql_or_with_non_fts_branch_matches_full_scan() {
        // `title = 'alpha' OR category = 'go'` is un-boundable (the
        // category branch could match any row), so the planner falls back
        // to a full scan + FilterExec — and must still be exact.
        let st = Supertable::create(options_id_cat_title()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_cat_batch(
            0,
            &["rust", "python", "go", "go"],
            &["alpha", "beta", "gamma", "delta"],
        ))
        .expect("append");
        w.commit().expect("commit");

        // alpha (1 row) ∪ category=go (2 rows), disjoint → 3.
        assert_eq!(
            run_count(
                &st,
                "SELECT COUNT(*) FROM supertable WHERE title = 'alpha' OR category = 'go'",
            ),
            3,
        );
    }

    #[test]
    fn query_sql_select_orders_ids_across_superfiles() {
        // Verifies row identity round-trips through MemTable +
        // DataFusion: rows planted across two superfiles come back
        // in monotonic _id order under ORDER BY. The _id values
        // are auto-injected by the supertable (timestamp +
        // worker + counter), so we don't assert specific
        // values — only strict-increasing order.
        let st = Supertable::create(options_id_cat_title()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_cat_batch(100, &["a", "b"], &["t1", "t2"]))
            .expect("a1");
        w.commit().expect("c1");
        w.append(&build_cat_batch(200, &["c"], &["t3"]))
            .expect("a2");
        w.commit().expect("c2");

        let batches = st
            .reader()
            .query_sql("SELECT _id FROM supertable ORDER BY _id")
            .expect("query");
        let ids: Vec<i128> = batches
            .iter()
            .flat_map(|b| {
                let a = b
                    .column(0)
                    .as_any()
                    .downcast_ref::<Decimal128Array>()
                    .expect("_id is Decimal128");
                (0..a.len()).map(|i| a.value(i)).collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(ids.len(), 3);
        for w in ids.windows(2) {
            assert!(w[0] < w[1], "expected strictly increasing _id");
        }
    }

    #[test]
    fn query_sql_select_star_exposes_only_user_columns_plus_id() {
        // The supertable is a thin SQL skin over scalar columns —
        // `inf.*` KV metadata stays invisible. The injected `_id`
        // column is part of the visible schema.
        let st = Supertable::create(options_id_cat_title()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_cat_batch(0, &["x"], &["t"])).expect("a");
        w.commit().expect("c");

        let batches = st
            .reader()
            .query_sql("SELECT * FROM supertable LIMIT 1")
            .expect("query");
        let schema = batches[0].schema();
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(names, vec!["_id", "category", "title"]);
    }

    #[test]
    fn query_sql_runtime_is_cached_across_calls() {
        // Two queries on the same supertable must share one
        // Runtime — the OnceLock guarantees this; we assert by
        // checking that both calls succeed without spawning a
        // fresh Runtime per call (observed indirectly via the
        // `.await` over `block_on` not double-allocating; if the
        // cache regressed, tests would still pass but would leak
        // a Runtime per call. The functional check below is
        // adequate for correctness; benchmarks would catch leak).
        let st = Supertable::create(options_id_cat_title()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_cat_batch(0, &["x"], &["t"])).expect("a");
        w.commit().expect("c");
        for _ in 0..3 {
            let n = run_count(&st, "SELECT COUNT(*) FROM supertable");
            assert_eq!(n, 1);
        }
    }

    #[test]
    fn query_sql_invalid_sql_returns_plan_error() {
        let st = Supertable::create(options_id_cat_title()).expect("create");
        let err = st
            .reader()
            .query_sql("SELECT NOT_A_REAL_FN(*) FROM supertable")
            .expect_err("expected a plan error");
        assert!(
            matches!(err, QueryError::Plan(_)),
            "expected Plan variant; got {err:?}"
        );
    }

    // ---- vector schema integration ----------------------------------

    /// Build a schema that includes a vector column. The supertable
    /// strips it at commit time; SQL surface only sees the scalar
    /// columns. `query_sql` SELECTing the vector column must error
    /// (DataFusion's planner rejects unknown column).
    fn schema_with_vector(dim: usize) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new(
                "emb",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dim as i32,
                ),
                false,
            ),
        ]))
    }

    fn options_with_vector(dim: usize) -> SupertableOptions {
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("rayon pool"),
        );
        SupertableOptions::new(
            schema_with_vector(dim),
            vec![FtsConfig {
                column: "title".into(),
            }],
            vec![VectorConfig {
                column: "emb".into(),
                dim,
                n_cent: 4,
                rot_seed: 0,
                metric: Metric::Cosine,
                rerank_codec: RerankCodec::Fp32,
            }],
            Some(tok()),
        )
        .expect("valid options")
        .with_writer_pool(pool)
    }

    fn build_vector_batch(_start: u64, n: usize, dim: usize) -> RecordBatch {
        let titles = LargeStringArray::from((0..n).map(|i| format!("doc {i}")).collect::<Vec<_>>());
        let mut flat = Vec::<f32>::with_capacity(n * dim);
        for i in 0..n {
            for d in 0..dim {
                flat.push(((i + d) as f32) / 100.0);
            }
        }
        let item_field = Arc::new(Field::new("item", DataType::Float32, true));
        let values = Float32Array::from(flat);
        let emb = FixedSizeListArray::try_new(
            item_field,
            dim as i32,
            Arc::new(values) as Arc<dyn Array>,
            None,
        )
        .expect("FixedSizeList build");
        RecordBatch::try_new(
            schema_with_vector(dim),
            vec![Arc::new(titles), Arc::new(emb)],
        )
        .expect("build batch")
    }

    #[test]
    fn query_sql_hides_vector_columns_from_sql_surface() {
        let st = Supertable::create(options_with_vector(16)).expect("create");
        let mut w = st.writer().expect("writer");
        // n=8 ≥ n_cent=4 so kmeans has data to cluster.
        w.append(&build_vector_batch(0, 8, 16)).expect("append");
        w.commit().expect("commit");

        let batches = st
            .reader()
            .query_sql("SELECT * FROM supertable LIMIT 1")
            .expect("query");
        let schema = batches[0].schema();
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        // `emb` was stripped by `vector_split` at commit time and
        // lives in the embedded vector blob — not visible to SQL.
        // The supertable-injected `_id` is visible.
        assert_eq!(names, vec!["_id", "title"]);
    }

    #[test]
    fn query_sql_referencing_vector_column_returns_plan_error() {
        let st = Supertable::create(options_with_vector(16)).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_vector_batch(0, 8, 16)).expect("append");
        w.commit().expect("commit");

        let err = st
            .reader()
            .query_sql("SELECT emb FROM supertable")
            .expect_err("vector column should not be in the SQL schema");
        assert!(
            matches!(err, QueryError::Plan(_)),
            "expected Plan variant; got {err:?}"
        );
    }
}
