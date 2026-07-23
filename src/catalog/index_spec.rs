// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! [`IndexSpec`] — declares which columns of a table are full-text
//! (BM25) indexed and which are vector (IVF kNN) indexed. Passed to
//! [`Connection::create_table`](crate::Connection::create_table) alongside
//! the Arrow schema.

use crate::superfile::{
    builder::FtsConfig,
    vector::{builder::VectorConfig, distance::Metric},
};

/// Default rotation-matrix RNG seed for vector columns. The seed only
/// has to be stable for a given table; the public API does not vary it.
const DEFAULT_ROT_SEED: u64 = 0x5EED_5EED_5EED_5EED;

/// A vector index declaration: column, dimensionality, IVF centroid
/// count, and distance metric.
#[derive(Debug, Clone)]
struct VectorIndex {
    column: String,
    dim: usize,
    n_cent: usize,
    metric: Metric,
}

/// Declares the search indexes to build over a table's columns.
///
/// Built fluently; every column named here must exist in the table's
/// Arrow schema. Columns with no index are still stored and queryable
/// via SQL — they just have no BM25 / vector index.
///
/// ```
/// use infino::{IndexSpec, Metric};
/// let spec = IndexSpec::new()
///     .fts("body")
///     .vector("embedding", 384, 256, Metric::Cosine);
/// # let _ = spec;
/// ```
#[derive(Debug, Clone, Default)]
pub struct IndexSpec {
    fts: Vec<String>,
    vectors: Vec<VectorIndex>,
    cluster_by: Vec<String>,
}

impl IndexSpec {
    /// An empty spec — no FTS, no vector indexes.
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark `column` as full-text (BM25) indexed. The column must be a
    /// UTF-8 string column in the schema.
    pub fn fts(mut self, column: impl Into<String>) -> Self {
        self.fts.push(column.into());
        self
    }

    /// Mark `column` as vector (IVF kNN) indexed. `dim` is the vector
    /// dimensionality, `n_cent` the IVF centroid count (governs the
    /// recall/latency trade-off — size it to the table's scale), and
    /// `metric` the distance metric. The column must be a
    /// `FixedSizeList<Float32, dim>` column in the schema.
    pub fn vector(
        mut self,
        column: impl Into<String>,
        dim: usize,
        n_cent: usize,
        metric: Metric,
    ) -> Self {
        self.vectors.push(VectorIndex {
            column: column.into(),
            dim,
            n_cent,
            metric,
        });
        self
    }

    /// Declare the table's clustering key: an ordered list of column
    /// names each commit physically sorts its rows by before writing
    /// (lexicographic on the list, ascending, nulls last). Rows with
    /// nearby key values land next to each other on disk, which is
    /// what range queries on the key want. Every named column must
    /// exist in the schema as a sortable scalar type — vector
    /// columns are rejected at `create_table`. The key is fixed at
    /// creation for the table's lifetime.
    pub fn cluster_by<I, S>(mut self, columns: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.cluster_by = columns.into_iter().map(Into::into).collect();
        self
    }

    /// FTS column names, in declaration order.
    pub(crate) fn fts_columns(&self) -> &[String] {
        &self.fts
    }

    /// Clustering-key column names, in sort-precedence order.
    /// Empty means the table is unclustered.
    pub(crate) fn cluster_by_columns(&self) -> &[String] {
        &self.cluster_by
    }

    /// Lower to the internal `(FtsConfig, VectorConfig)` lists the
    /// supertable options take. `rot_seed` / `rerank_codec` are not part
    /// of the public spec — defaults are applied here.
    pub(crate) fn to_configs(&self) -> (Vec<FtsConfig>, Vec<VectorConfig>) {
        let fts = self
            .fts
            .iter()
            .map(|column| FtsConfig {
                column: column.clone(),
                positions: false,
            })
            .collect();
        let vectors = self
            .vectors
            .iter()
            .map(|v| {
                VectorConfig::new(
                    v.column.clone(),
                    v.dim,
                    v.n_cent,
                    DEFAULT_ROT_SEED,
                    v.metric,
                )
            })
            .collect();
        (fts, vectors)
    }

    /// Has at least one FTS column (so a tokenizer is required).
    pub(crate) fn has_fts(&self) -> bool {
        !self.fts.is_empty()
    }
}
