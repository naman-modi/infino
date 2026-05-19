//! Full-text search subsystem — the BM25 + posting list + FST term
//! dictionary stack lives here.

pub mod bm25;
pub mod builder;
pub mod dict;
pub(crate) mod fst_value;
pub mod posting;
pub mod reader;
pub mod tokenize;
