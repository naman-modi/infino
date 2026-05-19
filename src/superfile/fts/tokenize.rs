//! Tokenization.
//!
//! Ships one tokenizer: [`AsciiLowerTokenizer`]. The [`Tokenizer`]
//! trait is the extension point for ICU / language-aware stemmers /
//! custom char filters under the same trait without touching FTS
//! code.
//!
//! Semantics:
//!   - Split on any byte that isn't `[A-Za-z0-9]`.
//!   - Lowercase each ASCII letter (bytes `b'A'..=b'Z'` → `b'a'..=b'z'`).
//!   - Drop any token that contains a non-ASCII byte (high-bit set).
//!     Non-ASCII tokens are silently dropped (not an error) — the
//!     ASCII-only design is intentional; richer tokenizers can opt
//!     into the trait without changing the FTS pipeline.
//!   - Empty tokens are never emitted.

/// Trait every tokenizer impl must satisfy.
///
/// Two entry points:
///
///   - [`Tokenizer::tokenize`] — iterator-shaped, yields owned
///     `String`s. Convenient for query-side / one-off use, but
///     allocates one heap `String` per token.
///
///   - [`Tokenizer::tokenize_each`] — callback-shaped, hands the
///     callback a `&str` borrowed from an internal scratch buffer
///     (valid only for the duration of the call). Zero-alloc on the
///     hot ingest path. The default impl wraps `tokenize`; impls
///     that can do better (like [`AsciiLowerTokenizer`]) override.
pub trait Tokenizer: Send + Sync {
    /// Yield each token as an owned `String` lower-cased per the
    /// implementation's rules.
    fn tokenize<'a>(&'a self, text: &'a str) -> Box<dyn Iterator<Item = String> + 'a>;

    /// Call `f(&token)` for each token. The `&str` passed to `f` is
    /// valid only for that call — copy it (e.g. into a bump arena) if
    /// you need to keep it.
    ///
    /// Default impl iterates `self.tokenize(...)` and calls `f` on
    /// each `String` (one heap alloc per token). Impls that can be
    /// zero-alloc should override.
    fn tokenize_each(&self, text: &str, f: &mut dyn FnMut(&str)) {
        for s in self.tokenize(text) {
            f(&s);
        }
    }
}

/// ASCII whitespace + punctuation split, ASCII lowercase, no stemming,
/// no stopwords. The simplest tokenizer that's still useful.
#[derive(Debug, Clone, Copy, Default)]
pub struct AsciiLowerTokenizer;

impl AsciiLowerTokenizer {
    pub fn new() -> Self {
        Self
    }
}

impl Tokenizer for AsciiLowerTokenizer {
    fn tokenize<'a>(&'a self, text: &'a str) -> Box<dyn Iterator<Item = String> + 'a> {
        Box::new(AsciiLowerIter::new(text.as_bytes()))
    }

    /// Zero-alloc emission: walks the input once, reusing a 32-byte
    /// stack-or-heap scratch buffer for each token. The `&str` passed
    /// to the callback aliases that buffer; the next callback
    /// invocation overwrites it.
    fn tokenize_each(&self, text: &str, f: &mut dyn FnMut(&str)) {
        let bytes = text.as_bytes();
        let mut buf: Vec<u8> = Vec::with_capacity(32);
        let mut pos = 0;
        loop {
            // Skip non-token bytes.
            while pos < bytes.len() && !is_token_byte(bytes[pos]) {
                pos += 1;
            }
            if pos >= bytes.len() {
                return;
            }
            // Accumulate one token.
            buf.clear();
            let mut had_non_ascii = false;
            while pos < bytes.len() {
                let b = bytes[pos];
                if is_token_byte(b) {
                    buf.push(b.to_ascii_lowercase());
                    pos += 1;
                } else if b >= 0x80 {
                    had_non_ascii = true;
                    pos += 1;
                } else {
                    break;
                }
            }
            if had_non_ascii || buf.is_empty() {
                continue;
            }
            // SAFETY-equivalent: we only push ASCII letters/digits (and
            // their lowercased forms), so `buf` is always valid UTF-8.
            let s = std::str::from_utf8(&buf).expect("ASCII-only by construction");
            f(s);
        }
    }
}

/// Internal iterator that walks the input byte slice once, emitting
/// lowercased tokens. Skips tokens containing non-ASCII bytes per the
/// v1 ASCII-only rule.
struct AsciiLowerIter<'a> {
    src: &'a [u8],
    pos: usize,
    buf: Vec<u8>,
}

impl<'a> AsciiLowerIter<'a> {
    fn new(src: &'a [u8]) -> Self {
        Self {
            src,
            pos: 0,
            buf: Vec::with_capacity(32),
        }
    }
}

impl Iterator for AsciiLowerIter<'_> {
    type Item = String;

    fn next(&mut self) -> Option<String> {
        loop {
            // Skip non-token bytes.
            while self.pos < self.src.len() && !is_token_byte(self.src[self.pos]) {
                self.pos += 1;
            }
            if self.pos >= self.src.len() {
                return None;
            }

            // Accumulate one token.
            self.buf.clear();
            let mut had_non_ascii = false;
            while self.pos < self.src.len() {
                let b = self.src[self.pos];
                if is_token_byte(b) {
                    self.buf.push(b.to_ascii_lowercase());
                    self.pos += 1;
                } else if b >= 0x80 {
                    // Non-ASCII byte inside a contiguous "word-ish" run —
                    // mark this run as non-ASCII and consume until a true
                    // separator. Drop the whole token.
                    had_non_ascii = true;
                    self.pos += 1;
                } else {
                    break;
                }
            }

            if had_non_ascii || self.buf.is_empty() {
                continue;
            }

            // SAFETY: we only push ASCII letters and digits via
            // is_token_byte + to_ascii_lowercase, so the buffer is
            // guaranteed valid UTF-8.
            let s = std::str::from_utf8(&self.buf)
                .expect("ASCII-only by construction")
                .to_owned();
            return Some(s);
        }
    }
}

/// `[A-Za-z0-9]` — the v1 token alphabet.
#[inline]
fn is_token_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(text: &str) -> Vec<String> {
        AsciiLowerTokenizer.tokenize(text).collect()
    }

    #[test]
    fn empty_input_yields_nothing() {
        assert_eq!(tokens(""), Vec::<String>::new());
    }

    #[test]
    fn whitespace_only_yields_nothing() {
        assert_eq!(tokens("   \t\n\r"), Vec::<String>::new());
    }

    #[test]
    fn single_token_lowercased() {
        assert_eq!(tokens("Hello"), vec!["hello"]);
    }

    #[test]
    fn multiple_tokens_split_on_whitespace() {
        assert_eq!(
            tokens("Rust async runtime"),
            vec!["rust", "async", "runtime"]
        );
    }

    #[test]
    fn punctuation_splits_tokens() {
        assert_eq!(
            tokens("hello,world!foo;bar.baz?"),
            vec!["hello", "world", "foo", "bar", "baz"]
        );
    }

    #[test]
    fn case_folding_applies_to_uppercase_only() {
        assert_eq!(tokens("ABC abc XyZ"), vec!["abc", "abc", "xyz"]);
    }

    #[test]
    fn alphanumerics_kept_together() {
        assert_eq!(tokens("foo123 bar456"), vec!["foo123", "bar456"]);
    }

    #[test]
    fn pure_numeric_tokens_kept() {
        assert_eq!(tokens("404 200 500"), vec!["404", "200", "500"]);
    }

    #[test]
    fn underscore_is_a_separator_in_v1() {
        // `_` is not in `[A-Za-z0-9]` — it splits tokens. v2 may revisit.
        assert_eq!(tokens("foo_bar"), vec!["foo", "bar"]);
    }

    #[test]
    fn dash_is_a_separator() {
        assert_eq!(tokens("rust-async"), vec!["rust", "async"]);
    }

    #[test]
    fn non_ascii_token_is_dropped() {
        // ASCII-only tokenizer: "café" has a non-ASCII byte, so the
        // entire token is dropped.
        assert_eq!(tokens("café"), Vec::<String>::new());
    }

    #[test]
    fn non_ascii_token_drops_only_that_token() {
        // Surrounding ASCII tokens still come through.
        assert_eq!(tokens("hello café world"), vec!["hello", "world"]);
    }

    #[test]
    fn cjk_input_yields_nothing() {
        assert_eq!(tokens("日本語"), Vec::<String>::new());
    }

    #[test]
    fn emoji_input_yields_nothing() {
        assert_eq!(tokens("hello 🚀 world"), vec!["hello", "world"]);
    }

    #[test]
    fn multiple_consecutive_separators_are_collapsed() {
        assert_eq!(tokens("foo,,,bar"), vec!["foo", "bar"]);
        assert_eq!(tokens("foo   bar"), vec!["foo", "bar"]);
    }

    #[test]
    fn leading_and_trailing_separators_are_skipped() {
        assert_eq!(tokens("  foo bar  "), vec!["foo", "bar"]);
        assert_eq!(tokens("...foo..."), vec!["foo"]);
    }

    #[test]
    fn tokenizer_is_send_and_sync() {
        // Compile-time assertion via the Tokenizer trait bound.
        fn is_send_sync<T: Send + Sync>() {}
        is_send_sync::<AsciiLowerTokenizer>();
    }

    #[test]
    fn tokenizer_used_via_dyn_trait() {
        // The trait object form is what the FtsBuilder will hold.
        let tok: Box<dyn Tokenizer> = Box::new(AsciiLowerTokenizer);
        let v: Vec<String> = tok.tokenize("Hello WORLD").collect();
        assert_eq!(v, vec!["hello", "world"]);
    }

    #[test]
    fn stress_long_input_does_not_panic() {
        // Rough scale-test: 1 MB of pseudo-text.
        let chunk = "lorem ipsum dolor sit amet, consectetur adipiscing elit. ";
        let big = chunk.repeat(20_000);
        let count = AsciiLowerTokenizer.tokenize(&big).count();
        // 8 tokens per chunk × 20_000 = 160_000.
        assert_eq!(count, 8 * 20_000);
    }
}
