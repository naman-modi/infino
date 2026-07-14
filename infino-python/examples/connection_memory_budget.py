# Bound a connection's heap and recover from an over-budget refusal.
#
# `connect(connection_memory_budget_bytes=...)` caps the heap one connection may
# use for query, ingest, and vector work. A request that would cross the limit
# raises MemoryError instead of risking an out-of-memory crash, so the caller can
# back off (narrow the query, raise the budget) rather than lose the process.
#
# Storage is a local directory, so the data is durable and inspectable. We use
# a TemporaryDirectory so the example is self-cleaning and always writes to a
# path this process owns (created 0700 under the system temp root); a real
# deployment would point at a persistent path or an object-store URI. Key-free.
#
# Run:  pip install infino pyarrow && python connection_memory_budget.py

import os
import tempfile

import infino
import pyarrow as pa

TITLE_SCHEMA = pa.schema([pa.field("title", pa.large_utf8(), nullable=False)])


def batch(titles):
    return pa.record_batch([pa.array(titles, type=pa.large_utf8())], schema=TITLE_SCHEMA)


def main():
    with tempfile.TemporaryDirectory(prefix="infino-budget-") as root:
        # Separate catalog dirs so the two connections don't share state.
        # Create them up front so we know the process owns them.
        tight_dir = os.path.join(root, "tight")
        ample_dir = os.path.join(root, "ample")
        os.makedirs(tight_dir)
        os.makedirs(ample_dir)

        # A 1-byte budget floors the enforced gate to 0, so any build is refused.
        # Real deployments pass a fraction of the process's RAM here.
        tight = infino.connect(tight_dir, connection_memory_budget_bytes=1)
        docs = tight.create_table("docs", TITLE_SCHEMA, infino.IndexSpec().fts("title"))
        try:
            docs.append(batch(["the quick brown fox", "a lazy dog"]))
            print("unexpected: ingest was admitted under a 1-byte budget")
        except MemoryError as e:
            print(f"ingest refused, over budget: {e}")

        # A generous budget admits the same work.
        ample = infino.connect(ample_dir, connection_memory_budget_bytes=1 << 30)
        ok = ample.create_table("docs", TITLE_SCHEMA, infino.IndexSpec().fts("title"))
        ok.append(batch(["the quick brown fox"]))
        hits = ok.bm25_search("title", "fox", 10)
        print(f"under an ample budget, ingest committed: {hits.num_rows} row(s)")


if __name__ == "__main__":
    main()
