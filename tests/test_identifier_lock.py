"""Per-identifier serialization of adds (#486).

The Zotero API lock protects the single-threaded local API. It never promised
to make check-then-create atomic — that was a side effect of how wide it used
to be, and narrowing it (to keep CrossRef and page fetches out of a held lock)
removed it wherever a fetch now sits between the check and the write:

    lock -> find_existing -> unlock -> fetch -> lock -> create -> unlock

Two callers can both pass the dedup check and both create. There is no version
to conflict on, so the version-checked retry cannot help: two ``create_items()``
POSTs yield two new keys.

The fix the maintainer chose on #486 is a short-lived in-process lock keyed on
the *normalized* identifier, held across a re-check and the create. These tests
pin the four properties that decision came with.
"""

import threading
import time

from zotero_mcp.tools._helpers import identifier_lock, identifier_lock_key

# ---------------------------------------------------------------------------
# Keying: the same work must take the same lock, and different work must not
# ---------------------------------------------------------------------------


class TestIdentifierLockKey:
    def test_isbn10_and_isbn13_of_one_book_share_a_key(self):
        """The reason keying is on the normalized form, not the raw string.

        0-306-40615-2 and 978-0-306-40615-7 are the same book; keyed raw they
        would take different locks and race each other.
        """
        assert identifier_lock_key("isbn", "0-306-40615-2") == identifier_lock_key(
            "isbn", "9780306406157"
        )

    def test_doi_spellings_share_a_key(self):
        forms = [
            "10.1234/abcd",
            "doi:10.1234/abcd",
            "https://doi.org/10.1234/abcd",
            "  10.1234/abcd  ",
            "10.1234/abcd.",
        ]
        keys = {identifier_lock_key("doi", f) for f in forms}
        assert len(keys) == 1, keys

    def test_different_identifiers_get_different_keys(self):
        assert identifier_lock_key("doi", "10.1234/aaa") != identifier_lock_key(
            "doi", "10.1234/bbb"
        )

    def test_kinds_do_not_collide(self):
        """A DOI and a URL that stringify alike must not share a lock."""
        assert identifier_lock_key("doi", "10.1234/abcd") != identifier_lock_key(
            "url", "10.1234/abcd"
        )

    def test_urls_are_exact_after_strip(self):
        """URLs have no normalizer, so the key is the stripped string.

        Matching what #443 settled on for collapsing repeated identifiers —
        deliberately not doing case or trailing-slash folding here, because a
        URL that differs in either can legitimately be a different page.
        """
        assert identifier_lock_key("url", "  https://example.org/a  ") == (
            identifier_lock_key("url", "https://example.org/a")
        )
        assert identifier_lock_key("url", "https://example.org/a") != (
            identifier_lock_key("url", "https://example.org/a/")
        )

    def test_unnormalizable_identifier_has_no_key(self):
        """A junk token takes no lock rather than sharing one.

        Sharing a single "unnormalizable" lock would make a batch of bad
        identifiers serialize against each other for no benefit — they cannot
        collide, because none of them will dedup-match anything.
        """
        for kind, raw in (
            ("doi", "not-a-doi"),
            ("doi", ""),
            ("doi", None),
            ("isbn", "12345"),
            ("url", "   "),
        ):
            assert identifier_lock_key(kind, raw) is None, (kind, raw)


# ---------------------------------------------------------------------------
# Mutual exclusion
# ---------------------------------------------------------------------------


class TestIdentifierLock:
    def test_same_identifier_serializes(self):
        """Two threads on one DOI must not overlap inside the lock."""
        overlap = []
        inside = []
        barrier = threading.Barrier(2)

        def worker():
            barrier.wait()
            with identifier_lock("doi", "10.1234/same"):
                overlap.append(len(inside))
                inside.append(1)
                time.sleep(0.05)
                inside.pop()

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert overlap == [0, 0], f"threads overlapped inside the lock: {overlap}"

    def test_different_identifiers_do_not_serialize(self):
        """Distinct DOIs must proceed concurrently.

        A global add lock would also fix the race and would also pass the test
        above; this is what distinguishes per-identifier from that.
        """
        barrier = threading.Barrier(2, timeout=2)
        reached = []

        def worker(doi):
            with identifier_lock("doi", doi):
                # Deadlocks (raising BrokenBarrierError) if these serialize.
                barrier.wait()
                reached.append(doi)

        threads = [
            threading.Thread(target=worker, args=(f"10.1234/{n}",))
            for n in ("aaa", "bbb")
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sorted(reached) == ["10.1234/aaa", "10.1234/bbb"]

    def test_unnormalizable_identifiers_do_not_serialize(self):
        """The no-key path must not funnel junk through one shared lock."""
        barrier = threading.Barrier(2, timeout=2)
        reached = []

        def worker(token):
            with identifier_lock("doi", token):
                barrier.wait()
                reached.append(token)

        threads = [
            threading.Thread(target=worker, args=(t,))
            for t in ("garbage-one", "garbage-two")
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sorted(reached) == ["garbage-one", "garbage-two"]

    def test_lock_is_reentrant_for_one_thread(self):
        """Nested acquisition of the same identifier must not self-deadlock.

        The add paths call into helpers that may take the same lock again; a
        plain Lock would wedge the process rather than fail a test.
        """
        with identifier_lock("doi", "10.1234/nested"):
            with identifier_lock("doi", "10.1234/nested"):
                pass

    def test_registry_does_not_leak_entries(self):
        """A long-running server indexing a large library must not accumulate
        one lock object per identifier it has ever seen."""
        from zotero_mcp.tools import _helpers

        before = len(_helpers._identifier_locks)
        for i in range(50):
            with identifier_lock("doi", f"10.1234/leak{i}"):
                pass
        assert len(_helpers._identifier_locks) == before

    def test_entry_survives_while_a_waiter_holds_it(self):
        """Cleanup must not drop an entry another thread is queued on."""
        from zotero_mcp.tools import _helpers

        started = threading.Event()
        release = threading.Event()

        def holder():
            with identifier_lock("doi", "10.1234/held"):
                started.set()
                release.wait(timeout=2)

        t = threading.Thread(target=holder)
        t.start()
        started.wait(timeout=2)
        assert "doi:10.1234/held" in _helpers._identifier_locks
        release.set()
        t.join()
        assert "doi:10.1234/held" not in _helpers._identifier_locks


# ---------------------------------------------------------------------------
# The add paths must actually take the lock, and re-check inside it
# ---------------------------------------------------------------------------
#
# Reproducing the race by timing turned out to be the wrong test to write: it
# needs two threads to interleave at a specific point inside a path that does
# a lot of other work, and every version of it either passed with the lock
# removed (proving nothing) or deadlocked on its own scaffolding. A flaky or
# vacuous concurrency test is worse than none — it is the same shape of
# mistake as the comment #486 was filed about, a guard that is load-bearing in
# the argument and absent in fact.
#
# What is deterministic, and is what the fix actually claims: each add path
# enters the identifier lock with the right key, and performs its dedup
# re-check *while inside it*. Ordering is the property; these pin it directly.


class _LockSpy:
    """Records identifier_lock entries and what happened inside them."""

    def __init__(self, real):
        self._real = real
        self.entered = []
        self.events = []

    def __call__(self, kind, raw):
        import contextlib as _cl

        spy = self

        @_cl.contextmanager
        def _wrapper():
            spy.entered.append((kind, raw))
            spy.events.append(("enter", kind, raw))
            with spy._real(kind, raw) as key:
                yield key
            spy.events.append(("exit", kind, raw))

        return _wrapper()


class TestAddPathsTakeTheLock:
    def _spy(self, monkeypatch):
        from zotero_mcp.tools import _helpers

        spy = _LockSpy(_helpers.identifier_lock)
        monkeypatch.setattr(_helpers, "identifier_lock", spy)
        return spy

    def test_add_by_isbn_locks_on_the_normalized_isbn(self, monkeypatch):
        from conftest import DummyContext, FakeZotero

        from zotero_mcp.tools import write

        spy = self._spy(monkeypatch)
        zot = FakeZotero()
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (zot, zot))
        monkeypatch.setattr(write, "_lookup_isbn_openlibrary",
                            lambda isbn, ctx: {"title": "A Book"})

        write.add_by_isbn("0-306-40615-2", ctx=DummyContext())

        # ISBN-10 in, ISBN-13 on the lock: keyed on the normalized form, so
        # the other spelling of the same book takes the same lock.
        assert ("isbn", "9780306406157") in spy.entered

    def test_dedup_recheck_happens_inside_the_lock(self, monkeypatch):
        """Ordering is the whole fix: a re-check outside the lock is no fix.

        Records the sequence of lock-enter / dedup-check / create and asserts
        the check and the create both fall between enter and exit.
        """
        from conftest import DummyContext, FakeZotero

        from zotero_mcp.tools import _helpers, write

        spy = self._spy(monkeypatch)
        zot = FakeZotero()
        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                            lambda ctx: (zot, zot))
        monkeypatch.setattr(write, "_lookup_isbn_openlibrary",
                            lambda isbn, ctx: {"title": "A Book"})

        real_find = _helpers.find_existing_items

        def recording_find(*args, **kwargs):
            spy.events.append(("dedup-check", kwargs.get("isbn"), None))
            return real_find(*args, **kwargs)

        monkeypatch.setattr(_helpers, "find_existing_items", recording_find)

        def recording_create(items, **kwargs):
            spy.events.append(("create", None, None))
            return {"success": {"0": "KEY1"}}

        monkeypatch.setattr(zot, "create_items", recording_create)

        # if_exists="skip" is what asks for dedup at all; the default
        # "duplicate" deliberately performs no check, so it cannot show the
        # ordering this test is about.
        write.add_by_isbn("9780306406157", if_exists="skip", ctx=DummyContext())

        kinds = [e[0] for e in spy.events]
        assert "enter" in kinds and "create" in kinds, spy.events
        enter_at = kinds.index("enter")
        exit_at = kinds.index("exit")
        create_at = kinds.index("create")
        rechecks = [i for i, k in enumerate(kinds)
                    if k == "dedup-check" and enter_at < i < exit_at]

        assert rechecks, (
            f"no dedup re-check inside the lock: {spy.events}"
        )
        assert enter_at < create_at < exit_at, (
            f"create happened outside the identifier lock: {spy.events}"
        )
        assert rechecks[0] < create_at, (
            f"re-check did not precede the create: {spy.events}"
        )
