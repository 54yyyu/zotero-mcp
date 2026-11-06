"""Local duplicate scans bypass the HTTP ceiling without changing grouping."""

import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch

import _search_corpus as corpus

from zotero_mcp import library
from zotero_mcp.local_db import LocalZoteroReader
from zotero_mcp.tools import write


class ApiItems:
    def __init__(self, items):
        self.rows = items
        self.items = Mock(side_effect=self.page)
        self.collection_items = Mock(side_effect=self.collection_page)
        self.children = Mock(return_value=[])

    def page(self, *, start=0, limit=100):
        return self.rows[start:start + limit]

    def collection_page(self, key, *, start=0, limit=100):
        rows = [item for item in self.rows if key in item['data'].get('collections', [])]
        return rows[start:start + limit]


def members(groups):
    return {key: sorted(item['key'] for item in items) for key, items in groups.items()}


class TestSqliteDuplicateScan(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'zotero.sqlite'
        self.rows = [
            corpus.Item('DUP00001', title='A Study', doi='10.1000/ABC', abstract='Abstract', collections=['COLLA001']),
            corpus.Item('DUP00002', title='A study!', doi='https://doi.org/10.1000/abc', collections=['COLLA001']),
            corpus.Item('TITLE001', title='Title only', collections=['COLLB001']),
            corpus.Item('TITLE002', title='Title only.', collections=['COLLB001']),
            corpus.Item('ATTACH01', item_type='attachment', title='A Study', doi='10.1000/ABC'),
            corpus.Item('NOTE0001', item_type='note', title='A Study', doi='10.1000/ABC'),
            corpus.Item('DELETED1', title='A Study', doi='10.1000/ABC'),
            corpus.Item('GROUP001', title='Group paper', doi='10.1000/GROUP', library_id=corpus.GROUP_LIBRARY_ID),
            corpus.Item('GROUP002', title='Group paper', doi='10.1000/GROUP', library_id=corpus.GROUP_LIBRARY_ID),
            # API records use caseName, not title. Keep the existing grouping policy.
            corpus.Item('CASE0001', item_type='case', title='Case name'),
            corpus.Item('CASE0002', item_type='case', title='Case name'),
        ]

    def open_reader(self, rows=None):
        corpus.build_sqlite(self.path, self.rows if rows is None else rows)
        env = patch.dict(os.environ, {'ZOTERO_MCP_DB_SNAPSHOT': '0'})
        env.start()
        self.addCleanup(env.stop)
        reader = LocalZoteroReader(db_path=str(self.path))
        self.addCleanup(reader.close)
        return reader

    def local_backend(self, reader, group_id=0):
        return library.FallbackBackend(library.SqliteBackend(reader, group_id), Mock(side_effect=AssertionError('HTTP fallback')))

    def test_groups_match_api_for_every_method_and_collection(self):
        reader = self.open_reader()
        api = ApiItems(corpus.build_api_items(self.rows))
        for method in ('doi', 'title', 'both'):
            for collection in (None, 'COLLA001', 'COLLB001'):
                with self.subTest(method=method, collection=collection):
                    with patch.object(library, 'get_library_backend', return_value=library.ApiBackend(api)):
                        expected, error = write._collect_duplicate_groups(api, method, collection)
                    self.assertIsNone(error)
                    no_http = ApiItems([])
                    no_http.items.side_effect = AssertionError('HTTP items')
                    no_http.collection_items.side_effect = AssertionError('HTTP collection')
                    with patch.object(library, 'get_library_backend', return_value=self.local_backend(reader)):
                        actual, error = write._collect_duplicate_groups(no_http, method, collection)
                    self.assertIsNone(error)
                    self.assertEqual(members(actual), members(expected))
                    no_http.items.assert_not_called()
                    no_http.collection_items.assert_not_called()

    def test_local_scan_exceeds_web_ceiling(self):
        rows = self.rows[:2] + [corpus.Item(f'F{i:07d}', title=f'Unique {i}') for i in range(write._DUP_SCAN_MAX_ITEMS + 1)]
        reader = self.open_reader(rows)
        api = ApiItems(corpus.build_api_items(rows))
        with patch.object(library, 'get_library_backend', return_value=self.local_backend(reader)):
            groups, error = write._collect_duplicate_groups(api, 'doi')
        self.assertIsNone(error)
        self.assertEqual(members(groups), {'doi:10.1000/abc': ['DUP00001', 'DUP00002']})
        api.items.assert_not_called()

    def test_api_ceiling_is_preserved(self):
        api = ApiItems(corpus.build_api_items([corpus.Item(f'F{i:07d}') for i in range(write._DUP_SCAN_MAX_ITEMS + 1)]))
        with patch.object(library, 'get_library_backend', return_value=library.ApiBackend(api)):
            groups, error = write._collect_duplicate_groups(api, 'doi')
        self.assertEqual(groups, {})
        self.assertIn('too large', error)

    def test_active_group_does_not_leak_personal_items(self):
        reader = self.open_reader()
        with patch.object(library, 'get_library_backend', return_value=self.local_backend(reader, corpus.GROUP_ID)):
            groups, error = write._collect_duplicate_groups(ApiItems([]), 'doi')
        self.assertIsNone(error)
        self.assertEqual(members(groups), {'doi:10.1000/group': ['GROUP001', 'GROUP002']})

    def test_invalid_or_out_of_scope_collection_is_explicit(self):
        reader = self.open_reader()
        for key, group in [('MISSING1', 0), ('COLLA001', corpus.GROUP_ID)]:
            with self.subTest(key=key, group=group):
                with patch.object(library, 'get_library_backend', return_value=self.local_backend(reader, group)):
                    with self.assertRaisesRegex(ValueError, 'Collection'):
                        write._collect_duplicate_groups(ApiItems([]), 'doi', key)

    def test_auto_merge_keeps_the_same_plan_and_abstract_preference(self):
        reader = self.open_reader()
        api = ApiItems(corpus.build_api_items(self.rows))
        with patch.object(library, 'get_library_backend', return_value=library.ApiBackend(api)):
            expected = write._auto_merge_groups(api, api, 'doi', None, 200)
        with patch.object(library, 'get_library_backend', return_value=self.local_backend(reader)):
            actual = write._auto_merge_groups(api, api, 'doi', None, 200)
        self.assertEqual(actual, expected)
        self.assertEqual(actual[0][0]['keeper_key'], 'DUP00001')

    def test_sqlite_failure_does_not_silently_scan_the_api(self):
        reader = self.open_reader()
        backend = self.local_backend(reader)
        api = ApiItems([])
        with patch.object(reader, 'get_duplicate_scan_items', side_effect=sqlite3.OperationalError('broken database')):
            with patch.object(library, 'get_library_backend', return_value=backend):
                with self.assertRaisesRegex(sqlite3.OperationalError, 'broken database'):
                    write._collect_duplicate_groups(api, 'doi')
        api.items.assert_not_called()

    def test_reader_missing_keeps_bounded_api_fallback(self):
        api = ApiItems(corpus.build_api_items(self.rows))
        with patch.object(library, 'configured_backend', return_value='sqlite'):
            with patch.object(library, '_sqlite_reader', return_value=None):
                groups, error = write._collect_duplicate_groups(api, 'doi')
        self.assertIsNone(error)
        self.assertTrue(groups)
        api.items.assert_called()

    def test_find_tool_reads_sqlite_without_requesting_api_items(self):
        reader = self.open_reader()
        api = ApiItems([])
        api.items.side_effect = AssertionError('Zotero is closed')
        with patch.object(write._client, 'get_zotero_client', return_value=api):
            with patch.object(library, 'get_library_backend', return_value=self.local_backend(reader)):
                output = write.find_duplicates(method='doi', ctx=Mock())
        self.assertIn('Found 1 duplicate groups', output)
        self.assertIn('DUP00001', output)
        self.assertIn('DUP00002', output)
        api.items.assert_not_called()

    @unittest.skipUnless(hasattr(sqlite3.Connection, 'setlimit'), 'requires Python 3.11 SQLite limit API')
    def test_scan_does_not_bind_every_item_id(self):
        reader = self.open_reader(self.rows[:2] + [corpus.Item(f'F{i:07d}') for i in range(30)])
        reader._get_connection().setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 10)
        with patch.object(library, 'get_library_backend', return_value=self.local_backend(reader)):
            groups, error = write._collect_duplicate_groups(ApiItems([]), 'doi')
        self.assertIsNone(error)
        self.assertEqual(members(groups), {'doi:10.1000/abc': ['DUP00001', 'DUP00002']})
