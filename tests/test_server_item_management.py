from zotero_mcp import server


class DummyContext:
    def info(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None

    def warn(self, *_args, **_kwargs):
        return None


class FakeZoteroItemManagement:
    def __init__(self):
        self.collection_names = {
            "A": "Alpha",
            "B": "Beta",
            "C": "Gamma",
            "COLL001": "PhD Research",
            "COLL002": "Reading List",
        }
        self.collection_name_to_key = {
            "phd research": "COLL001",
            "reading list": "COLL002",
        }
        self.items_by_key = {
            "ITEM1": {
                "key": "ITEM1",
                "data": {
                    "title": "Test Item",
                    "itemType": "journalArticle",
                    "collections": ["A", "B"],
                    "tags": [],
                    "creators": [],
                },
            }
        }
        self.add_calls = []
        self.remove_calls = []
        self.updated_items = []
        self.created_items = []

    def item(self, key):
        if key not in self.items_by_key:
            raise KeyError(key)
        return self.items_by_key[key]

    def update_item(self, item):
        self.updated_items.append(item)
        return {"success": True}

    def collections(self):
        output = []
        for name_lower, key in self.collection_name_to_key.items():
            output.append({"key": key, "data": {"name": self.collection_names[key]}})
        return output

    def collection(self, key):
        if key not in self.collection_names:
            raise KeyError(key)
        return {"key": key, "data": {"name": self.collection_names[key]}}

    def addto_collection(self, collection_key, item_keys):
        if collection_key not in self.collection_names:
            return False
        self.add_calls.append((collection_key, list(item_keys)))
        return True

    def deletefrom_collection(self, collection_key, item_keys):
        if collection_key not in self.collection_names:
            return False
        self.remove_calls.append((collection_key, list(item_keys)))
        return True

    def create_items(self, items):
        self.created_items.extend(items)
        return {"success": {"0": "ITEMNEW1"}}


def test_update_item_collections_strict_replace_with_overlap(monkeypatch):
    fake_zot = FakeZoteroItemManagement()
    monkeypatch.setattr(server, "get_zotero_client", lambda: fake_zot)

    result = server.update_item(
        item_key="ITEM1",
        collections=["B", "C"],
        ctx=DummyContext(),
    )

    assert ("A", ["ITEM1"]) in fake_zot.remove_calls
    assert ("C", ["ITEM1"]) in fake_zot.add_calls
    assert ("B", ["ITEM1"]) not in fake_zot.add_calls
    assert "**Removed from collection(s):** Alpha" in result
    assert "**Added to collection(s):** Gamma" in result


def test_update_item_collections_clear_all(monkeypatch):
    fake_zot = FakeZoteroItemManagement()
    monkeypatch.setattr(server, "get_zotero_client", lambda: fake_zot)

    result = server.update_item(
        item_key="ITEM1",
        collections=[],
        ctx=DummyContext(),
    )

    assert ("A", ["ITEM1"]) in fake_zot.remove_calls
    assert ("B", ["ITEM1"]) in fake_zot.remove_calls
    assert fake_zot.add_calls == []
    assert "**Removed from collection(s):** Alpha, Beta" in result


def test_update_item_collections_invalid_key_best_effort(monkeypatch):
    fake_zot = FakeZoteroItemManagement()
    monkeypatch.setattr(server, "get_zotero_client", lambda: fake_zot)

    result = server.update_item(
        item_key="ITEM1",
        collections=["A", "INVALID"],
        ctx=DummyContext(),
    )

    assert ("B", ["ITEM1"]) in fake_zot.remove_calls
    assert ("A", ["ITEM1"]) not in fake_zot.add_calls
    assert "Collection warnings" in result
    assert "INVALID" in result


def test_create_item_accepts_plain_collection_name_string(monkeypatch):
    fake_zot = FakeZoteroItemManagement()
    monkeypatch.setattr(server, "get_zotero_client", lambda: fake_zot)

    result = server.create_item(
        item_type="journalArticle",
        title="Plain Collection Name",
        collection_names="PhD Research",
        ctx=DummyContext(),
    )

    assert "Successfully created journalArticle" in result
    assert ("COLL001", ["ITEMNEW1"]) in fake_zot.add_calls


def test_create_item_accepts_comma_separated_collection_names(monkeypatch):
    fake_zot = FakeZoteroItemManagement()
    monkeypatch.setattr(server, "get_zotero_client", lambda: fake_zot)

    result = server.create_item(
        item_type="journalArticle",
        title="Comma Collections",
        collection_names="PhD Research, Reading List",
        ctx=DummyContext(),
    )

    assert "Successfully created journalArticle" in result
    assert ("COLL001", ["ITEMNEW1"]) in fake_zot.add_calls
    assert ("COLL002", ["ITEMNEW1"]) in fake_zot.add_calls


def test_update_item_rejects_json_scalar_tags(monkeypatch):
    fake_zot = FakeZoteroItemManagement()
    monkeypatch.setattr(server, "get_zotero_client", lambda: fake_zot)

    result = server.update_item(
        item_key="ITEM1",
        tags='"ml"',
        ctx=DummyContext(),
    )

    assert "tags must be a list of strings" in result
    assert fake_zot.updated_items == []


def test_update_item_rejects_json_object_add_tags(monkeypatch):
    fake_zot = FakeZoteroItemManagement()
    monkeypatch.setattr(server, "get_zotero_client", lambda: fake_zot)

    result = server.update_item(
        item_key="ITEM1",
        add_tags='{"not":"a-list"}',
        ctx=DummyContext(),
    )

    assert "add_tags must be a list of strings" in result
    assert fake_zot.updated_items == []


def test_update_item_rejects_json_scalar_remove_tags(monkeypatch):
    fake_zot = FakeZoteroItemManagement()
    monkeypatch.setattr(server, "get_zotero_client", lambda: fake_zot)

    result = server.update_item(
        item_key="ITEM1",
        remove_tags='"ml"',
        ctx=DummyContext(),
    )

    assert "remove_tags must be a list of strings" in result
    assert fake_zot.updated_items == []


def test_update_item_rejects_non_dict_extra_fields(monkeypatch):
    fake_zot = FakeZoteroItemManagement()
    monkeypatch.setattr(server, "get_zotero_client", lambda: fake_zot)

    result = server.update_item(
        item_key="ITEM1",
        extra_fields='["a"]',
        ctx=DummyContext(),
    )

    assert "extra_fields must be a dictionary" in result
    assert fake_zot.updated_items == []


def test_update_item_accepts_plain_string_tags(monkeypatch):
    fake_zot = FakeZoteroItemManagement()
    monkeypatch.setattr(server, "get_zotero_client", lambda: fake_zot)

    result = server.update_item(
        item_key="ITEM1",
        tags="ml",
        ctx=DummyContext(),
    )

    assert "Successfully updated item" in result
    assert len(fake_zot.updated_items) == 1
    assert fake_zot.updated_items[0]["data"]["tags"] == [{"tag": "ml"}]
