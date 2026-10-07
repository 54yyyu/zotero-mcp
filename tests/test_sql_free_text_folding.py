"""Free-text search on the SQLite backend folds case and accents like Zotero.

Zotero's own quick search (the pyzotero path, ``items(q=...)``) matches
``índice``, ``Índice`` and ``indice`` alike. ``search_items_sql`` compared raw
values with SQLite's ``LIKE``, which folds ASCII case only, so a lowercase
query missed a capitalised accented title and ``SUCESIÓN`` missed
``sucesión``. Advanced search already folds both sides through
``zsearch_norm`` (#417); free-text search now does too.
"""

import pytest
from _search_corpus import CORPUS, Item, build_sqlite

from zotero_mcp.local_db import LocalZoteroReader

EXTRA = [
    Item("SPAIN001", title="Sudáfrica - Índice de precios de consumo", date="2022"),
    Item("SPAIN002", title="VENTA DE UNIDAD PRODUCTIVA Y SUCESIÓN DE EMPRESA", date="2021"),
    Item("SPAIN003", title="La sucesión de empresa", date="2020",
         abstract="Estudio sobre ÉTICA laboral", tags=["Política"]),
]


@pytest.fixture
def reader(tmp_path):
    db_path = tmp_path / "zotero.sqlite"
    build_sqlite(db_path, CORPUS + EXTRA)
    r = LocalZoteroReader(db_path=str(db_path))
    yield r
    r.close()


def _keys(reader, query, qmode="titleCreatorYear"):
    results = reader.search_items_sql(query, qmode=qmode, group_id=0)
    assert results is not None
    return {r["key"] for r in results}


@pytest.mark.parametrize("query", ["índice", "Índice", "ÍNDICE", "indice", "INDICE"])
def test_title_matches_whatever_the_case_and_accents(reader, query):
    assert "SPAIN001" in _keys(reader, query)


@pytest.mark.parametrize("query", ["sucesión", "SUCESIÓN", "Sucesión", "sucesion"])
def test_uppercase_and_lowercase_accented_titles_both_match(reader, query):
    assert {"SPAIN002", "SPAIN003"} <= _keys(reader, query)


@pytest.mark.parametrize("query", ["müller", "MÜLLER", "Müller", "muller"])
def test_creator_matches_whatever_the_case_and_accents(reader, query):
    assert {"ACCENT01", "ACCENT02", "ACCENT03"} <= _keys(reader, query)


def test_umlaut_expansion_still_matches(tmp_path):
    # _generate_search_variants also turns Müller into Mueller, so an
    # accented query still finds a name stored in its expanded spelling.
    db_path = tmp_path / "zotero.sqlite"
    build_sqlite(db_path, [Item("MUELLER1", title="Expanded", creators=[("Jo", "Mueller")])])
    r = LocalZoteroReader(db_path=str(db_path))
    try:
        assert _keys(r, "MÜLLER") == {"MUELLER1"}
    finally:
        r.close()


@pytest.mark.parametrize("query", ["ética", "ÉTICA", "etica"])
def test_everything_mode_folds_abstracts(reader, query):
    assert "SPAIN003" in _keys(reader, query, qmode="everything")


@pytest.mark.parametrize("query", ["política", "POLÍTICA", "politica"])
def test_everything_mode_folds_tags(reader, query):
    assert "SPAIN003" in _keys(reader, query, qmode="everything")


def test_like_metacharacters_stay_literal(reader):
    assert _keys(reader, "50%") == {"PCTSIGN1"}
    assert _keys(reader, "Underscore_Separated") == {"PCTSIGN2"}


@pytest.mark.parametrize("query", ["\U0001F600", "   "])
def test_query_that_folds_to_nothing_matches_nothing(reader, query):
    # unidecode drops an emoji, and LIKE '%%' would have matched every item.
    assert reader.search_items_sql(query, qmode="everything", group_id=0) == []
