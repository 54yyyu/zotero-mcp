"""The CSL JSON importer against Zotero's own CSL crosswalk.

The coverage tests walk the vendored schema: for every Zotero item type the
CSL importer can produce, every CSL text, date and name variable Zotero maps
is fed in at once, and each value must land in the field Zotero would use, or,
when the type has no such field, in Extra. A type or variable a schema update
adds is checked the same way, with no test to write.
"""

import json

import pytest

from zotero_mcp import schema
from zotero_mcp.citation_import import (
    CSL_TYPE_MAP,
    bibtex_entry_to_zotero,
    csl_json_to_zotero,
    parse_bibtex,
)


@pytest.fixture(autouse=True)
def _vendored_table():
    """Resolve from the shipped table, not a developer's on-disk cache."""
    saved = schema._table_cache
    schema._table_cache = schema._load_vendored()
    yield
    schema._table_cache = saved


def schema_template(item_type: str) -> dict:
    return {
        "itemType": item_type,
        **{f: "" for f in schema.valid_fields(item_type)},
        "creators": [],
        "tags": [],
        "collections": [],
        "relations": {},
    }


def _import_type(csl_type: str) -> str:
    return CSL_TYPE_MAP.get(csl_type) or schema.csl_item_type(csl_type) or "document"


CSL_TYPES = sorted({*CSL_TYPE_MAP, *schema._load_vendored()["csl"]["types"]})


def _csl_type_for(item_type: str) -> str:
    return next(c for c in CSL_TYPES if _import_type(c) == item_type)


REACHABLE = sorted({_import_type(c) for c in CSL_TYPES})


@pytest.mark.parametrize("csl_type", sorted(schema._load_vendored()["csl"]["types"]))
def test_every_csl_type_zotero_knows_imports_as_a_type_zotero_maps_it_from(csl_type):
    item = csl_json_to_zotero({"type": csl_type, "title": "T"}, schema_template)

    assert item["itemType"] in schema._load_vendored()["csl"]["types"][csl_type]
DATE = {"date-parts": [[2001, 2, 3]]}


def _everything(csl_type: str) -> dict:
    csl = {"type": csl_type}
    csl.update({var: f"v-{var}" for var in schema.csl_text_fields() if var != "note"})
    csl.update({var: DATE for var in schema.csl_date_fields()})
    csl.update({var: [{"family": f"F-{var}", "given": "G"}] for var in schema.csl_name_variables()})
    return csl


def _extra_lines(item: dict) -> set[str]:
    return set((item.get("extra") or "").splitlines())


@pytest.mark.parametrize("item_type", REACHABLE)
def test_every_text_variable_lands_in_its_field_or_in_extra(item_type):
    item = csl_json_to_zotero(_everything(_csl_type_for(item_type)), schema_template)
    fields = {k: v for k, v in item.items() if isinstance(v, str) and k not in ("extra", "itemType")}

    for var, bases in schema.csl_text_fields().items():
        if var == "note":
            continue
        value = f"v-{var}"
        if value in fields.values():
            continue
        targets = [schema.resolve_field(item_type, b) for b in bases]
        targets = [t for t in targets if t in fields and t != "extra"]
        # Not in a field: only right when every field it could fill already
        # holds another variable's value, and then it must be kept in Extra.
        assert all(fields[t] for t in targets), f"{var} not written although {targets} were free"
        assert f"{var}: {value}" in _extra_lines(item), f"{var} lost for {item_type}"


@pytest.mark.parametrize("item_type", REACHABLE)
def test_every_date_variable_lands_in_its_field_or_in_extra(item_type):
    item = csl_json_to_zotero(_everything(_csl_type_for(item_type)), schema_template)

    for var, base in schema.csl_date_fields().items():
        field = schema.resolve_field(item_type, base)
        if field in schema.valid_fields(item_type):
            assert item[field] == "2001-02-03", f"{var} -> {field} for {item_type}"
        else:
            assert f"{var}: {json.dumps(DATE)}" in _extra_lines(item), f"{var} lost for {item_type}"


@pytest.mark.parametrize("item_type", REACHABLE)
def test_every_name_variable_becomes_a_valid_creator_or_stays_in_extra(item_type):
    item = csl_json_to_zotero(_everything(_csl_type_for(item_type)), schema_template)
    by_name = {c["lastName"]: c["creatorType"] for c in item["creators"]}
    valid = schema.creator_types(item_type)

    assert set(by_name.values()) <= set(valid), f"invalid creator type for {item_type}"
    for var in schema.csl_name_variables():
        expected = schema.creator_type_for(item_type, var) or (valid[0] if var == "author" else None)
        if expected:
            assert by_name.get(f"F-{var}") == expected, f"{var} for {item_type}"
        else:
            assert any(line.startswith(f"{var}: ") for line in _extra_lines(item)), f"{var} lost for {item_type}"


# ---------------------------------------------------------------------------
# The cases that showed the gap
# ---------------------------------------------------------------------------

def test_csl_standard_keeps_its_number_and_issuing_body():
    item = csl_json_to_zotero(
        {"type": "standard", "title": "Level of Information Need. Part 1", "number": "EN ISO 7817-1:2024",
         "authority": "CEN", "issued": {"date-parts": [[2024, 7, 12]]}},
        schema_template,
    )

    assert item["itemType"] == "standard"
    assert item["number"] == "EN ISO 7817-1:2024"
    assert item["organization"] == "CEN"
    assert "authority" not in item["extra"]


def test_csl_report_number_still_goes_to_report_number():
    item = csl_json_to_zotero({"type": "report", "title": "R", "number": "TR-7"}, schema_template)

    assert item["reportNumber"] == "TR-7"


def test_csl_variable_the_type_has_no_field_for_is_kept_in_extra():
    item = csl_json_to_zotero({"type": "standard", "title": "S", "volume": "3"}, schema_template)

    assert "volume: 3" in _extra_lines(item)


def test_csl_book_section_container_author_is_the_book_author():
    item = csl_json_to_zotero(
        {"type": "chapter", "title": "C", "container-author": [{"family": "Ed", "given": "B"}]}, schema_template
    )

    assert item["creators"] == [{"creatorType": "bookAuthor", "firstName": "B", "lastName": "Ed"}]


def test_csl_accessed_date_is_the_access_date():
    item = csl_json_to_zotero(
        {"type": "webpage", "title": "W", "accessed": {"date-parts": [[2026, 11, 16]]}}, schema_template
    )

    assert item["accessDate"] == "2026-11-16"


def _bibtex(text: str) -> dict:
    return bibtex_entry_to_zotero(parse_bibtex(text)[0], schema_template)


def test_bibtex_standard_is_a_standard_with_its_number_and_organization():
    item = _bibtex("@standard{iso, title={Level of Information Need}, number={EN ISO 7817-1:2024}, "
                   "organization={CEN}, year={2024}}")

    assert item["itemType"] == "standard"
    assert item["number"] == "EN ISO 7817-1:2024"
    assert item["organization"] == "CEN"


def test_bibtex_report_number_goes_to_report_number_and_article_number_to_issue():
    assert _bibtex("@techreport{r, title={R}, number={TR-7}, institution={I}, year={2020}}")["reportNumber"] == "TR-7"
    assert _bibtex("@article{a, title={A}, journal={J}, number={4}, year={2020}}")["issue"] == "4"


# ---------------------------------------------------------------------------
# The crosswalk travels with the schema table
# ---------------------------------------------------------------------------

def test_a_cache_written_before_the_crosswalk_falls_back_to_the_vendored_one():
    schema._table_cache = {"version": 1, "itemTypes": schema._load_vendored()["itemTypes"]}

    assert schema.csl_text_fields()["number"]
    assert schema.creator_type_for("bookSection", "container-author") == "bookAuthor"


def test_refresh_builds_the_crosswalk_from_the_schema_document():
    doc = {
        "version": 99,
        "itemTypes": [{"itemType": "standard", "fields": [{"field": "organization", "baseField": "authority"}],
                       "creatorTypes": [{"creatorType": "author", "primary": True}]}],
        "csl": {"fields": {"text": {"authority": ["authority"]}, "date": {}}, "names": {"author": "author"}},
    }

    table = schema._build_table(doc)

    assert table["creatorTypes"] == {"standard": ["author"]}
    assert table["csl"]["fields"]["text"] == {"authority": ["authority"]}
