"""Creator display names must not gain separators for empty name components."""

import pytest

from zotero_mcp.utils import format_creators, format_item_result


@pytest.mark.parametrize(
    ("creators", "expected"),
    [
        pytest.param([], "No authors listed", id="no-creators"),
        pytest.param([{"name": "中川武"}], "中川武", id="literal-person"),
        pytest.param([{"name": "京都林泉協会"}], "京都林泉協会", id="literal-institution"),
        pytest.param(["Doe, Jane"], "Doe, Jane", id="plain-string"),
        pytest.param([{"firstName": "Jane", "lastName": "Doe"}], "Doe, Jane", id="two-part-name"),
        pytest.param([{"firstName": "", "lastName": "中川武"}], "中川武", id="empty-first-name"),
        pytest.param([{"firstName": "Plato", "lastName": ""}], "Plato", id="empty-last-name"),
        pytest.param([{"lastName": "中川武"}], "中川武", id="missing-first-name"),
        pytest.param([{"firstName": "Plato"}], "Plato", id="missing-last-name"),
        pytest.param([{"firstName": "", "lastName": ""}], "No authors listed", id="empty-components"),
        pytest.param([{"name": ""}], "No authors listed", id="empty-literal-name"),
        pytest.param([""], "No authors listed", id="empty-string"),
        pytest.param([{}], "No authors listed", id="missing-name-fields"),
        pytest.param(
            [{"firstName": "Jane", "lastName": "Doe", "name": "Other Name"}],
            "Doe, Jane",
            id="two-part-name-precedence",
        ),
        pytest.param(
            [{"firstName": "", "lastName": "", "name": "京都林泉協会"}],
            "京都林泉協会",
            id="literal-name-with-empty-components",
        ),
        pytest.param(
            [{"firstName": "Jane", "lastName": "Doe, Jr."}],
            "Doe, Jr., Jane",
            id="preserve-name-punctuation",
        ),
        pytest.param(
            [
                "",
                {"firstName": "", "lastName": "中川武"},
                {"name": ""},
                {"firstName": "", "lastName": ""},
                {"firstName": "Jane", "lastName": "Doe"},
                {"name": "京都林泉協会"},
                "Smith, John",
                {},
            ],
            "中川武; Doe, Jane; 京都林泉協会; Smith, John",
            id="mixed-creators-with-empty-entries",
        ),
    ],
)
def test_format_creators(creators, expected):
    assert format_creators(creators) == expected


def test_item_summary_formats_creator_without_changing_other_punctuation():
    item = {
        "key": "TEST1234",
        "data": {
            "title": "Space, Memory, Language",
            "creators": [{"creatorType": "author", "firstName": "", "lastName": "中川武"}],
            "abstractNote": "A house, a garden, a memory.",
        },
    }

    lines = format_item_result(item)

    assert [line for line in lines if line.startswith("**Authors:**")] == ["**Authors:** 中川武"]
    assert "## Space, Memory, Language" in lines
    assert "**Abstract:** A house, a garden, a memory." in lines
