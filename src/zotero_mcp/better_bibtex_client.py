"""
Helper for accessing Zotero via Better BibTeX JSON-RPC API.
Provides direct access to Zotero's annotations without requiring PDF extraction.
"""

import json
import os
import re
from typing import Any

import requests

# Matches the opening ``@type{`` line of a BibTeX entry where the citekey is
# either absent or followed immediately by the comma + newline that a
# missing-citekey entry produces. Used to inject the citekey when BBT's
# ``item.export`` strips it from the output (#293).
_EMPTY_CITEKEY_LINE = re.compile(r"(@[A-Za-z]+\s*\{)(\s*,?\s*\n)")


def _inject_citekey(bibtex_str: str, citation_key: str) -> str:
    """Insert *citation_key* on the @-line of the first BibTeX entry if missing.

    Only touches entries whose @-line is empty (``@article{``) or has a bare
    leading comma (``@article{,``); an entry that already carries a key is
    left alone. Safe to call repeatedly.
    """
    if not citation_key or not bibtex_str:
        return bibtex_str

    def _replace(match: re.Match) -> str:
        return f"{match.group(1)}{citation_key},\n"

    return _EMPTY_CITEKEY_LINE.sub(_replace, bibtex_str, count=1)

class ZoteroBetterBibTexAPI:
    """Class to interact with Zotero's local Better BibTeX JSON-RPC API"""

    def __init__(self, port="23119", database="Zotero"):
        """
        Initialize the API connection.

        Args:
            port: The port number Zotero is running on (default: 23119 for Zotero, 24119 for Juris-M)
            database: The database type ('Zotero' or 'Juris-M')
        """
        self.port = port
        if database == "Juris-M":
            self.port = "24119"

        self.base_url = f"http://127.0.0.1:{self.port}/better-bibtex/json-rpc"
        self.headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'python/zotero-mcp',
            'Accept': 'application/json',
            'Connection': 'keep-alive',
        }

    def _make_request(self, method: str, params: list[Any] | dict[str, Any]) -> dict[str, Any]:
        """
        Make a JSON-RPC request to the Better BibTeX API.

        Args:
            method: The JSON-RPC method to call (e.g. ``item.citationkey``).
            params: Either a positional list or a named-parameter dict. BBT's
                JSON-RPC handler expects *named* params for most methods —
                e.g. ``{"item_keys": [...]}`` rather than ``[[...]]``. See
                https://retorque.re/zotero-better-bibtex/exporting/json-rpc/.

        Returns:
            The response data
        """
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": 1  # Adding an ID to the request
        }

        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                data=json.dumps(payload),
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                error_msg = str(data['error'].get('message', 'Unknown error'))
                error_data = data['error'].get('data', '')
                if error_data:
                    error_msg += f": {error_data}"
                raise Exception(f"API error: {error_msg}")

            return data.get("result", {})

        except requests.exceptions.RequestException as e:
            raise Exception(f"Connection error: {str(e)}. Is Zotero running with Better BibTeX installed?")

    def is_zotero_running(self) -> bool:
        """Check if Zotero is running and accessible."""
        try:
            response = requests.get(
                f"http://127.0.0.1:{self.port}/better-bibtex/cayw?probe=true",
                headers=self.headers,
                timeout=5
            )
            return response.text == "ready"
        except Exception:
            return False

    def get_item_by_citekey(self, citekey: str) -> dict[str, Any]:
        """
        Export the CSL-JSON item data for a citation key.

        Uses ``item.export`` directly with the CSL-JSON translator. The
        previous implementation called ``item.search`` to discover the
        item key first, but that BBT JSON-RPC method does not exist in
        current versions and always returned -32601 Method not found.
        Going straight to ``item.export`` skips the broken probe.

        Args:
            citekey: The citation key of the item

        Returns:
            The item data (CSL-JSON dict)
        """
        csl_json_translator = "36a3b0b5-bad0-4a04-b79b-441c7cef77db"
        try:
            export_result = self._make_request(
                "item.export", [[citekey], csl_json_translator]
            )
        except Exception as e:
            raise Exception(f"Could not export item for citekey {citekey}: {e}")

        if not export_result:
            raise Exception(f"Failed to export item data for citekey: {citekey}")

        # The result shape varies across Better BibTeX versions.
        payload: str | None = None
        if isinstance(export_result, list):
            if len(export_result) > 2 and isinstance(export_result[2], str):
                payload = export_result[2]
            elif export_result and isinstance(export_result[0], str):
                payload = export_result[0]
        elif isinstance(export_result, str):
            payload = export_result
        elif isinstance(export_result, dict) and "items" in export_result:
            items = export_result.get("items") or []
            if items:
                return items[0]

        if payload is None:
            raise Exception(f"Unexpected item.export response shape: {type(export_result).__name__}")

        try:
            items = json.loads(payload).get("items", [])
        except (json.JSONDecodeError, AttributeError) as e:
            raise Exception(f"Could not parse item.export payload for {citekey}: {e}")
        if not items:
            raise Exception(f"No items returned for citekey: {citekey}")
        return items[0]

    def get_attachments(self, citekey: str, library_id: int) -> list[dict[str, Any]]:
        """
        Get all attachments for an item.

        Args:
            citekey: The citation key of the item
            library_id: The library ID

        Returns:
            A list of attachment data
        """
        try:
            return self._make_request("item.attachments", [citekey, library_id])
        except Exception as e:
            print(f"Warning: Could not get attachments: {e}")
            return []

    def get_annotations_from_attachment(self, attachment: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Extract annotations from an attachment.

        Args:
            attachment: The attachment data

        Returns:
            A list of annotations
        """
        # Return empty list if attachment has no annotations
        if not attachment.get('annotations'):
            return []

        return attachment.get('annotations', [])

    def search_citekeys(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """
        Search for items in Zotero by a search query and return their citation keys.

        Args:
            query: Search term to find items
            limit: Maximum number of results to return (default: 10)

        Returns:
            A list of dictionaries containing cite keys and basic item information
        """
        try:
            # Use the general item.search method with the query
            search_results = self._make_request("item.search", [query])

            # If no results found, return empty list
            if not search_results:
                return []

            # Process and filter results
            cite_key_results = []
            for item in search_results[:limit]:
                # Ensure we have a cite key
                if item.get('citekey'):
                    cite_key_results.append({
                        'citekey': item['citekey'],
                        'title': item.get('title', 'No Title'),
                        'creators': item.get('creators', []),
                        'year': item.get('year', 'N/A'),
                        'libraryID': item.get('libraryID')
                    })

            return cite_key_results

        except Exception as e:
            print(f"Error searching for cite keys: {e}")
            return []

    def export_bibtex(self, item_key: str, library_id: int = 1) -> str:
        """
        Export BibTeX for a specific item using its item key.

        Args:
            item_key: Zotero item key to export
            library_id: Library ID (default: 1 = Personal Library)

        Returns:
            BibTeX formatted string
        """
        try:
            # Better BibTeX translator ID for BibTeX export
            translator_id = "ca65189f-8815-4afe-8c8b-8c7c15f0edca"  # Better BibTeX

            # Step 1: Get citation key from item key. BBT's JSON-RPC expects
            # named params (``{"item_keys": [...]}``) and bare item keys —
            # not the ``library_id:item_key`` form — see #293.
            citation_mapping = self._make_request(
                "item.citationkey", {"item_keys": [item_key]}
            )

            if not citation_mapping:
                raise Exception(f"No citation key found for item: {item_key}")

            citation_key = citation_mapping.get(item_key)

            if not citation_key:
                raise Exception(f"Citation key not found for item: {item_key}")

            # Step 2: Export BibTeX using the citation key.
            export_result = self._make_request(
                "item.export",
                [[citation_key], translator_id]
            )

            # Handle different response formats
            if isinstance(export_result, str):
                bibtex_str = export_result
            elif isinstance(export_result, list) and len(export_result) > 0:
                # Sometimes the result is wrapped in an array
                bibtex_str = (
                    export_result[0]
                    if isinstance(export_result[0], str)
                    else str(export_result[0])
                )
            elif isinstance(export_result, dict) and 'bibtex' in export_result:
                bibtex_str = export_result['bibtex']
            else:
                bibtex_str = str(export_result)

            # BBT's ``item.export`` omits the citekey from the @-line in some
            # versions (#293 Bug 2) — entries come back as ``@article{`` with
            # an empty key. Inject the citekey we already resolved above.
            return _inject_citekey(bibtex_str, citation_key)

        except Exception as e:
            print(f"Error exporting BibTeX: {e}")
            return ""


def process_annotation(annotation: dict[str, Any], attachment: dict[str, Any], format_type: str = 'markdown') -> dict[str, Any]:
    """
    Process a raw Zotero annotation into a more usable format.

    Args:
        annotation: The raw annotation data from Zotero
        attachment: The attachment this annotation belongs to
        format_type: Output format (raw or markdown)

    Returns:
        A processed annotation object
    """
    try:
        annotation_type = annotation.get('annotationType', 'unknown')
        color = annotation.get('annotationColor', '')

        # Extract text content
        text = annotation.get('annotationText', '')
        comment = annotation.get('annotationComment', '')

        # Handle page information
        page_label = annotation.get('annotationPageLabel', '1')
        page = 1

        # Get position data
        position = annotation.get('annotationPosition', {})

        if isinstance(position, str):
            try:
                position = json.loads(position)
            except (json.JSONDecodeError, ValueError):
                position = {}

        if position:
            # Get page index if available
            if 'pageIndex' in position:
                page = position['pageIndex'] + 1

            # Get coordinates if available
            if 'rects' in position and position['rects'] and len(position['rects'][0]) >= 2:
                x, y = position['rects'][0][0], position['rects'][0][1]
            else:
                x, y = 0, 0
        else:
            x, y = 0, 0

        # Create result object
        result = {
            'id': annotation.get('key', ''),
            'type': annotation_type,
            'color': color,
            'annotatedText': text,
            'comment': comment,
            'page': page,
            'pageLabel': page_label,
            'x': x,
            'y': y,
            'date': annotation.get('dateModified', ''),
            'attachment': {
                'key': attachment.get('itemKey', ''),
                'filename': os.path.basename(attachment.get('path', '')),
                'title': attachment.get('title', 'PDF'),
                'path': attachment.get('path', ''),
            }
        }

        # If markdown format is requested, format the output
        if format_type == 'markdown':
            result['markdown'] = format_annotation_markdown(result)

        return result

    except Exception as e:
        print(f"Error processing annotation: {e}")
        return {}

def format_annotation_markdown(annotation: dict[str, Any]) -> str:
    """
    Format an annotation as markdown.

    Args:
        annotation: The processed annotation object

    Returns:
        A markdown string representing the annotation
    """
    md = []

    # Format the citation with text and page number
    if annotation['annotatedText']:
        color_str = f" {annotation['color']}" if annotation['color'] else ""
        md.append(f"> \"{annotation['annotatedText']}\"{color_str} {annotation['type'].capitalize()} [Page {annotation['pageLabel']}]")

    # Add the comment if available
    if annotation['comment']:
        md.append(f"\n{annotation['comment']}")

    return "\n".join(md)

def get_color_category(hex_color: str) -> str:
    """
    Get a color category name from a hex color code.

    Args:
        hex_color: The hex color code

    Returns:
        A color category name
    """
    # Simple implementation based on common annotation colors
    color_map = {
        "#ffd400": "Yellow",
        "#ff6666": "Red",
        "#5fb236": "Green",
        "#2ea8e5": "Blue",
        "#a28ae5": "Purple",
        "#e56eee": "Magenta",
        "#f19837": "Orange",
        "#aaaaaa": "Gray"
    }

    return color_map.get(hex_color.lower(), "")
