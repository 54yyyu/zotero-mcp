"""
Helper for accessing Zotero via Better BibTeX JSON-RPC API.
Provides direct access to Zotero's annotations without requiring PDF extraction.
"""

import json
import requests
import os
import sys
from typing import Dict, Any, List, Optional

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

    def _make_request(self, method: str, params: list[Any]) -> dict[str, Any]:
        """
        Make a JSON-RPC request to the Zotero API.

        Args:
            method: The JSON-RPC method to call
            params: The parameters for the method

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
        except:
            return False

    def get_item_by_citekey(self, citekey: str) -> dict[str, Any]:
        """
        Get item data by citation key.

        Args:
            citekey: The citation key of the item

        Returns:
            The item data
        """
        # First, search for the item to get its ID and library ID
        search_results = self._make_request("item.search", [citekey])

        if not search_results:
            raise Exception(f"No items found with citekey: {citekey}")

        item = next((item for item in search_results if item.get('citekey') == citekey), None)

        if not item:
            raise Exception(f"No exact match found for citekey: {citekey}")

        library_id = item.get('libraryID')

        # Now export the full item data
        try:
            export_result = self._make_request(
                "item.export",
                [[citekey], "36a3b0b5-bad0-4a04-b79b-441c7cef77db", library_id]
            )

            if not export_result:
                raise Exception(f"Failed to export item data for citekey: {citekey}")

            # The result might be an array or a string depending on the Better BibTeX version
            if isinstance(export_result, list):
                if len(export_result) > 2 and export_result[2]:
                    try:
                        return json.loads(export_result[2]).get('items', [])[0]
                    except:
                        # Try to use the first element if it's a string
                        if isinstance(export_result[0], str):
                            return json.loads(export_result[0]).get('items', [])[0]
            elif isinstance(export_result, str):
                return json.loads(export_result).get('items', [])[0]
            elif isinstance(export_result, dict) and 'items' in export_result:
                return export_result.get('items', [])[0]

            # Fall back to using the search result
            return item

        except Exception as e:
            print(f"Warning: Could not export full item data: {e}")
            # Return basic item data from search
            return item

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

            # Step 1: Get citation key from item key
            item_keys = [f"{library_id}:{item_key}"]
            citation_mapping = self._make_request("item.citationkey", [item_keys])

            if not citation_mapping:
                raise Exception(f"No citation key found for item: {item_key}")

            # Step 2: Extract citation key from mapping
            full_item_key = f"{library_id}:{item_key}"
            citation_key = citation_mapping.get(full_item_key)

            if not citation_key:
                raise Exception(f"Citation key not found for item: {item_key}")

            # Step 3: Export BibTeX using citation key
            export_result = self._make_request(
                "item.export",
                [[citation_key], translator_id]
            )

            # Handle different response formats
            if isinstance(export_result, str):
                return export_result
            elif isinstance(export_result, list) and len(export_result) > 0:
                # Sometimes the result is wrapped in an array
                return export_result[0] if isinstance(export_result[0], str) else str(export_result[0])
            elif isinstance(export_result, dict) and 'bibtex' in export_result:
                return export_result['bibtex']
            else:
                return str(export_result)

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
            except:
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
