"""Search tools supporting multiple providers."""

import os

from rich.table import Table

from tavily import TavilyClient
from multi_search_api import SmartSearchTool

from ..ui.console import console


class HybridSearchTool:
    """Supports both Tavily and Multi-Search API providers."""

    def __init__(self, provider="multi-search"):
        self.provider = provider
        self.provider_usage = {}  # Track which providers were used

        # Initialize clients
        if provider in ["tavily", "auto"]:
            self.tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

        if provider in ["multi-search", "auto"]:
            self.multi_search = SmartSearchTool(
                serper_api_key=os.getenv("SERPER_API_KEY"),
                brave_api_key=os.getenv("BRAVE_API_KEY"),
                enable_cache=True,  # Thread-safe since multi-search-api v0.1.1
                quiet=True,  # Suppress all logging output for clean UI
            )

    def normalize_multi_search_response(self, response):
        """Convert multi-search-api format to Tavily-compatible format."""
        return {
            "query": response.get("query", ""),
            "results": [
                {
                    "title": result.get("title", ""),
                    "content": result.get("snippet", ""),  # snippet -> content
                    "url": result.get("link", ""),  # link -> url
                    "score": 0.9,  # Multi-search doesn't provide scores
                }
                for result in response.get("results", [])
            ],
            "_provider": response.get("provider", "unknown"),
            "_cache_hit": response.get("cache_hit", False),
        }

    def search(self, query: str, max_results: int = 7, topic: str = "general"):
        """Execute search with selected provider."""

        if self.provider == "tavily":
            result = self.tavily.search(query, max_results=max_results, topic=topic)
            self.provider_usage["Tavily"] = self.provider_usage.get("Tavily", 0) + 1
            result["_actual_provider"] = "Tavily"
            return result

        elif self.provider == "multi-search":
            response = self.multi_search.search(query=query, num_results=max_results)
            normalized = self.normalize_multi_search_response(response)
            provider_name = normalized.get("_provider", "Unknown")
            self.provider_usage[provider_name] = (
                self.provider_usage.get(provider_name, 0) + 1
            )
            normalized["_actual_provider"] = provider_name
            return normalized

        elif self.provider == "auto":
            # Try multi-search first (free), fallback to Tavily
            try:
                response = self.multi_search.search(
                    query=query, num_results=max_results
                )
                normalized = self.normalize_multi_search_response(response)
                provider_name = normalized.get("_provider", "Unknown")
                self.provider_usage[provider_name] = (
                    self.provider_usage.get(provider_name, 0) + 1
                )
                normalized["_actual_provider"] = provider_name
                return normalized
            except Exception as e:
                console.print(
                    f"[yellow]Multi-search failed, using Tavily: {e}[/yellow]"
                )
                result = self.tavily.search(query, max_results=max_results, topic=topic)
                self.provider_usage["Tavily (fallback)"] = (
                    self.provider_usage.get("Tavily (fallback)", 0) + 1
                )
                result["_actual_provider"] = "Tavily (fallback)"
                return result

    def clear_cache(self):
        """Clear the search cache (multi-search only)."""
        if self.provider in ["multi-search", "auto"] and hasattr(self, "multi_search"):
            self.multi_search.clear_cache()
            console.print("[green][/green] Cache cleared successfully")
        else:
            console.print("[yellow]Cache not available for this provider[/yellow]")

    def get_cache_stats(self):
        """Get cache statistics (multi-search only)."""
        if self.provider in ["multi-search", "auto"] and hasattr(self, "multi_search"):
            status = self.multi_search.get_status()
            cache_info = status.get("cache", {})
            return {
                "total_entries": cache_info.get("total_entries", 0),
                "oldest_entry": cache_info.get("oldest_entry", "N/A"),
                "newest_entry": cache_info.get("newest_entry", "N/A"),
                "cache_file": cache_info.get("cache_file", "N/A"),
            }
        return None

    def display_cache_stats(self):
        """Display cache statistics in a nice format."""
        stats = self.get_cache_stats()
        if stats is None:
            console.print(
                "[yellow]Cache stats not available for this provider[/yellow]"
            )
            return

        table = Table(title="Cache Statistics", show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Entries", str(stats["total_entries"]))
        table.add_row("Oldest Entry", stats["oldest_entry"])
        table.add_row("Newest Entry", stats["newest_entry"])
        table.add_row("Cache Location", stats["cache_file"])

        console.print(table)
