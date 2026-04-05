from __future__ import annotations

import functools
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from src.models.schemas import ToolCategory


@dataclass
class ToolDefinition:
    name: str
    description: str
    category: ToolCategory
    handler: Callable[..., Any | Coroutine[Any, Any, Any]]
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    is_async: bool = False


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool_def: ToolDefinition) -> None:
        if tool_def.name in self._tools:
            raise ValueError(f"Tool '{tool_def.name}' is already registered")
        self._tools[tool_def.name] = tool_def

    def get(self, name: str) -> ToolDefinition:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in registry")
        return self._tools[name]

    def list_tools(self, category: ToolCategory | None = None) -> list[ToolDefinition]:
        if category is None:
            return list(self._tools.values())
        return [t for t in self._tools.values() if t.category == category]

    def get_schemas(self) -> list[dict[str, Any]]:
        schemas = []
        for tool in self._tools.values():
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema or {"type": "object", "properties": {}},
                },
            })
        return schemas

    def get_schemas_for_categories(
        self, categories: list[ToolCategory]
    ) -> list[dict[str, Any]]:
        schemas = []
        for tool in self._tools.values():
            if tool.category in categories:
                schemas.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema
                        or {"type": "object", "properties": {}},
                    },
                })
        return schemas

    async def execute(self, name: str, **kwargs: Any) -> Any:
        tool = self.get(name)
        start = time.perf_counter()
        try:
            if tool.is_async:
                result = await tool.handler(**kwargs)
            else:
                result = tool.handler(**kwargs)
            return result
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            _ = elapsed  # available for tracing integration

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


# Global registry instance
_global_registry = ToolRegistry()


def get_global_registry() -> ToolRegistry:
    return _global_registry


def register_tool(
    name: str,
    description: str,
    category: ToolCategory,
    input_schema: dict[str, Any] | None = None,
    output_schema: dict[str, Any] | None = None,
) -> Callable:
    def decorator(func: Callable) -> Callable:
        import asyncio

        is_async = asyncio.iscoroutinefunction(func)
        tool_def = ToolDefinition(
            name=name,
            description=description,
            category=category,
            handler=func,
            input_schema=input_schema or {},
            output_schema=output_schema or {},
            is_async=is_async,
        )
        _global_registry.register(tool_def)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            return await func(*args, **kwargs)

        wrapper._tool_definition = tool_def  # type: ignore[attr-defined]
        async_wrapper._tool_definition = tool_def  # type: ignore[attr-defined]

        return async_wrapper if is_async else wrapper

    return decorator
