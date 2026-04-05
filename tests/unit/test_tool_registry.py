from __future__ import annotations

from src.models.schemas import ToolCategory
from src.tools.registry import ToolDefinition, ToolRegistry


class TestToolRegistry:
    def test_register_and_get(self):
        reg = ToolRegistry()
        tool_def = ToolDefinition(
            name="test_tool",
            description="A test tool",
            category=ToolCategory.MARKET_DATA,
            handler=lambda x: x + 1,
        )
        reg.register(tool_def)
        tool = reg.get("test_tool")
        assert tool.name == "test_tool"

    def test_list_tools(self):
        reg = ToolRegistry()
        reg.register(ToolDefinition(name="t1", description="d1", category=ToolCategory.MARKET_DATA, handler=lambda: None))
        reg.register(ToolDefinition(name="t2", description="d2", category=ToolCategory.NEWS, handler=lambda: None))
        assert len(reg.list_tools()) == 2

    def test_get_schemas(self):
        reg = ToolRegistry()
        reg.register(ToolDefinition(
            name="schema_tool",
            description="desc",
            category=ToolCategory.INDICATOR,
            handler=lambda x: x,
            input_schema={"type": "object", "properties": {"x": {"type": "integer"}}},
        ))
        schemas = reg.get_schemas()
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "schema_tool"

    def test_get_nonexistent(self):
        reg = ToolRegistry()
        try:
            reg.get("nope")
            assert False, "Should have raised KeyError"
        except KeyError:
            pass

    def test_filter_by_category(self):
        reg = ToolRegistry()
        reg.register(ToolDefinition(name="a", description="a", category=ToolCategory.MARKET_DATA, handler=lambda: None))
        reg.register(ToolDefinition(name="b", description="b", category=ToolCategory.INDICATOR, handler=lambda: None))
        schemas = reg.get_schemas_for_categories([ToolCategory.MARKET_DATA])
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "a"
