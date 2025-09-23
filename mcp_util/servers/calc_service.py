from fastmcp import FastMCP
from config.loader import settings
import ast

mcp = FastMCP("CalcService", stateless_http=True, json_response=True)

# Safe expression evaluator using ast
ALLOWED_NODES = {
    'Expression','BinOp','UnaryOp','Constant','Num',
    'Add','Sub','Mult','Div','Pow','Mod','UAdd','USub','Load','Tuple','List'
}

class SafeEval(ast.NodeVisitor):
    def generic_visit(self, node):
        nodename = type(node).__name__
        if nodename not in ALLOWED_NODES:
            raise ValueError(f"Disallowed expression element: {nodename}")
        super().generic_visit(node)

@mcp.tool()
def calc(expr: str) -> dict:
    """
    Evaluate a simple arithmetic expression safely and return result.
    Supports +,-,*,/,**,%, unary +/-. No function calls.
    """
    try:
        tree = ast.parse(expr, mode='eval')
        SafeEval().visit(tree)
        code = compile(tree, filename="<ast>", mode="eval")
        result = eval(code, { }, { })
        return {"expression": expr, "result": result}
    except Exception as e:
        return {"error": str(e), "expression": expr}

if __name__ == "__main__":
    cfg = settings.get('mcp', {}).get('calc', {})
    port = cfg.get('port', 8002)
    print(f"🚀 Starting CalcService on port {port}")
    mcp.run(transport="streamable-http", host="0.0.0.0", port=port)

