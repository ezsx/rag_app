from __future__ import annotations

import ast
import operator
import math
import logging
from typing import Any, Dict, Union

logger = logging.getLogger(__name__)

# Разрешенные математические операции
ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Разрешенные математические функции
ALLOWED_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "max": max,
    "min": min,
    "sum": sum,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "ceil": math.ceil,
    "floor": math.floor,
    "pi": math.pi,
    "e": math.e,
}


def _safe_eval(node: ast.AST) -> Union[int, float]:
    """Безопасное вычисление AST узла"""
    if isinstance(node, ast.Constant):  # Python 3.8+
        return node.value
    elif isinstance(node, ast.Num):  # Python < 3.8
        return node.n
    elif isinstance(node, ast.BinOp):
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        op = ALLOWED_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"Неразрешенная операция: {type(node.op).__name__}")
        return op(left, right)
    elif isinstance(node, ast.UnaryOp):
        operand = _safe_eval(node.operand)
        op = ALLOWED_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(
                f"Неразрешенная унарная операция: {type(node.op).__name__}"
            )
        return op(operand)
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Разрешены только простые вызовы функций")

        func_name = node.func.id
        if func_name not in ALLOWED_FUNCTIONS:
            raise ValueError(f"Неразрешенная функция: {func_name}")

        func = ALLOWED_FUNCTIONS[func_name]
        args = [_safe_eval(arg) for arg in node.args]

        # Специальная обработка для некоторых функций
        if (
            func_name in ("max", "min", "sum")
            and len(args) == 1
            and isinstance(args[0], (list, tuple))
        ):
            return func(args[0])

        return func(*args)
    elif isinstance(node, ast.Name):
        # Разрешаем только математические константы
        if node.id in ALLOWED_FUNCTIONS:
            return ALLOWED_FUNCTIONS[node.id]
        else:
            raise ValueError(f"Неразрешенная переменная: {node.id}")
    elif isinstance(node, ast.List):
        return [_safe_eval(elem) for elem in node.elts]
    else:
        raise ValueError(f"Неразрешенный тип узла: {type(node).__name__}")


def math_eval(expression: str) -> Dict[str, Any]:
    """Безопасное вычисление математических выражений.

    Поддерживает базовые арифметические операции и математические функции.
    Использует AST парсинг для безопасности.

    Args:
        expression: Математическое выражение для вычисления

    Returns:
        {result: Any, expression: str, success: bool}
    """
    if not expression.strip():
        return {
            "result": None,
            "expression": expression,
            "success": False,
            "error": "Пустое выражение",
        }

    try:
        # Очистка выражения
        clean_expr = expression.strip()

        # Парсинг в AST
        try:
            tree = ast.parse(clean_expr, mode="eval")
        except SyntaxError as e:
            return {
                "result": None,
                "expression": clean_expr,
                "success": False,
                "error": f"Синтаксическая ошибка: {str(e)}",
            }

        # Безопасное вычисление
        try:
            result = _safe_eval(tree.body)

            # Форматирование результата
            if isinstance(result, float):
                if result.is_integer():
                    result = int(result)
                else:
                    result = round(result, 10)  # Ограничиваем точность

            return {
                "result": result,
                "expression": clean_expr,
                "success": True,
                "type": type(result).__name__,
            }

        except (ValueError, TypeError, ZeroDivisionError, OverflowError) as e:
            return {
                "result": None,
                "expression": clean_expr,
                "success": False,
                "error": f"Ошибка вычисления: {str(e)}",
            }

    except Exception as e:
        logger.error(f"Неожиданная ошибка при вычислении выражения '{expression}': {e}")
        return {
            "result": None,
            "expression": expression,
            "success": False,
            "error": f"Неожиданная ошибка: {str(e)}",
        }


# Примеры использования для документации
EXAMPLES = {
    "basic": "2 + 3 * 4",
    "functions": "sqrt(16) + sin(pi/2)",
    "complex": "max([1, 2, 3]) + min([4, 5, 6])",
    "constants": "pi * e",
}
