"""
Expression evaluator for SplinalTap.

This module provides secure expression evaluation using Python's AST,
allowing math expressions to be used in keyframe values and variables.
"""

import ast
from typing import Dict, Callable, Any

from .backends import BackendManager, get_math_functions


class ExpressionEvaluator:
    """A class that transforms AST expressions to safe callable functions."""
    
    def __init__(self, variables: Dict[str, Any] = None):
        """Initialize the expression evaluator.
        
        Args:
            variables: Dictionary of variable name to value/callable
        """
        # Get math functions from the current backend
        math_funcs = get_math_functions()
        
        # Mapping of functions that can be used in expressions
        self.safe_funcs = {
            'sin': math_funcs['sin'], 
            'cos': math_funcs['cos'], 
            'tan': math_funcs['tan'],
            'sqrt': math_funcs['sqrt'], 
            'log': math_funcs['log'], 
            'exp': math_funcs['exp'],
            'pow': math_funcs['pow'],
            'abs': abs,
            'max': max,
            'min': min,
            'round': round
        }
        
        # Constants that can be used in expressions
        self.safe_constants = {
            'pi': math_funcs['pi'], 
            'e': math_funcs['e']
        }
        
        # Variables that can be used in expressions
        self.variables = variables or {}
    
    def parse_expression(self, expr: str) -> Callable[[float, Dict[str, Any]], float]:
        """Parse an expression into a safe lambda function using AST transformation.
        
        Args:
            expr: The expression string to parse
            
        Returns:
            A callable that evaluates the expression
            
        Raises:
            ValueError: If the expression contains unsafe operations
            SyntaxError: If the expression has invalid syntax
        """
        # Replace ^ with ** for power operator
        expr = expr.replace('^', '**')
        # Replace @ with t for evaluating at position
        expr = expr.replace('@', 't')
        
        # Parse the expression to an AST
        try:
            tree = ast.parse(expr, mode='eval')
        except SyntaxError as e:
            raise ValueError(f"Invalid expression syntax: {e}")
        
        # Validate the expression is safe
        self._validate_expression_safety(tree)
        
        # Transform the AST to a callable
        transformer = self.ExpressionTransformer(self.safe_funcs, self.safe_constants, self.variables)
        expr_func = transformer.visit(tree.body)
        
        def evaluator(t: float, channels: Dict[str, Any] = None) -> float:
            """Evaluate the expression at position t with optional channels.
            
            Args:
                t: The position to evaluate at
                channels: Optional channel values to use in the expression
                
            Returns:
                The result of the expression evaluation
            """
            channels = channels or {}
            # Create a context with t and channels
            context = {'t': t}
            context.update(channels)
            return expr_func(context)
        
        return evaluator
    
    def _validate_expression_safety(self, tree: ast.AST) -> None:
        """Validate that an AST only contains safe operations.
        
        Args:
            tree: The AST to validate
            
        Raises:
            ValueError: If the AST contains unsafe operations
        """
        class SafetyValidator(ast.NodeVisitor):
            def __init__(self, parent):
                self.parent = parent
                self.used_vars = set()
                
                # Set of allowed AST node types
                self.allowed_nodes = {
                    # Expression types
                    ast.Expression, ast.Num, ast.UnaryOp, ast.BinOp, ast.Name, ast.Call, ast.Load, ast.Constant,
                    # Binary operators
                    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, 
                    # Comparison operators
                    ast.IfExp, ast.Compare, ast.Eq, ast.Mod, ast.Lt, ast.Gt, ast.LtE, ast.GtE, ast.NotEq,
                    # Logical operators
                    ast.BoolOp, ast.And, ast.Or
                }
                
                # Common variable names allowed in expressions
                self.allowed_var_names = {'t', 'x', 'y', 'z', 'a', 'b', 'c', 'd'}
            
            def generic_visit(self, node):
                if type(node) not in self.allowed_nodes:
                    raise ValueError(f"Unsafe operation: {type(node).__name__}")
                super().generic_visit(node)
            
            def visit_Name(self, node):
                if node.id in self.parent.variables:
                    self.used_vars.add(node.id)
                elif node.id in self.parent.safe_funcs or node.id in self.parent.safe_constants:
                    # Functions and constants are fine
                    pass
                elif node.id in self.allowed_var_names:
                    # Common variable names are allowed
                    pass
                else:
                    raise ValueError(f"Unknown name: {node.id}")
                super().generic_visit(node)
            
            def visit_Call(self, node):
                if not isinstance(node.func, ast.Name) or node.func.id not in self.parent.safe_funcs:
                    raise ValueError(f"Unsafe function call: {node.func}")
                super().generic_visit(node)
        
        validator = SafetyValidator(self)
        validator.visit(tree)
    
    class ExpressionTransformer(ast.NodeTransformer):
        """Transforms AST nodes into callable functions."""
        
        def __init__(self, safe_funcs, safe_constants, variables):
            self.safe_funcs = safe_funcs
            self.safe_constants = safe_constants
            self.variables = variables
        
        def visit_Expression(self, node):
            # Return the transformed body
            return self.visit(node.body)
        
        def visit_Num(self, node):
            # For constant numbers, just return the value
            return lambda ctx: node.n
        
        def visit_Constant(self, node):
            # For constant values, just return the value
            return lambda ctx: node.value
        
        def visit_Name(self, node):
            # Handle variable names
            name = node.id
            
            if name in self.safe_constants:
                # For constants like pi, e
                constant_value = self.safe_constants[name]
                return lambda ctx: constant_value
            elif name in self.variables:
                # For variables defined in the interpolator
                var_func = self.variables[name]
                if callable(var_func):
                    return lambda ctx: var_func(ctx.get('t', 0), ctx)
                else:
                    return lambda ctx: var_func
            else:
                # For t and channel variables
                return lambda ctx: ctx.get(name, 0)
        
        def visit_BinOp(self, node):
            # Handle binary operations
            left = self.visit(node.left)
            right = self.visit(node.right)
            
            if isinstance(node.op, ast.Add):
                return lambda ctx: left(ctx) + right(ctx)
            elif isinstance(node.op, ast.Sub):
                return lambda ctx: left(ctx) - right(ctx)
            elif isinstance(node.op, ast.Mult):
                return lambda ctx: left(ctx) * right(ctx)
            elif isinstance(node.op, ast.Div):
                return lambda ctx: left(ctx) / right(ctx)
            elif isinstance(node.op, ast.Pow):
                return lambda ctx: left(ctx) ** right(ctx)
            elif isinstance(node.op, ast.Mod):
                return lambda ctx: left(ctx) % right(ctx)
            else:
                raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        
        def visit_UnaryOp(self, node):
            # Handle unary operations
            operand = self.visit(node.operand)
            
            if isinstance(node.op, ast.USub):
                return lambda ctx: -operand(ctx)
            elif isinstance(node.op, ast.UAdd):
                return lambda ctx: +operand(ctx)
            else:
                raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        
        def visit_Call(self, node):
            # Handle function calls
            if not isinstance(node.func, ast.Name) or node.func.id not in self.safe_funcs:
                raise ValueError(f"Unsafe function call: {node.func}")
            
            func = self.safe_funcs[node.func.id]
            args = [self.visit(arg) for arg in node.args]
            
            return lambda ctx: func(*(arg(ctx) for arg in args))
        
        def visit_IfExp(self, node):
            # Handle conditional expressions (x if condition else y)
            test = self.visit(node.test)
            body = self.visit(node.body)
            orelse = self.visit(node.orelse)
            
            return lambda ctx: body(ctx) if test(ctx) else orelse(ctx)
        
        def visit_Compare(self, node):
            # Handle comparisons (a < b, a == b, etc.)
            if len(node.ops) != 1 or len(node.comparators) != 1:
                raise ValueError("Only simple comparisons are supported")
            
            left = self.visit(node.left)
            op = node.ops[0]
            right = self.visit(node.comparators[0])
            
            if isinstance(op, ast.Eq):
                return lambda ctx: left(ctx) == right(ctx)
            elif isinstance(op, ast.NotEq):
                return lambda ctx: left(ctx) != right(ctx)
            elif isinstance(op, ast.Lt):
                return lambda ctx: left(ctx) < right(ctx)
            elif isinstance(op, ast.LtE):
                return lambda ctx: left(ctx) <= right(ctx)
            elif isinstance(op, ast.Gt):
                return lambda ctx: left(ctx) > right(ctx)
            elif isinstance(op, ast.GtE):
                return lambda ctx: left(ctx) >= right(ctx)
            else:
                raise ValueError(f"Unsupported comparison operator: {type(op).__name__}")
        
        def visit_BoolOp(self, node):
            # Handle boolean operations (and, or)
            values = [self.visit(value) for value in node.values]
            
            if isinstance(node.op, ast.And):
                def eval_and(ctx):
                    for value in values:
                        result = value(ctx)
                        if not result:
                            return result
                    return result
                return eval_and
            
            elif isinstance(node.op, ast.Or):
                def eval_or(ctx):
                    for value in values:
                        result = value(ctx)
                        if result:
                            return result
                    return result
                return eval_or
            
            else:
                raise ValueError(f"Unsupported boolean operator: {type(node.op).__name__}")