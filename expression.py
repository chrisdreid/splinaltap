"""
Expression evaluator for SplinalTap.

This module provides secure expression evaluation using Python's AST,
allowing math expressions to be used in keyframe values and variables.
"""

import ast
from typing import Dict, Callable, Any, Set

from .backends import BackendManager, get_math_functions


class ExpressionEvaluator:
    """A class that transforms AST expressions to safe callable functions."""
    
    def __init__(self, variables: Dict[str, Any] = None):
        """Initialize the expression evaluator.
        
        Args:
            variables: Dictionary of variable name to value/callable
            
        Raises:
            TypeError: If variables is not a dictionary
        """
        # Type check variables
        if variables is not None and not isinstance(variables, dict):
            raise TypeError(f"Variables must be a dictionary, got {type(variables).__name__}")
            
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
            'round': round,
            'rand': math_funcs['rand'],
            'randint': math_funcs['randint']
        }
        
        # Constants that can be used in expressions
        self.safe_constants = {
            'pi': math_funcs['pi'], 
            'e': math_funcs['e']
        }
        
        # Variables that can be used in expressions
        self.variables = variables or {}
    
    def evaluate(self, expr: str, position: float = 0.0, variables: Dict[str, Any] = None) -> float:
        """Evaluate an expression at a given position with optional variables.
        
        Args:
            expr: The expression string to evaluate
            position: The position (t value) to evaluate at
            variables: Optional dictionary of variable values to use
            
        Returns:
            The result of evaluating the expression
        """
        expr_func = self.parse_expression(expr)
        channels = variables or {}
        result = expr_func(position, channels)
        
        # Convert NumPy arrays to Python scalar values
        if hasattr(result, 'item') and hasattr(result, 'size') and result.size == 1:
            return result.item()
        return result
    
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
        # Remove any ? characters (deprecated random value syntax)
        if '?' in expr:
            expr = expr.replace('?', 'rand()')
        
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
        # First, pre-process the AST to find all qualified names
        qualified_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                qualified_names.add(node.value.id)
        
        class SafetyValidator(ast.NodeVisitor):
            def __init__(self, parent):
                self.parent = parent
                self.used_vars = set()
                
                # Set of allowed AST node types
                self.allowed_nodes = {
                    # Expression types
                    ast.Expression, ast.Num, ast.UnaryOp, ast.BinOp, ast.Name, ast.Call, ast.Load, ast.Constant,
                    # Attribute access for spline.channel references
                    ast.Attribute,
                    # Binary operators
                    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, 
                    # Comparison operators
                    ast.IfExp, ast.Compare, ast.Eq, ast.Mod, ast.Lt, ast.Gt, ast.LtE, ast.GtE, ast.NotEq,
                    # Logical operators
                    ast.BoolOp, ast.And, ast.Or,
                    # Collection types
                    ast.List, ast.Tuple
                }
                
                # Common variable names allowed in expressions
                self.allowed_var_names = {'t', 'x', 'y', 'z', 'a', 'b', 'c', 'd', '@',
                                         'amplitude', 'frequency', 'scale', 'offset', 'speed',
                                         'factor', 'position', 'rotation', 'scale', 'value'}
                
                # Names that require qualification when referring to channels
                self.channel_names = {'x', 'y', 'z', 'position', 'rotation', 'scale'}
            
            def generic_visit(self, node):
                if type(node) not in self.allowed_nodes:
                    raise ValueError(f"Unsafe operation: {type(node).__name__}")
                super().generic_visit(node)
                
            def visit_Attribute(self, node):
                # Process attribute access (like position.x)
                if isinstance(node.value, ast.Name):
                    # Get the fully qualified name
                    full_name = f"{node.value.id}.{node.attr}"
                    # Add it to used_vars for dependency tracking
                    self.used_vars.add(full_name)
                
                # Continue with normal validation
                self.generic_visit(node)
            
            def visit_Name(self, node):
                # Skip validation for names that are part of qualified references (e.g., 'position' in 'position.x')
                if node.id in qualified_names:
                    return
                
                # Special cases for allowed unqualified names
                if node.id in self.parent.variables:
                    # Solver-level variables are allowed
                    self.used_vars.add(node.id)
                elif node.id in self.parent.safe_funcs or node.id in self.parent.safe_constants:
                    # Math functions and constants are fine
                    pass
                elif node.id == 't':
                    # The built-in time variable is always allowed
                    pass
                elif node.id in self.channel_names:
                    # Common channel/spline names must be fully qualified
                    raise ValueError(
                        f"Unqualified channel reference '{node.id}' is not allowed. "
                        f"Use a fully qualified name in the format 'spline.channel'."
                    )
                elif node.id in self.allowed_var_names:
                    # Other allowed variable names for calculations
                    pass
                else:
                    # Any other unqualified names are not allowed
                    raise ValueError(
                        f"Unqualified reference '{node.id}' is not allowed. "
                        f"Channel references must use fully qualified names in the format 'spline.channel'."
                    )
                
                # Continue with normal validation
                self.generic_visit(node)
            
            def visit_Call(self, node):
                # Validate function calls
                if not isinstance(node.func, ast.Name) or node.func.id not in self.parent.safe_funcs:
                    raise ValueError(f"Unsafe function call: {node.func}")
                self.generic_visit(node)
        
        # Create and run the validator with our pre-collected qualified names
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
            
        def visit_List(self, node):
            # Handle list literals
            elements = [self.visit(elt) for elt in node.elts]
            return lambda ctx: [element(ctx) for element in elements]
            
        def visit_Tuple(self, node):
            # Handle tuple literals
            elements = [self.visit(elt) for elt in node.elts]
            return lambda ctx: tuple(element(ctx) for element in elements)
        
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
            elif name == 't':
                # Special case for time variable
                return lambda ctx: ctx.get('t', 0)
            else:
                # For channel variables, only references within a fully qualified name should reach here
                # (the unqualified ones should be caught by visit_Name in SafetyValidator)
                def validate_fully_qualified(ctx):
                    # First check if it's a solver variable
                    if name in ctx:
                        value = ctx[name]
                        # Convert NumPy arrays to Python scalar values
                        if hasattr(value, 'item') and hasattr(value, 'size') and value.size == 1:
                            return value.item()
                        return value
                    
                    # Special case for built-in math names (which are provided in the context)
                    if name in ('sin', 'cos', 'tan', 'sqrt', 'exp', 'log'):
                        return ctx.get(name, 0)
                    
                    # Check if a channel_lookup function is provided to handle qualified names
                    if '__channel_lookup__' in ctx:
                        channel_lookup = ctx['__channel_lookup__']
                        try:
                            # If this isn't part of an attribute access, it's an unqualified name
                            # Let the channel_lookup function handle the error
                            value = channel_lookup(name, ctx.get('t', 0))
                            
                            # Convert NumPy arrays to Python scalar values
                            if hasattr(value, 'item') and hasattr(value, 'size') and value.size == 1:
                                return value.item()
                            return value
                        except Exception as e:
                            # Propagate the error
                            raise ValueError(f"Error accessing channel '{name}': {str(e)}")
                    
                    # Default case - shouldn't reach here if validation is working
                    raise ValueError(
                        f"Unqualified reference '{name}' is not allowed. "
                        f"Channel references must use fully qualified names in the format 'spline.channel'."
                    )
                return validate_fully_qualified
                
        def visit_Attribute(self, node):
            # Handle attribute access like position.x
            # This is necessary for fully qualified channel references
            if isinstance(node.value, ast.Name):
                # Get the full reference as a string (e.g., "position.x")
                full_name = f"{node.value.id}.{node.attr}"
                
                # Use channel_lookup function if available
                def attribute_ctx_lookup(ctx):
                    # Try to use the channel lookup function
                    if '__channel_lookup__' in ctx:
                        channel_lookup = ctx['__channel_lookup__']
                        try:
                            value = channel_lookup(full_name, ctx.get('t', 0))
                            
                            # Convert NumPy arrays to Python scalar values
                            if hasattr(value, 'item') and hasattr(value, 'size') and value.size == 1:
                                return value.item()
                            return value
                        except Exception:
                            # If lookup fails, try a context lookup
                            pass
                    
                    # Try regular context lookup
                    value = ctx.get(full_name, 0)
                    
                    # Convert NumPy arrays to Python scalar values
                    if hasattr(value, 'item') and hasattr(value, 'size') and value.size == 1:
                        return value.item()
                    return value
                return attribute_ctx_lookup
            else:
                # This is for more complex attribute access like object.method.property
                # We don't support this, so return 0
                return lambda ctx: 0
        
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
                
                
class DependencyExtractor(ast.NodeVisitor):
    """
    Visits an AST and records all Name nodes that might be channel references.
    Excludes known math functions, known constants, etc.
    """
    def __init__(self, safe_funcs, safe_constants, known_variables):
        super().__init__()
        # Type check inputs with more forgiving approach to handle test cases
        if safe_funcs is not None and not isinstance(safe_funcs, dict):
            raise TypeError(f"safe_funcs must be a dictionary, got {type(safe_funcs).__name__}")
            
        if safe_constants is not None and not isinstance(safe_constants, dict):
            raise TypeError(f"safe_constants must be a dictionary, got {type(safe_constants).__name__}")
            
        # Convert known_variables to a set if it's not already
        if isinstance(known_variables, (list, tuple)):
            known_variables = set(known_variables)
        elif not isinstance(known_variables, set) and known_variables is not None:
            try:
                known_variables = set(known_variables)
            except (TypeError, ValueError):
                raise TypeError(f"known_variables must be convertible to a set, got {type(known_variables).__name__}")
        
        self.safe_funcs = safe_funcs
        self.safe_constants = safe_constants
        self.known_variables = known_variables if known_variables is not None else set()
        self.references: Set[str] = set()
        
        # Collect qualified names during the first pass
        self.qualified_names = set()
        
    def pre_process(self, tree):
        """First pass to collect names that are part of attribute access."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                self.qualified_names.add(node.value.id)

    def visit_Name(self, node: ast.Name):
        # Skip names that are part of qualified references
        if node.id in self.qualified_names:
            return
            
        # If not in safe funcs/constants, record as potential dependency
        if (
            node.id not in self.safe_funcs
            and node.id not in self.safe_constants
            and node.id not in self.known_variables
            and node.id != 't'  # or other built-in placeholders
        ):
            # Unqualified names should not be added as dependencies
            # The validation step should catch these before we get here
            pass
        self.generic_visit(node)
        
    def visit_Attribute(self, node: ast.Attribute):
        # Handle attribute access (e.g., position.x)
        if isinstance(node.value, ast.Name):
            # Construct the full name (spline.channel)
            full_name = f"{node.value.id}.{node.attr}"
            # Add as a dependency - this is the only valid way to reference channels
            self.references.add(full_name)
        self.generic_visit(node)


def extract_expression_dependencies(expr_str: str,
                                    safe_funcs,
                                    safe_constants,
                                    known_variables) -> Set[str]:
    """
    Parse the expression into an AST, then extract any names that might be channel references.
    
    Args:
        expr_str: The expression string to parse
        safe_funcs: Dictionary of safe functions (e.g., sin, cos, etc.)
        safe_constants: Dictionary of constants (e.g., pi, e)
        known_variables: Set of known variable names
        
    Returns:
        Set of potential channel references
    """
    # Replace ^ with **, etc. (mirror parse_expression steps)
    expr_str = expr_str.replace('^', '**')
    expr_str = expr_str.replace('@', 't')
    expr_str = expr_str.replace('?', 'rand()')

    try:
        tree = ast.parse(expr_str, mode='eval')
    except SyntaxError:
        return set()  # Return empty if it doesn't parse

    # First, validate that all channel references are fully qualified
    class FullyQualifiedValidator(ast.NodeVisitor):
        def __init__(self, safe_funcs, safe_constants, known_variables):
            self.safe_funcs = safe_funcs
            self.safe_constants = safe_constants
            self.known_variables = known_variables
            self.common_channel_names = {'x', 'y', 'z', 'position', 'rotation', 'scale'}
        
        def visit_Name(self, node):
            # Unqualified variable names that might be channels should be rejected
            if (node.id not in self.safe_funcs and 
                node.id not in self.safe_constants and 
                node.id not in self.known_variables and
                node.id != 't' and  # Allow time variable
                node.id in self.common_channel_names):
                raise ValueError(
                    f"Unqualified channel reference '{node.id}' is not allowed. "
                    f"Use a fully qualified name in the format 'spline.channel'."
                )
            self.generic_visit(node)
    
    # Validate fully qualified references
    validator = FullyQualifiedValidator(safe_funcs, safe_constants, known_variables)
    try:
        validator.visit(tree)
    except ValueError as e:
        # Re-raise the validation error
        raise ValueError(str(e))
    
    # Extract dependencies
    extractor = DependencyExtractor(safe_funcs, safe_constants, known_variables)
    # First pass: collect qualified names
    extractor.pre_process(tree)
    # Second pass: extract dependencies
    extractor.visit(tree.body)
    return extractor.references