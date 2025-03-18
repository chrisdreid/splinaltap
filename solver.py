"""
KeyframeSolver class for SplinalTap interpolation.

A KeyframeSolver is a collection of Splines that can be evaluated together.
It represents a complete animation or property set, like a scene in 3D software.
"""

import os
import json
import pickle
from collections import defaultdict
from typing import Dict, List, Optional, Union, Any, Tuple, Set, Callable

from .spline import Spline
from .expression import ExpressionEvaluator, extract_expression_dependencies
from .backends import get_math_functions

# KeyframeSolver file format version
KEYFRAME_SOLVER_FORMAT_VERSION = "2.0"

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    yaml = None
    HAS_YAML = False


class KeyframeSolver:
    """A solver containing multiple splines for complex animation."""
    
    def __init__(self, name: str = "Untitled"):
        """Initialize a new solver.
        
        Args:
            name: The name of the solver
            
        Raises:
            TypeError: If name is not a string
        """
        if not isinstance(name, str):
            raise TypeError(f"Name must be a string, got {type(name).__name__}")
            
        self.name = name
        self.splines: Dict[str, Spline] = {}
        self.metadata: Dict[str, Any] = {}
        self.variables: Dict[str, Any] = {}
        self.range: Tuple[float, float] = (0.0, 1.0)
        self.publish: Dict[str, List[str]] = {}
        # Cache for topological solver
        self._dependency_graph = None
        self._topo_order = None
        self._evaluation_cache = {}
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, splines={self.splines}, metadata={self.metadata}, variables={self.variables}, range={self.range}, publish={self.publish})"
    
    def create_spline(self, name: str) -> Spline:
        """Create a new spline in this solver.
        
        Args:
            name: The name of the spline
            
        Returns:
            The newly created spline
        """
        spline = Spline()
        self.splines[name] = spline
        return spline
    
    def get_spline(self, name: str) -> Spline:
        """Get a spline by name.
        
        Args:
            name: The name of the spline to get
            
        Returns:
            The requested spline
            
        Raises:
            KeyError: If the spline does not exist
        """
        if name not in self.splines:
            raise KeyError(f"Spline '{name}' does not exist in this solver")
        return self.splines[name]
    
    def get_spline_names(self) -> List[str]:
        """Get the names of all splines in this solver.
        
        Returns:
            A list of spline names
        """
        return list(self.splines.keys())
    
    def set_metadata(self, key: str, value: Any) -> None:
        """Set a metadata value.
        
        Args:
            key: The metadata key
            value: The metadata value
        """
        self.metadata[key] = value
    
    def set_variable(self, name: str, value: Any) -> None:
        """Set a variable value.
        
        Args:
            name: The variable name
            value: The variable value
        """
        self.variables[name] = value
        
    def set_publish(self, source: str, targets: List[str]) -> None:
        """Set up a publication channel for cross-channel or cross-spline access.
        
        Args:
            source: The source channel in "spline.channel" format
            targets: A list of targets that can access the source ("spline.channel" format or "*" for global)
        
        Raises:
            ValueError: If source format is incorrect
        """
        if '.' not in source:
            raise ValueError(f"Source must be in 'spline.channel' format, got {source}")
            
        self.publish[source] = targets
        
        # Reset dependency graph and topo order since publish relationships changed
        self._dependency_graph = None
        self._topo_order = None
        
    def _build_dependency_graph(self) -> Dict[str, Set[str]]:
        """
        Build a directed graph of channel dependencies (node = 'spline.channel').
        Edge from X->Y if Y depends on X (i.e. Y's expression references X or Y is published by X).
        """
        graph = defaultdict(set)
        
        # 1) Gather references for math/variables from ExpressionEvaluator
        math_funcs = get_math_functions()
        
        safe_funcs = {
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
        safe_constants = {'pi': math_funcs['pi'], 'e': math_funcs['e']}
        known_variables = set(self.variables.keys())  # solver-level variables

        def node_key(spline_name, channel_name):
            return f"{spline_name}.{channel_name}"

        # 2) Build a list of all nodes
        all_nodes = []
        for spline_name, spline in self.splines.items():
            for channel_name in spline.channels:
                all_nodes.append(node_key(spline_name, channel_name))
        
        # Ensure each node appears in graph with an empty set (in case no dependencies)
        for node in all_nodes:
            graph[node] = set()
        
        # 3) Inspect each channel for expression references
        for spline_name, spline in self.splines.items():
            for channel_name, channel in spline.channels.items():
                current_node = node_key(spline_name, channel_name)

                for kf in channel.keyframes:
                    # If the keyframe is an expression (string), parse it
                    # If your system stores an already-compiled callable, you may want
                    # to store the original expression string so we can re-parse it for deps
                    expr_str = None
                    if isinstance(kf.value, str):
                        expr_str = kf.value
                    # If kf.value is a callable that wraps a string expression
                    elif hasattr(kf.value, '__splinaltap_expr__'):
                        expr_str = kf.value.__splinaltap_expr__
                    
                    if expr_str:
                        deps = extract_expression_dependencies(
                            expr_str,
                            safe_funcs,
                            safe_constants,
                            known_variables
                        )
                        # For each dep, if it has a '.', treat it as "spline.channel"
                        # else treat as "currentSpline.dep"
                        for ref in deps:
                            if '.' in ref:
                                dependency_node = ref
                            else:
                                # same spline
                                dependency_node = node_key(spline_name, ref)
                                
                            # Only add if the dependency node actually exists
                            if dependency_node in all_nodes:
                                # Add edge dependency_node -> current_node
                                graph[dependency_node].add(current_node)

                # 4) Also handle "publish" list
                # If this channel publishes to otherChannel,
                # we interpret that otherChannel depends on this channel
                if hasattr(channel, 'publish') and channel.publish:
                    for target_ref in channel.publish:
                        if target_ref == "*":
                            # Global means "anyone can see it," but we don't know the specifics
                            # Usually we skip or handle differently
                            continue
                        else:
                            if '.' in target_ref:
                                target_node = target_ref
                            else:
                                target_node = node_key(spline_name, target_ref)
                                
                            # Only add if the target node actually exists
                            if target_node in all_nodes:
                                # current_node -> target_node (current publishes to target)
                                graph[current_node].add(target_node)

        # 5) Handle solver-level publish directives
        for source, targets in self.publish.items():
            if source in all_nodes:
                for target in targets:
                    if target == "*":
                        # Global publish - all channels can depend on this source
                        for node in all_nodes:
                            if node != source:  # Avoid self-dependencies
                                graph[source].add(node)
                    elif '.' in target and target in all_nodes:
                        # Specific target channel
                        graph[source].add(target)
                    elif target.endswith(".*"):
                        # Wildcard for all channels in a spline
                        spline_prefix = target[:-1]  # Remove the ".*"
                        for node in all_nodes:
                            if node.startswith(spline_prefix) and node != source:
                                graph[source].add(node)

        return graph
        
    def _topological_sort(self, graph: Dict[str, Set[str]]) -> List[str]:
        """
        Returns a list of node_keys (e.g., 'spline.channel') in valid topological order.
        Raises ValueError if there's a cycle.
        """
        # 1) Compute in-degrees
        in_degree = {node: 0 for node in graph}
        for node, dependents in graph.items():
            for dep in dependents:
                in_degree[dep] += 1

        # 2) Initialize queue with all nodes of in-degree 0
        queue = [n for n, deg in in_degree.items() if deg == 0]
        topo_order = []

        while queue:
            current = queue.pop()
            topo_order.append(current)

            # Decrement in-degree of all dependents
            for dep in graph[current]:
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    queue.append(dep)

        if len(topo_order) != len(graph):
            raise ValueError("Cycle detected in channel dependencies.")

        return topo_order
    
    def _normalize_position(self, position: float) -> float:
        """Apply range normalization to a position.
        
        Args:
            position: The position to normalize
            
        Returns:
            Normalized position in 0-1 range
        """
        min_t, max_t = self.range
        if min_t != 0.0 or max_t != 1.0:
            # Normalize the position to the 0-1 range
            if position >= min_t and position <= max_t:
                return (position - min_t) / (max_t - min_t)
            elif position < min_t:
                return 0.0
            else:  # position > max_t
                return 1.0
        else:
            return position

    def _evaluate_channel_at_time(self, node_key: str, t: float,
                                  external_channels: Dict[str, Any]) -> float:
        """
        Evaluate a single channel at time 't' (0-1 normalized), using the caching dict.
        If we have a cached value, use it; otherwise, compute via channel's get_value.
        This method also ensures that if the channel references another channel at time offset,
        we do a sub-call back into _evaluate_channel_at_time(...) with a different t_sub.
        """
        cache_key = (node_key, t)
        if cache_key in self._evaluation_cache:
            return self._evaluation_cache[cache_key]

        # Parse node_key into (splineName, channelName)
        spline_name, channel_name = node_key.split('.', 1)
        spline = self.splines[spline_name]
        channel = spline.channels[channel_name]

        # Create a channel lookup function for expression evaluation
        def channel_lookup(sub_chan_name, sub_t):
            """Helper function that uses _evaluate_channel_at_time to get dependencies at sub_t."""
            # Always require fully qualified names for cross-channel references
            # Special cases: built-in variables and solver-level variables
            if sub_chan_name == 't':
                return sub_t
            elif sub_chan_name in self.variables:
                # Allow access to solver-level variables without qualification
                return self.variables[sub_chan_name]
                
            if '.' not in sub_chan_name:
                # This is an unqualified name (e.g., "x"), which is no longer allowed
                # First, try to find if any channels with this name are published to this target
                matching_published_channels = []
                
                # Check solver-level publish directives
                for source, targets in self.publish.items():
                    source_spline, source_channel = source.split('.', 1)
                    if source_channel == sub_chan_name:
                        channel_path = f"{spline_name}.{channel_name}"
                        can_access = (
                            "*" in targets or
                            channel_path in targets or
                            any(target.endswith(".*") and channel_path.startswith(target[:-1]) for target in targets)
                        )
                        if can_access:
                            matching_published_channels.append(source)
                
                # Check channel-level publish directives
                for other_spline_name, other_spline in self.splines.items():
                    for other_channel_name, other_channel in other_spline.channels.items():
                        if other_channel_name != sub_chan_name:
                            continue
                        
                        if not hasattr(other_channel, 'publish') or not other_channel.publish:
                            continue
                        
                        target_path = f"{spline_name}.{channel_name}"
                        can_access = (
                            "*" in other_channel.publish or
                            target_path in other_channel.publish or
                            any(pattern.endswith(".*") and target_path.startswith(pattern[:-1]) 
                                for pattern in other_channel.publish)
                        )
                        
                        if can_access:
                            matching_published_channels.append(f"{other_spline_name}.{other_channel_name}")
                
                # Check if a local channel with this name exists in this spline
                local_channel_exists = (
                    spline_name in self.splines and 
                    sub_chan_name in self.splines[spline_name].channels
                )
                
                # Provide a helpful error message based on what's available
                if local_channel_exists:
                    # Suggest the local channel
                    suggestions = [f"{spline_name}.{sub_chan_name}"]
                    if matching_published_channels:
                        suggestions.extend(matching_published_channels)
                    
                    raise ValueError(
                        f"Unqualified channel reference '{sub_chan_name}' is not allowed. "
                        f"Use a fully qualified name such as: {', '.join(suggestions)}"
                    )
                elif matching_published_channels:
                    # Suggest the published channels
                    channels_str = ", ".join(matching_published_channels)
                    raise ValueError(
                        f"Unqualified channel reference '{sub_chan_name}' is not allowed. "
                        f"Use a fully qualified name such as: {channels_str}"
                    )
                else:
                    # No matching channels found
                    raise ValueError(
                        f"Unqualified channel reference '{sub_chan_name}' is not allowed. "
                        f"Use a fully qualified name in the format 'spline.channel'."
                    )
            else:
                # This is a fully qualified reference (e.g., "position.x")
                sub_node_key = sub_chan_name
            
            # Check if the node key exists before evaluating
            spline_part, channel_part = sub_node_key.split('.', 1)
            if spline_part in self.splines and channel_part in self.splines[spline_part].channels:
                return self._evaluate_channel_at_time(sub_node_key, sub_t, external_channels)
            else:
                # If the specified spline.channel doesn't exist, return 0
                return 0

        # Prepare variable context for channel evaluation
        combined_vars = {}
        # Add external_channels
        if external_channels:
            combined_vars.update(external_channels)
        # Add solver-level variables
        combined_vars.update(self.variables)
        # Add the channel lookup function for cross-channel references
        combined_vars['__channel_lookup__'] = channel_lookup

        # Evaluate the channel with the combined variables
        val = channel.get_value(t, combined_vars)

        # Store in cache
        self._evaluation_cache[cache_key] = val
        return val

    def solve(self, position: Union[float, List[float]], external_channels: Optional[Dict[str, Any]] = None, method: str = "topo") -> Union[Dict[str, Dict[str, Any]], List[Dict[str, Dict[str, Any]]]]:
        """Solve all splines at one or more positions using topological ordering by default.
        
        Args:
            position: The position to solve at, either a single float or a list of floats
            external_channels: Optional external channel values
            method: Solver method ('topo' or 'ondemand', default: 'topo')
            
        Returns:
            If position is a float:
                A dictionary of spline names to channel value dictionaries
            If position is a list of floats:
                A list of dictionaries, each mapping spline names to channel value dictionaries
                
        Raises:
            TypeError: If position is neither a float nor a list of floats
        """
        # Check if we're solving for multiple positions
        if isinstance(position, list):
            # Make sure all elements are numbers
            if not all(isinstance(pos, (int, float)) for pos in position):
                raise TypeError("When position is a list, all elements must be numbers")
            # Apply solve to each position (we can optimize this implementation in the future)
            return [self.solve(pos, external_channels, method) for pos in position]
            
        # From here on, position should be a single float
        if not isinstance(position, (int, float)):
            raise TypeError(f"Position must be a float or a list of floats, got {type(position).__name__}")
            
        # Allow specifying the solver method
        if method == "ondemand":
            return self.solve_on_demand(position, external_channels)
        
        # Use topological ordering by default
        # Build or reuse graph
        if self._dependency_graph is None or self._topo_order is None:
            try:
                self._dependency_graph = self._build_dependency_graph()
                self._topo_order = self._topological_sort(self._dependency_graph)
            except ValueError as e:
                # If there's a cycle, fall back to on-demand method
                print(f"Warning: {e}. Falling back to on-demand evaluation.")
                return self.solve_on_demand(position, external_channels)

        # Clear evaluation cache for this solve
        self._evaluation_cache = {}

        # Normalize the position
        normalized_t = self._normalize_position(position)

        # Evaluate in topological order
        result_by_node = {}
        external_channels = external_channels or {}

        for node in self._topo_order:
            val = self._evaluate_channel_at_time(node, normalized_t, external_channels)
            result_by_node[node] = val

        # Initialize the output with all splines and channels, ensuring every channel is in the result
        out = {}
        for spline_name, spline in self.splines.items():
            if spline_name not in out:
                out[spline_name] = {}
            for channel_name in spline.channels:
                # Create an entry for every channel in every spline
                node_key = f"{spline_name}.{channel_name}"
                if node_key in result_by_node:
                    out[spline_name][channel_name] = result_by_node[node_key]
                else:
                    # If the channel wasn't evaluated in the topological sort, evaluate it now
                    val = self._evaluate_channel_at_time(node_key, normalized_t, external_channels)
                    out[spline_name][channel_name] = val

        return out

    def solve_on_demand(self, position: Union[float, List[float]], external_channels: Optional[Dict[str, Any]] = None) -> Union[Dict[str, Dict[str, Any]], List[Dict[str, Dict[str, Any]]]]:
        """Solve all splines at one or more positions using the original on-demand method.
        
        Args:
            position: The position to solve at, either a single float or a list of floats
            external_channels: Optional external channel values
            
        Returns:
            If position is a float:
                A dictionary of spline names to channel value dictionaries
            If position is a list of floats:
                A list of dictionaries, each mapping spline names to channel value dictionaries
                
        Raises:
            TypeError: If position is neither a float nor a list of floats
        """
        # Check if we're solving for multiple positions
        if isinstance(position, list):
            # Make sure all elements are numbers
            if not all(isinstance(pos, (int, float)) for pos in position):
                raise TypeError("When position is a list, all elements must be numbers")
            # Apply solve to each position
            return [self.solve_on_demand(pos, external_channels) for pos in position]
            
        # From here on, position should be a single float
        if not isinstance(position, (int, float)):
            raise TypeError(f"Position must be a float or a list of floats, got {type(position).__name__}")
            
        result = {}
        
        # Apply range normalization if needed
        normalized_position = self._normalize_position(position)
            
        # First pass: calculate channel values without expressions that might depend on other channels
        channel_values = {}
        
        # Initialize result structure with all splines and channels
        for spline_name, spline in self.splines.items():
            if spline_name not in result:
                result[spline_name] = {}
        
        # First pass: evaluate channels without expressions
        for spline_name, spline in self.splines.items():
            for channel_name, channel in spline.channels.items():
                # For simple numeric keyframes, evaluate them first
                if all(not isinstance(kf.value, str) and not hasattr(kf.value, '__splinaltap_expr__') for kf in channel.keyframes):
                    # Combine variables with external channels for non-expression evaluation
                    combined_channels = {}
                    if external_channels:
                        combined_channels.update(external_channels)
                    combined_channels.update(self.variables)
                    
                    # Evaluate the channel at the normalized position
                    value = channel.get_value(normalized_position, combined_channels)
                    result[spline_name][channel_name] = value
                    
                    # Store the channel value for expression evaluation
                    channel_values[f"{spline_name}.{channel_name}"] = value
        
        # Second pass: evaluate channels with expressions that might depend on other channels
        for spline_name, spline in self.splines.items():                
            for channel_name, channel in spline.channels.items():
                # Skip channels already evaluated in the first pass
                if channel_name in result[spline_name]:
                    continue
                    
                # Create an accessible channels dictionary based on publish rules
                accessible_channels = {}
                
                # Add external channels
                if external_channels:
                    accessible_channels.update(external_channels)
                    
                # Add solver variables
                accessible_channels.update(self.variables)
                
                # Add channels from the same spline (always accessible)
                for ch_name, ch_value in result.get(spline_name, {}).items():
                    accessible_channels[ch_name] = ch_value
                
                # Add published channels
                for source, targets in self.publish.items():
                    # Check if this channel can access the published channel
                    channel_path = f"{spline_name}.{channel_name}"
                    can_access = False
                    
                    # Check for global access with "*"
                    if "*" in targets:
                        can_access = True
                    # Check for specific access
                    elif channel_path in targets:
                        can_access = True
                    # Check for spline-level access (spline.*)
                    elif any(target.endswith(".*") and channel_path.startswith(target[:-1]) for target in targets):
                        can_access = True
                        
                    if can_access and source in channel_values:
                        # Extract just the channel name for easier access in expressions
                        source_parts = source.split(".")
                        if len(source_parts) == 2:
                            # Make the channel value accessible using the full path and just the channel name
                            accessible_channels[source] = channel_values[source]
                            accessible_channels[source_parts[1]] = channel_values[source]
                
                # Check channel-level publish list
                for other_spline_name, other_spline in self.splines.items():
                    for other_channel_name, other_channel in other_spline.channels.items():
                        if hasattr(other_channel, 'publish') and other_channel.publish:
                            source_path = f"{other_spline_name}.{other_channel_name}"
                            target_path = f"{spline_name}.{channel_name}"
                            
                            # Check if this channel is in the publish list using different matching patterns
                            can_access = False
                            
                            # Check for direct exact match
                            if target_path in other_channel.publish:
                                can_access = True
                            # Check for global "*" wildcard access
                            elif "*" in other_channel.publish:
                                can_access = True
                            # Check for spline-level wildcard "spline.*" access
                            elif any(pattern.endswith(".*") and target_path.startswith(pattern[:-1]) for pattern in other_channel.publish):
                                can_access = True
                                
                            if can_access and source_path in channel_values:
                                # If the other channel has been evaluated, make it accessible
                                accessible_channels[source_path] = channel_values[source_path]
                                # Also make it accessible by just the channel name
                                accessible_channels[other_channel_name] = channel_values[source_path]
                
                # Set up channel lookup function to handle references to published channels that weren't processed yet
                def channel_lookup(sub_chan_name, sub_t):
                    """Helper function to look up channel values."""
                    if sub_chan_name in accessible_channels:
                        return accessible_channels[sub_chan_name]
                    elif sub_chan_name == 't':
                        return sub_t
                    elif sub_chan_name in self.variables:
                        return self.variables[sub_chan_name]
                    elif '.' in sub_chan_name:
                        # Handle fully qualified reference
                        try:
                            ref_spline_name, ref_channel_name = sub_chan_name.split('.', 1)
                            ref_channel = self.splines[ref_spline_name].channels[ref_channel_name]
                            return ref_channel.get_value(normalized_position, accessible_channels)
                        except (KeyError, ValueError):
                            return 0
                    else:
                        # Unqualified references should have been caught by validation
                        return 0
                
                # Add channel lookup function to accessible channels
                accessible_channels['__channel_lookup__'] = channel_lookup
                
                # Evaluate the channel with the accessible channels
                value = channel.get_value(normalized_position, accessible_channels)
                result[spline_name][channel_name] = value
                
                # Store the value for later channel access
                channel_values[f"{spline_name}.{channel_name}"] = value
        
        return result
        
    def solve_multiple(self, positions: List[float], external_channels: Optional[Dict[str, Any]] = None, method: str = "topo") -> List[Dict[str, Dict[str, Any]]]:
        """Solve all splines at multiple positions.
        
        Args:
            positions: List of positions to solve at
            external_channels: Optional external channel values
            method: Solver method ('topo' or 'ondemand', default: 'topo')
            
        Returns:
            A list of result dictionaries, one for each position
        
        Note:
            This is a wrapper around the solve method for backward compatibility.
            Using solve() directly with a list of positions is now possible and preferred.
        """
        # Simply delegate to the solve method, which now supports lists of positions
        return self.solve(positions, external_channels, method=method)
        
    def get_plot(
        self, 
        samples: Optional[int] = None, 
        filter_channels: Optional[Dict[str, List[str]]] = None, 
        theme: str = "dark",
        save_path: Optional[str] = None,
        overlay: bool = True
    ) -> 'matplotlib.figure.Figure':
        """Generate a plot for the solver's splines and channels.
        
        Args:
            samples: Number of evenly spaced samples to use (defaults to 100)
            filter_channels: Optional dictionary mapping spline names to lists of channel names to include
                            (e.g., {'position': ['x', 'y'], 'rotation': ['angle']})
            theme: Plot theme - 'light' or 'dark'
            save_path: Optional file path to save the plot (e.g., 'plot.png')
            overlay: If True, all splines are plotted in a single graph; if False, each spline gets its own subplot
            
        Returns:
            The matplotlib figure
            
        Raises:
            ImportError: If matplotlib is not available
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
        except ImportError:
            raise ImportError("Plotting requires matplotlib. Install it with: pip install matplotlib")
            
        # Default number of samples
        if samples is None:
            samples = 100
            
        # Generate sample positions
        positions = [i / (samples - 1) for i in range(samples)]
        
        # If no filter is provided, include all splines and channels
        if filter_channels is None:
            filter_channels = {}
            for spline_name in self.splines:
                filter_channels[spline_name] = list(self.splines[spline_name].channels.keys())
                
        # Determine the number of splines to plot
        num_splines = len(filter_channels)
        
        # Set theme first before creating any plots
        if theme == "dark":
            plt.style.use('dark_background')
            color_palette = ['#ff9500', '#00b9f1', '#fb02fe', '#01ff66', '#fffd01', '#ff2301']
            grid_color = '#444444'
            plt.rcParams.update({
                'text.color': '#ffffff',
                'axes.labelcolor': '#ffffff',
                'axes.edgecolor': '#444444',
                'axes.facecolor': '#121212',
                'figure.facecolor': '#121212',
                'grid.color': '#444444',
                'xtick.color': '#aaaaaa',
                'ytick.color': '#aaaaaa',
                'figure.edgecolor': '#121212',
                'savefig.facecolor': '#121212',
                'savefig.edgecolor': '#121212',
                'legend.facecolor': '#121212',
                'legend.edgecolor': '#444444',
                'patch.edgecolor': '#444444'
            })
        elif theme == "medium":
            plt.style.use('default')  # Base on default style
            color_palette = ['#ff9500', '#00b9f1', '#fb02fe', '#01ff66', '#fffd01', '#ff2301']
            grid_color = '#666666'
            plt.rcParams.update({
                'text.color': '#e0e0e0',
                'axes.labelcolor': '#e0e0e0',
                'axes.edgecolor': '#666666',
                'axes.facecolor': '#333333',
                'figure.facecolor': '#222222',
                'grid.color': '#666666',
                'xtick.color': '#cccccc',
                'ytick.color': '#cccccc',
                'figure.edgecolor': '#222222',
                'savefig.facecolor': '#222222',
                'savefig.edgecolor': '#222222',
                'legend.facecolor': '#333333',
                'legend.edgecolor': '#666666',
                'patch.edgecolor': '#666666'
            })
        else:  # light theme
            plt.style.use('default')
            color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            grid_color = 'lightgray'
            plt.rcParams.update({
                'text.color': '#333333',
                'axes.labelcolor': '#333333',
                'axes.edgecolor': '#bbbbbb',
                'axes.facecolor': '#ffffff',
                'figure.facecolor': '#ffffff',
                'grid.color': '#dddddd',
                'xtick.color': '#666666',
                'ytick.color': '#666666',
                'figure.edgecolor': '#ffffff',
                'savefig.facecolor': '#ffffff',
                'savefig.edgecolor': '#ffffff',
                'legend.facecolor': '#ffffff',
                'legend.edgecolor': '#cccccc',
                'patch.edgecolor': '#cccccc'
            })
            
        # Evaluate all channels at the sample positions
        results = []
        for position in positions:
            results.append(self.solve(position))
        
        if overlay:
            # Create a single figure for all splines/channels
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)
            
            # Used to track all channel names for a combined legend
            all_lines = []
            all_labels = []
            
            # Keep track of color index across all splines
            color_index = 0
            
            # Plot all splines and channels on the same axis
            for spline_name, channel_names in filter_channels.items():
                if spline_name not in self.splines:
                    continue
                    
                # Extract channel values at each sample position
                for channel_name in channel_names:
                    if channel_name in self.splines[spline_name].channels:
                        values = []
                        for j, position in enumerate(positions):
                            if spline_name in results[j] and channel_name in results[j][spline_name]:
                                values.append(results[j][spline_name][channel_name])
                            else:
                                # If channel isn't in result, use 0 or the previous value
                                prev_value = values[-1] if values else 0
                                values.append(prev_value)
                        
                        # Plot this channel with a unique color
                        color = color_palette[color_index % len(color_palette)]
                        color_index += 1
                        
                        # Create full label with spline and channel name
                        full_label = f"{spline_name}.{channel_name}"
                        line, = ax.plot(positions, values, label=full_label, color=color, linewidth=2)
                        all_lines.append(line)
                        all_labels.append(full_label)
                        
                        # Add keyframe markers
                        channel = self.splines[spline_name].channels[channel_name]
                        keyframe_positions = [kf.at for kf in channel.keyframes]
                        keyframe_values = []
                        for pos in keyframe_positions:
                            # Get the value at this keyframe position
                            normal_pos = self._normalize_position(pos)
                            result = self.solve(normal_pos)
                            if spline_name in result and channel_name in result[spline_name]:
                                keyframe_values.append(result[spline_name][channel_name])
                            else:
                                keyframe_values.append(0)  # Fallback
                                
                        ax.scatter(keyframe_positions, keyframe_values, color=color, s=60, zorder=5)
            
            # Set labels and title for the single plot
            ax.set_xlabel('Position')
            ax.set_ylabel('Value')
            ax.set_title(f"Solver: {self.name}")
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7, color=grid_color)
            
            # Add legend with all channels
            if theme == "dark":
                ax.legend(facecolor='#121212', edgecolor='#444444', labelcolor='white')
            elif theme == "medium":
                ax.legend(facecolor='#333333', edgecolor='#666666', labelcolor='#e0e0e0')
            else:  # light theme
                ax.legend(facecolor='#ffffff', edgecolor='#cccccc', labelcolor='#333333')
            
            # Set x-axis to 0-1 range
            ax.set_xlim(0, 1)
            
        else:
            # Create a figure with separate subplots for each spline (original behavior)
            fig = plt.figure(figsize=(12, 4 * num_splines))
            gs = GridSpec(num_splines, 1, figure=fig)
            
            # Plot each spline in its own subplot
            for i, (spline_name, channel_names) in enumerate(filter_channels.items()):
                if spline_name not in self.splines:
                    continue
                    
                # Create subplot
                ax = fig.add_subplot(gs[i])
                
                # Ensure subplot has dark theme in dark mode
                if theme == "dark":
                    ax.set_facecolor('#121212')
                
                # Set subplot title
                ax.set_title(f"Spline: {spline_name}")
                
                # Extract channel values at each sample position
                channel_values = {}
                for channel_name in channel_names:
                    if channel_name in self.splines[spline_name].channels:
                        channel_values[channel_name] = []
                        for j, position in enumerate(positions):
                            if spline_name in results[j] and channel_name in results[j][spline_name]:
                                channel_values[channel_name].append(results[j][spline_name][channel_name])
                            else:
                                # If channel isn't in result, use 0 or the previous value
                                prev_value = channel_values[channel_name][-1] if channel_values[channel_name] else 0
                                channel_values[channel_name].append(prev_value)
                
                # Plot each channel
                for j, (channel_name, values) in enumerate(channel_values.items()):
                    color = color_palette[j % len(color_palette)]
                    ax.plot(positions, values, label=channel_name, color=color, linewidth=2)
                    
                    # Add markers at keyframe positions
                    channel = self.splines[spline_name].channels[channel_name]
                    keyframe_positions = [kf.at for kf in channel.keyframes]
                    keyframe_values = []
                    for pos in keyframe_positions:
                        # Get the value at this keyframe position
                        normal_pos = self._normalize_position(pos)
                        result = self.solve(normal_pos)
                        if spline_name in result and channel_name in result[spline_name]:
                            keyframe_values.append(result[spline_name][channel_name])
                        else:
                            keyframe_values.append(0)  # Fallback
                            
                    ax.scatter(keyframe_positions, keyframe_values, color=color, s=60, zorder=5)
                
                # Set labels
                ax.set_xlabel('Position')
                ax.set_ylabel('Value')
                
                # Add grid and legend
                ax.grid(True, linestyle='--', alpha=0.7, color=grid_color)
                
                # Use custom legend style for each theme
                if theme == "dark":
                    ax.legend(facecolor='#121212', edgecolor='#444444', labelcolor='white')
                elif theme == "medium":
                    ax.legend(facecolor='#333333', edgecolor='#666666', labelcolor='#e0e0e0')
                else:  # light theme
                    ax.legend(facecolor='#ffffff', edgecolor='#cccccc', labelcolor='#333333')
                
                # Set x-axis to 0-1 range
                ax.set_xlim(0, 1)
            
            # Add solver name as figure title
            fig.suptitle(f"Solver: {self.name}", fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        if not overlay:
            plt.subplots_adjust(top=0.95)
        
        # Save the plot if a save path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        return fig
    
    def save_plot(
        self,
        filepath: str,
        samples: Optional[int] = None,
        filter_channels: Optional[Dict[str, List[str]]] = None,
        theme: str = "dark",
        overlay: bool = True
    ) -> None:
        """Save a plot of the solver's splines and channels to a file.
        
        Args:
            filepath: The file path to save the plot to (e.g., 'plot.png')
            samples: Number of evenly spaced samples to use (defaults to 100)
            filter_channels: Optional dictionary mapping spline names to lists of channel names to include
            theme: Plot theme - 'light' or 'dark'
            overlay: If True, all splines are plotted in a single graph; if False, each spline gets its own subplot
            
        Raises:
            ImportError: If matplotlib is not available
        """
        # Get the plot and save it
        self.get_plot(samples, filter_channels, theme, save_path=filepath, overlay=overlay)
    
    def plot(
        self, 
        samples: Optional[int] = None, 
        filter_channels: Optional[Dict[str, List[str]]] = None, 
        theme: str = "dark",
        save_path: Optional[str] = None,
        overlay: bool = True
    ):
        """Plot the solver's splines and channels.
        
        Args:
            samples: Number of evenly spaced samples to use (defaults to 100)
            filter_channels: Optional dictionary mapping spline names to lists of channel names to include
            theme: Plot theme - 'light' or 'dark'
            save_path: Optional file path to save the plot (e.g., 'plot.png')
            overlay: If True, all splines are plotted in a single graph; if False, each spline gets its own subplot
            
        Returns:
            None - displays the plot
            
        Raises:
            ImportError: If matplotlib is not available
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Plotting requires matplotlib. Install it with: pip install matplotlib")
            
        fig = self.get_plot(samples, filter_channels, theme, save_path, overlay)
        plt.show()
        return None
        
    def show(
            self, 
            samples: Optional[int] = None, 
            filter_channels: Optional[Dict[str, List[str]]] = None, 
            theme: str = "dark",
            save_path: Optional[str] = None,
            overlay: bool = True):
        self.plot(samples, filter_channels, theme, save_path, overlay)
    
    def save(self, filepath: str, format: Optional[str] = None) -> None:
        """Save the solver to a file.
        
        Args:
            filepath: The path to save to
            format: The format to save in (json, pickle, yaml, or numpy)
        """
        # Determine format from extension if not specified
        if format is None:
            ext = os.path.splitext(filepath)[1].lower()
            if ext == '.json':
                format = 'json'
            elif ext == '.pkl':
                format = 'pickle'
            elif ext == '.yaml' or ext == '.yml':
                format = 'yaml'
            elif ext == '.npz':
                format = 'numpy'
            else:
                format = 'json'  # Default to JSON
        
        # Convert to dictionary representation
        data = self._serialize()
        
        # Save in the appropriate format
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        elif format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        elif format == 'yaml':
            if not HAS_YAML:
                raise ImportError("PyYAML is required for YAML support")
            with open(filepath, 'w') as f:
                yaml.dump(data, f)
        elif format == 'numpy':
            if not HAS_NUMPY:
                raise ImportError("NumPy is required for NumPy support")
            metadata = json.dumps(data)
            np.savez(filepath, metadata=metadata)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _serialize(self) -> Dict[str, Any]:
        """Serialize the solver to a dictionary.
        
        Returns:
            Dictionary representation of the solver
        """
        # Start with basic information
        data = {
            "version": KEYFRAME_SOLVER_FORMAT_VERSION,  # Update version for new publish feature
            "name": self.name,
            "metadata": self.metadata,
            "range": self.range,
            "variables": {}
        }
        
        # Add publish directives if present
        if self.publish:
            data["publish"] = self.publish
        
        # Add variables (with conversion for NumPy types)
        for name, value in self.variables.items():
            if HAS_NUMPY and isinstance(value, np.ndarray):
                data["variables"][name] = value.tolist()
            elif HAS_NUMPY and isinstance(value, np.number):
                data["variables"][name] = float(value)
            else:
                data["variables"][name] = value
        
        # Add splines
        data["splines"] = {}
        for spline_name, spline in self.splines.items():
            # Create a dictionary for this spline
            spline_data = {
                "channels": {}
            }
            
            # Add channels
            for channel_name, channel in spline.channels.items():
                # Create a dictionary for this channel
                channel_data = {
                    "interpolation": channel.interpolation,
                    "keyframes": []
                }
                
                # Add min/max if set
                if channel.min_max is not None:
                    channel_data["min_max"] = channel.min_max
                
                # Add publish list if present
                if hasattr(channel, 'publish') and channel.publish:
                    channel_data["publish"] = channel.publish
                
                # Add keyframes
                for keyframe in channel.keyframes:
                    # Create a dictionary for this keyframe
                    # Convert function values to strings to avoid serialization errors
                    # We need to handle the fact that keyframe.value is a callable
                    # Let's try to convert it to a string representation if possible
                    value = None
                    
                    if isinstance(keyframe.value, (int, float)):
                        value = keyframe.value
                    elif isinstance(keyframe.value, str):
                        value = keyframe.value
                    else:
                        # This is probably a callable, so we'll just use a string representation
                        value = "0"  # Default fallback
                    
                    keyframe_data = {
                        "@": keyframe.at,  # Use @ instead of position
                        "value": value
                    }
                    
                    # Add interpolation if different from channel default
                    if keyframe.interpolation is not None:
                        keyframe_data["interpolation"] = keyframe.interpolation
                    
                    # Add parameters
                    params = {}
                    if keyframe.derivative is not None:
                        params["deriv"] = keyframe.derivative
                    if keyframe.control_points is not None:
                        params["cp"] = keyframe.control_points
                    if params:
                        keyframe_data["parameters"] = params
                    
                    # Add this keyframe to the channel data
                    channel_data["keyframes"].append(keyframe_data)
                
                # Add this channel to the spline data
                spline_data["channels"][channel_name] = channel_data
            
            # Add this spline to the data
            data["splines"][spline_name] = spline_data
        
        return data
    
    @classmethod
    def from_file(cls, filepath: str, format: Optional[str] = None) -> 'KeyframeSolver':
        """Load a solver from a file.
        
        Args:
            filepath: The path to load from
            format: Optional format override, one of ('json', 'pickle', 'yaml', 'numpy')
            
        Returns:
            The loaded Solver
        """
        
        # Determine format from extension if not specified
        if format is None:
            ext = os.path.splitext(filepath)[1].lower()
            if ext == '.json':
                format = 'json'
            elif ext == '.pkl':
                format = 'pickle'
            elif ext == '.yaml' or ext == '.yml':
                format = 'yaml'
            elif ext == '.npz':
                format = 'numpy'
            else:
                format = 'json'  # Default to JSON
        
        # Load based on format
        if format == 'json':
            with open(filepath, 'r') as f:
                solver_data = json.load(f)
        elif format == 'pickle':
            with open(filepath, 'rb') as f:
                solver_data = pickle.load(f)
        elif format == 'yaml':
            if not HAS_YAML:
                raise ImportError("PyYAML is required for YAML support")
            with open(filepath, 'r') as f:
                solver_data = yaml.safe_load(f)
        elif format == 'numpy':
            if not HAS_NUMPY:
                raise ImportError("NumPy is required for NumPy support")
            np_data = np.load(filepath)
            solver_data = json.loads(np_data['metadata'].item())
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        return cls._deserialize(solver_data)
    
   
    def load(self, filepath: str, format: Optional[str] = None) -> 'KeyframeSolver':
        """Load a solver from a file.
            Updates the solver in place # not efficient with copy attrs

        Args:
            filepath: The path to load from
            format: Optional format override, one of ('json', 'pickle', 'yaml', 'numpy')
            
        Returns:
            The loaded Solver
        """
        
        # Load the data into the current instance
        loaded_instance = self.__class__.from_file(filepath, format=format)
        
        # Copy all attributes from the loaded instance to self
        loaded_dict = vars(loaded_instance)
        
        # Copy all attributes to self
        for attr_name, attr_value in loaded_dict.items():
            setattr(self, attr_name, attr_value)
        
        # delete loaded_instance and loaded_dict
        del loaded_instance
        del loaded_dict
        
        # for backward compatibility return the instance
        return self
    
    @classmethod
    def _deserialize(cls, data: Dict[str, Any]) -> 'KeyframeSolver':
        """Deserialize a solver from a dictionary.
        
        Args:
            data: Dictionary representation of the solver
            
        Returns:
            The deserialized Solver
        """
        # Check version - require version KeyframeSolverFormatVersion
        if "version" in data:
            file_version = data["version"]
            if file_version != KEYFRAME_SOLVER_FORMAT_VERSION:
                raise ValueError(f"Unsupported KeyframeSolver file version: {file_version}. Current version is {KEYFRAME_SOLVER_FORMAT_VERSION}.")
        
        # Create a new solver
        solver = cls(name=data.get("name", "Untitled"))
        
        # Set range
        if "range" in data:
            solver.range = tuple(data["range"])
        
        # Set metadata
        solver.metadata = data.get("metadata", {})
        
        # Set variables
        for name, value in data.get("variables", {}).items():
            solver.set_variable(name, value)
            
        # Set publish directives
        if "publish" in data:
            for source, targets in data["publish"].items():
                solver.publish[source] = targets
        
        # Create splines
        splines_data = data.get("splines", {})
        
        # Handle both array and dictionary formats for splines
        if isinstance(splines_data, list):
            # Array format (each spline has a name field)
            for spline_item in splines_data:
                spline_name = spline_item.get("name", f"spline_{len(solver.splines)}")
                spline = solver.create_spline(spline_name)
                
                # Process channels
                channels_data = spline_item.get("channels", [])
                
                # Handle channels as array
                if isinstance(channels_data, list):
                    for channel_item in channels_data:
                        channel_name = channel_item.get("name", f"channel_{len(spline.channels)}")
                        interpolation = channel_item.get("interpolation", "cubic")
                        min_max = channel_item.get("min_max")
                        publish = channel_item.get("publish")
                        
                        # Convert list min_max to tuple (needed for test assertions)
                        if isinstance(min_max, list) and len(min_max) == 2:
                            min_max = tuple(min_max)
                        
                        channel = spline.add_channel(
                            name=channel_name,
                            interpolation=interpolation,
                            min_max=min_max,
                            replace=True,  # Add replace=True to handle duplicates
                            publish=publish
                        )
                        
                        # Add keyframes
                        keyframes_data = channel_item.get("keyframes", [])
                        for kf_data in keyframes_data:
                            # Handle keyframe as array [position, value] or as object
                            if isinstance(kf_data, list):
                                position = kf_data[0]
                                value = kf_data[1]
                                interp = None
                                control_points = None
                                derivative = None
                            else:
                                # Object format - only support "@" key for positions
                                position = kf_data.get("@", 0)
                                value = kf_data.get("value", 0)
                                interp = kf_data.get("interpolation")
                                params = kf_data.get("parameters", {})
                                
                                control_points = None
                                derivative = None
                                
                                if params:
                                    if "cp" in params:
                                        control_points = params["cp"]
                                    if "deriv" in params:
                                        derivative = params["deriv"]
                            
                            channel.add_keyframe(
                                at=position,
                                value=value,
                                interpolation=interp,
                                control_points=control_points,
                                derivative=derivative
                            )
                
                # Handle channels as dictionary (backward compatibility)
                elif isinstance(channels_data, dict):
                    for channel_name, channel_data in channels_data.items():
                        interpolation = channel_data.get("interpolation", "cubic")
                        min_max = channel_data.get("min_max")
                        
                        # Convert list min_max to tuple (needed for test assertions)
                        if isinstance(min_max, list) and len(min_max) == 2:
                            min_max = tuple(min_max)
                        
                        channel = spline.add_channel(
                            name=channel_name,
                            interpolation=interpolation,
                            min_max=min_max,
                            replace=True  # Add replace=True to handle duplicates
                        )
                        
                        # Add keyframes
                        for keyframe_data in channel_data.get("keyframes", []):
                            position = keyframe_data.get("position", 0)
                            value = keyframe_data.get("value", 0)
                            interp = keyframe_data.get("interpolation")
                            params = keyframe_data.get("parameters", {})
                            
                            control_points = None
                            derivative = None
                            
                            if params:
                                if "cp" in params:
                                    control_points = params["cp"]
                                if "deriv" in params:
                                    derivative = params["deriv"]
                            
                            channel.add_keyframe(
                                at=position,
                                value=value,
                                interpolation=interp,
                                control_points=control_points,
                                derivative=derivative
                            )
        else:
            # Dictionary format (backward compatibility)
            for spline_name, spline_data in splines_data.items():
                # Create a new spline
                spline = solver.create_spline(spline_name)
                
                # Process the spline data
                if isinstance(spline_data, dict):
                    # Check if there's a 'channels' key in the spline data (new format)
                    channels_data = spline_data.get("channels", {})
                    
                    # Process channels dictionary
                    if channels_data:
                        for channel_name, channel_data in channels_data.items():
                            # Create a channel
                            interpolation = channel_data.get("interpolation", "cubic")
                            min_max = channel_data.get("min_max")
                            publish = channel_data.get("publish")
                            
                            channel = spline.add_channel(
                                name=channel_name,
                                interpolation=interpolation,
                                min_max=min_max,
                                replace=True,  # Replace existing channel if it exists
                                publish=publish
                            )
                            
                            # Add keyframes
                            for keyframe_data in channel_data.get("keyframes", []):
                                # Support both old "position" key and new "@" key
                                position = keyframe_data.get("@", keyframe_data.get("position", 0))
                                value = keyframe_data.get("value", 0)
                                interp = keyframe_data.get("interpolation")
                                params = keyframe_data.get("parameters", {})
                                
                                control_points = None
                                derivative = None
                                
                                if params:
                                    if "cp" in params:
                                        control_points = params["cp"]
                                    if "deriv" in params:
                                        derivative = params["deriv"]
                                
                                channel.add_keyframe(
                                    at=position,
                                    value=value,
                                    interpolation=interp,
                                    control_points=control_points,
                                    derivative=derivative
                                )
                    else:
                        # Legacy format - channels directly in spline
                        for channel_name, channel_data in spline_data.items():
                            if channel_name != "name":  # Skip name field
                                # Create a channel
                                interpolation = "cubic"
                                min_max = None
                                
                                if isinstance(channel_data, dict):
                                    interpolation = channel_data.get("interpolation", "cubic")
                                    min_max = channel_data.get("min_max")
                                
                                channel = spline.add_channel(
                                    name=channel_name,
                                    interpolation=interpolation,
                                    min_max=min_max,
                                    replace=True  # Replace existing channel if it exists
                                )
                                
                                # Add keyframes if available
                                if isinstance(channel_data, dict) and "keyframes" in channel_data:
                                    for keyframe_data in channel_data["keyframes"]:
                                        # Support both old "position" key and new "@" key
                                        position = keyframe_data.get("@", keyframe_data.get("position", 0))
                                        value = keyframe_data.get("value", 0)
                                        interp = keyframe_data.get("interpolation")
                                        params = keyframe_data.get("parameters", {})
                                        
                                        control_points = None
                                        derivative = None
                                        
                                        if params:
                                            if "cp" in params:
                                                control_points = params["cp"]
                                            if "deriv" in params:
                                                derivative = params["deriv"]
                                        
                                        channel.add_keyframe(
                                            at=position,
                                            value=value,
                                            interpolation=interp,
                                            control_points=control_points,
                                            derivative=derivative
                                        )
        
        return solver
        
    def copy(self):
        """Create a deep copy of this solver.
        
        Returns:
            A new KeyframeSolver with the same data
        """
        # Create a new solver with the same name
        copied_solver = KeyframeSolver(name=self.name)
        
        # Copy range
        copied_solver.range = self.range
        
        # Copy metadata
        copied_solver.metadata = self.metadata.copy()
        
        # Copy variables
        for name, value in self.variables.items():
            copied_solver.set_variable(name, value)
        
        # Copy splines and their channels/keyframes
        for spline_name, spline in self.splines.items():
            copied_spline = copied_solver.create_spline(spline_name)
            
            # Copy channels
            for channel_name, channel in spline.channels.items():
                # Create new channel with same properties
                copied_channel = copied_spline.add_channel(
                    name=channel_name,
                    interpolation=channel.interpolation,
                    min_max=channel.min_max,
                    replace=True  # Add replace parameter
                )
                
                # Copy keyframes
                for kf in channel.keyframes:
                    copied_channel.add_keyframe(
                        at=kf.at,
                        value=kf.value(kf.at, {}),  # Extract the actual value
                        interpolation=kf.interpolation,
                        control_points=kf.control_points,
                        derivative=kf.derivative
                    )
        
        return copied_solver