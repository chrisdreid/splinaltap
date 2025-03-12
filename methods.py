"""
Interpolation methods for the splinaltap library.
"""

from typing import Dict, List, Tuple

try:
    import numpy as np
except ImportError:
    np = None

def nearest_neighbor(self, t: float, channels: Dict[str, float] = {}) -> float:
    """Nearest neighbor interpolation."""
    if not self.keyframes:
        raise ValueError("No keyframes defined")
    points = self._get_keyframe_points(channels)
    # Find closest point by time, return the value (second element)
    return min(points, key=lambda p: abs(p[0] - t))[1]

def linear_interpolate(self, t: float, channels: Dict[str, float] = {}) -> float:
    """Linear interpolation at time t."""
    if not self.keyframes:
        raise ValueError("No keyframes defined")
    points = self._get_keyframe_points(channels)
    if t <= points[0][0]:
        return points[0][1]
    if t >= points[-1][0]:
        return points[-1][1]
    for i in range(len(points) - 1):
        # Unpack x, y values (point is now (index, value, method))
        x0, y0, _ = points[i]
        x1, y1, _ = points[i + 1]
        if x0 <= t <= x1:
            return y0 + (y1 - y0) * (t - x0) / (x1 - x0)
    raise ValueError(f"Time {t} out of bounds")

def polynomial_interpolate(self, t: float, channels: Dict[str, float] = {}) -> float:
    """Lagrange polynomial interpolation."""
    if not self.keyframes:
        raise ValueError("No keyframes defined")
    points = self._get_keyframe_points(channels)
    # Extract x and y values, ignoring the method
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    result = 0.0
    for i in range(len(points)):
        term = y[i]
        for j in range(len(points)):
            if i != j:
                term *= (t - x[j]) / (x[i] - x[j])
        result += term
    return result

def quadratic_spline(self, t: float, channels: Dict[str, float] = {}) -> float:
    """Quadratic spline interpolation."""
    if not self.keyframes:
        raise ValueError("No keyframes defined")
    points = self._get_keyframe_points(channels)
    if len(points) < 2:
        raise ValueError("Quadratic spline requires at least 2 keyframes")
    if t <= points[0][0]:
        return points[0][1]
    if t >= points[-1][0]:
        return points[-1][1]

    cache_key = f"quadratic_coeffs_{hash(tuple(channels.items()))}"
    if cache_key not in self._precomputed:
        # Extract x and y values, ignoring the method
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        n = len(points) - 1
        coeffs = []
        for i in range(n):
            # Unpack x, y values (point is now (index, value, method))
            x0, y0, _ = points[i]
            x1, y1, _ = points[i + 1]
            if i == 0:
                a = 0
                b = (y1 - y0) / (x1 - x0)
                c = y0
            else:
                x_prev, y_prev, _ = points[i - 1]
                b_prev = coeffs[-1][1]
                a = (y1 - y0 - b_prev * (x1 - x0)) / ((x1 - x0) ** 2)
                b = b_prev
                c = y0 - a * (x0 - x_prev) ** 2 - b * (x0 - x_prev)
            coeffs.append((a, b, c))
        self._precomputed[cache_key] = (x, coeffs)

    x, coeffs = self._precomputed[cache_key]
    for i in range(len(coeffs)):
        if x[i] <= t <= x[i + 1]:
            a, b, c = coeffs[i]
            dt = t - x[i]
            return a * dt * dt + b * dt + c
    raise ValueError(f"Time {t} out of bounds")

def hermite_interpolate(self, t: float, channels: Dict[str, float] = {}) -> float:
    """Hermite cubic interpolation with optional derivatives."""
    if not self.keyframes:
        raise ValueError("No keyframes defined")
    points = [(x, self._evaluate_keyframe(x, x, channels), d or 0.0) for x, (f, d, _, _) in sorted(self.keyframes.items())]
    if len(points) < 2:
        raise ValueError("Hermite requires at least 2 keyframes")
    if t <= points[0][0]:
        return points[0][1]
    if t >= points[-1][0]:
        return points[-1][1]

    for i in range(len(points) - 1):
        x0, y0, m0 = points[i]
        x1, y1, m1 = points[i + 1]
        if x0 <= t <= x1:
            h = x1 - x0
            t_prime = (t - x0) / h
            h00 = 2 * t_prime**3 - 3 * t_prime**2 + 1
            h10 = t_prime**3 - 2 * t_prime**2 + t_prime
            h01 = -2 * t_prime**3 + 3 * t_prime**2
            h11 = t_prime**3 - t_prime**2
            return h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1
    raise ValueError(f"Time {t} out of bounds")

def bezier_interpolate(self, t: float, channels: Dict[str, float] = {}) -> float:
    """Cubic Bezier interpolation with control points."""
    if not self.keyframes:
        raise ValueError("No keyframes defined")
    points = [(x, self._evaluate_keyframe(x, x, channels), cp or (x + 0.1, self._evaluate_keyframe(x, x, channels) + 0.1, x + 0.2, self._evaluate_keyframe(x, x, channels) + 0.2)) 
              for x, (f, _, cp, _) in sorted(self.keyframes.items())]
    if len(points) < 2:
        raise ValueError("Bezier requires at least 2 keyframes")
    if t <= points[0][0]:
        return points[0][1]
    if t >= points[-1][0]:
        return points[-1][1]

    for i in range(len(points) - 1):
        x0, y0, (c1x, c1y, c2x, c2y) = points[i]
        x1, y1, _ = points[i + 1]
        if x0 <= t <= x1:
            t_prime = (t - x0) / (x1 - x0) if x1 != x0 else 0
            mt = 1 - t_prime
            return (mt**3 * y0 + 3 * mt**2 * t_prime * c1y + 
                    3 * mt * t_prime**2 * c2y + t_prime**3 * y1)
    raise ValueError(f"Time {t} out of bounds")

def gaussian_interpolate(self, t: float, channels: Dict[str, float] = {}) -> float:
    """Gaussian Process (Kriging) interpolation."""
    if not np:
        raise ImportError("Gaussian Process interpolation requires NumPy")
    if not self.keyframes:
        raise ValueError("No keyframes defined")
    points = self._get_keyframe_points(channels)
    
    cache_key = f"gaussian_weights_{hash(tuple(channels.items()))}"
    if cache_key not in self._precomputed:
        # Extract x and y values, ignoring the method
        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])
        sigma = 1.0
        K = np.exp(-((x[:, None] - x[None, :]) ** 2) / (2 * sigma ** 2))
        weights = np.linalg.solve(K + 1e-6 * np.eye(len(x)), y)
        self._precomputed[cache_key] = (x, weights, sigma)
    
    x, weights, sigma = self._precomputed[cache_key]
    k = np.exp(-((t - x) ** 2) / (2 * sigma ** 2))
    return np.dot(k, weights)

def pchip_interpolate(self, t: float, channels: Dict[str, float] = {}) -> float:
    """Piecewise Cubic Hermite Interpolation preserving monotonicity."""
    if not self.keyframes:
        raise ValueError("No keyframes defined")
    points = self._get_keyframe_points(channels)
    if len(points) < 2:
        raise ValueError("PCHIP requires at least 2 keyframes")
    if t <= points[0][0]:
        return points[0][1]
    if t >= points[-1][0]:
        return points[-1][1]

    cache_key = f"pchip_coeffs_{hash(tuple(channels.items()))}"
    if cache_key not in self._precomputed:
        # Extract x and y values, ignoring the method
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        n = len(points) - 1
        h = [x[i + 1] - x[i] for i in range(n)]
        delta = [(y[i + 1] - y[i]) / h[i] for i in range(n)]
        d = [0] * len(points)
        for i in range(1, n):
            if delta[i-1] * delta[i] > 0:
                w1 = 2 * h[i-1] + h[i]
                w2 = h[i-1] + 2 * h[i]
                d[i] = (w1 + w2) / (w1 / delta[i-1] + w2 / delta[i])
            else:
                d[i] = 0
        d[0] = 0 if delta[0] == 0 else (3 * delta[0] / 2 if delta[0] * (delta[0] + delta[1]) <= 0 else delta[0])
        d[-1] = 0 if delta[-1] == 0 else (3 * delta[-1] / 2 if delta[-1] * (delta[-2] + delta[-1]) <= 0 else delta[-1])
        coeffs = [(y[i], d[i], (3 * delta[i] - 2 * d[i] - d[i+1]) / h[i], (d[i] + d[i+1] - 2 * delta[i]) / (h[i] ** 2)) 
                  for i in range(n)]
        self._precomputed[cache_key] = (x, coeffs)

    x, coeffs = self._precomputed[cache_key]
    for i in range(len(coeffs)):
        if x[i] <= t <= x[i + 1]:
            dt = t - x[i]
            a, b, c, d = coeffs[i]
            return a + dt * (b + dt * (c + dt * d))
    raise ValueError(f"Time {t} out of bounds")

def cubic_spline(self, t: float, channels: Dict[str, float] = {}) -> float:
    """Cubic spline interpolation at time t (simplified natural spline)."""
    if not self.keyframes:
        raise ValueError("No keyframes defined")
    points = self._get_keyframe_points(channels)
    if len(points) < 2:
        raise ValueError("Cubic spline requires at least 2 keyframes")
    if t <= points[0][0]:
        return points[0][1]
    if t >= points[-1][0]:
        return points[-1][1]

    n = len(points) - 1
    # Extract x and y values, ignoring the method
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    h = [x[i + 1] - x[i] for i in range(n)]
    a = y[:-1]
    b, d = [0] * n, [0] * n
    c = [0] * (n + 1)
    alpha = [0] + [(3/h[i] * (y[i+1] - y[i]) - 3/h[i-1] * (y[i] - y[i-1])) for i in range(1, n)]
    l, mu, z = [1] + [0] * n, [0] * (n + 1), [0] * (n + 1)
    for i in range(1, n):
        l[i] = 2 * (x[i+1] - x[i-1]) - h[i-1] * mu[i-1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i-1] * z[i-1]) / l[i]
    for j in range(n-1, -1, -1):
        c[j] = z[j] - mu[j] * c[j+1]
        b[j] = (y[j+1] - y[j]) / h[j] - h[j] * (c[j+1] + 2 * c[j]) / 3
        d[j] = (c[j+1] - c[j]) / (3 * h[j])

    for i in range(n):
        if x[i] <= t <= x[i + 1]:
            s = t - x[i]
            return a[i] + b[i] * s + c[i] * s**2 + d[i] * s**3
    raise ValueError(f"Time {t} out of bounds")