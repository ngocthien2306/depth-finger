import numpy as np
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d
from typing import List, Tuple, Optional

class StrokeSmoothing:
    @staticmethod
    def kalman_smooth(points: List[Tuple[float, float]], 
                     process_noise: float = 0.001,
                     measurement_noise: float = 0.1) -> List[Tuple[float, float]]:
        """
        Kalman filter for stroke smoothing.
        
        Args:
            points: List of (x, y) coordinates
            process_noise: How much we trust the model prediction (Q)
            measurement_noise: How much we trust the measurements (R)
        """
        n_points = len(points)
        if n_points < 3:
            return points
            
        # Separate x and y coordinates
        x_coords, y_coords = zip(*points)
        
        # State transition matrix
        A = np.array([[1, 1], [0, 1]])
        
        # Measurement matrix
        H = np.array([[1, 0]])
        
        # Process noise covariance
        Q = np.array([[process_noise, 0], [0, process_noise]])
        
        # Measurement noise covariance
        R = np.array([[measurement_noise]])
        
        # Initial state and covariance
        x_state = np.array([[x_coords[0]], [0]])
        y_state = np.array([[y_coords[0]], [0]])
        P = np.eye(2)
        
        smoothed_points = []
        
        for i in range(n_points):
            # Predict
            x_state = A @ x_state
            y_state = A @ y_state
            P = A @ P @ A.T + Q
            
            # Update
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            
            x_measurement = np.array([[x_coords[i]]])
            y_measurement = np.array([[y_coords[i]]])
            
            x_state = x_state + K @ (x_measurement - H @ x_state)
            y_state = y_state + K @ (y_measurement - H @ y_state)
            P = (np.eye(2) - K @ H) @ P
            
            smoothed_points.append((float(x_state[0]), float(y_state[0])))
            
        return smoothed_points

    @staticmethod
    def spline_smooth(points: List[Tuple[float, float]], 
                     smoothing: float = 0.1,
                     num_points: Optional[int] = None) -> List[Tuple[float, float]]:
        """
        B-spline smoothing for more natural curves.
        
        Args:
            points: List of (x, y) coordinates
            smoothing: Smoothing factor (0 = interpolation, larger = smoother)
            num_points: Number of points in output (default: same as input)
        """
        if len(points) < 4:
            return points
            
        x_coords, y_coords = zip(*points)
        
        # Convert to numpy arrays
        points_combined = np.column_stack([x_coords, y_coords])
        
        # Fit B-spline
        tck, u = splprep([points_combined[:, 0], points_combined[:, 1]], 
                        s=smoothing, k=3)
        
        # Generate smooth curve
        if num_points is None:
            num_points = len(points)
        
        u_new = np.linspace(0, 1, num_points)
        smoothed_x, smoothed_y = splev(u_new, tck)
        
        return list(zip(smoothed_x, smoothed_y))

    @staticmethod
    def gaussian_smooth(points: List[Tuple[float, float]], 
                       sigma: float = 2.0) -> List[Tuple[float, float]]:
        """
        Gaussian smoothing with edge preservation.
        
        Args:
            points: List of (x, y) coordinates
            sigma: Standard deviation for Gaussian kernel
        """
        if len(points) < 3:
            return points
            
        x_coords, y_coords = zip(*points)
        
        # Apply Gaussian filter separately to x and y coordinates
        smoothed_x = gaussian_filter1d(x_coords, sigma=sigma)
        smoothed_y = gaussian_filter1d(y_coords, sigma=sigma)
        
        return list(zip(smoothed_x, smoothed_y))

    @staticmethod
    def hybrid_smooth(points: List[Tuple[float, float]], 
                     method1="spline",
                     method2="gaussian") -> List[Tuple[float, float]]:
        """
        Combines multiple smoothing methods for better results.
        
        Args:
            points: List of (x, y) coordinates
            method1: First smoothing method ('spline', 'gaussian', or 'kalman')
            method2: Second smoothing method ('spline', 'gaussian', or 'kalman')
        """
        smoother = StrokeSmoothing()
        
        # First pass
        if method1 == "spline":
            points = smoother.spline_smooth(points, smoothing=0.05)
        elif method1 == "gaussian":
            points = smoother.gaussian_smooth(points, sigma=1.5)
        elif method1 == "kalman":
            points = smoother.kalman_smooth(points)
            
        # Second pass
        if method2 == "spline":
            points = smoother.spline_smooth(points, smoothing=0.05)
        elif method2 == "gaussian":
            points = smoother.gaussian_smooth(points, sigma=1.5)
        elif method2 == "kalman":
            points = smoother.kalman_smooth(points)
            
        return points


# def visualize_smoothing_comparison():
#     import matplotlib.pyplot as plt
    
#     # Create noisy test data
#     t = np.linspace(0, 2*np.pi, 100)
#     x = t + np.random.normal(0, 0.1, 100)
#     y = np.sin(t) + np.random.normal(0, 0.1, 100)
#     original_points = list(zip(x, y))
    
#     generator = StrokeSampleGenerator()

#     original_points = generator.generate_handwriting(
#         num_points=10,
#         noise=0.05      
#     )
    
    
    
#     smoother = StrokeSmoothing()
    
#     # Apply different smoothing methods
#     import time 
#     s = time.time()
#     kalman_smoothed = smoother.kalman_smooth(original_points)
#     print('kalman_smoothed: ', str(round((time.time() - s) * 1000, 6)) + ' ms')
#     s = time.time()
#     spline_smoothed = smoother.spline_smooth(original_points)
#     print('spline_smoothed: ', str(round((time.time() - s) * 1000, 6)) + ' ms')
#     s = time.time()
#     gaussian_smoothed = smoother.gaussian_smooth(original_points)
#     print('gaussian_smoothed: ', str(round((time.time() - s) * 1000, 6))+ ' ms')
#     s = time.time()
#     hybrid_smoothed = smoother.hybrid_smooth(original_points)
#     print('hybrid_smoothed: ', str(round((time.time() - s) * 1000, 6))+ ' ms')
    
#     # Plot results
#     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))
    
#     # Original vs Kalman
#     ax1.plot(*zip(*original_points), 'b-', label='Original')
#     ax1.plot(*zip(*kalman_smoothed), 'r-', label='Kalman')
#     ax1.set_title('Kalman Filter Smoothing')
#     ax1.legend()
    
#     # Original vs Spline
#     ax2.plot(*zip(*original_points), 'b-', label='Original')
#     ax2.plot(*zip(*spline_smoothed), 'g-', label='Spline')
#     ax2.set_title('B-Spline Smoothing')
#     ax2.legend()
    
#     # Original vs Gaussian
#     ax3.plot(*zip(*original_points), 'b-', label='Original')
#     ax3.plot(*zip(*gaussian_smoothed), 'm-', label='Gaussian')
#     ax3.set_title('Gaussian Smoothing')
#     ax3.legend()
    
#     # Original vs Hybrid
#     ax4.plot(*zip(*original_points), 'b-', label='Original')
#     ax4.plot(*zip(*hybrid_smoothed), 'c-', label='Hybrid')
#     ax4.set_title('Hybrid Smoothing')
#     ax4.legend()
    
#     plt.tight_layout()
#     plt.show()
    


class StrokeSampleGenerator:
    @staticmethod
    def generate_circle(num_points: int = 100, radius: float = 1.0, noise: float = 0.05) -> List[Tuple[float, float]]:
        """Generate a noisy circle."""
        t = np.linspace(0, 2*np.pi, num_points)
        x = radius * np.cos(t) + np.random.normal(0, noise, num_points)
        y = radius * np.sin(t) + np.random.normal(0, noise, num_points)
        return list(zip(x, y))

    @staticmethod
    def generate_spiral(num_points: int = 100, noise: float = 0.05) -> List[Tuple[float, float]]:
        """Generate a noisy spiral."""
        t = np.linspace(0, 4*np.pi, num_points)
        r = t/4
        x = r * np.cos(t) + np.random.normal(0, noise, num_points)
        y = r * np.sin(t) + np.random.normal(0, noise, num_points)
        return list(zip(x, y))

    @staticmethod
    def generate_zigzag(num_points: int = 100, amplitude: float = 1.0, noise: float = 0.05) -> List[Tuple[float, float]]:
        """Generate a noisy zigzag pattern."""
        t = np.linspace(0, 4*np.pi, num_points)
        x = t
        y = amplitude * np.abs(np.sin(t)) + np.random.normal(0, noise, num_points)
        return list(zip(x, y))

    @staticmethod
    def generate_handwriting(num_points: int = 100, noise: float = 0.05) -> List[Tuple[float, float]]:
        """Generate a pattern similar to handwriting."""
        t = np.linspace(0, 4*np.pi, num_points)
        x = t + 0.4 * np.sin(2*t)
        y = np.sin(t) + 0.4 * np.cos(3*t)
        x += np.random.normal(0, noise, num_points)
        y += np.random.normal(0, noise, num_points)
        return list(zip(x, y))

    @staticmethod
    def generate_random_curve(num_points: int = 100, noise: float = 0.05) -> List[Tuple[float, float]]:
        """Generate a random smooth curve with noise."""
        t = np.linspace(0, 1, num_points)
        # Generate random coefficients for polynomial
        coeffs_x = np.random.randn(4)
        coeffs_y = np.random.randn(4)
        
        x = np.polyval(coeffs_x, t) + np.random.normal(0, noise, num_points)
        y = np.polyval(coeffs_y, t) + np.random.normal(0, noise, num_points)
        return list(zip(x, y))


generator = StrokeSampleGenerator()

points = generator.generate_handwriting(
    num_points=100,
    noise=0.05      
)

smoother = StrokeSmoothing()

smoothed_points = smoother.kalman_smooth(points)

smoothed_points = smoother.spline_smooth(points, smoothing=0.1)

smoothed_points = smoother.gaussian_smooth(points, sigma=2.0)

smoothed_points = smoother.hybrid_smooth(points, method1="spline", method2="gaussian")

