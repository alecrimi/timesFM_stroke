import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib.cm as cm
from scipy.special import comb

def bernstein_poly(i, n, t):
    """
    Bernstein polynomial for Bézier curve calculation.
    """
    return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))

def bezier_curve(points, n_times=100):
    """
    Generate a Bézier curve from control points.
    
    Parameters:
    points: control points for the curve
    n_times: number of time steps for the curve
    """
    n_points = len(points)
    t = np.linspace(0, 1, n_times)
    
    curve = np.zeros((n_times, 3))
    for i in range(n_points):
        curve += np.outer(bernstein_poly(i, n_points-1, t), points[i])
    
    return curve

def generate_coupled_logistic_maps(n=100, r=3.8, alpha=0.02, c1=0.1, c2=0.2, c3=0.3, epsilon_range=0.01):
    """
    Generate three unidirectionally coupled logistic map time series.
    """
    # Initialize arrays
    X1 = np.zeros(n + 1)
    X2 = np.zeros(n + 1)
    X3 = np.zeros(n + 1)
    
    # Set initial conditions with random perturbations
    np.random.seed(42)  # For reproducible results
    X1[0] = c1 + np.random.uniform(-epsilon_range, epsilon_range)
    X2[0] = c2 + np.random.uniform(-epsilon_range, epsilon_range)
    X3[0] = c3 + np.random.uniform(-epsilon_range, epsilon_range)
    
    # Generate time series using update equations
    for t in range(1, n + 1):
        # Standard logistic map for X1
        X1[t] = r * X1[t-1] * (1 - X1[t-1])
        
        # Coupled logistic maps for X2 and X3
        X2[t] = r * X2[t-1] * (1 - X2[t-1]) + alpha * X1[t-1]
        X3[t] = r * X3[t-1] * (1 - X3[t-1]) + alpha * X2[t-1]
        
        # Keep values in reasonable bounds to prevent divergence
        X1[t] = np.clip(X1[t], 0, 1.2)
        X2[t] = np.clip(X2[t], -0.5, 1.5)
        X3[t] = np.clip(X3[t], -0.5, 1.5)
        
        # Check for NaN or infinite values
        if not np.isfinite(X1[t]):
            X1[t] = 0.5
        if not np.isfinite(X2[t]):
            X2[t] = 0.5
        if not np.isfinite(X3[t]):
            X3[t] = 0.5
    
    return X1[1:], X2[1:], X3[1:]

def create_smooth_bezier_trajectory(X1, X2, X3, segments=5, points_per_segment=10):
    """
    Create smooth Bézier curve trajectories.
    
    Parameters:
    X1, X2, X3: original time series data
    segments: number of points to use for each Bézier segment
    points_per_segment: number of interpolated points per segment
    """
    n_points = len(X1)
    points_3d = np.column_stack((X1, X2, X3))
    
    # Create Bézier curve segments
    bezier_curves = []
    for i in range(0, n_points - segments, segments):
        control_points = points_3d[i:i+segments+1]
        curve = bezier_curve(control_points, points_per_segment)
        bezier_curves.append(curve)
    
    # Combine all curves
    full_curve = np.vstack(bezier_curves)
    
    return full_curve[:, 0], full_curve[:, 1], full_curve[:, 2]

def create_3d_animation(X1, X2, X3, save_gif=False, filename='coupled_logistic_3d.gif'):
    """
    Create an animated 3D plot with Bézier curves.
    """
    # Create smooth Bézier curves
    X1_smooth, X2_smooth, X3_smooth = create_smooth_bezier_trajectory(X1, X2, X3)
    n_points = len(X1_smooth)
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create color map based on time
    colors = cm.viridis(np.linspace(0, 1, n_points))
    
    # Calculate robust axis limits
    x_min, x_max = np.percentile(X1, [1, 99])
    y_min, y_max = np.percentile(X2, [1, 99])
    z_min, z_max = np.percentile(X3, [1, 99])
    
    # Add padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    
    x_padding = max(0.1, x_range * 0.1)
    y_padding = max(0.1, y_range * 0.1)
    z_padding = max(0.1, z_range * 0.1)
    
    def animate(frame):
        ax.clear()
        
        # Set labels and title
        ax.set_xlabel('X₁(t)', fontsize=12)
        ax.set_ylabel('X₂(t)', fontsize=12)
        ax.set_zlabel('X₃(t)', fontsize=12)
        ax.set_title(f'Coupled Logistic Maps 3D Trajectory with Bézier Curves\n(r=3.8, α=0.1, n={n_points})', 
                    fontsize=14, pad=20)
        
        # Plot the trajectory up to current frame
        if frame > 0:
            # Plot smooth Bézier curve
            ax.plot(X1_smooth[:frame], X2_smooth[:frame], X3_smooth[:frame], 
                   'k-', alpha=0.3, linewidth=1.5)
            
            # Plot points with time-based colors
            scatter = ax.scatter(X1_smooth[:frame], X2_smooth[:frame], X3_smooth[:frame], 
                               c=colors[:frame], s=30, alpha=0.8, cmap='viridis')
        
        # Highlight current point
        if frame > 0:
            ax.scatter(X1_smooth[frame-1], X2_smooth[frame-1], X3_smooth[frame-1], 
                      c='red', s=80, alpha=1.0, marker='o', edgecolors='black')
        
        # Set axis limits using robust ranges
        ax.set_xlim([x_min - x_padding, x_max + x_padding])
        ax.set_ylim([y_min - y_padding, y_max + y_padding])
        ax.set_zlim([z_min - z_padding, z_max + z_padding])
        
        # Rotate the view
        ax.view_init(elev=20, azim=frame * 2)  # Rotate 2 degrees per frame
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add frame number
        ax.text2D(0.02, 0.98, f'Time step: {frame}', transform=ax.transAxes, 
                  fontsize=10, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=n_points + 20, 
                                  interval=100, blit=False, repeat=True)
    
    # Save as GIF if requested
    if save_gif:
        try:
            print(f"Attempting to save animation as {filename}...")
            writergif = animation.PillowWriter(fps=10)
            anim.save(filename, writer=writergif)
            print(f"Animation saved as {filename}")
        except Exception as e:
            print(f"Could not save GIF: {e}")
            print("You can still view the animation in the notebook")
    
    plt.tight_layout()
    plt.show()
    
    return anim

def plot_time_series(X1, X2, X3):
    """
    Plot the individual time series for visualization.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    time = np.arange(len(X1))
    
    axes[0].plot(time, X1, 'b-', linewidth=1.5, label='X₁(t)')
    axes[0].set_ylabel('X₁(t)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].plot(time, X2, 'g-', linewidth=1.5, label='X₂(t)')
    axes[1].set_ylabel('X₂(t)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    axes[2].plot(time, X3, 'r-', linewidth=1.5, label='X₃(t)')
    axes[2].set_xlabel('Time step')
    axes[2].set_ylabel('X₃(t)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.suptitle('Coupled Logistic Map Time Series', fontsize=16)
    plt.tight_layout()
    plt.show()

def main():
    # Parameters
    n = 100  # number of time points
    r = 3.8  # logistic map parameter (chaotic regime)
    alpha = 0.02  # reduced coupling strength to prevent instability
    
    # Generate the coupled time series
    print("Generating coupled logistic map time series...")
    print(f"Parameters: r={r}, α={alpha}, n={n}")
    X1, X2, X3 = generate_coupled_logistic_maps(n=n, r=r, alpha=alpha)
    
    print(f"Generated {n} time points for each series")
    print(f"X1 range: [{X1.min():.6f}, {X1.max():.6f}]")
    print(f"X2 range: [{X2.min():.6f}, {X2.max():.6f}]")
    print(f"X3 range: [{X3.min():.6f}, {X3.max():.6f}]")
    
    # Check for any problematic values
    print(f"X1 finite values: {np.isfinite(X1).sum()}/{len(X1)}")
    print(f"X2 finite values: {np.isfinite(X2).sum()}/{len(X2)}")
    print(f"X3 finite values: {np.isfinite(X3).sum()}/{len(X3)}")
    
    # Plot individual time series
    plot_time_series(X1, X2, X3)
    
    # Create 3D animation with Bézier curves
    print("\nCreating 3D animation with Bézier curves...")
    anim = create_3d_animation(X1, X2, X3, save_gif=True, filename='coupled_logistic_bezier_3d.gif')
    
    return X1, X2, X3, anim

if __name__ == "__main__":
    X1, X2, X3, animation_obj = main()
