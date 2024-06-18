import numpy as np
import scipy
import matplotlib.pyplot as plt
def matrix_sqrt(Sigma):
    """Compute the matrix square root of a covariance matrix Sigma."""
    return scipy.linalg.sqrtm(Sigma)
    # # Compute the matrix square root of Sigma to get F
    # F_sqrtm = np.linalg.cholesky(Sigma)

def boundary_point(theta, Sigma, pi, mu):
    """
    Calculate the point on the boundary of the uncertainty-aware safe-reachable set.
    
    Parameters:
    - theta: A parameter that parameterizes the boundary of the ellipse.
    - Sigma: The covariance matrix of the uncertainty region.
    - pi: The position of the pursuer.
    - mu: The center of the uncertainty ellipse.
    
    Returns:
    - A numpy array representing the point on the boundary.
    """
    # Compute the matrix square root of Sigma
    F = matrix_sqrt(Sigma)
    
    # Parameterize the boundary of the ellipse
    d_theta = np.array([np.cos(theta), np.sin(theta)])  # 2D case, adjust for higher dimensions
    
    # Compute the closest point 'a' on the boundary of the ellipse to the point 'q'
    a = mu + F @ d_theta
    
    # Compute the outward normal 'na' of the ellipse at point 'a'
    na = Sigma @ d_theta / np.linalg.norm(Sigma @ d_theta)
    
    # Calculate 'r' using the formula provided in the document
    r = (np.linalg.norm(pi - a)**2) / (2 * (pi - a).T @ na)
    
    # Calculate the point 'q' on the boundary of the safe-reachable set
    q = a + r * na
    
    return q

# Example usage:
# Define parameters
mu = np.array([0, 0])  # Example center of the uncertainty ellipse
Sigma = np.array([[1, 0], [0, 4]])  # Example covariance matrix
# # Define the covariance matrix Sigma for the 2D ellipse
# Sigma = np.array([[3, 1], [1, 2]])
# pursuers = [[4, 0], [-2, 4], [-2, -4]]  # Example pursuer position
pursuers = [[-2, -4]]  # Example pursuer position

# Parameters for the plot
# theta_range = np.linspace(0, 2 * np.pi, 100)  # 100 points around the ellipse
theta_range = np.linspace(0, np.pi, 100)  # 100 points around the ellipse

# Create a figure
fig, ax = plt.subplots()

# Function to draw an ellipse
def draw_ellipse(F_sqrtm, levels=10):
    # Generate a mesh grid
    theta = np.linspace(0, 2 * np.pi, 100)
    x = F_sqrtm[0, 0] * np.cos(theta)
    y = F_sqrtm[1, 1] * np.sin(theta)
    return x, y

# Draw the ellipse
F_sqrtm = np.linalg.cholesky(Sigma)
x, y = draw_ellipse(F_sqrtm)
ax.plot(x, y, color='b', label='Ellipse')
for pi in pursuers:
    # Plot the pursuer's position p_i
    ax.scatter(pi[0], pi[1], color='r', zorder=5, label='Pursuer')

    # Plot the boundary by varying theta
    for theta in theta_range:
        d_theta = np.array([np.cos(theta), np.sin(theta)])
        a = boundary_point(theta, Sigma, pi, mu)
        # a = F_sqrtm @ d_theta
        ax.plot([a[0]], [a[1]], 'go-', markersize=3, zorder=5)  # Green dots on the ellipse

# Set plot limits
ax.set_xlim(-8, 8)
ax.set_ylim(-8, 8)

# Add legend
ax.legend()

# Set plot title and labels
plt.title('2D Ellipsoid (Ellipse) with Pursuer Position')
plt.xlabel('X axis')
plt.ylabel('Y axis')
# Save the figure
plt.savefig('ellipse_with_pursuer.png', bbox_inches='tight', dpi=300)

# Show the plot
plt.grid(True)
plt.show()
