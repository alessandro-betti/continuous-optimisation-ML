#I want to write a program to sample from the following probability distribution in 2 dimensions:
#\pi(A) = \frac{1}{\pi}\int_{A} e^{-(x-y)^2} e^{-x^2}\, dxdy
#This means that the corresponding density function is:
#f(x,y) = e^{-(x-y)^2} e^{-x^2}
#I want to sample from this distribution using the Metropolis-Hastings algorithm.
#I want to use a Gaussian proposal distribution with mean (0,0) and standard deviation 1.
#I want to run the algorithm for 100000 iterations and then plot the samples.

import numpy as np
import matplotlib.pyplot as plt

def target_distribution(x,y):
    return np.exp(-(x-y)**2) * np.exp(-x**2)

def proposal_distribution():
    # Return a 2D proposal
    return np.random.normal(0, 1, size=2)

def metropolis_hastings(num_samples):
    samples = np.zeros((num_samples, 2))  # Pre-allocate array for efficiency
    current_x = 0
    current_y = 0
    accepted = 0
    
    for i in range(num_samples):
        # Propose a new sample using 2D proposal
        proposal = proposal_distribution()
        proposed_x = current_x + proposal[0]
        proposed_y = current_y + proposal[1]

        # Calculate the acceptance probability
        # Add small constant to avoid division by zero
        acceptance_probability = min(
            1,
            target_distribution(proposed_x, proposed_y) / 
            (target_distribution(current_x, current_y) + 1e-10)
        )
        
        # Accept or reject the proposed sample
        if np.random.rand() < acceptance_probability:
            current_x = proposed_x
            current_y = proposed_y
            accepted += 1
            
        samples[i] = [current_x, current_y]
    
    acceptance_rate = accepted / num_samples
    print(f"Acceptance rate: {acceptance_rate:.2%}")
    return samples

def plot_samples(samples):
    """Function to plot the samples and their distributions"""
    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.scatter(samples[:,0], samples[:,1], alpha=0.1, s=1)
    plt.title('Samples from Target Distribution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()

    # Plot marginal distributions
    plt.figure(figsize=(12, 4))

    plt.subplot(121)
    plt.hist(samples[:,0], bins=50, density=True, alpha=0.7)
    plt.title('Marginal Distribution of x')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.grid(True)

    plt.subplot(122)
    plt.hist(samples[:,1], bins=50, density=True, alpha=0.7)
    plt.title('Marginal Distribution of y')
    plt.xlabel('y')
    plt.ylabel('Density')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # This code only runs when sample.py is run directly
    samples = metropolis_hastings(100000)
    plot_samples(samples)
    # Save the samples to a file
    np.savetxt('samples.txt', samples)


