import numpy as np
import matplotlib.pyplot as plt

def ksdensity(data, width=0.3):
    """Returns kernel smoothing function from data points in data"""
    def ksd(x_axis):
        def n_pdf(x, mu=0., sigma=1.):  # normal pdf
            u = (x - mu) / abs(sigma)
            y = (1 / (np.sqrt(2 * np.pi) * abs(sigma)))
            y *= np.exp(-u * u / 2)
            return y
        
        prob = [n_pdf(x_i, data, width) for x_i in x_axis]
        pdf = [np.average(pr) for pr in prob]  # each row is one x value
        return np.array(pdf)
    
    return ksd


def rvs_1():
    """Plots KDEs for Gaussian and Uniform random variables, compares with exact distributions, 
    and shows histograms for Uniform distributions at different sample sizes (N = 100, 1000, 10000)."""
    # Plot for Gaussian random numbers
    # Generate Gaussian random data
    gaussian_data = np.random.randn(1000)

    # Create a new figure
    plt.figure(figsize=(6, 6))
    plt.hist(gaussian_data, bins=30, density=True, alpha=0.5, label='Histogram')

    # Estimate density using ksdensity
    ks_density_gaussian = ksdensity(gaussian_data, width=0.4)
    x_values_gaussian = np.linspace(-5, 5, 100)
    plt.plot(x_values_gaussian, ks_density_gaussian(x_values_gaussian), label='KDE', color='red')

    # Overlay exact Gaussian curve
    exact_gaussian = (1 / (np.sqrt(2 * np.pi))) * np.exp(-0.5 * x_values_gaussian**2)
    plt.plot(x_values_gaussian, exact_gaussian, label='Exact Gaussian', color='blue')

    plt.title('KDE for Gaussian Random Numbers')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()

    # Save the Gaussian plot
    plt.savefig('kde_gaussian_random_numbers.png')
    plt.close()  # Close the figure to free up memory

    # Plot for Uniform random numbers
    # Generate Uniform random data
    uniform_data = np.random.rand(1000)

    # Create a new figure
    plt.figure(figsize=(6, 6))
    plt.hist(uniform_data, bins=20, density=True, alpha=0.5, label='Histogram')

    # Estimate density using ksdensity
    ks_density_uniform = ksdensity(uniform_data, width=0.2)
    x_values_uniform = np.linspace(-1, 2, 100)
    plt.plot(x_values_uniform, ks_density_uniform(x_values_uniform), label='KDE', color='red')

    # Overlay exact Gaussian curve
    plt.plot(x_values_uniform, exact_gaussian[:len(x_values_uniform)], label='Exact Gaussian', color='blue')
    plt.title('KDE for Uniform Random Numbers')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()

    # Save the Uniform plot
    plt.savefig('kde_uniform_random_numbers.png')
    plt.close()  # Close the figure to free up memory



    # plotting uniform histogram for different values of N

    uniform_100 = np.random.rand(100)
    uniform_1000 = np.random.rand(1000)
    uniform_10000 = np.random.rand(10000)
    
    uniform_mean = 0.5
    uniform_sd = 1 / np.sqrt(12)

    fig, ax = plt.subplots(3)
    ax[0].hist(uniform_100, bins = 30, density = True, alpha = 0.5, label = 'N = 100')

    ax[1].hist(uniform_1000, bins = 30, density = True, alpha = 0.5, label = 'N = 1000')

    ax[2].hist(uniform_10000, bins = 30, density = True, alpha = 0.5, label = 'N = 10000')

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()

    fig.suptitle('Histogram of N uniformly distributed numbers')
    #plt.savefig('histograms_of_N_uniformly_distributed_numbers.png')
    plt.show()


def rvs_mult():
    """Plots histograms of Uniform[0,1] random samples of various sizes and overlays theoretical mean 
    and ±3 standard deviation lines for bin entries to compare empirical and expected frequencies."""
    num_bins = 25
    # Plot for Uniform random numbers with different sample sizes (uniform_100, uniform_1000, uniform_10000)
    uniform_100 = np.random.rand(100)
    uniform_1000 = np.random.rand(1000)
    uniform_10000 = np.random.rand(10000)

    fig, ax = plt.subplots(3, figsize=(12, 6))  # Create 3 subplots (one for each sample size)

    # Iterate through each sample size
    for j, uniform_data in enumerate([uniform_100, uniform_1000, uniform_10000]):
        counts, bin_edges, _ = ax[j].hist(uniform_data, bins=num_bins, alpha=0.5, label=f'N = {len(uniform_data)}')


        # Theoretical mean and standard deviation for bin entries
        N = len(uniform_data)  # Total number of data points
        bin_prob = 1 / num_bins  # Probability for each bin under uniform distribution

        theoretical_mean = N * bin_prob  # Theoretical mean for the number of entries in each bin
        theoretical_std = np.sqrt(N * bin_prob * (1 - bin_prob))  # Theoretical standard deviation

        # Plot the theoretical mean line (this will be constant across bins)
        ax[j].axhline(theoretical_mean, color='green', linestyle='--', label='Theoretical Mean (' + str(theoretical_mean) + ')')
        # Plot the theoretical SD lines (the same for each bin)
        ax[j].axhline(theoretical_mean + theoretical_std, color='orange', linestyle='--', label=f'Theoretical +3 SD ({theoretical_mean + 3*theoretical_std:.2f})')
        ax[j].axhline(theoretical_mean - theoretical_std, color='orange', linestyle='--', label=f'Theoretical -3 SD ({theoretical_mean - 3*theoretical_std:.2f})')

        # Set titles and labels for each subplot
        ax[j].set_title(f'Histogram for Uniform Random Numbers (N={N})')
        ax[j].set_xlabel('Value')
        ax[j].set_ylabel('Frequency')
        ax[j].legend(loc = 'upper left', bbox_to_anchor=(1,1))
    
    plt.subplots_adjust(right=0.8, hspace=1.5)

    # Set the overall title for the figure
    fig.suptitle('Histograms of Uniformly Distributed Numbers with Theoretical Mean and ±3 SD of Bin Entries')

    # Save the plot as an image file
    plt.savefig('histograms_of_N_uniformly_distributed_numbers.png')
    plt.show()  # Display the plot

def f_rvs():
    """Demonstrates random variable transformations:
    1. Linear transformation of a standard Gaussian (Y = aX + b),
    2. Nonlinear transformation (Y = X²) showing how the distributions change."""
    std_gauss_data = np.random.randn(1000)
    x_values = np.linspace(-5, 35, 1000)
    lin_tf_data = 5 * std_gauss_data + 15
    exact_lin_tf_pdf = (1 /(5 * (np.sqrt(2 * np.pi)))) * np.exp(-0.5 * (((x_values-15)/5)**2))
    
    plt.figure(figsize=(10,6))
    plt.hist(lin_tf_data, bins=30, density=True, alpha = 0.5, label = 'Histogram of ax+b', color='blue')
    plt.plot(x_values, exact_lin_tf_pdf, label='Exact Gaussian', color='red')
    plt.title('Transformed Standard Gaussian using y = ax + b')
    plt.legend()
    plt.savefig('rv_transform_ax+b')
    plt.close()


    x_values = np.linspace(1e-6, 6, 1000)
    sq_tf_data = std_gauss_data**2
    exact_sq_tf_pdf = (1 / (np.sqrt(2 * np.pi* x_values))) * np.exp(-0.5 * x_values)

    plt.figure(figsize=(10,6))
    plt.hist(sq_tf_data, bins=30, density=True, alpha = 0.5, label = 'Histogram of x^2', color='blue')
    plt.plot(x_values, exact_sq_tf_pdf, label='Exact pdf p(y)', color='red')
    plt.title('Transformed Standard Gaussian using y = x^2')
    plt.legend()
    plt.ylim(0,2)
    plt.xlim(0, 7)
    plt.savefig('rv_transform_x^2')


def inv_cdf():
    """Generates samples from an exponential distribution using the inverse CDF (-ln(X)) 
    method applied to Uniform[0,1] data and compares with the exact exponential PDF."""
    x_values = np.random.rand(1000)
    y_values = np.linspace(0, 7, 1000)
    true_pdf = np.exp(-1* y_values)
    inv_cdf_pdf = -1 * np.log(x_values) 

    plt.figure(figsize=(10,6))
    plt.hist(inv_cdf_pdf, bins=30, density=True, alpha = 0.5, label = 'Histogram of -ln(X), where X~U(0, 1)', color='blue')
    plt.plot(y_values, true_pdf, label='Exact pdf p(y) = exp(-y)', color='red')
    plt.title('Generating samples of the exponential dist. from sampling the uniform dist. and transforming with ln(.)')
    plt.legend()
    plt.xlim(0,6.5)
    plt.savefig('inverse_cdf_method')
    plt.show()

    return

def sim(N, alpha, tol=200):
    """Simulates random variables from a class of complex, heavy-tailed distributions 
    for different skewness values (beta), using an alpha-stable distribution formulation. 
    Crops out extreme values to reduce outlier influence and visualizes the result with histograms."""
    betas = [-1, -0.5, 0, 0.5, 1]
    gen_init = int(N + tol)

    # produce lists of b and s values corresponding to beta
    B = [(1/alpha) * np.arctan(beta * np.tan(np.pi * alpha /2)) for beta in betas]
    S = [( 1 + beta**2 * np.tan(np.pi * alpha /2)**2)**(1/(2*alpha)) for beta in betas]

    # produce the specified rvs
    U = np.random.uniform(low = -np.pi/2, high = np.pi/2 , size = gen_init)
    V = np.random.exponential(scale=2, size = gen_init)
    #bs = [(b, s) for b in B for s in S]

    # initialise x as a 5 row array to store the values generated by the rv X
    # each row of x corresponds to a value of beta
    x = np.zeros((5, gen_init))
    for i in range(len(betas)):
        b = B[i]
        s = S[i]
    
        x[i, :] = [
            s * ((np.sin(alpha * (u + b)) / (np.cos(u))**(1 / alpha))) *
            (((np.cos(u - alpha * ((u + b))) / v))**(1 - alpha / alpha))
            for u, v in zip(U, V) 
        ]

    # sort x and take the central values to account for spurious huge values
    # dne by cropping out a total of 'tol' fringe datapoints 
    x_sorted = np.sort(x, axis=1)
    crop = int(tol/2)
    middle = x_sorted[:, crop:-crop]

    x_min = np.min(middle)
    x_max = np.max(middle)

    fig, axes = plt.subplots(5, 1, figsize=(10, 12)) 

    for i in range(len(betas)):
        axes[i].hist(middle[i, :], bins=500, alpha=0.5, color='blue')
        axes[i].set_title('Histogram for Beta = ' + str(betas[i]))
        axes[i].set_xlabel('Value of x')
        axes[i].set_ylabel('Frequency')    
        axes[i].set_xlim(x_min, x_max)

    plt.subplots_adjust(hspace = 1.5)
    plt.savefig('difficult_density_alpha_' + str(alpha) + '_tol_' + str(tol)+ '.png') 

    return

