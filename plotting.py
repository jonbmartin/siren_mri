import matplotlib.pyplot as plt

# plot the weights of the network 
def plot_weight_distribution(model, savefile):
    """Plots the distribution of weights in a PyTorch model."""

    all_weights = []
    for param in model.parameters():
        if param.requires_grad:
            all_weights.extend(param.data.view(-1).cpu().numpy())

    plt.hist(all_weights, bins=500, range=(-0.015, 0.015))
    plt.title("Weight Distribution")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    #plt.yscale('log')  # Set y-axis to log scale
    plt.xlim([-0.015,0.015])
    # plt.show()
    plt.savefig(savefile)

