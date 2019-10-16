import matplotlib.pyplot as plt
def figFontSizes(small=14, medium=16, large=20):

    plt.rc('font', size=small)  # controls default text sizes
    plt.rc('axes', titlesize=large)  # fontsize of the axes title
    plt.rc('axes', labelsize=medium)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=small)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=small)  # fontsize of the tick labels
    plt.rc('legend', fontsize=small)  # legend fontsize
    plt.rc('figure', titlesize=large)  # fontsize of the figure title


figFontSizes()