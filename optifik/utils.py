class OptimizeResult(dict):
    """ Represents the optimization result.

    Notes
    -----
    This class has been copied from scipy.optimize

    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())




def is_latex_installed():
    import shutil
    return shutil.which("latex") is not None or shutil.which("pdflatex") is not None


def setup_matplotlib():
    """
    Configure matplotlib with LaTeX text rendering and custom font sizes.

    """
    import matplotlib.pyplot as plt
    plt.rc('text', usetex=is_latex_installed())
    plt.rcParams.update({
        'figure.dpi': 300,
        'figure.figsize': (10, 6),
        'axes.labelsize': 26,
        'xtick.labelsize': 32,
        'ytick.labelsize': 32,
        'legend.fontsize': 23,
    })
