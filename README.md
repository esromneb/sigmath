# Sigmath
A bunch of helper functions jammed into the same file when they should not be.

# Setup Ubuntu
Get pip:
```bash
sudo apt -y install python-pip build-essential python-dev
```

After you get pip:
```bash
pip install scikit-commpy python-interface
```

On Ubuntu 20.04 you may get this error:

```bash
UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
```

Solve it with
```bash
apt-get install python3.8-tk
```

# Show plots :100:
Show visual test coverage of all plotting types we have sugar for.

> `make plot`  

![Image of available plots](/screenshots/plot_test.png "Available Plots")


# Socket
* `nonblock_socket` - Allows python to read from a socket without blocking when there is no data

# Plotting, Nplot
Wrappers around matplotlib that make my life easier
* `nplot` - Basic plot
* `ncplot` - Basic plot, shows real in blue and complex in red
* `nplotdots` - Plot dots without line segments

# OsiBase
A horrible class that should not have been written.  This should be simplified to be single directional
