# TDSE-project

## Motivation

Solving the TDSE using the finite differences method for the simple case of a free-particle.

## Importing libaries

```python
import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation, rc
```

## Creating a gaussian shaped initial wavefunction

```python
def initial_wavefunction(a,x):
    return math.exp(-a * (x ** 2))
```

## Converting a wavefunction into a probability density function

```python
def probability_density(wavefunction):
    wavefunction = np.array(wavefunction)
    conjugate_wavefunction = np.conjugate(wavefunction)
    probability_density = wavefunction * conjugate_wavefunction
    probability_density_real = list(probability_density.real)
    return probability_density_real
```

## Applying normalisation condition to a probability density function

```python
def normalisation(prob_density, x_gap):
    total = np.sum(np.array(prob_density) * x_gap)
    normalised_prob_density = []
    for point in prob_density:
        point /= total
        normalised_prob_density.append(point)
    return normalised_prob_density
```   
    
## Defining the first order step in time for a point in space

```python
def first_order(i, delta_x, delta_t, old_wavefunction):
    return complex(0,1) * delta_t * ((old_wavefunction[i-1] - (2 * old_wavefunction[i]) + old_wavefunction[i + 1])/(2 * (delta_x**2)))
```

## Defining a second order step in time for a point in space

```python
def second_order(i, delta_x, delta_t, old_wavefunction):
    return complex(0,1) * delta_t * ((first_order(i-1, delta_x, delta_t, old_wavefunction) - (2 * first_order(i, delta_x, delta_t, old_wavefunction)) + first_order(i+1, delta_x, delta_t, old_wavefunction))/(2 * (delta_x**2)))
```

## Applying a first order step in time to every point on a wavefunction

```python
def time_step_forward_first(delta_x, delta_t, old_wavefunction):
    new_wavefunction = []
    old_wavefunction = [0] + old_wavefunction + [0]
    for i in range(1, (len(old_wavefunction) - 1)):
        new_wavefunction.append(old_wavefunction[i] + first_order(i, delta_x, delta_t, old_wavefunction))
    new_probability_density = normalisation(probability_density(new_wavefunction),1)
    return new_probability_density
```

## Applying a second order step in time to every point on a wavefunction

```python
def time_step_forward_second(delta_x, delta_t, old_wavefunction):
    new_wavefunction = []
    old_wavefunction = [0] + [0] + old_wavefunction + [0] + [0]
    for i in range(2, (len(old_wavefunction) - 2)):
        new_wavefunction.append(old_wavefunction[i] + first_order(i, delta_x, delta_t, old_wavefunction) + (0.5 * second_order(i, delta_x, delta_t, old_wavefunction) * (delta_t**2)))
    new_probability_density = normalisation(probability_density(new_wavefunction),1)
    return new_probability_density
```

## Creating a set of axes for the animation

```python
fig = plt.figure(figsize = (12,12))
ax = plt.axes(xlim=(-1, 1), ylim=(0, 0.0025))
plt.title('Evolution of the Probability Density for a Free Particle with Time')
plt.xlabel('Position')
plt.ylabel('Probability Density')
line, = ax.plot([], [], lw=2)
plt.close()
```

## Initialising the animation

```python
def init():
    line.set_data([], [])
    return line,
```

## Defining a set of frames for the animation

```python
def animate(i):
    x = positions
    y = time_step_forward_second(0.25, i, initial_wavefunction)
    line.set_data(x, y)
    return line,
```

## Outputting the animation

```python
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=60, interval=150, blit=True)
%matplotlib inline
rc('animation', html='html5')
anim
```
