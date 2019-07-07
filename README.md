# TDSE-project

## Motivation

Solving the one-dimensional Time Dependent Schrodinger Equation using the finite differences method for the simple case of a free-particle.

## Importing libaries

```python
import math
import numpy as np
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
    return list(probability_density.real)
```

## Applying normalisation condition to a probability density function

```python
def norm(pdf):
    total = np.sum(np.array(pdf))
    norm_pdf = []
    for point in pdf:
        point /= total
        norm_pdf.append(point)
    return norm_pdf
```   
    
## Defining a third order approximation for the time derivative


```python
def time_deriv(i, old_wf, delta_x):
    if i == 0:
        return complex(0,1) * ((0 - (2 * old_wf[i]) + old_wf[i+1])/(2 * (delta_x**2)))
    if i == (len(old_wf) - 1):
        return complex(0,1) * ((old_wf[i-1] - (2 * old_wf[i]) + 0)/(2 * (delta_x**2)))
    else:
        return complex(0,1) * ((old_wf[i-1] - (2 * old_wf[i]) + old_wf[i+1])/(2 * (delta_x**2)))
```

## Defining a first order approximation for a step in time

```python
def time_step_first_order(old_wf, delta_x, delta_t):
    new_wf = []
    for i in range(len(old_wf)):
        new_wf.append(old_wf[i] + (delta_t * time_deriv(i, old_wf, delta_x)))
    return new_wf
```

## Plotting a first order approximation for a step in time

```python
positions = [0.01 * i for i in range(-5000,5001)]

wf_initial = [initial_wf(0.5, x) for x in positions]
pdf_initial = norm(pdf(wf_initial))

wf_one_step = time_step_first_order(wf_initial, 0.01, 1)
pdf_one_step = norm(pdf(wf_one_step))

plt.plot(positions, pdf_initial, positions, pdf_one_step)
plt.axis([-3, 3, 0, 0.006])
```

## Defining a fourth order Runge Kutta approximation for a step in time

```python
def time_step_rk4(wf_initial, x_coords, delta_x, delta_t):
    k_1 = []
    k_2 = []
    k_3 = []
    k_4 = []
    wf_updated = []
    for i in range(len(x_coords)):
        k_1.append(delta_t * time_deriv(i, wf_initial, delta_x))
    for i in range(len(x_coords)):
        wf_updated.append(wf_initial[i] + (0.5 * k_1[i]))
    for i in range(len(x_coords)):
        k_2.append(delta_t * time_deriv(i, wf_updated, delta_x))
    for i in range(len(x_coords)):
        wf_updated[i] = wf_initial[i] + k_2[i]/2
    for i in range(len(x_coords)):
        k_3.append(delta_t * time_deriv(i, wf_updated, delta_x))
    for i in range(len(x_coords)):
        wf_updated[i] = wf_initial[i] + k_3[i]
    for i in range(len(x_coords)):
        k_4.append(delta_t * time_deriv(i, wf_updated, delta_x))
    for i in range(len(x_coords)):
        wf_updated[i] = wf_initial[i] + (k_1[i] + 2*(k_2[i] + k_3[i]) + k_4[i])/6
    return(wf_updated) 
```

# Plotting a fourth order approximation for a step in time

```python
positions = [(0.01 * i) for i in range(-5000,5001)]
wf_initial = [initial_wf(0.5, x) for x in positions]
pdf_initial = norm(pdf(wf_initial))
wf_one_step_2 = rk4(wf_initial, positions, 0.01, 0.5)
pdf_one_step_2 = norm(pdf(wf_one_step_2))
plt.plot(positions, pdf_initial, positions, pdf_one_step_2)
plt.axis([-3, 3, 0, 0.006])
```
