# EMI from a single moving charge

Here is given the math behind calculation of EMI parameters at specific target point, caused by a
single charge at another point, that moves with speed given as a vector. This is performed by
[emi_calc.calc_emi_dif()](../emi_calc.py#L34-L101).

## Parameters

The EMI parameters include the `base parameters` only, i.e. the magnetic field and its derivative,
a.k.a. Jacobian matrix.

> The `base parameters` are the ones that obey the superposition principle. In order to get the
> combined effect of two separate EM sources, over the same target point, they can be simply added.


### Magnetic field

The magnetic field is calculated from the Biot–Savart law for a point charge, see
[point charge at constant velocity](https://en.wikipedia.org/wiki/Biot%E2%80%93Savart_law#Point_charge_at_constant_velocity):

 ![$$\vec B = C . \vec v \times \frac{\vec r}{|\vec r|^3}$$](https://render.githubusercontent.com/render/math?math=%5Cvec%20B%20%3D%20C%20.%20%5Cvec%20v%20%5Ctimes%20%5Cfrac%7B%5Cvec%20r%7D%7B%7C%5Cvec%20r%7C%5E3%7D)

Where **C** is a constant equal to ![$$\frac{\mu_0 q}{4\pi}$$](https://render.githubusercontent.com/render/math?math=%5Cfrac%7B%5Cmu_0%20q%7D%7B4%5Cpi%7D).
This is the [`coef`](../emi_calc.py#L34) argument, that currently defaults to `1`.

#### Notes on the field calculation implementation

- The [`src_dir`](../emi_calc.py#L34) argument is the ![$$\vec v$$](https://render.githubusercontent.com/render/math?math=%5Cvec%20v), when zero - [all-zero params](../emi_calc.py#L52) are returned.

- The [`r`](../emi_calc.py#L48) variable is the ![$$\vec r$$](https://render.githubusercontent.com/render/math?math=%5Cvec%20r), when zero - [`None`](../emi_calc.py#L60) is returned.

- The magnetic field vector ![$$\vec B$$](https://render.githubusercontent.com/render/math?math=%5Cvec%20B)
is calculated [here](../emi_calc.py#L64), then it is scaled with the `coef` (a.k.a. **C**).

### Jacobian matrix

The Jacobian matrix is combined from the partial derivatives of the field vector, from the movement
of the target point along the 3 axes. To simplify the calculations, the derivative calculactions are
made in a coordinate system with axes:
- first, called **`l`**: along the `src_dir`
- second, called **`R`**: in the plane of `r` and `src_dir`, toward `r`, but perpendicular to `src_dir`
- third, called **`B`**: perpedicular to **`l`** and **`R`**, following the right hand rule, i.e.
 it is along the field vector

> In this coordinate system the field has only **`B`** component, and only it is changed when target
> is moving toward **`l`** or **`R`**. When target is moving along **`B`**, there is a change in
> **`R`** component of the field only (if movement approaches zero).

Finally, this matrix is transformed to the main coordinate system. The actual matix construction and
transformation is made by [build_jacobian()](../emi_calc.py#L8-L32). It needs the two gradient
components (`l_comp`, `R_comp`) and the vectors along the 3 axes.

#### Notes on the Jacobian components calculation

- The [`l`](../emi_calc.py#L55) and [`R`](../emi_calc.py#L56) vector-variables are the projections
 of `r` to the axes of this system, i.e. ![$$\vec r=\vec l + \vec R$$](https://render.githubusercontent.com/render/math?math=%5Cvec%20r%3D%5Cvec%20l%20%2B%20%5Cvec%20R)

- Given that ![$$\vec v$$](https://render.githubusercontent.com/render/math?math=%5Cvec%20v) and
![$$\vec R$$](https://render.githubusercontent.com/render/math?math=%5Cvec%20R) are perpendicular
(also the field magnitude only is changed),
![$$|\vec v \times \vec R| = |\vec v||\vec R|$$](https://render.githubusercontent.com/render/math?math=%7C%5Cvec%20v%20%5Ctimes%20%5Cvec%20R%7C%20%3D%20%7C%5Cvec%20v%7C%7C%5Cvec%20R%7C).
Thus, the derivative function is simplified to:

 ![$$\frac{|v||R|}{\sqrt{l^2 + |R|^2}^3}$$](https://render.githubusercontent.com/render/math?math=%5Cfrac%7B%7Cv%7C%7CR%7C%7D%7B%5Csqrt%7Bl%5E2%20%2B%20%7CR%7C%5E2%7D%5E3%7D)


- Partial gradient along the **`l`** axis:

 ![$$\frac{\partial\frac{|v||R|}{\sqrt{l^2 + |R|^2}^3}}{\partial l}$$](https://render.githubusercontent.com/render/math?math=%5Cfrac%7B%5Cpartial%5Cfrac%7B%7Cv%7C%7CR%7C%7D%7B%5Csqrt%7Bl%5E2%20%2B%20%7CR%7C%5E2%7D%5E3%7D%7D%7B%5Cpartial%20l%7D)

is calculated by using derivative calculator https://www.derivative-calculator.net (substitute `l`
with `x`):

```
Input: v*R / sqrt(x^2 + R^2)^3
Result: -3*v*R*x / (x^2 + R^2)^(5/2)
```

 ![$$\frac{d}{d x}\frac{v R}{\sqrt{x^2 + R^2}^3} = \frac{-3 v R x}{(x^2 + R^2)^\frac{5}{2}}$$](https://render.githubusercontent.com/render/math?math=%5Cfrac%7Bd%7D%7Bd%20x%7D%5Cfrac%7Bv%20R%7D%7B%5Csqrt%7Bx%5E2%20%2B%20R%5E2%7D%5E3%7D%20%3D%20%5Cfrac%7B-3%20v%20R%20x%7D%7B%28x%5E2%20%2B%20R%5E2%29%5E%5Cfrac%7B5%7D%7B2%7D%7D)


Substitute back `x` to `l`, then `(l^2 + R^2)^(1/2)` to `r_len` (this is the length of `r`):

```
Result: -3*v*R*l / r_len^5
```

This `l_comp` calculation is made [here](../emi_calc.py#L84).


- Partial gradient along the **`R`** axis