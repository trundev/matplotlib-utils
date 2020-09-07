<!--
This source is edited with VSCode, with help of:
- "Markdown+Math" (vscode:extension/goessner.mdmath) -- Preview during editing
- "Markdown Preview Github Styling" (vscode:extension/bierner.markdown-preview-github-styles) -- Match GitHub style
- Task "Math to image" (docs/math2img.py) -- Generate markdown-image tag URLs from LaTeX math expressions
-->
# Math behind EMI calculations

Main calculations are made by [emi_calc.calc_all_emis()](../emi_calc.py#L259-L321).

The EMI parameters for each target point are calculated by summing the EM effect caused by each of
the source line, where electric current flows.

> TODO: The module includes *differential* [calc_emi_dif()](../emi_calc.py#L34-L101) and *integral*
> [calc_emi()](../emi_calc.py#L103-L236) calculation routines. The first one is to calculate the EM
> effect caused by a single charged particle, the second one -- by a electric current along a line.
> Currently, only the *integral* option is used by the ```calc_all_emis()``` routine.


## Superposition principle

If we wave two separate EM sources that, on the same target point, provoke magnetic fields
![$$\vec B_1$$](https://render.githubusercontent.com/render/math?math=%5Cvec%20B_1) and ![$$\vec B_2$$](https://render.githubusercontent.com/render/math?math=%5Cvec%20B_2).
The result field is a sum of individual fields:

 ![$$\vec B = \vec B_1 + \vec B_2$$](https://render.githubusercontent.com/render/math?math=%5Cvec%20B%20%3D%20%5Cvec%20B_1%20%2B%20%5Cvec%20B_2)

The same is valid for the jacobian ![$$\mathbf J_B = \frac{\partial \vec B}{\partial \vec r}$$](https://render.githubusercontent.com/render/math?math=%5Cmathbf%20J_B%20%3D%20%5Cfrac%7B%5Cpartial%20%5Cvec%20B%7D%7B%5Cpartial%20%5Cvec%20r%7D),
or ![$$d\vec B = \mathbf J_B \cdot d\vec r$$](https://render.githubusercontent.com/render/math?math=d%5Cvec%20B%20%3D%20%5Cmathbf%20J_B%20%5Ccdot%20d%5Cvec%20r):

![$$d\vec B = d\vec B_1 + d\vec B_2 \Rightarrow d\vec B = \mathbf J_{B_1} \cdot d\vec r + \mathbf J_{B_2} \cdot d\vec r = (\mathbf J_{B_1} + \mathbf J_{B_2}) \cdot d \vec r$$](https://render.githubusercontent.com/render/math?math=d%5Cvec%20B%20%3D%20d%5Cvec%20B_1%20%2B%20d%5Cvec%20B_2%20%5CRightarrow%20d%5Cvec%20B%20%3D%20%5Cmathbf%20J_%7BB_1%7D%20%5Ccdot%20d%5Cvec%20r%20%2B%20%5Cmathbf%20J_%7BB_2%7D%20%5Ccdot%20d%5Cvec%20r%20%3D%20%28%5Cmathbf%20J_%7BB_1%7D%20%2B%20%5Cmathbf%20J_%7BB_2%7D%29%20%5Ccdot%20d%20%5Cvec%20r)

> Note that ![$$d\vec r$$](https://render.githubusercontent.com/render/math?math=d%5Cvec%20r)
> is the same for both fields, as we are interested in the same movement of the same target point,
> but individual fields are provoked by different EM sources.

However, this principle is not valid for the gradient ![$$\frac{d|\vec B|}{d \vec r}$$](https://render.githubusercontent.com/render/math?math=%5Cfrac%7Bd%7C%5Cvec%20B%7C%7D%7Bd%20%5Cvec%20r%7D)
(actually the gradient of the field's magnitude). Thus, it must be handled separately, together with
similar ones.


## Parameters

The overall EM effect is calculated from the `base parameters` of individual source lines. These
are the (linear) parameters, that can be simply added to calculate the overall result, see
[code](../emi_calc.py#L310-L311).

Then, some useful `derived parameters` are calculated from the `base` ones. These are the
(non-linear) parameters, that can not be simply added, see [code](../emi_calc.py#L318-L319).


### Base parameters

The parameter to describe the EM field caused by current along line **L**, to a point at ![$$\vec{r}$$](https://render.githubusercontent.com/render/math?math=%5Cvec%7Br%7D) are:

- Magnetic field vector ```emi_params['B']```:

 ![$$\vec{\boldsymbol B(\vec{r})} = C \int_\boldsymbol L \frac{d \vec{\boldsymbol \ell} \times (\vec{r} - \vec{\boldsymbol \ell})}{|\vec{r} - \vec{\boldsymbol \ell}|^3}$$](https://render.githubusercontent.com/render/math?math=%5Cvec%7B%5Cboldsymbol%20B%28%5Cvec%7Br%7D%29%7D%20%3D%20C%20%5Cint_%5Cboldsymbol%20L%20%5Cfrac%7Bd%20%5Cvec%7B%5Cboldsymbol%20%5Cell%7D%20%5Ctimes%20%28%5Cvec%7Br%7D%20-%20%5Cvec%7B%5Cboldsymbol%20%5Cell%7D%29%7D%7B%7C%5Cvec%7Br%7D%20-%20%5Cvec%7B%5Cboldsymbol%20%5Cell%7D%7C%5E3%7D)


- Magnetic field Jacobian matrix ```emi_params['jacob']```:

 ![$$\mathbf J_B = \frac{\partial \vec{B}}{\partial \vec{r}} = \begin{bmatrix} \dfrac {\partial B_x}{\partial r_x} & \dfrac {\partial B_x}{\partial r_y} & \dfrac {\partial B_x}{\partial r_z} \\ \dfrac {\partial B_y}{\partial r_x} & \dfrac {\partial B_y}{\partial r_y} & \dfrac {\partial B_y}{\partial r_z} \\ \dfrac {\partial B_z}{\partial r_x} & \dfrac {\partial B_z}{\partial r_y} & \dfrac {\partial B_z}{\partial r_z}\end{bmatrix}$$](https://render.githubusercontent.com/render/math?math=%5Cmathbf%20J_B%20%3D%20%5Cfrac%7B%5Cpartial%20%5Cvec%7BB%7D%7D%7B%5Cpartial%20%5Cvec%7Br%7D%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20%5Cdfrac%20%7B%5Cpartial%20B_x%7D%7B%5Cpartial%20r_x%7D%20%26%20%5Cdfrac%20%7B%5Cpartial%20B_x%7D%7B%5Cpartial%20r_y%7D%20%26%20%5Cdfrac%20%7B%5Cpartial%20B_x%7D%7B%5Cpartial%20r_z%7D%20%5C%5C%20%5Cdfrac%20%7B%5Cpartial%20B_y%7D%7B%5Cpartial%20r_x%7D%20%26%20%5Cdfrac%20%7B%5Cpartial%20B_y%7D%7B%5Cpartial%20r_y%7D%20%26%20%5Cdfrac%20%7B%5Cpartial%20B_y%7D%7B%5Cpartial%20r_z%7D%20%5C%5C%20%5Cdfrac%20%7B%5Cpartial%20B_z%7D%7B%5Cpartial%20r_x%7D%20%26%20%5Cdfrac%20%7B%5Cpartial%20B_z%7D%7B%5Cpartial%20r_y%7D%20%26%20%5Cdfrac%20%7B%5Cpartial%20B_z%7D%7B%5Cpartial%20r_z%7D%5Cend%7Bbmatrix%7D)


A detailed description of how these parameters are calculated can be found here: [differential](emi_calc-dif.md)
and [integral](emi_calc-int.md) calculation routines.


### Derived parameters

These parameters are re-calculated from the `base parameters` (
![$$\vec B$$](https://render.githubusercontent.com/render/math?math=%5Cvec%20B)
and ![$$\mathbf J_B$$](https://render.githubusercontent.com/render/math?math=%5Cmathbf%20J_B)
), after a new EM source is added to the model.

- Magnetic field gradient vector ```emi_params['gradB']```, or the the Jacobian of the scalar function ![$$|\vec B|$$](https://render.githubusercontent.com/render/math?math=%7C%5Cvec%20B%7C):

 ![$$\nabla \vec B = \frac{d|\vec B|}{d \vec r} = ... = \frac{\vec B}{|\vec B|} \cdot \mathbf J_B$$](https://render.githubusercontent.com/render/math?math=%5Cnabla%20%5Cvec%20B%20%3D%20%5Cfrac%7Bd%7C%5Cvec%20B%7C%7D%7Bd%20%5Cvec%20r%7D%20%3D%20...%20%3D%20%5Cfrac%7B%5Cvec%20B%7D%7B%7C%5Cvec%20B%7C%7D%20%5Ccdot%20%5Cmathbf%20J_B)

#### Steps in details

 ![$$\frac{d|\vec B|}{d \vec r}=\begin{bmatrix} \frac{\partial \sqrt{B_x^2+B_y^2+B_z^2}}{\partial r_x} && \frac{\partial \sqrt{B_x^2+B_y^2+B_z^2}}{\partial r_y} && \frac{\partial \sqrt{B_x^2+B_y^2+B_z^2}}{\partial r_z} \end{bmatrix}=$$](https://render.githubusercontent.com/render/math?math=%5Cfrac%7Bd%7C%5Cvec%20B%7C%7D%7Bd%20%5Cvec%20r%7D%3D%5Cbegin%7Bbmatrix%7D%20%5Cfrac%7B%5Cpartial%20%5Csqrt%7BB_x%5E2%2BB_y%5E2%2BB_z%5E2%7D%7D%7B%5Cpartial%20r_x%7D%20%26%26%20%5Cfrac%7B%5Cpartial%20%5Csqrt%7BB_x%5E2%2BB_y%5E2%2BB_z%5E2%7D%7D%7B%5Cpartial%20r_y%7D%20%26%26%20%5Cfrac%7B%5Cpartial%20%5Csqrt%7BB_x%5E2%2BB_y%5E2%2BB_z%5E2%7D%7D%7B%5Cpartial%20r_z%7D%20%5Cend%7Bbmatrix%7D%3D)

> given that
>
> ![$$\frac {\partial \sqrt{F(x)}}{\partial x} = \frac {\partial \sqrt F}{\partial F} \frac {\partial F}{\partial x} = \frac {1}{2 \sqrt F} \frac {\partial F}{\partial x}$$](https://render.githubusercontent.com/render/math?math=%5Cfrac%20%7B%5Cpartial%20%5Csqrt%7BF%28x%29%7D%7D%7B%5Cpartial%20x%7D%20%3D%20%5Cfrac%20%7B%5Cpartial%20%5Csqrt%20F%7D%7B%5Cpartial%20F%7D%20%5Cfrac%20%7B%5Cpartial%20F%7D%7B%5Cpartial%20x%7D%20%3D%20%5Cfrac%20%7B1%7D%7B2%20%5Csqrt%20F%7D%20%5Cfrac%20%7B%5Cpartial%20F%7D%7B%5Cpartial%20x%7D)

 ![$$= \frac {1}{2 |\vec B|} \begin{bmatrix} \frac{\partial(B_x^2+B_y^2+B_z^2)}{\partial r_x} && \frac{\partial(B_x^2+B_y^2+B_z^2)}{\partial r_y} && ...\end{bmatrix} = \frac {1}{2 |\vec B|} \begin{bmatrix} \frac{\partial B_x^2}{\partial r_x} + \frac{\partial B_y^2}{\partial r_x} + \frac{\partial B_z^2}{\partial r_x} && \frac{\partial B_x^2}{\partial r_y} + \frac{\partial B_y^2}{\partial r_y} + \frac{\partial B_z^2}{\partial r_y} && ...\end{bmatrix} =$$](https://render.githubusercontent.com/render/math?math=%3D%20%5Cfrac%20%7B1%7D%7B2%20%7C%5Cvec%20B%7C%7D%20%5Cbegin%7Bbmatrix%7D%20%5Cfrac%7B%5Cpartial%28B_x%5E2%2BB_y%5E2%2BB_z%5E2%29%7D%7B%5Cpartial%20r_x%7D%20%26%26%20%5Cfrac%7B%5Cpartial%28B_x%5E2%2BB_y%5E2%2BB_z%5E2%29%7D%7B%5Cpartial%20r_y%7D%20%26%26%20...%5Cend%7Bbmatrix%7D%20%3D%20%5Cfrac%20%7B1%7D%7B2%20%7C%5Cvec%20B%7C%7D%20%5Cbegin%7Bbmatrix%7D%20%5Cfrac%7B%5Cpartial%20B_x%5E2%7D%7B%5Cpartial%20r_x%7D%20%2B%20%5Cfrac%7B%5Cpartial%20B_y%5E2%7D%7B%5Cpartial%20r_x%7D%20%2B%20%5Cfrac%7B%5Cpartial%20B_z%5E2%7D%7B%5Cpartial%20r_x%7D%20%26%26%20%5Cfrac%7B%5Cpartial%20B_x%5E2%7D%7B%5Cpartial%20r_y%7D%20%2B%20%5Cfrac%7B%5Cpartial%20B_y%5E2%7D%7B%5Cpartial%20r_y%7D%20%2B%20%5Cfrac%7B%5Cpartial%20B_z%5E2%7D%7B%5Cpartial%20r_y%7D%20%26%26%20...%5Cend%7Bbmatrix%7D%20%3D)

 ![$$= \frac {1}{2 |\vec B|} \begin{bmatrix} 2(B_x\frac{\partial B_x}{\partial r_x}+B_y\frac{\partial B_y}{\partial r_x}+B_z\frac{\partial B_z}{\partial r_x}) && 2(B_x\frac{\partial B_x}{\partial r_y}+B_y\frac{\partial B_y}{\partial r_y}+B_z\frac{\partial B_z}{\partial r_y}) && ...\end{bmatrix} =$$](https://render.githubusercontent.com/render/math?math=%3D%20%5Cfrac%20%7B1%7D%7B2%20%7C%5Cvec%20B%7C%7D%20%5Cbegin%7Bbmatrix%7D%202%28B_x%5Cfrac%7B%5Cpartial%20B_x%7D%7B%5Cpartial%20r_x%7D%2BB_y%5Cfrac%7B%5Cpartial%20B_y%7D%7B%5Cpartial%20r_x%7D%2BB_z%5Cfrac%7B%5Cpartial%20B_z%7D%7B%5Cpartial%20r_x%7D%29%20%26%26%202%28B_x%5Cfrac%7B%5Cpartial%20B_x%7D%7B%5Cpartial%20r_y%7D%2BB_y%5Cfrac%7B%5Cpartial%20B_y%7D%7B%5Cpartial%20r_y%7D%2BB_z%5Cfrac%7B%5Cpartial%20B_z%7D%7B%5Cpartial%20r_y%7D%29%20%26%26%20...%5Cend%7Bbmatrix%7D%20%3D)

 ![$$= \frac {1}{|\vec B|} \begin{bmatrix} B_x\frac{\partial B_x}{\partial r_x}+B_y\frac{\partial B_y}{\partial r_x}+B_z\frac{\partial B_z}{\partial r_x} && B_x\frac{\partial B_x}{\partial r_y}+B_y\frac{\partial B_y}{\partial r_y}+B_z\frac{\partial B_z}{\partial r_y} && ...\end{bmatrix} = \frac{\vec B \cdot \mathbf J_B}{|\vec B|} =$$](https://render.githubusercontent.com/render/math?math=%3D%20%5Cfrac%20%7B1%7D%7B%7C%5Cvec%20B%7C%7D%20%5Cbegin%7Bbmatrix%7D%20B_x%5Cfrac%7B%5Cpartial%20B_x%7D%7B%5Cpartial%20r_x%7D%2BB_y%5Cfrac%7B%5Cpartial%20B_y%7D%7B%5Cpartial%20r_x%7D%2BB_z%5Cfrac%7B%5Cpartial%20B_z%7D%7B%5Cpartial%20r_x%7D%20%26%26%20B_x%5Cfrac%7B%5Cpartial%20B_x%7D%7B%5Cpartial%20r_y%7D%2BB_y%5Cfrac%7B%5Cpartial%20B_y%7D%7B%5Cpartial%20r_y%7D%2BB_z%5Cfrac%7B%5Cpartial%20B_z%7D%7B%5Cpartial%20r_y%7D%20%26%26%20...%5Cend%7Bbmatrix%7D%20%3D%20%5Cfrac%7B%5Cvec%20B%20%5Ccdot%20%5Cmathbf%20J_B%7D%7B%7C%5Cvec%20B%7C%7D%20%3D)

 ![$$=\frac{\vec B}{|\vec B|} \cdot \mathbf J_B$$](https://render.githubusercontent.com/render/math?math=%3D%5Cfrac%7B%5Cvec%20B%7D%7B%7C%5Cvec%20B%7C%7D%20%5Ccdot%20%5Cmathbf%20J_B)


- Field-relative target movement vector ```emi_params['dr_dI']```, it is actually ![$$I \frac{dr}{dI}$$](https://render.githubusercontent.com/render/math?math=I%20%5Cfrac%7Bdr%7D%7BdI%7D):

 ![$$I \frac{d \vec r}{d I} = \frac{\vec B \cdot \mathbf J_B}{\lvert{\nabla \vec B}\rvert^2}$$](https://render.githubusercontent.com/render/math?math=I%20%5Cfrac%7Bd%20%5Cvec%20r%7D%7Bd%20I%7D%20%3D%20%5Cfrac%7B%5Cvec%20B%20%5Ccdot%20%5Cmathbf%20J_B%7D%7B%5Clvert%7B%5Cnabla%20%5Cvec%20B%7D%5Crvert%5E2%7D)

Field-relative movement, is a *virtual movement* of the target (![$$d \vec r$$](https://render.githubusercontent.com/render/math?math=d%20%5Cvec%20r)
in a constant magnetic field, that provokes the same EM effect, as the effect from a change in the
electric current (**dI**).

 
> Here is assumed that the *virtual movement* is toward the closest point, where previously the
> magnitude of **B** ( ![$$|\vec B|$$](https://render.githubusercontent.com/render/math?math=%7C%5Cvec%20B%7C) )
> was the same as the new value at the current point, after the change in the electric current.

#### Steps in details

The *virtual movement* is in the same direction as the gradient. This is because, a change in the
current, proportionally changes the magnitude of **B**, but not its direction. Thus, to find the
closest point with specific **d|B|**, must find a point **r** where:

![$$d|\vec B| = \nabla{\vec B} . d \vec \boldsymbol r$$](https://render.githubusercontent.com/render/math?math=d%7C%5Cvec%20B%7C%20%3D%20%5Cnabla%7B%5Cvec%20B%7D%20.%20d%20%5Cvec%20%5Cboldsymbol%20r)

I.e. the *"movement"* is inversely proportional to the gradient (note that ![$$d|\vec B|$$](https://render.githubusercontent.com/render/math?math=d%7C%5Cvec%20B%7C) is a scalar):

 ![$$d \vec r = d|\vec B| \frac{1}{\lvert{\nabla{\vec B}}\rvert} \frac{\nabla{\vec B}}{\lvert{\nabla{\vec B}}\rvert} = \frac{d|\vec B|}{\lvert{\nabla{\vec B}}\rvert^2} \nabla{\vec B}$$](https://render.githubusercontent.com/render/math?math=d%20%5Cvec%20r%20%3D%20d%7C%5Cvec%20B%7C%20%5Cfrac%7B1%7D%7B%5Clvert%7B%5Cnabla%7B%5Cvec%20B%7D%7D%5Crvert%7D%20%5Cfrac%7B%5Cnabla%7B%5Cvec%20B%7D%7D%7B%5Clvert%7B%5Cnabla%7B%5Cvec%20B%7D%7D%5Crvert%7D%20%3D%20%5Cfrac%7Bd%7C%5Cvec%20B%7C%7D%7B%5Clvert%7B%5Cnabla%7B%5Cvec%20B%7D%7D%5Crvert%5E2%7D%20%5Cnabla%7B%5Cvec%20B%7D)

> Note that, the points where the gradient is smaller, have larger *virtual movement*. Moreover,
> in case of nearly homogeneous field, this movement must be huge.

Thus:

 ![$$I \frac{d \vec r}{d I} = I \frac{\nabla{\vec B}}{\lvert{\nabla{\vec B}}\rvert^2} \frac{d |\vec B|}{d I} =$$](https://render.githubusercontent.com/render/math?math=I%20%5Cfrac%7Bd%20%5Cvec%20r%7D%7Bd%20I%7D%20%3D%20I%20%5Cfrac%7B%5Cnabla%7B%5Cvec%20B%7D%7D%7B%5Clvert%7B%5Cnabla%7B%5Cvec%20B%7D%7D%5Crvert%5E2%7D%20%5Cfrac%7Bd%20%7C%5Cvec%20B%7C%7D%7Bd%20I%7D%20%3D)

Since **|B|** and **I** are proportional, we can substutute ![$$|\vec B| = I \frac{d |\vec B|}{d I}$$](https://render.githubusercontent.com/render/math?math=%7C%5Cvec%20B%7C%20%3D%20I%20%5Cfrac%7Bd%20%7C%5Cvec%20B%7C%7D%7Bd%20I%7D):

 ![$$= \frac{|\vec B| . \nabla{\vec B}}{\lvert{\nabla{\vec B}}\rvert^2} = \frac{\vec B \cdot \mathbf J_B}{\lvert{\nabla \vec B}\rvert^2}$$](https://render.githubusercontent.com/render/math?math=%3D%20%5Cfrac%7B%7C%5Cvec%20B%7C%20.%20%5Cnabla%7B%5Cvec%20B%7D%7D%7B%5Clvert%7B%5Cnabla%7B%5Cvec%20B%7D%7D%5Crvert%5E2%7D%20%3D%20%5Cfrac%7B%5Cvec%20B%20%5Ccdot%20%5Cmathbf%20J_B%7D%7B%5Clvert%7B%5Cnabla%20%5Cvec%20B%7D%5Crvert%5E2%7D)
