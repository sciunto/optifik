Snippets
========

Basic Optimization
------------------

Here's a simple example using Optifik to solve a quadratic function.

.. code-block:: python

    from optifik import minimize

    def quadratic(x):
        return (x - 3) ** 2

    result = minimize(quadratic, x0=0)
    print(result)

Constrained Optimization
------------------------

.. code-block:: python

    from optifik import minimize

    def objective(x):
        return x[0] ** 2 + x[1] ** 2

    def constraint(x):
        return x[0] + x[1] - 1

    result = minimize(objective, x0=[0, 0], constraints=[constraint])
    print(result)
