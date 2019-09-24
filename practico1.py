import numpy as np
import matplotlib.pyplot as plt

def lvpp_model(t, y):
    """
    Lokta-Volterra predators and preys model.
    
    ·R(t) = a R(t) - b R(t)F(t)
    ·F(t) = -c F(t) + d R(t)F(t)

    Here the parameters of the model are:

        A = ( 0.1 0.02
              0.3 0.01 )

    Parameters:
    t: float, time in which the sistem is
    y: value of the R and F functions at time t
    """
    dr = 0.1*y[0] - 0.02* y[0]*y[1]
    df = -0.3*y[1] + 0.01* y[0]*y[1]
    return np.array([dr, df])

def runge_kutta_iter(fun, t, yn, h=0.5):
    k1 = h* fun(t, yn)
    k2 = h* fun(t+0.5*h, yn + 0.5*k1)
    k3 = h* fun(t+0.5*h, yn + 0.5*k2)
    k4 = h* fun(t+h, yn + k3)

    yn1 = yn + (1/6)* (k1 + 2*k2 + 2*k3 + k4)
    tn1 = t + h
    return yn1, tn1

def runge_kutta(fun, t0, y0, h, tmax):
    yn = y0
    tn = t0
    R = [y0[0]]
    F = [y0[1]]
    while tn<=tmax:
        yn, tn = runge_kutta_iter(fun, tn, yn, h)
        R.append(yn[0])
        F.append(yn[1])

    return R, F

def plot_marginal(data, t, fn, fname):
    plt.plot(t, data)
    plt.xlabel("tiempo t")
    plt.ylabel(fn)
    plt.savefig(fname)
    plt.clf()

if __name__ == "__main__":
    t0 = 0.0
    y0 = np.array([40,9])
    tmax = 200.0
    h = 0.05

    R, F = runge_kutta(lvpp_model, t0, y0, h, tmax)

    t = np.linspace(0,200, len(R))

    plot_marginal(R, t, "R(t)", "rt.png")
    plot_marginal(F, t, "F(t)", "ft.png")

    plt.plot(R,F)
    plt.xlabel("R(t)")
    plt.ylabel("F(t)")
    plt.savefig("lvpp_model.png")


    

    
