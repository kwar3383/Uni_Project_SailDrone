# imports necessary libraries for numerical calculations and graph plotting
import numpy as np
import matplotlib.pyplot as plt

# imports file given in brief so get_vals() can be accessed to find hydrodynamic forces
import saildrone_hydro


def state_deriv(current_t, current_state):
    """
    Computes the derivative of the current state vector

    Parameters
    ----------
    current_t : float64
        Current time in seconds.

    current_state : numpy.ndarray of shape (6,1)
        6x1 Matrix corresponding to the state vector of the sail drone at the current time:
            [0,0] = x displacement at current time, m
            [1,0] = x velocity at current time, m/s
            [2,0] = y displacement at current time, m
            [3,0] = y velocity at current time, m/s
            [4,0] = angular displacement at current time, rad
            [5,0] = angular velocity at current time, rad/s

    Returns
    -------
    dz : numpy.ndarray of shape (6,)
        The current time derivative of the state vector, forming the first-order ODE representation of the system:
            [0] = dx/dt, m/s
            [1] = dVx/dt, m/s^2
            [2] = dy/dt, m/s
            [3] = dVy/dt, m/s^2
            [4] = dtheta/dt, rad/s
            [5] = dVtheta/dt, rad/s^2
    """

    # known constants
    V_wind = np.array([-6.7, 0])
    density = 1.225
    mass = 2500
    sail_area = 15
    rot_inertia = 10000

    # sail and rudder angles depending on what part of the journey the drone is at, A, B or C
    beta_sail = np.radians(-45) if current_t < 60 else np.radians(-22.5)
    beta_rudder = np.radians(2.1) if 60 <= current_t < 65 else np.radians(0)

    # flattens current_state vector into a 1D array, so it can be indexed properly, ie, current_state now has shape (6,)
    current_state = np.asarray(current_state).flatten()

    # names and stores the current state values for easier and clearer use later on
    x = current_state[0]
    Vx = current_state[1]
    y = current_state[2]
    Vy = current_state[3]
    theta = current_state[4]
    Vtheta = current_state[5]

    # finds velocity and speed of drone relative to wind so aerodynamic forces can be computed
    V_drone = np.array([Vx, Vy])
    V_rel = V_wind - V_drone
    Speed_rel = np.linalg.norm(V_rel)

    # finds angle of relative wind direction, phi, for use in resolving drag and lift components
    phi = np.arctan2(V_rel[1], V_rel[0])

    # finds angle of attack relative to wind, alpha, for use in finding drag and lift coefficients
    alpha = phi - theta - beta_sail

    # finds aerodynamic drag and lift coefficients
    Cd = 1 - np.cos(2 * alpha)
    Cl = 1.5 * np.sin((2 * alpha) + (0.5 * np.sin(2 * alpha)))

    # finds aerodynamic drag and lift forces and separates into x and y components
    Fd = 0.5 * density * sail_area * (Speed_rel**2) * Cd
    Fl = 0.5 * density * sail_area * (Speed_rel ** 2) * Cl

    Fdx, Fdy = Fd * np.cos(phi), Fd * np.sin(phi)
    Flx, Fly = -Fl * np.sin(phi), Fl * np.cos(phi)

    # finds total aerodynamic force in x and y
    Fax = Fdx + Flx
    Fay = Fdy + Fly

    # calculates total aerodynamic torque
    rx = -0.1 * np.cos(theta)
    ry = 0.1 * np.sin(theta)
    Tx, Ty = Fax*rx, Fay*ry
    Ta = -Tx-Ty

    # fetches hydrodynamic force and torque values from imported function given in brief
    [Fh, Th] = saildrone_hydro.get_vals(np.array([float(Vx), float(Vy)]), float(theta), float(Vtheta), float(beta_rudder))

    Fhx, Fhy = Fh[0], Fh[1]

    # finds and returns the first order reduction of the second order ODEs of resultant forces in x,y and theta
    dz1 = Vx
    dz2 = (Fax + Fhx) / mass
    dz3 = Vy
    dz4 = (Fay + Fhy) / mass
    dz5 = Vtheta
    dz6 = (Ta + Th) / rot_inertia

    dz = np.array([dz1, dz2, dz3, dz4, dz5, dz6])

    return dz


def step_rk(state_deriv, current_t, step, current_state):
    """
       Applies 4th Order Runge-Kutta method for one time step to solve the current state equations numerically

       Parameters
       ----------
       state_deriv : callable
           Returns the derivative of the inputted state vector.

       current_t : float64
           Current time in seconds.

       step : float64
           Time step.

       current_state : numpy.ndarray of shape (6,1)
           6x1 Matrix corresponding to the state vector of the saildrone at the current time:
               [0,0] = x displacement at current time, m
               [1,0] = x velocity at current time, m/s
               [2,0] = y displacement at current time, m
               [3,0] = y velocity at current time, m/s
               [4,0] = angular displacement at current time, rad
               [5,0] = angular velocity at current time, rad/s

       Returns
       -------
       updated_state : numpy.ndarray of shape (6,1)
           6x1 Matrix corresponding to the state vector at the next time step (t+dt):
               [0,0] = x displacement at current time + step, m
               [1,0] = x velocity at current time + step, m/s
               [2,0] = y displacement at current time + step, m
               [3,0] = y velocity at current time + step, m/s
               [4,0] = angular displacement at current time + step, rad
               [5,0] = angular velocity at current time + step, rad/s

       """

    # RK Parameters for four steps
    A = step * state_deriv(current_t, current_state)
    B = step * state_deriv(current_t + step / 2, current_state + A / 2)
    C = step * state_deriv(current_t + step / 2, current_state + B / 2)
    D = step * state_deriv(current_t + step, current_state + C)

    # Runge-Kutta update equation to find next state vector
    updated_state = current_state + (A + 2 * B + 2 * C + D) / 6

    return updated_state


def solve_ivp(state_deriv, start_time, end_time, step, initial_state):
    """
    Solves initial value problems (IVPs) for ordinary differential equations (ODEs)

    Parameters
    ----------
    state_deriv : callable
        Returns the derivative of the inputted state vector

    start_time : float64
        Start time.

    end_time : float64
        End time/length of simulation.

    step : float64
        Time step.

    initial_state : numpy.ndarray with shape (6,1)
        Initial state vector at start time, time=0 :
            [0,0] = initial x displacement, m
            [1,0] = initial x velocity, m/s
            [2,0] = initial y displacement, m
            [3,0] = initial y velocity, m/s
            [4,0] = initial angular displacement, rad
            [5,0] = initial angular velocity, rad/s


    Returns
    -------
    time : numpy.ndarray of shape (N,)
        Time axis, N = number of time steps

    state : numpy.ndarray with shape (6, N)
        Solution as a 6xN matrix, with rows corresponding to the following six state vectors, and columns to time steps:
            [0,time] = x displacement at each time step, m
            [1,time] = x velocity at each time step, m/s
            [2,time] = y displacement at each time step, m
            [3,time] = y velocity at each time step, m/s
            [4,time] = angular displacement at each time step, rad
            [5,time] = angular velocity at each time step, rad/s

            (where time = [0, 1, ..., N-1]. N = total number of time steps)
    """

    # creates time and state arrays based on initial values
    time = np.array([start_time])
    state = initial_state

    # continues incrementing by one time step until the end time is exceeded
    n = 0
    while time[n] <= end_time:
        # adds one time step and appends to the time axis
        time = np.append(time, time[-1] + step)

        # obtains next state vector using Range-Kutta 4th Order method
        next_state = step_rk(state_deriv, time[n], step, state[:, n])

        # appends to the state solution matrix
        state = np.append(state, next_state[:, np.newaxis], axis=1)

        n = n + 1

    return time, state


def plot_trajectory(results_matrix):
    """
    Plots the saildrone's x displacement against y displacement to show its overall trajectory

    Parameters
    ----------
    results_matrix : numpy.ndarray with shape (6, N)
        Solution as a 6xN matrix, with rows corresponding to the following six state vectors, and columns to time steps:
            [0,t] = x displacement at time t, m
            [1,t] = x velocity at time t, m/s
            [2,t] = y displacement at time t, m
            [3,t] = y velocity at time t, m/s
            [4,t] = angular displacement at time t, rad
            [5,t] = angular velocity at time t, rad/s

            (where t = [0, 1, ..., N-1]. N = total number of time steps)

    """

    # plots x displacement values against y displacement values, labelling line 'trajectory' and labelling axes
    plt.figure()
    plt.plot(results_matrix[0, :], results_matrix[2, :], linewidth=2, label='Trajectory')
    plt.xlabel('X Displacement (m)', fontsize=12)
    plt.ylabel('Y Displacement (m)', fontsize=12)
    plt.title('Sail Drone Trajectory', fontsize=14)

    # labels start and end points of the journey for clarity
    plt.scatter(results_matrix[0, 0], results_matrix[2, 0], color='green', s=20, label='Start Position')
    plt.scatter(results_matrix[0, -1], results_matrix[2, -1], color='red', s=20, label='End position')

    # tidies graph and adds a grid for readability
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.6)

    # displays the final x displacement, y displacement, and angular displacement (heading) values
    end_x, end_y = results_matrix[0, -1], results_matrix[2, -1]
    plt.annotate(f'({end_x:.2f}m, {end_y:.2f}m)',
                 xy=(end_x, end_y),
                 xytext=(end_x + 5, end_y))

    end_heading = results_matrix[4, -1]
    plt.annotate(f'(Heading: {end_heading:.2f}rad)',
                 xy=(end_x, end_y),
                 xytext=(50, 140))

    # displays the graph with a legend
    plt.legend()
    plt.show()


def plot_headingvstime(time_axis, results_matrix):
    """
    Plots the saildrone's angular displacement against time to show its heading throughout the journey.

    Parameters
    ----------
    time_axis : numpy.ndarray of shape (N,)
        Time axis, N = number of time steps
    results_matrix : numpy.ndarray with shape (6, N)
        Solution as a 6xN matrix, with rows corresponding to the following six state vectors, and columns to time steps:
            [0,t] = x displacement at time t, m
            [1,t] = x velocity at time t, m/s
            [2,t] = y displacement at time t, m
            [3,t] = y velocity at time t, m/s
            [4,t] = angular displacement at time t, rad
            [5,t] = angular velocity at time t, rad/s

            (where t = [0, 1, ..., N-1]. N = total number of time steps)
    """

    # plots time values against angular displacement (heading) values, labelling axes
    plt.figure()
    plt.plot(time_axis, results_matrix[4, :], 'blue')
    plt.xlabel('Time, s')
    plt.ylabel('Heading, rad')
    plt.title('Sail Drone Heading against Time', fontsize=14)

    # sets size of graph to ensure labels all fit clearly within axes and adds grid for readability
    plt.xlim(-2, 80)
    plt.grid(True, linestyle='--', alpha=0.6)

    # separates the graph into the 3 different legs of the journey
    course_sections = [[0, "A"], [60, "B"], [65, "C"]]
    for i in range(3):
        time = course_sections[i][0]
        section = course_sections[i][1]

        # displays a vertical line showing the start of each new leg
        plt.axvline(x=time, linestyle=':', color='orange', alpha=0.8)

        # finds the angular displacement at the start of each leg
        x_val = int(time / 0.1 if time != 0 else 0)
        theta = results_matrix[4, x_val]

        # labels each leg and displays the angular displacement (heading) at the start of each leg
        # labels are right of the lines for legs A and C, but left for B to avoid overlap for a clearer display
        plt.annotate(f'{section}', color='orange', fontsize='18',
                     xy=(time, 1.45),
                     xytext=(time + 0.5, 1.45) if section != 'B' else (time - 3, 1.45))

        plt.annotate(f'({theta:.2f}rad)', color='orange',
                     xy=(time, 1.4),
                     xytext=(time + 0.5, 1.42) if section != 'B' else (time - 12, 1.42))

    # displays the graph
    plt.show()


def plot_velocityvstime(time_axis, results_matrix):
    """
    Plots the saildrone's x and y velocities against time to show how they change throughout the journey

    Parameters
    ----------
    time_axis : numpy.ndarray of shape (N,)
        Time axis, N = number of time steps
    results_matrix : numpy.ndarray with shape (6, N)
        Solution as a 6xN matrix, with rows corresponding to the following six state vectors, and columns to time steps:
            [0,t] = x displacement at time t, m
            [1,t] = x velocity at time t, m/s
            [2,t] = y displacement at time t, m
            [3,t] = y velocity at time t, m/s
            [4,t] = angular displacement at time t, rad
            [5,t] = angular velocity at time t, rad/s

            (where t = [0, 1, ..., N-1]. N = total number of time steps)
    """

    # plots x velocity and y velocity values separeately against time, labelling axes
    plt.figure()
    plt.plot(time_axis, results_matrix[1, :], 'blue')
    plt.plot(time_axis, results_matrix[3, :], 'red')
    plt.xlabel('Time, s')
    plt.ylabel('Velocity, m/s')
    plt.legend(['X Velocity', 'Y Velocity'])
    plt.title('Velocity Components against Time', fontsize=14)

    # sets size of graph to ensure labels all fit clearly within axes and adds grid for readability
    plt.xlim(-2, 80)
    plt.grid(True, linestyle='--', alpha=0.6)

    # separates the graph into the 3 different legs of the journey
    course_sections = [[0, "A"], [60, "B"], [65, "C"]]
    for i in range(3):
        time = course_sections[i][0]
        section = course_sections[i][1]

        # displays a vertical line showing the start of each new leg
        plt.axvline(x=time, linestyle=':', color='orange', alpha=0.8)

        # finds the x and y velocities at the start of each leg
        x_val = int(time / 0.1 if time != 0 else 0)
        vx, vy = results_matrix[1, x_val], results_matrix[3, x_val]

        # labels each leg and displays the velocity components at the start of each leg
        # labels are right of the lines for legs A and C, but left for B to avoid overlap for a clearer display
        plt.annotate(f'{section}', color='orange', fontsize='18',
                     xy=(time, 2.5),
                     xytext=(time + 0.5, 2.5) if section != 'B' else (time - 3, 2.5))

        plt.annotate(f'X Vel: {vx:.2f}\nY Vel: {vy:.2f}', color='orange',
                     xy=(time, 1.25),
                     xytext=(time + 0.5, 2) if section != 'B' else (time - 13, 2))

    # displays the graph
    plt.show()


if __name__ == '__main__':
    # defines all fixed simulation parameters given in brief: time values, and position and velocities of sail drone

    # sets start time, end time, and time step in seconds
    t0 = 0
    tmax = 65
    dt = 0.1

    # sets initial values for position/velocity states of the saildrone
    x0 = 0
    Vx0 = 0
    y0 = 0
    Vy0 = 2.9
    theta0 = np.pi/2
    Vtheta0 = 0

    # stores initial state vector
    z0 = np.array([[x0], [Vx0], [y0], [Vy0], [theta0], [Vtheta0]])

    # runs the solver to get results
    [t, z] = solve_ivp(state_deriv, t0, tmax, dt, z0)

    # plots the graphs of results
    plot_trajectory(z)
    plot_headingvstime(t, z)
    plot_velocityvstime(t, z)
