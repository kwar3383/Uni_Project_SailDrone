# Only NEW code has been commented in order for changes to be seen clearly. See main code for full comments.

# Changes:
#   - original step_rk() function replaced by step_rk_5th_order(), see Extension Task 1: Updated Range-Kutta Method.
#   - get_betas() function created, see Extension Task 2: Added Sail and Rudder Actuation Rates.
#   - get_v_wind() function created, see Extension Task 3: Time-Varying Wind and Ocean Currents.
#   - get_v_current() function created, see Extension Task 3: Time-Varying Wind and Ocean Currents.
#   - lines added in state_deriv() to implement varying sail/rudder angles, wind velocity and current velocity.


import numpy as np
import matplotlib.pyplot as plt

import saildrone_hydro


def get_betas(prev_beta_sail, prev_beta_rudder, current_t, step):
    """
        Calculates the sail and rudder angles at the current time, limiting them by maximum actuation rate

        Parameters
        ----------
        prev_beta_sail : float64
            The value of the sail angle at the previous time step.

        prev_beta_rudder : float64
            The value of the rudder angle at the previous time step.

        current_t : float64
            Current time in seconds.

        step : float64
            Time step.


        Returns
        -------
        beta_sail : float64
            The sail angle at the current time.

        beta_rudder : float64
            The rudder angle at the current time.

        """

    # finds target sail and rudder angles depending on what part of the journey the drone is at, A, B or C.
    beta_sail_target = np.radians(-45) if current_t < 60 else np.radians(-22.5)
    beta_rudder_target = np.radians(2.1) if 60 <= current_t < 65 else np.radians(0)

    # calculates maximum actuation rate for sail and rudder, using reasonable actuation rate values
    max_sail_rate = np.radians(5) * step
    max_rudder_rate = np.radians(20) * step

    # finds current sail/rudder angles ensuring they are as close as the targets as possible without varying too quickly
    beta_sail = np.clip(beta_sail_target, prev_beta_sail - max_sail_rate, prev_beta_sail + max_sail_rate)
    beta_rudder = np.clip(beta_rudder_target, prev_beta_rudder - max_rudder_rate, prev_beta_rudder + max_rudder_rate)

    # stores current angles as previous angles for the next time step
    get_betas.prev_beta_sail = beta_sail
    get_betas.prev_beta_rudder = beta_rudder

    # returns current sail and rudder angles
    return beta_sail, beta_rudder


def get_v_wind(current_t):
    """
        Calculates the velocity of the wind at the current time, using a time-varying function.

        Parameters
        ----------
        current_t : float64
            Current time in seconds.


        Returns
        -------
        vx : float64
            The x component of the wind's velocity at the current time.

        vy : float64
            The y component of the wind's velocity at the current time.

        """

    # sets the mean velocity of the wind, assumed to be the initial velocity given in brief
    v_mean_x, v_mean_y = -6.7, 0

    # large-scale oscillation amplitude for velocity in x, assumes velocity variation in range ±15% of mean velocity
    LSa_x = 0.15 * v_mean_x
    # gust amplitude for velocity in x, assumes gust factor=1.2, so velocity variation in range ±20% of mean velocity
    Ga_x = 0.2 * v_mean_x

    # large-scale oscillation amplitude for velocity in y, assumes velocity variation in range ±10% of mean velocity
    LSa_y = 0.1 * v_mean_y
    # gust amplitude for velocity in y, assumes gust factor=1.2, so velocity variation in range ±20% of mean velocity
    Ga_y = 0.2 * v_mean_y

    # calculates frequency assuming an average wind variation period of 3 minutes (180 seconds)
    omega = 2 * np.pi / 180

    # sets gust frequency multiplier corresponding to moderate gusts, assuming moderate wind conditions
    b = 2

    # calculates velocity using time-varying sinusoidal function for sum of varying large-scale oscillations and gusts
    vx = v_mean_x + LSa_x * np.sin(omega * current_t) + Ga_x * np.sin(b * omega * current_t)
    vy = v_mean_y + LSa_y * np.cos(omega * current_t) + Ga_y * np.cos(b * omega * current_t)

    # returns current wind velocity components
    return vx, vy


def get_v_current(current_t):
    """
            Calculates the velocity of the ocean current at the current time, using a time-varying function.

            Parameters
            ----------
            current_t : float64
                Current time in seconds.


            Returns
            -------
            vx : float64
                The x component of the ocean current's velocity at the current time.

            vy : float64
                The y component of the ocean current's velocity at the current time.

            """

    # sets the mean velocity of the ocean current, assuming a speed of ~0.5m/s in a southwesterly direction
    v_mean_x, v_mean_y = -0.35, -0.35

    # sets tidal modulation variation amplitude, assuming velocity variation in range ±30% of mean velocity
    Ax = 0.3 * v_mean_x
    Ay = 0.3 * v_mean_y

    # finds angular frequency assuming a tidal variation period of 350 seconds
    omega = 2 * np.pi / 350

    # calculates ocean current velocity using simple sinusoidal time-varying function
    vx = v_mean_x + Ax * np.sin(omega * current_t)
    vy = v_mean_y + Ay * np.cos(omega * current_t)

    # returns ocean current velocity components at current time
    return vx, vy


def state_deriv(current_t, current_state):

    density = 1.225
    mass = 2500
    sail_area = 15
    rot_inertia = 10000

    # fetches current wind velocity based on time
    v_wind_x, v_wind_y = get_v_wind(current_t)
    V_wind = np.array([v_wind_x, v_wind_y])

    # fetches current ocean current velocity based on time
    v_current_x, v_current_y = get_v_current(current_t)
    V_current = np.array([v_current_x, v_current_y])

    # fetches previous sail and rudder angles, 0 at the start of the journey, ie, when current time = 0
    prev_beta_sail = 0 if current_t == 0 else get_betas.prev_beta_sail
    prev_beta_rudder = 0 if current_t == 0 else get_betas.prev_beta_rudder

    # finds the current sail and rudder angles based on time
    beta_sail, beta_rudder = get_betas(prev_beta_sail, prev_beta_rudder, current_t, dt)


    current_state = np.asarray(current_state).flatten()

    x = current_state[0]
    Vx = current_state[1]
    y = current_state[2]
    Vy = current_state[3]
    theta = current_state[4]
    Vtheta = current_state[5]

    V_drone = np.array([Vx,Vy])

    # finds velocity of drone relative to water (using ocean current velocity) in order to find hydrodynamic forces
    V_drone_hydro = V_drone - V_current

    V_rel = V_wind - V_drone
    Speed_rel = np.linalg.norm(V_rel)

    phi = np.arctan2(V_rel[1], V_rel[0])

    alpha = phi - theta - beta_sail

    Cd = 1 - np.cos(2 * alpha)
    Cl = 1.5 * np.sin((2 * alpha) + (0.5 * np.sin(2 * alpha)))

    Fd = 0.5 * density * sail_area * (Speed_rel**2) * Cd
    Fdx = Fd * np.cos(phi)
    Fdy = Fd * np.sin(phi)

    Fl = 0.5 * density * sail_area * (Speed_rel**2) * Cl
    Flx = -Fl * np.sin(phi)
    Fly = Fl * np.cos(phi)

    Fax = Fdx + Flx
    Fay = Fdy + Fly

    rx = -0.1 * np.cos(theta)
    ry = 0.1 * np.sin(theta)
    Tx, Ty = Fax*rx, Fay*ry
    Ta = -Tx-Ty

    # fetches hydrodynamic forces and torque using the drone's velocity relative to the water
    [Fh, Th] = saildrone_hydro.get_vals(np.array([float(V_drone_hydro[0]), float(V_drone_hydro[1])]), float(theta), float(Vtheta), float(beta_rudder))
    Fhx, Fhy = Fh[0], Fh[1]

    dz1 = Vx
    dz2 = (Fax + Fhx) / mass
    dz3 = Vy
    dz4 = (Fay + Fhy) / mass
    dz5 = Vtheta
    dz6 = (Ta + Th) / rot_inertia

    dz = np.array([dz1, dz2, dz3, dz4, dz5, dz6])

    return dz


def step_rk_5th_order(state_deriv, current_t, step, current_state):
    """
    Applies 5th Order Runge-Kutta method for one time step to solve the current state equations numerically

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

    # RK Parameters for six steps
    A = step * state_deriv(current_t, current_state)
    B = step * state_deriv(current_t + step / 4, current_state + A / 4)
    C = step * state_deriv(current_t + step / 4, current_state + A / 8 + B / 8)
    D = step * state_deriv(current_t + step / 2, current_state - B / 2 + C)
    E = step * state_deriv(current_t + 3 * step / 4, current_state + 3 * A / 16 + 9 * D / 16)
    F = step * state_deriv(current_t + step, current_state - 3 * A / 7 + 2 * B / 7 + 12 * C / 7 - 12 * D / 7 + 8 * E / 7)

    # Runge-Kutta update equation to find next state vector
    updated_state = current_state + (7 * A + 32 * C + 12 * D + 32 * E + 7 * F) / 90

    return updated_state


def solve_ivp(state_deriv, start_time, end_time, step, initial_state):

    time = np.array([start_time])
    state = initial_state

    n = 0
    while time[n] <= end_time:

        time = np.append(time, time[-1] + step)

        next_state = step_rk_5th_order(state_deriv, time[n], step, state[:, n])
        state = np.append(state, next_state[:, np.newaxis], axis=1)

        n = n + 1

    return time, state


def plot_trajectory(results_matrix):
    plt.figure()
    plt.plot(results_matrix[0, :], results_matrix[2, :], "blue", linewidth=2, label='Trajectory')

    plt.scatter(results_matrix[0, 0], results_matrix[2, 0], color='green', s=20, label='Start Position')
    plt.scatter(results_matrix[0, -1], results_matrix[2, -1], color='red', s=20, label='End position')

    plt.xlabel('X Displacement (m)', fontsize=12)
    plt.ylabel('Y Displacement (m)', fontsize=12)
    plt.title('Sail Drone Trajectory', fontsize=14)

    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.6)

    end_x, end_y = results_matrix[0, -1], results_matrix[2, -1]
    plt.annotate(f'({end_x:.2f}m, {end_y:.2f}m)',
                 xy=(end_x, end_y),
                 xytext=(end_x + 5, end_y))

    end_heading = results_matrix[4, -1]
    plt.annotate(f'(Heading: {end_heading:.2f}rad)',
                 xy=(end_x, end_y),
                 xytext=(end_x + 7, end_y - 10))

    plt.legend()
    plt.show()


def plot_headingvstime(time_axis, results_matrix):
    plt.figure()
    plt.plot(time_axis, results_matrix[4, :], 'blue')
    plt.xlabel('Time, s')
    plt.ylabel('Heading, rad')
    plt.title('Sail Drone Heading against Time', fontsize=14)

    plt.grid(True, linestyle='--', alpha=0.6)

    course_sections = [[0, "A"], [60, "B"], [65, "C"]]

    for i in range(3):
        time = course_sections[i][0]
        section = course_sections[i][1]

        plt.axvline(x=time, linestyle=':', color='orange', alpha=0.8)

        x_val = int(time / 0.1 if time != 0 else 0)
        theta = results_matrix[4, x_val]

        plt.annotate(f'{section}', color='orange', fontsize='18',
                     xy=(time, 1.45),
                     xytext=(time + 0.5, 1.45) if section != 'B' else (time - 3, 1.45))

        plt.annotate(f'({theta:.2f}rad)', color='orange',
                     xy=(time, 1.4),
                     xytext=(time + 0.5, 1.42) if section != 'B' else (time - 12, 1.42))

    plt.xlim(-2, 80)
    plt.show()


def plot_velocityvstime(time_axis, results_matrix):

    plt.figure()
    plt.plot(time_axis, results_matrix[1, :], 'blue')
    plt.plot(time_axis, results_matrix[3, :], 'red')
    plt.xlabel('Time, s')
    plt.ylabel('Velocity, m/s')
    plt.legend(['X Velocity', 'Y Velocity'])
    plt.title('Velocity Components against Time', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)

    course_sections = [[0, "A"], [60, "B"], [65, "C"]]

    for i in range(3):
        time = course_sections[i][0]
        section = course_sections[i][1]

        plt.axvline(x=time, linestyle=':', color='orange', alpha=0.8)

        x_val = int(time / 0.1 if time != 0 else 0)
        vx, vy = results_matrix[1, x_val], results_matrix[3, x_val]

        plt.annotate(f'{section}', color='orange', fontsize='18',
                     xy=(time, 2.5),
                     xytext=(time + 0.5, 2.5) if section != 'B' else (time - 3, 2.5))

        plt.annotate(f'X Vel: {vx:.2f}\nY Vel: {vy:.2f}', color='orange',
                     xy=(time, 1.25),
                     xytext=(time + 0.5, 2) if section != 'B' else (time - 13, 2))

    plt.xlim(-2, 80)
    plt.show()


if __name__ == '__main__':

    t0 = 0
    tmax = 65
    dt = 0.1

    x0 = 0
    Vx0 = 0
    y0 = 0
    Vy0 = 2.9
    theta0 = np.pi/2
    Vtheta0 = 0

    z0 = np.array([[x0], [Vx0], [y0], [Vy0], [theta0], [Vtheta0]])

    [t, z] = solve_ivp(state_deriv, t0, tmax, dt, z0)

    plot_trajectory(z)
    plot_headingvstime(t,z)
    plot_velocityvstime(t,z)
