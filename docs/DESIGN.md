# Design notes

## Frames and signs

World is east-north-up. The body frame is x forward, y left, z up. A quaternion `[w, x, y, z]` rotates body vectors into the world. Attitude errors are small rotations in **body** axes: `q_true = q_est (x) exp(dtheta)`.

Motors are numbered front-right, front-left, rear-left, rear-right, spinning counter-clockwise, clockwise, counter-clockwise, clockwise. The matrix in `Params.allocation` turns four motor thrusts into collective thrust and three torques, and the mixer is its inverse, so the physics and the controller cannot disagree about signs. A test checks the yaw torque direction against the propeller spin.

## Physics

Motor thrust follows its command through a 20 ms first-order lag. The rigid-body equation is `J w' = tau - w x J w`; the cross term matters for the asymmetric-inertia conservation test and costs nothing. A fourth-order Runge-Kutta step at 1 ms is exact enough that a torque-free body keeps its angular momentum and energy to six digits over two seconds.

The IMU model needs the *specific force*: acceleration with gravity left out, in body axes. The dynamics keep that value from the last derivative evaluation. On the floor, the floor pushes up with exactly one g, so a resting accelerometer reads +g; the first version of the sim got this wrong (it read zero), and the filter started the flight 95 degrees out. Initial attitude is taken from gravity and the compass while the vehicle is still.

## The estimator

Fifteen error states: position, velocity, attitude, accelerometer bias, gyro bias. The nominal state is the IMU integrated forward. Between measurements the error covariance grows through the linearised error dynamics:

```
dp     = dv
dv     = -R [f]x dtheta - R d(ba)  + noise
dtheta = -[w]x dtheta - d(bg)     + noise
```

The second line is what makes tilt observable. If the attitude estimate is tilted, the filter thinks the thrust pushes sideways when it does not, its velocity drifts, and GPS velocity tells it so. This is how the filter finds roll and pitch while accelerating, when the accelerometer alone points at the wrong "down". A gravity-vector correction would be wrong in a banked turn, which is why I did not use one.

Updates: GPS position and velocity (10 Hz), barometer (25 Hz), compass heading (25 Hz). Each is applied in Joseph form, the error is folded into the nominal state, and the covariance is re-expressed in the new attitude frame.

Things that are deliberately simple: heading comes straight from a compass angle (no magnetometer calibration), the compass Jacobian is found numerically, and measurements arrive with no delay.

## The controller

Loops, outermost first, with the bandwidths the gains produce:

1. **Position P** gives a velocity command (1.4 per second horizontally, so about 0.7 s time constant). The reference velocity is added as a feed-forward, which is what lets it follow a moving target without lagging.
2. **Velocity PI** gives an acceleration command. The reference acceleration is added as a feed-forward. The integrator clamps at 3 m/s^2 and stays at zero while the vehicle is on the floor.
3. The commanded acceleration plus gravity is a **thrust vector**. Its length, projected on the current body z axis, becomes collective thrust. Its direction, together with the yaw reference, defines the desired attitude.
4. **Attitude P** works on the quaternion error and gives a body-rate command (9 per second for roll and pitch).
5. **Rate PID** gives a torque, scaled by the inertia so that gains are in 1/s. The derivative term works on a low-pass filtered rate, not on the error, so reference changes do not kick it.
6. **Mixer** inverts the allocation matrix. When a motor would go outside 0 to 8 N it scales the torque part down, not the collective thrust, so the vehicle keeps its height and gives up some agility. Tilt is limited to 40 degrees.

On the floor with a floor reference the controller idles the motors, so sensor noise cannot lift the vehicle off: a test (`test_resting_on_the_ground_does_not_wander`) found it creeping to 7 cm before the rule existed.

## Why a figure of eight

It asks for continuous acceleration in changing directions, which exposes lag in the loops, any mismatch between the feed-forward and the real response, and attitude estimation errors in a banked turn. The reference is eased in over four seconds so the vehicle starts from rest.

## Testing approach

The physics is tested against closed-form results (free fall, torque-limited roll rate, conserved quantities). The filter is tested only through whole flights, comparing its output with the truth the simulator keeps, since the ways it can be wrong show up as bad flights. The flight tests use wide margins on purpose: they are there to catch a broken change, not to pin down exact numbers.

## Next

- Sensor delay with a state buffer in the filter.
- Wind estimation as extra states.
- A mode for tuning the loops from a step response automatically.
