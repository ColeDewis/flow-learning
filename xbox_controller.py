import numpy as np
import pygame
from robosuite.devices.device import Device


class XboxControllerDevice(Device):
    """
    A device class for interfacing with an Xbox One controller.
    """

    def __init__(self, env, deadzone=0.2, invert_xy=False):
        """
        Initialize the Xbox controller device.

        Args:
            env (RobotEnv): The environment which contains the robot(s) to control using this device.
            deadzone (float): Threshold below which joystick or trigger inputs are ignored.
        """
        super().__init__(env)

        # Initialize pygame for controller input
        pygame.init()
        pygame.joystick.init()

        # Ensure at least one joystick is connected
        if pygame.joystick.get_count() == 0:
            raise RuntimeError(
                "No Xbox controller detected. Please connect a controller."
            )

        # Use the first joystick
        self.controller = pygame.joystick.Joystick(0)
        self.controller.init()

        # Deadzone threshold
        self.deadzone = deadzone

        self.invert_xy = invert_xy

        # Internal state
        self._reset_internal_state()

    def _reset_internal_state(self):
        """
        Resets internal state related to robot control.
        """
        super()._reset_internal_state()
        self._prev_grasp = False
        self._grasp_state = False  # Current grasp state (default: not grasping)

    def start_control(self):
        """
        Method that should be called externally before the controller can start receiving commands.
        """
        print("Xbox controller is ready for input.")

    def _apply_deadzone(self, value):
        """
        Applies the deadzone to a single input value.

        Args:
            value (float): The input value to process.

        Returns:
            float: The processed value, set to 0 if within the deadzone.
        """
        if abs(value) < self.deadzone:
            return 0.0
        return value

    def get_controller_state(self):
        """
        Returns the current state of the Xbox controller.

        Returns:
            Dict: A dictionary containing dpos, rotation, raw_drotation, grasp, and reset.
        """
        pygame.event.pump()  # Process controller events

        # Get joystick axes with deadzone applied
        dpos = np.array(
            [
                self._apply_deadzone(self.controller.get_axis(1)),  # Left stick X-axis
                self._apply_deadzone(
                    self.controller.get_axis(0)
                ),  # Left stick Y-axis (invert for natural movement)
                self._apply_deadzone(
                    self.controller.get_axis(5) - self.controller.get_axis(2)
                )
                * 0.5,  # Right trigger - Left trigger for Z-axis
            ]
        )

        if self.invert_xy:
            dpos[0], dpos[1] = -dpos[0], -dpos[1]

        # Get rotation (right stick) with deadzone applied
        raw_drotation = np.array(
            [
                0.0,
                0.0,
                -self._apply_deadzone(self.controller.get_axis(3)),
                # self._apply_deadzone(self.controller.get_axis(4)),  # Right stick X-axis
                # -self._apply_deadzone(
                #     self.controller.get_axis(3)
                # ),  # Right stick Y-axis (invert for natural movement)
                # 0.0,  # No roll control for now
            ]
        )

        # Grasp (A button)
        a_button = self.controller.get_button(0)  # A button
        if a_button and not self._prev_grasp:  # Detect button press (not hold)
            self._grasp_state = not self._grasp_state  # Toggle grasp state
        self._prev_grasp = a_button  # Update previous state of the A button

        # Reset (Start button)
        reset = self.controller.get_button(7)  # Start button

        # Base mode (optional, controlled by B button)
        base_mode = self.controller.get_button(1)  # B button

        # PROPER AXES: [left x, left y, left trigger, right x, right y, right trigger]
        # print("dpos", dpos)
        # print("raw_drotation", raw_drotation)
        # print("axes", [self.controller.get_axis(i) for i in range(6)])

        return {
            "dpos": dpos * 3,
            "rotation": np.eye(3),  # Identity matrix for absolute rotation
            "raw_drotation": raw_drotation * 2,
            "grasp": self._grasp_state,
            "reset": reset,
            "base_mode": base_mode,
        }

    def _postprocess_device_outputs(self, dpos, drotation):
        """
        Postprocess raw device outputs to scale them appropriately.

        Args:
            dpos (np.ndarray): Raw delta position from the controller.
            drotation (np.ndarray): Raw delta rotation from the controller.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Processed dpos and drotation.
        """
        # Scale dpos and drotation for smoother control
        dpos *= 0.05  # Scale position changes
        drotation *= 0.15  # Scale rotation changes
        return dpos, drotation
