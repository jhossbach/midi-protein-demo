import numpy as np
import pygame.midi as midi
from enum import IntEnum
import time
import threading
import espressomd

import espressomd.visualization
import espressomd.polymer
import espressomd.interactions

import matplotlib.pyplot as plt

np.random.seed(seed=43)


class Lever(IntEnum):
    """Midi event IDs."""

    TEMP = 224
    WEIGHT = 225
    ROTATION = 176
    IDLE = 0

    @classmethod
    def has_value(cls, value):
        return value in list(cls.__members__.values())


class MidiControls:
    def __init__(self, init_temp=127, init_weight=127):
        self._get_midi_devices()
        self._connect_to_midi()
        self._reset_values(init_temp, init_weight)

    def _get_midi_devices(self):
        if not midi.get_init():
            midi.init()
        for i in range(midi.get_count()):
            interf, name, input, output, opened = midi.get_device_info(i)
            if input:
                if "Through Port" not in name.decode():
                    self.input_midi_device = i
                    break
        for i in range(midi.get_count()):
            interf, name, input, output, opened = midi.get_device_info(i)
            if output:
                if "Through Port" not in name.decode():
                    self.output_midi_device = i
                    break

    def _reset_values(self, init_temp, init_weight):
        self.midi_output.write_short(Lever.TEMP, init_temp, init_temp)
        self.midi_output.write_short(Lever.WEIGHT, init_weight, init_weight)

    def _connect_to_midi(self):
        self.midi_input = midi.Input(self.input_midi_device)
        self.midi_output = midi.Output(self.output_midi_device)

    def poll(self):
        """Return whether the MIDI state has changed."""
        return self.midi_input.poll()

    def read(self):
        """
        Read the input of the MIDI controller, return the corresponding lever and its data.
        We use the first lever for the temperature (event ID 124) and the jog for the rotation (event ID 176).
        The data of the lever contains values between 0 and 127 while data2 of jog takes values of either 1 or
         65 depending on the rotation direction.
        """
        event = self.midi_input.read(1)
        event = event[0][0]
        status, data1, data2, _ = event
        if Lever.has_value(status):
            return [Lever(status), data2]
        return [Lever.IDLE, 0]


class SystemProteinDemo:
    def __init__(self):
        self.num_integrator_steps = 100
        self.messagehandler = {
            Lever.TEMP: self._change_midi_temp,
            Lever.ROTATION: self._change_midi_rotation,
            Lever.WEIGHT: self._change_midi_weight,
        }

        # TODO: Improve values for better visualization effect
        self.TEMP_MIN = 0.001
        self.TEMP_MAX = 0.02
        self.MIDI_TEMP = (self.TEMP_MAX + self.TEMP_MIN) / 2

        self.WEIGHT_MIN = 0.009
        self.WEIGHT_MAX = 0.07
        self.MIDI_WEIGHT = (self.WEIGHT_MAX + self.WEIGHT_MIN) / 2

        self.MIDI_ROTATION = 0.0
        self.midi = MidiControls(
            init_temp=int(self.MIDI_TEMP / (self.TEMP_MAX + self.TEMP_MIN) * 127),
            init_weight=int(
                self.MIDI_WEIGHT / (self.WEIGHT_MAX + self.WEIGHT_MIN) * 127
            ),
        )

        self.flag_dict = {Lever.TEMP: False, Lever.ROTATION: False, Lever.WEIGHT: False}

        self.end_to_end_distance = []
        self.temperature = [self.MIDI_TEMP]
        self.weight = [self.MIDI_WEIGHT]

        self.set_up_system()
        self.set_up_camera()
        self.init_plots()

    def set_up_system(self):
        """
        Create the system, add particles and bonds.
        Rotate the system and move all particles so that the first particle is
         in the upper center of the box
        Fix position of the first particle
        """
        self.system = espressomd.System(box_l=[35, 35, 35])

        self.system.time_step = 0.002

        positions = espressomd.polymer.linear_polymer_positions(
            n_polymers=1, beads_per_chain=25, bond_length=1.0, seed=3610
        )
        part_list = []
        positions = positions[0]
        for pos in positions:
            part = self.system.part.add(pos=pos)
            part_list.append(part)

        # Rotate the system so that the vector between the first and last particle
        #  point towards the negative z direction
        distance_vec = (positions[-1] - positions[0]) / np.linalg.norm(
            positions[-1] - positions[0]
        )
        theta = np.arccos(distance_vec[2] / np.linalg.norm(distance_vec))
        phi = np.arctan2(distance_vec[1], distance_vec[0])
        self.system.rotate_system(
            alpha=np.pi - theta, theta=np.pi / 2, phi=phi + np.pi / 2
        )

        # Move particles to center of box
        box_l = self.system.box_l[0]
        move_vec = self.system.distance_vec(
            part_list[0], [box_l / 2, box_l / 2, box_l - 1]
        )
        for part in self.system.part.all():
            part.pos = part.pos + move_vec

        # Fix first and last particle, apply external force
        part_list[0].fix = 3 * [True]
        part_list[-1].fix = [True, True, False]

        part_list[-1].ext_force = [0, 0, -self.MIDI_WEIGHT]
        # For the weight update
        self.first_particle = part_list[0]
        self.weight_particle = part_list[-1]

        # Get the first value for the end to end distance
        self.end_to_end_distance.append(
            self.system.distance(self.first_particle, self.weight_particle)
            / self.system.box_l[0]
        )

        # Add bonds between particles
        fene = espressomd.interactions.HarmonicBond(k=1.0, r_cut=2.0, r_0=1.0)
        self.system.bonded_inter.add(fene)
        previous_part = None
        for part in part_list:
            if previous_part:
                part.add_bond((fene, previous_part))
            previous_part = part

        # Apply initial temperature
        self.system.thermostat.set_langevin(kT=self.MIDI_TEMP, gamma=1.0, seed=42)

    def change_temperature(self):
        """Change the temperature of the system to the new value."""
        if self.flag_dict[Lever.TEMP]:
            self.system.thermostat.set_langevin(kT=self.MIDI_TEMP, gamma=1.0)
        self.flag_dict[Lever.TEMP] = False

    def update_weight(self):
        """Adapt to new weight."""
        if self.flag_dict[Lever.WEIGHT]:
            self.weight_particle.ext_force = [0, 0, -self.MIDI_WEIGHT]

    def update_observables(self):
        """Update the end-to-end distance for the plots."""
        # Scale to numbers between 0 and 1
        self.end_to_end_distance.append(
            np.linalg.norm(self.weight_particle.pos - self.first_particle.pos)
            / self.system.box_l[0]
        )
        self.temperature.append(
            (self.MIDI_TEMP - self.TEMP_MIN) / (self.TEMP_MAX - self.TEMP_MIN)
        )
        self.weight.append(
            (self.MIDI_WEIGHT - self.WEIGHT_MIN) / (self.WEIGHT_MAX - self.WEIGHT_MIN)
        )

    #############################################################
    #      MIDI Interface                                       #
    #############################################################

    def _change_midi_temp(self, kT_new_midi):
        self.MIDI_TEMP = self.TEMP_MIN + kT_new_midi / 127.0 * (
            self.TEMP_MAX - self.TEMP_MIN
        )

    def _change_midi_weight(self, weight_new):
        self.MIDI_WEIGHT = self.WEIGHT_MIN + weight_new / 127.0 * (
            self.WEIGHT_MAX - self.WEIGHT_MIN
        )

    def _change_midi_rotation(self, new_rot):
        # MIDI_ROTATION is either negative or positive depending on the direction
        #  of rotation of the jog
        self.MIDI_ROTATION = (32.0 - new_rot) * 0.01

    def midi_thread(self):
        """Target method of the MIDI thread."""
        while True:
            if self.midi.poll():
                msg = self.midi.read()
                if not msg[0] == Lever.IDLE:
                    self.messagehandler[msg[0]](msg[1])
                    self.flag_dict[msg[0]] = True
            time.sleep(0.01)

    def adapt_new_values(self):
        """Adapt the current system to the new parameters."""
        self.change_temperature()
        self.update_weight()
        self.change_angle()

    #############################################################
    #      Visualizer                                           #
    #############################################################

    def set_up_camera(self):
        """Initialize the visualizer, rotate the camera."""
        self.visualizer = espressomd.visualization.openGLLive(
            self.system, camera_position=[15, -40, 15]
        )
        self.visualizer.register_callback(self.adapt_new_values, interval=100)
        self.visualizer.register_callback(self.update_plots, interval=100)

    def change_angle(self):
        """Change to new angle of view."""
        if self.flag_dict[Lever.ROTATION]:
            self.visualizer.camera.rotate_system_y(self.MIDI_ROTATION)
        self.flag_dict[Lever.ROTATION] = False

    #############################################################
    #      Plots                                                #
    #############################################################

    def init_plots(self):
        self.fig = plt.figure()
        (self.plot_end_to_end_dist,) = plt.plot(
            [self.system.time], self.end_to_end_distance, label="Laenge"
        )
        (self.plot_temperature,) = plt.plot(
            [self.system.time], self.temperature, label="Temperatur"
        )
        (self.plot_weight,) = plt.plot([self.system.time], self.weight, label="Gewicht")
        plt.ylim(0.0, 1.1)
        plt.xlim(0.0, 100)
        plt.legend()
        plt.xlabel("Time")
        self.fig.canvas.draw()
        plt.show(block=False)

    def update_plots(self):
        """Update the plots."""
        self.update_observables()
        time_array = np.linspace(0, self.system.time, num=len(self.end_to_end_distance))
        self.plot_end_to_end_dist.set_xdata(time_array)
        self.plot_end_to_end_dist.set_ydata(self.end_to_end_distance)
        self.plot_temperature.set_xdata(time_array)
        self.plot_temperature.set_ydata(self.temperature)
        self.plot_weight.set_xdata(time_array)
        self.plot_weight.set_ydata(self.weight)
        if np.round(self.system.time % 100) < 7:
            plt.xlim(right=self.system.time + 100)
        plt.draw()
        plt.pause(0.01)

    #############################################################
    #      Integration                                          #
    #############################################################

    def main_thread(self):
        while True:
            self.system.integrator.run(self.num_integrator_steps)
            self.visualizer.update()

    def run(self):
        """Run the simulation."""
        self.thread_midi = threading.Thread(target=self.midi_thread, daemon=True)
        self.thread_main = threading.Thread(target=self.main_thread, daemon=True)
        self.thread_midi.start()
        self.thread_main.start()
        self.visualizer.start()


if __name__ == "__main__":
    demo = SystemProteinDemo()
    demo.run()
