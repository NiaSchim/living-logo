#-------------------------------------------------------------------------------
# Name:        LavalampParticles.py
# Purpose: pretty visualizer
#
# Author:      The Schim
#
# Created:     16/03/2023
# Copyright:   (c) The Schim 2023
# Licence:     CC0
#-------------------------------------------------------------------------------
import pygame
import random
import math
from colorsys import hsv_to_rgb, rgb_to_hsv
import uuid

# Constants
WIDTH, HEIGHT = 600, 600
BG_COLOR = (0, 0, 0)
FPS = 60
MIN_RADIUS = 33.3
MAX_RADIUS = 99.9
SPLIT_PROB = 0.29
DEPTH = 600
cooldown = random.randint(314, 6400)
INITIAL_GLOBS = 25
MAX_NUMBER_GLOBS = 115
SPEED_DIVISOR = .5
AGE_FACTOR = 0.1
TRANSFER = 0.00075

from scipy.signal import firwin, lfilter
import numpy as np
import simpleaudio as sa
import scipy.signal
import threading
import time
import sounddevice as sd
import scipy.optimize as opt

# Define audio constants
SAMPLE_RATE = int(44100//100)
BIT_DEPTH = -16
NUM_CHANNELS = 2
AUDIO_BUFFER_SIZE = round(1024*2)
NUM_CHANNELS = 2  # mono audio with psuedo surround
AMPLITUDE = 3000 # maximum amplitude of the audio signal
PENTATONIC_SCALE = [2, 4, 6, 9, 11]  # pentatonic scale intervals in semitones
OCTAVES = 8  # number of octaves to span with the pentatonic scale
BASE_FREQ = 466.16 / FPS
SEMITONE_RATIO =  2**(1/12)
FRAME_INTERVAL = FPS/math.pi*2/1.3333333333
FRAME_INTERVAL2 =  FPS/(math.sqrt(2)*2)*6/2
BUFFER_OVERLAP = AUDIO_BUFFER_SIZE
start = True
thread_running = True
BIT_DEPTH = 16

# Add these lines at the beginning of the script
pygame.mixer.init()
pygame.mixer.set_num_channels(2)
# Define global variables
current_buffer = bytes()
current_buffer_lock = threading.Lock()

def generate_next_buffer(globs):
    num_samples = int(FRAME_INTERVAL * SAMPLE_RATE)
    next_signal = np.zeros((0, NUM_CHANNELS), dtype=np.float32)

    for glob in globs:
        # Calculate pan value based on the x-coordinate
        pan = glob.x / WIDTH

        # Calculate the note index based on the y-coordinate
        note_index = int(36 * (HEIGHT - glob.y) / HEIGHT)

        # Calculate the frequency of the note
        frequency = BASE_FREQ * SEMITONE_RATIO ** (PENTATONIC_SCALE[note_index % len(PENTATONIC_SCALE)])

        # Calculate the detuning based on the glob size
        detuning = -50 * (glob.radius - MIN_RADIUS) / (MAX_RADIUS - MIN_RADIUS)
        if detuning > 0:
            detuning = -50 + detuning
        frequency *= 2 ** (detuning / 1200)

        # Calculate the amplitude based on the z-coordinate
        amplitude = AMPLITUDE * (1 - glob.z / DEPTH)

        # Generate a sine wave for this glob
        sine_wave = generate_triangle_wave(num_samples, frequency, amplitude)

        # Pan the sine wave
        left_channel_gain = 1 - pan
        right_channel_gain = pan
        panned_wave = np.array([left_channel_gain * sine_wave, right_channel_gain * sine_wave]).T

        # Crossfade between notes if necessary
        if glob.note_index != note_index:
            glob.note_index = note_index
            crossfade_duration = round(SAMPLE_RATE * 1/8)
            panned_wave = crossfade(glob.prev_wave[-crossfade_duration:], panned_wave, crossfade_duration)

        # Update the glob's previous wave
        glob.prev_wave = panned_wave

        # Add the panned wave to the mix
        next_signal = np.concatenate([next_signal, panned_wave])

    # Normalize the signal and apply amplification
    max_value = np.max(np.abs(next_signal))
    if max_value > 0:
        next_signal = (next_signal / max_value) * AMPLITUDE
    else:
        next_signal = np.zeros((0, NUM_CHANNELS), dtype=np.float32)

    fade_in_samples = round(SAMPLE_RATE * 1/8)  # ms fade in
    fade_out_samples = round(SAMPLE_RATE * 1/8)  # ms fade out
    next_signal = apply_fade(next_signal, fade_in_samples, fade_out_samples)

    next_signal_bytes = next_signal.astype(np.int16).tobytes()

    # Ensure the buffer size is a multiple of bytes-per-sample and the number of channels
    buffer_size = len(next_signal_bytes)
    remainder = buffer_size % (NUM_CHANNELS * 2)  # 2 bytes per sample for int16

    if remainder != 0:
        padding_size = NUM_CHANNELS * 2 - remainder
        next_signal_bytes = np.concatenate((next_signal_bytes, np.zeros((padding_size, 2), dtype=np.int8)))

    return next_signal_bytes

def amplify_audio(audio, factor):
    return np.clip(audio * factor, -32768, 32767).astype(np.int16)

def generate_next_buffer_thread(globs):
    global current_buffer
    next_buffer = generate_next_buffer(globs)
    current_buffer_lock.acquire()
    current_buffer = next_buffer
    current_buffer_lock.release()

def apply_fade(signal, fade_in_samples, fade_out_samples):
    num_samples = len(signal)
    if num_samples < fade_in_samples + fade_out_samples:
        raise ValueError("Signal length is shorter than the total fade in and fade out duration")

    for i in range(fade_in_samples):
        gain = i / fade_in_samples
        signal[i] *= gain

    for i in range(num_samples - fade_out_samples, num_samples):
        gain = (num_samples - i) / fade_out_samples
        signal[i] *= gain

    return signal

def process_buffer(buffer, prev_buffer):
    buffer_mono = np.mean(buffer, axis=1)
    prev_buffer_mono = np.mean(prev_buffer, axis=1)
    crossfaded_data = crossfade(prev_buffer[-BUFFER_OVERLAP:], buffer_mono[:BUFFER_OVERLAP], BUFFER_OVERLAP // 3)
    crossfaded_data_mono = np.mean(crossfaded_data, axis=1)

    # Convert mono crossfaded data to stereo
    crossfaded_data_stereo = np.column_stack((crossfaded_data, np.repeat(crossfaded_data_mono[:, np.newaxis], 2, axis=1)))

    # Resize crossfaded_data_stereo if necessary
    if crossfaded_data_stereo.shape[0] != buffer.shape[0]:
        crossfaded_data_stereo = resize_array(crossfaded_data_stereo, buffer.shape)

    buffer_stereo = np.column_stack((buffer, np.repeat(buffer_mono[:, np.newaxis], 2, axis=1)))
    prev_buffer_stereo = np.column_stack((prev_buffer, np.repeat(prev_buffer_mono[:, np.newaxis], 2, axis=1)))
    buffer_mono_sq = np.mean(buffer_stereo, axis=1)
    prev_buffer_mono_sq = np.mean(prev_buffer_stereo, axis=1)
    crossfaded_data_stereo_sq = np.mean(crossfaded_data_stereo, axis=1)
    buffer_stereo_sq = np.column_stack((buffer_stereo, np.repeat(buffer_mono_sq[:, np.newaxis], 2, axis=1)))
    prev_buffer_stereo_sq = np.column_stack((prev_buffer_stereo, np.repeat(prev_buffer_mono_sq[:, np.newaxis], 2, axis=1)))
    crossfaded_data_stereo_sq = np.column_stack((crossfaded_data_stereo, np.repeat(crossfaded_data_stereo_sq[:, np.newaxis], 2, axis=1)))
    buffer_mono_sq = np.mean(buffer_stereo_sq, axis=1)
    prev_buffer_mono_sq = np.mean(prev_buffer_stereo_sq, axis=1)
    crossfaded_data_stereo_sq = np.mean(crossfaded_data_stereo_sq, axis=1)
    crossfaded_data_stereo_sq = np.clip(crossfaded_data_stereo_sq, -1, 1)

    return buffer_mono_sq, prev_buffer_mono_sq, crossfaded_data_stereo_sq

def resize_array(array, new_shape):
    """
    Resizes a numpy array to a new shape using bilinear scaling algorithm.
    """
    if array.ndim != 2:
        raise ValueError("Input array must be 2-dimensional")
    if len(new_shape) != 2:
        raise ValueError("New shape must be a tuple of length 2")
    if not isinstance(new_shape[0], int) or not isinstance(new_shape[1], int):
        raise ValueError("New shape dimensions must be integers")

    if array.shape[0] == new_shape[0] and array.shape[1] == new_shape[1]:
        return array.copy()

    if array.shape[0] == 1 and new_shape[0] != 1:
        array = np.repeat(array, new_shape[0], axis=0)
    if array.shape[1] == 1 and new_shape[1] != 1:
        array = np.repeat(array, new_shape[1], axis=1)

    x_ratio = float(array.shape[1]) / new_shape[1]
    y_ratio = float(array.shape[0]) / new_shape[0]

    if array.ndim == 2:
        output_array = np.zeros(new_shape, dtype=array.dtype)
    else:
        output_array = np.zeros((new_shape[0], new_shape[1], array.shape[2]), dtype=array.dtype)

    for y in range(new_shape[0]):
        y_floor = int(np.floor(y * y_ratio))
        y_ceil = min(int(np.ceil(y * y_ratio)), array.shape[0]-1)
        y_offset = (y * y_ratio) - y_floor
        for x in range(new_shape[1]):
            x_floor = int(np.floor(x * x_ratio))
            x_ceil = min(int(np.ceil(x * x_ratio)), array.shape[1]-1)
            x_offset = (x * x_ratio) - x_floor
            y1_x1 = array[y_floor, x_floor]
            y1_x2 = array[y_floor, x_ceil] if x_ceil < array.shape[1] else y1_x1
            y2_x1 = array[y_ceil, x_floor] if y_ceil < array.shape[0] else y1_x1
            y2_x2 = array[y_ceil, x_ceil] if x_ceil < array.shape[1] and y_ceil < array.shape[0] else y1_x1
            if array.ndim == 2:
                output_array[y, x] = round((1 - y_offset) * ((1 - x_offset) * y1_x1 + x_offset * y1_x2) +
                                            y_offset * ((1 - x_offset) * y2_x1 + x_offset * y2_x2))
            else:
                output_array[y, x] = [round((1 - y_offset) * ((1 - x_offset) * y1_x1[c] + x_offset * y1_x2[c]) +
                                            y_offset * ((1 - x_offset) * y2_x1[c] + x_offset * y2_x2[c]))
                                      for c in range(array.shape[2])]
    return output_array

def play_buffer(channel, buffer, prev_buffer):
    if channel.get_queue() is None:
        sound = pygame.mixer.Sound(buffer)
        channel.queue(sound)
        return buffer
    else:
        return prev_buffer



def play_audio(globs, buffer1, buffer2, buffer3, thread_running, FRAME_INTERVAL):
    prev_buffer = np.zeros((AUDIO_BUFFER_SIZE * NUM_CHANNELS,), dtype=np.float64)

    while thread_running:
        channel_group1 = pygame.mixer.find_channel()
        channel_group2 = pygame.mixer.find_channel()
        channel_group3 = pygame.mixer.find_channel()

        prev_buffer = play_buffer(channel_group1, buffer1, prev_buffer)
        prev_buffer = play_buffer(channel_group2, buffer2, prev_buffer)
        prev_buffer = play_buffer(channel_group3, buffer3, prev_buffer)

        new_buffer = generate_next_buffer(globs)
        buffer1, buffer2, buffer3 = buffer2, buffer3, new_buffer
        pygame.time.wait(int(1000 * FRAME_INTERVAL))

def play_audio_second_interval(globs, buffer1, buffer2, buffer3, thread_running):
    play_audio(globs, buffer1, buffer2, buffer3, thread_running, FRAME_INTERVAL)

def brachistochrone_curve(t, total_time):
    def f(cycloid_param):
        return cycloid_param - total_time * (1 - np.sin(cycloid_param)) / 2

    cycloid_param = opt.newton(f, 1)
    return (1 - np.cos(t * cycloid_param / total_time)) / 2

def crossfade(buffer1, buffer2, fade_duration):
    fade_duration = min(fade_duration, len(buffer1), len(buffer2))
    t = np.linspace(0, fade_duration, fade_duration)
    fade_out = 1 - brachistochrone_curve(t, fade_duration)
    fade_in = brachistochrone_curve(t, fade_duration)

    # Convert mono buffers to stereo
    if len(buffer1.shape) == 1:
        buffer1 = np.stack((buffer1, buffer1), axis=1)
    if len(buffer2.shape) == 1:
        buffer2 = np.stack((buffer2, buffer2), axis=1)

    faded_buffer1 = buffer1[-fade_duration:] * fade_out[:, np.newaxis]
    faded_buffer2 = buffer2[:fade_duration] * fade_in.reshape(-1, 1)

    # Compress the longer buffer to the size of the shorter buffer
    if faded_buffer1.shape[0] != faded_buffer2.shape[0]:
        if faded_buffer1.shape[0] > faded_buffer2.shape[0]:
            faded_buffer1 = faded_buffer1[:faded_buffer2.shape[0], :]
        else:
            faded_buffer2 = faded_buffer2[:faded_buffer1.shape[0], :]

    return (faded_buffer1 + faded_buffer2)

#we're not really generating a triangle wave anymore, I'm just lazy, so I'm changing one function instead of 1 function and a billion name-calls
def generate_triangle_wave(num_samples, frequency, amplitude=1.0):
    """Generates a sine wave with the given frequency, amplitude, and sample rate."""
    t = np.linspace(0, num_samples / AUDIO_BUFFER_SIZE, num_samples, endpoint=False)
    triangle_wave = amplitude * np.sin(2 * np.pi * frequency * t)
    return triangle_wave

def generate_audio_signal(globs, VOLUME_SHRINK_FACTOR=0.85):
    VOLUME_SHRINK_FACTOR = 1
    num_samples = int(FRAME_INTERVAL * SAMPLE_RATE)
    signal = np.zeros((num_samples, 2), dtype=np.float32)

    for glob in globs:
        # Calculate pan value based on the x-coordinate
        pan = glob.x / WIDTH

        # Use the note_index attribute from the Glob object
        note_index = glob.note_index

        # Calculate the frequency of the note
        frequency = BASE_FREQ * SEMITONE_RATIO ** (PENTATONIC_SCALE[note_index % len(PENTATONIC_SCALE)])

        # Calculate the detuning based on the glob size
        detuning = -50 * (glob.radius - MIN_RADIUS) / (MAX_RADIUS - MIN_RADIUS)
        if detuning > 0:
            detuning = -50 + detuning
        frequency *= 2 ** (detuning / 1200)

        # Calculate the amplitude based on the z-coordinate
        amplitude = AMPLITUDE * (1 - glob.z / DEPTH)

        # Generate a triangle wave for this glob
        triangle_wave = generate_triangle_wave(num_samples, frequency, amplitude)

        # Pan the triangle wave
        left_channel_gain = 1 - pan
        right_channel_gain = pan
        panned_wave = np.array([left_channel_gain * triangle_wave, right_channel_gain * triangle_wave]).T

        # Crossfade between notes if necessary
        if glob.note_index != note_index:
            glob.note_index = note_index
            crossfade_duration = round(SAMPLE_RATE * 1/8)
            panned_wave = crossfade(glob.prev_wave[-crossfade_duration:], panned_wave, crossfade_duration)

        # Update the glob's previous wave
        glob.prev_wave = panned_wave

        # Add the panned wave to the signal
        signal += panned_wave

    # Normalize the signal and convert to int8
    max_value = np.max(np.abs(signal))
    if max_value > 0:
        signal /= max_value
    else:
        signal = np.zeros((num_samples, 2), dtype=np.float32)
    signal *= VOLUME_SHRINK_FACTOR
    signal_bytes = (signal * 127).astype(np.int8)

    return signal_bytes




def random_point_on_ellipsoid(a, b, c):
    while True:
        u = random.uniform(-1, 1)
        v = random.uniform(-1, 1)
        w = random.uniform(-1, 1)
        d = u**2/a**2 + v**2/b**2 + w**2/c**2

        if d <= 1:
            break

    x = (WIDTH / 2) + a * u
    y = (HEIGHT / 2) + b * v
    z = (DEPTH / 2) + c * w

    x = max(MIN_RADIUS, min(WIDTH - MIN_RADIUS, x))
    y = max(MIN_RADIUS, min(HEIGHT - MIN_RADIUS, y))
    z = max(MIN_RADIUS, min(DEPTH - MIN_RADIUS, z))

    return x, y, z

def color_difference(color1, color2):
    return sum(abs(color1[i] - color2[i]) for i in range(3))

def is_similar_color(color1, color2, threshold=32):
    return color_difference(color1, color2) < threshold

def calculate_mutation_range(globs):
    total_globs = len(globs)
    similar_color_count = 0

    for i in range(total_globs):
        for j in range(i+1, total_globs):
            if is_similar_color(globs[i].color, globs[j].color):
                similar_color_count += 1

    percentage_similar_color = similar_color_count / total_globs
    mutation_range = int(percentage_similar_color * 255)

    return mutation_range

def wild_color_mutation(parent_color, mutation_range):
    mutated_color = tuple(
        max(64, min(255, parent_color[i] + random.randint(-mutation_range, mutation_range)))
        for i in range(3)
    )
    return mutated_color

# Add a helper function to lerp between two values
def lerp(a, b, t):
    return a + (b - a) * t

class Glob:
    def __init__(self, x, y, z, radius, color, set_id=None, glob_sets=None):
        self.x = x
        self.y = y
        self.z = z
        self.radius = radius
        self.color = color
        self.glob_sets = glob_sets if glob_sets is not None else {}  # set default value
        self.creation_time = pygame.time.get_ticks()
        self.milestone1 = self.color
        self.milestone2 = self._get_next_milestone(self.color)
        self.lerp_t = 0
        self.lerp_speed = 0.0084

        if set_id is None:
            set_id = str(uuid.uuid4())

        self.set_id = set_id

        if self.set_id not in self.glob_sets:
            self.glob_sets[self.set_id] = set()
        self.glob_sets[self.set_id].add(self)

        speed_multiplier = 28.88 / self.radius
        self.vx = (random.uniform(-1, 1) / speed_multiplier) / SPEED_DIVISOR
        self.vy = (random.uniform(-1, 1) / speed_multiplier) / SPEED_DIVISOR
        self.vz = (random.uniform(-1, 1) / speed_multiplier) / SPEED_DIVISOR

        if self.radius == MAX_RADIUS:
            self.num_globs = len(INITIAL_GLOBS)
        else:
            self.num_globs = round(self.radius / (MAX_RADIUS / INITIAL_GLOBS))

        self.split_prob = SPLIT_PROB
        # Calculate the note index based on the y-coordinate
        self.note_index = int(36 * (HEIGHT - self.y) / HEIGHT)

    def _get_next_milestone(self, current_color):
        next_color = []
        for channel in current_color:
            min_val = max(0, channel - 128)
            max_val = min(255, channel + (255 - channel))
            next_channel = random.randint(min_val, max_val)
            next_color.append(next_channel)
        return tuple(next_color)

    def split(self, globs):
        if len(globs) < MAX_NUMBER_GLOBS and random.random() < self.split_prob:
            new_globs = []
            num_new_globs = random.randint(round(2*((self.radius/MAX_RADIUS*0.5)+1)), round(5*((self.radius/MAX_RADIUS*0.5)+1)))
            for _ in range(num_new_globs):
                new_x = self.x + random.uniform(-self.radius, self.radius)
                new_y = self.y + random.uniform(-self.radius, self.radius)
                new_z = self.z + random.uniform(-self.radius, self.radius)
                new_radius = self.radius / num_new_globs

                # Use wild color mutation for offspring
                mutation_range = calculate_mutation_range(globs)
                new_color = wild_color_mutation(self.color, mutation_range)

                new_glob = Glob(new_x, new_y, new_z, new_radius, new_color, self.set_id, self.glob_sets)
                new_glob.split_prob = self.split_prob
                new_globs.append(new_glob)
            return new_globs
        else:

            return None

    def draw(self, screen, bg_color):
        # Calculate the coordinate ratios of the glob's position relative to the room's center
        x_ratio = (self.x - WIDTH / 2) / (WIDTH / 2)
        y_ratio = (self.y - HEIGHT / 2) / (HEIGHT / 2)
        z_ratio = (self.z - DEPTH / 2) / (DEPTH / 2)

        # Calculate the amount by which to push in the glob's coordinate
        distance_from_center = math.sqrt(x_ratio ** 2 + y_ratio ** 2 + z_ratio ** 2)
        if distance_from_center == 0:
            push_in = 0
        else:
            a = 1.5  # Semi-major axis
            b = a / 2  # Semi-minor axis
            c = math.sqrt(a ** 2 - b ** 2)  # Distance from center to foci
            distance_from_focus = math.sqrt((x_ratio * a) ** 2 + (y_ratio * a) ** 2 + (z_ratio * b) ** 2)
            push_in = (distance_from_focus - c) / distance_from_center

        # Transform the glob's position based on the push-in value
        x_transformed = self.x + (WIDTH / 2 - self.x) * push_in
        y_transformed = self.y + (HEIGHT / 2 - self.y) * push_in
        z_transformed = self.z + (DEPTH / 2 - self.z) * push_in

        # Scale the transformed position based on the z-coordinate
        scale_factor = get_scale_factor(z_transformed, DEPTH)
        x_scaled = x_transformed * scale_factor + (1 - scale_factor) * (WIDTH / 2)
        y_scaled = y_transformed * scale_factor + (1 - scale_factor) * (HEIGHT / 2)

        # Calculate the scaled radius and fade color
        scaled_radius = int(self.radius * scale_factor)
        r = int(self.color[0])
        g = int(self.color[1])
        b = int(self.color[2])
        fade_color = (r, g, b)

        # Ensure fade_color is a valid RGB tuple
        fade_color = tuple(max(0, min(c, 255)) for c in fade_color)

        # Draw the glob on the screen
        pygame.draw.circle(screen, fade_color, (int(x_scaled), int(y_scaled)), scaled_radius)

    def update(self, globs, glob_sets):
        global TRANSFER
        removed = False
        # Move glob according to its speed
        self.x += self.vx
        self.y += self.vy
        self.z += self.vz

        # Apply boundary conditions
        self.x %= WIDTH
        self.y %= HEIGHT
        self.z %= DEPTH

        # Update color according to the current milestones
        if self.lerp_t < 1:
            self.color = tuple(int(lerp(self.milestone1[i], self.milestone2[i], self.lerp_t)) for i in range(3))
            self.lerp_t += self.lerp_speed
        else:
            self.milestone1 = self.milestone2
            self.milestone2 = self._get_next_milestone(self.color)
            self.lerp_t = 0

        # Move globs out of sibling set if they are far enough apart
        siblings = [g for g in self.glob_sets[self.set_id] if g != self]
        for sibling in siblings:
            distance = math.sqrt((self.x - sibling.x)**2 + (self.y - sibling.y)**2 + (self.z - sibling.z)**2)
            if distance > 2 * self.radius:
                self.glob_sets[self.set_id].remove(self)
                new_set_id = str(uuid.uuid4())
                self.set_id = new_set_id
                self.glob_sets[new_set_id] = {self}
                break

        # Handle glob collision and color blending
        for other in globs:
            if other != self:
                distance = math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)
                if distance <= self.radius + other.radius:
                    if self.radius > other.radius:
                        larger, smaller = self, other
                    else:
                        larger, smaller = other, self

                    transfer_rate = TRANSFER  # Adjust this value to control the transfer rate
                    transferred_radius = smaller.radius * transfer_rate
                    larger.radius += transferred_radius
                    smaller.radius -= transferred_radius

                    # Color blending
                    larger_area = math.pi * larger.radius**2
                    smaller_area = math.pi * smaller.radius**2
                    total_area = larger_area + smaller_area
                    new_color = tuple(int((larger_area * larger.color[i] + smaller_area * smaller.color[i]) / total_area) for i in range(3))
                    larger.color = new_color

                    # Remove smaller glob if its radius becomes zero
                    if smaller.radius <= 0:
                        globs.remove(smaller)
                        if smaller.set_id in glob_sets and smaller in glob_sets[smaller.set_id]:
                            glob_sets[smaller.set_id].remove(smaller)
                            self.num_globs -= 1 # decrement the num_globs of the parent glob
                        removed = True
                        break

        # Check if the glob should split, outside the loop
        if self.radius > MAX_RADIUS:
            new_globs = self.split(globs)
            if new_globs:
                globs.extend(new_globs)
                if not removed and self in globs:
                    globs.remove(self)
                    self.num_globs -= 1 # decrement the num_globs of the parent glob

def attract_smaller_globs(globs, min_radius):
    force = 0.3/4.6
    for glob1 in globs:
        if glob1.radius < min_radius:
            nearest_larger_glob = None
            nearest_distance = float('inf')
            for glob2 in globs:
                if glob2.radius >= min_radius and glob2 != glob1:
                    distance = math.sqrt((glob1.x - glob2.x) ** 2 + (glob1.y - glob2.y) ** 2 + (glob1.z - glob2.z) ** 2)
                    if distance < nearest_distance:
                        nearest_larger_glob = glob2
                        nearest_distance = distance
            if nearest_larger_glob is not None:
                attraction_force = force * (min_radius / nearest_distance)
                dx = nearest_larger_glob.x - glob1.x
                dy = nearest_larger_glob.y - glob1.y
                dz = nearest_larger_glob.z - glob1.z
                norm = math.sqrt(dx**2 + dy**2 + dz**2)
                glob1.vx += dx / norm * attraction_force
                glob1.vy += dy / norm * attraction_force
                glob1.vz += dz / norm * attraction_force

def get_attraction_force(color1, color2):
    h1, s1, v1 = rgb_to_hsv(*(c / 255 for c in color1))
    h2, s2, v2 = rgb_to_hsv(*(c / 255 for c in color2))

    hue_diff = abs(h1 - h2)
    saturation_diff = abs(s1 - s2)

    attraction_strength = (1 - hue_diff) * (1 - saturation_diff)
    attraction_force = 0.0002 * attraction_strength

    return attraction_force

def get_scale_factor(z, depth):
    return 1 - (z / depth)

def average_glob_hsv(globs):
    if len(globs) == 0:
        return (0, 0, 0)  # default background color if there are no globs

    num_globs = len(globs)
    total_h, total_s, total_v = 0, 0, 0
    for glob in globs:
        h, s, v = rgb_to_hsv(*(c / 255 for c in glob.color))
        total_h += h
        total_s += s
        total_v += v

    avg_h = total_h / num_globs
    avg_s = total_s / num_globs
    avg_v = total_v / num_globs

    return avg_h, avg_s, avg_v

def get_random_color():
    r = random.randint(100, 255)
    g = random.randint(100, 255)
    b = random.randint(100, 255)
    return (r, g, b)

def main():
    global start
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Nia.S & ChatGPT's Lavalamp")
    clock = pygame.time.Clock()

    a, b, c = WIDTH / 2, HEIGHT / 2, DEPTH / 2

    globs = [Glob(*random_point_on_ellipsoid(a, b, c),
                  random.uniform(MIN_RADIUS, MAX_RADIUS),
                  get_random_color(),
                  str(uuid.uuid4())) for _ in range(INITIAL_GLOBS)]

    glob_sets = {i: {glob} for i, glob in enumerate(globs)}

    pygame.mixer.init(frequency=SAMPLE_RATE, size=BIT_DEPTH, channels=NUM_CHANNELS + 1, buffer=AUDIO_BUFFER_SIZE)
    channel_group1 = pygame.mixer.Channel(0)
    channel_group2 = pygame.mixer.Channel(NUM_CHANNELS - 1)

    buffers1 = [generate_next_buffer(globs) for _ in range(3)]
    buffers2 = [generate_next_buffer(globs) for _ in range(3)]

    buffer_index1 = 0
    buffer_index2 = 0
    prev_buffer1 = buffers1[buffer_index1]
    prev_buffer2 = buffers2[buffer_index2]

    running = True
    first_frame = True
    while running:
        if first_frame:
            try:
                avg_h, avg_s, avg_v = average_glob_hsv(globs)
                bg_color = tuple(int(c * 255) for c in hsv_to_rgb(1 - avg_h, 1 - avg_s, 1 - avg_v))
                screen.fill(bg_color)
                last_valid_bg_color = bg_color
            except ValueError:
                if last_valid_bg_color is not None:
                    screen.fill(last_valid_bg_color)
                else:
                    r, g, b = bg_color
                    avg_value = (r + g + b) // 3
                    default_color = (64, 64, 64) if avg_value >= 128 else (255, 255, 255)
                    screen.fill(default_color)
            first_frame = False
        else:
            fg_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            fg_color = (*last_valid_bg_color, 1)
            fg_surf.fill(fg_color)
            screen.blit(fg_surf, (0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    if screen.get_flags() & pygame.FULLSCREEN:
                        pygame.display.set_mode((WIDTH, HEIGHT))
                    else:
                        pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)

        new_globs = []
        sorted_globs = sorted(globs, key=lambda g: g.z, reverse=False)
        attract_smaller_globs(globs, MIN_RADIUS)

        for glob in sorted_globs:
            result = glob.update(globs, glob_sets)
            if result:
                new_globs.extend(result)

            glob.draw(screen, bg_color)

        prev_buffer1 = play_buffer(channel_group1, buffers1[buffer_index1], prev_buffer1)
        buffer_index1 = (buffer_index1 + 1) % 3
        buffers1[buffer_index1] = generate_next_buffer(globs)

        prev_buffer2 = play_buffer(channel_group2, buffers2[buffer_index2], prev_buffer2)
        buffer_index2 = (buffer_index2 + 1) % 3
        buffers2[buffer_index2] = generate_next_buffer(globs)

        # Update the display and tick the clock
        pygame.display.flip()
        clock.tick(FPS)

        # Add new globs to the list
        globs.extend(new_globs)

        # Remove globs that have radius less than 1
        globs = [glob for glob in globs if glob.radius >= 1]

        # Update the num_globs attribute of all globs
        num_globs = len(globs)
        for glob in globs:
            glob.num_globs = num_globs

    pygame.quit()

if __name__ == "__main__":
    main()
