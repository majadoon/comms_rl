from scipy import constants


def clip(value, min_value, max_value):
    return max(min(value, max_value), min_value)


def calculate_thermal_noise(bw_mhz):
    t0_kelvin = 290
    return constants.Boltzmann * t0_kelvin * bw_mhz * 1E6 * 1000
