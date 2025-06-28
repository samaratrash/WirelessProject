# calculations.py
import math
import numpy as np

# Predefined list of valid cluster sizes in cellular systems
CLUSTER_SIZES = [1, 3, 4, 7, 9, 12, 13, 16, 19, 21, 28]

# Converts power from dB to watts
def convert_db_to_watt(db_value: float) -> float:
    return 10 ** (db_value / 10)

# Converts time to seconds based on the provided unit
def convert_time_to_second(duration: float, unit: str) -> float:
    mapping = {"seconds": 1, "minutes": 60, "hours": 3600}
    return duration * mapping.get(unit, 1)

# Converts distance to meters based on the unit ('km' or 'm')
def convert_distance_to_meters_from_unit(distance: float, unit: str) -> float:
    return distance * 1_000 if unit == "km" else distance

# Calculates the maximum communication range using path loss model
def compute_max_communication_distance(P0, receiver_sens, d0, path_loss_exp):
    return d0 * (P0 / receiver_sens) ** (1 / path_loss_exp)

# Computes the maximum area a single hexagonal cell can cover
def compute_max_cell_area(max_distance: float) -> float:
    return 3 * math.sqrt(3) / 2 * (max_distance ** 2)

# Computes how many cells are needed to cover a total area
def compute_required_cell_count(total_area: float, max_cell_size: float) -> int:
    return int(np.ceil(total_area / max_cell_size))

# Calculates average traffic per user in Erlangs
def compute_user_traffic_erlang(avg_call_sec: float, call_rate: float) -> float:
    return (avg_call_sec * call_rate) / 86400

# Determines minimum cluster size to meet SIR requirement
def compute_cluster_size_from_sir(SIR_linear: float, path_loss_exp: float) -> int:
    x = ((SIR_linear * 6) ** (2 / path_loss_exp)) / 3
    for N in CLUSTER_SIZES:
        if N >= x:
            return N
    return CLUSTER_SIZES[-1]

# Erlang B formula to calculate blocking probability
def erlang_b_formula(channels: int, traffic: float) -> float:
    inv_b = 1.0
    for i in range(1, channels + 1):
        inv_b = 1 + (i / traffic) * inv_b
    return 1 / inv_b

# Finds the minimum number of channels required to meet GoS (Grade of Service)
def compute_channels_for_gos(traffic_per_cell: float, GOS: float) -> int:
    erlang_traffic = traffic_per_cell
    for c in range(1, 100):
        if erlang_b_formula(c, erlang_traffic) <= GOS:
            return c
    return 100

# Calculates how many carriers are needed in one cell
def compute_carriers_per_cell(channels: int, time_slots: int) -> float:
    return math.ceil(channels / time_slots)

# Calculates the total number of carriers in the entire network
def compute_total_carriers_in_network(carriers_per_cell: float, cluster_size: int) -> float:
    return carriers_per_cell * cluster_size
