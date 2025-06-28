import os
import math # Added for math functions
import numpy as np
from flask import Flask, render_template, request
import google.generativeai as genai 
from dotenv import load_dotenv 
from calculations import (
    convert_db_to_watt,
    convert_time_to_second,
    convert_distance_to_meters_from_unit,
    compute_max_communication_distance,
    compute_max_cell_area,
    compute_required_cell_count,
    compute_user_traffic_erlang,
    compute_cluster_size_from_sir,
    compute_channels_for_gos,
    compute_carriers_per_cell,
    compute_total_carriers_in_network,
)
# --- Load Environment Variables ---
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = Flask(__name__)
# Set OpenAI API key 


if not GEMINI_API_KEY:
    print("❌ Missing GEMINI_API_KEY from environment.")
    gemini_model = None
else:
    genai.configure(api_key=GEMINI_API_KEY)
    try:
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")  # Or "gemini-pro"
    except Exception as e:
        print("❌ Gemini model init failed:", e)
        gemini_model = None


@app.route("/")
def index():
    return render_template("index.html")

# Route for Question 1: Wireless Communication System
@app.route('/question1', methods=['GET', 'POST'])
def q1():
    inputs = results = explanation = error = None

    if request.method == 'POST':
        try:
            # read inputs
            signal_bandwidth = float(request.form['signal_bandwidth'])
            unit = request.form['signal_bandwidth_unit']
            if unit == 'kHz':
                signal_bandwidth *= 1e3
            elif unit == 'MHz':
                signal_bandwidth *= 1e6

            quantizer_bits = int(request.form['quantizer_bits'])
            source_encoder = float(request.form['source_encoder'])
            channel_encoder = float(request.form['channel_encoder'])
            interleaver_bits = int(request.form['interleaver_bits'])
            overhead_bits = float(request.form['overhead_bits'])
            voice_duration_ms = float(request.form['speech_duration'])

           
            sampling_rate = 2 * signal_bandwidth  # Nyquist
            quantizer_output = sampling_rate * quantizer_bits / 1e3  # Kbps
            source_output = quantizer_output * source_encoder
            channel_output = source_output / channel_encoder
            interleaver_output = channel_output  # No change
            voice_duration_sec = voice_duration_ms / 1000
            channel_coded_bits = channel_output * 1e3 * voice_duration_sec
            total_bits = channel_coded_bits + overhead_bits
            burst_output = total_bits / voice_duration_sec / 1e3

           
            inputs = {
                "Signal_bandwidth_hz":       f"{signal_bandwidth:.3f} Hz",
                "Quantizer_bits":            quantizer_bits,
                "Source_comp_rate":          source_encoder,
                "Channel_code_rate":         channel_encoder,
                "Interleaver_bits":          interleaver_bits,
                "Overhead_bits":             overhead_bits,
                "Voice_segment_ms":          f"{voice_duration_ms:.1f} ms",
            }

            # ----------------  build results -----------------
            total_burst_bits_segment = total_bits        

            results = {
                "Sampling_rate_hz":          f"{sampling_rate:.3f} Hz",
                "Quantizer_output_kbps":     f"{quantizer_output:.3f} Kbps",
                "Source_output_kbps":        f"{source_output:.3f} Kbps",
                "Channel_output_kbps":       f"{channel_output:.3f} Kbps",
                "Interleaver_output_kbps":   f"{interleaver_output:.3f} Kbps",
                "Channel_coded_bits":        f"{channel_coded_bits:.3f} bits",
                "Total_burst_bits_segment":  f"{total_burst_bits_segment:.3f} bits",
                "Burst_output_kbps":         f"{burst_output:.3f} Kbps",
            }

            # --------------- story prompt --------------------
            story_prompt = (
                "Picture a high-tech factory where every machine handles the voice signal on its way to the airwaves.\n\n"
                f"🏁 **Arrival:** A {inputs['Signal_bandwidth_hz']} signal rolls in and heads straight to the "
                f"{inputs['Quantizer_bits']}-bit “Cut-and-Slice” station (quantizer).\n"
                f"⚙️ **Source Compression:** The bits are squeezed by a {inputs['Source_comp_rate']}× source encoder.\n"
                f"🛡️ **Channel Coding:** An error-armor forge works at a {inputs['Channel_code_rate']} code rate.\n"
                f"🔄 **Interleaver & Overhead:** {inputs['Interleaver_bits']} interleaver bits and "
                f"{inputs['Overhead_bits']} overhead bits are stitched in.\n"
                f"⏱️ Each voice segment lasts {inputs['Voice_segment_ms']}.\n\n"
                "### Production stats for one segment\n"
                f"• Sampling speed → {results['Sampling_rate_hz']}\n"
                f"• Quantizer output → {results['Quantizer_output_kbps']}\n"
                f"• After source encoding → {results['Source_output_kbps']}\n"
                f"• After channel coding → {results['Channel_output_kbps']}\n"
                f"• After interleaving (rate unchanged) → {results['Interleaver_output_kbps']}\n"
                f"• Channel-coded bits per segment → {results['Channel_coded_bits']}\n"
                f"• **Total bits per segment** (incl. overhead) → {results['Total_burst_bits_segment']}\n"
                f"• Final burst transmission rate → {results['Burst_output_kbps']}\n\n"
                "🎯 **Task:** In ~160 words, narrate the signal’s journey through each stage, naturally weaving in every figure above."
                
            )
            # ---------- Ask Gemini for the explanation ----------
            explanation = None
            if gemini_model:                       # model is configured
                try:
                    gemini_response = gemini_model.generate_content(story_prompt)
                    explanation = gemini_response.text
                except Exception as e:             # any API problem
                    error = f"Gemini API error: {e}"
                    explanation = "AI explanation could not be generated."
            else:
                explanation = "AI explanation feature is not configured."

        except ValueError as ve:
            error = f"Input validation error: {ve}"
            print(f"Validation Error in Q1: {ve}")
        except Exception as err:
            error = f"An unexpected error occurred during calculation: {err}"
            print(f"Unexpected Error in Q1: {err}")

    # Render the template, passing inputs, results, explanation, and error
    return render_template('question1.html', inputs=inputs, results=results, explanation=explanation, error=error)


# Route: Question 2  – OFDM System
@app.route("/question2", methods=["GET", "POST"])
def qn2():
    inputs = results = explanation = error = None

    if request.method == "POST":
        try:
            # Inputs 
            bandwidth_rb = float(request.form["bandwidth_rb"])
            unit_bw = request.form["bandwidth_rb_unit"]  # Hz / kHz
            if unit_bw == "Hz":
                bandwidth_rb /= 1e3  # convert to kHz

            subcarrier_spacing = float(request.form["subcarrier_spacing"])
            unit_sc = request.form["subcarrier_spacing_unit"]  # Hz / kHz
            if unit_sc == "Hz":
                subcarrier_spacing /= 1e3

            num_ofdm_symbols = int(request.form["num_ofdm_symbols"])
            duration_rb = float(request.form["duration_rb"])
            unit_dur = request.form["duration_rb_unit"]  # ms / seconds
            if unit_dur == "seconds":
                duration_rb *= 1e3  # to ms

            modulation_order = int(request.form["modulation_order"])
            num_parallel_rb = int(request.form["num_parallel_rb"])

            # Calculations 
            num_subcarriers = bandwidth_rb / subcarrier_spacing  # per RB
            bits_per_re = np.log2(modulation_order)
            bits_per_symbol = bits_per_re * num_subcarriers
            bits_per_rb = bits_per_symbol * num_ofdm_symbols
            max_tx_rate = (bits_per_rb * num_parallel_rb) / duration_rb  # kbits/ms == Mbit/s

            total_bw_hz = bandwidth_rb * 1e3 * num_parallel_rb
            spectral_eff = (max_tx_rate * 1e3) / total_bw_hz  # bps/Hz

            # Display Inputs
            inputs = {
                "Bandwidth_per_rb_khz":       f"{bandwidth_rb:.3f} kHz",
                "Subcarrier_spacing_khz":     f"{subcarrier_spacing:.3f} kHz",
                "Num_ofdm_symbols":           num_ofdm_symbols,
                "Duration_per_rb_ms":         f"{duration_rb:.3f} ms",
                "Modulation_order":           modulation_order,
                "Parallel_resource_blocks":   num_parallel_rb,
            }

            results = {
                "Bits_per_resource_element":  f"{bits_per_re:.3f} bits",
                "Bits_per_ofdm_symbol":       f"{bits_per_symbol:.3f} bits",
                "Bits_per_resource_block":    f"{bits_per_rb:.3f} bits",
                "Max_transmission_rate_bps":  f"{max_tx_rate * 1e3:.3f} bit/s",
                "Spectral_efficiency_bpshz":  f"{spectral_eff:.3f} bps/Hz",
            }
            story_prompt = (
                "The design of the OFDM system began with a few key parameters: each resource block had a bandwidth of "
                f"{inputs['Bandwidth_per_rb_khz']}, with a subcarrier spacing of {inputs['Subcarrier_spacing_khz']}. "
                f"{inputs['Num_ofdm_symbols']} OFDM symbols were fitted into each block, which spanned a duration of "
                f"{inputs['Duration_per_rb_ms']}. A modulation order of {inputs['Modulation_order']} was chosen, enabling each subcarrier to carry multiple bits. "
                f"To enhance capacity, the system transmitted data over {inputs['Parallel_resource_blocks']} resource blocks in parallel.\n\n"

                "Based on these specifications, the number of subcarriers per block was calculated, and the number of bits per subcarrier was derived from the modulation scheme. "
                "Multiplying those gave the total bits per OFDM symbol, which was further scaled by the number of symbols to compute the data per block. "
                f"This resulted in {results['Bits_per_resource_block']} per block. Parallel transmission led to an overall peak rate of {results['Max_transmission_rate_bps']}, "
                f"and normalizing this with bandwidth yielded a spectral efficiency of {results['Spectral_efficiency_bpshz']}.\n\n"

                "Write this as a clear ~160-word technical narrative that walks the reader from the input assumptions to the final throughput and efficiency results."
            )



            # ---------- Ask Gemini for the explanation ----------
            explanation = None
            if gemini_model:                       # model is configured
                try:
                    gemini_response = gemini_model.generate_content(story_prompt)
                    explanation = gemini_response.text
                except Exception as e:             # any API problem
                    error = f"Gemini API error: {e}"
                    explanation = "AI explanation could not be generated."
            else:
                explanation = "AI explanation feature is not configured."


        except ValueError as ve:
            error = f"Input validation error: {ve}"
            print(f"Validation Error in Q2: {ve}")
        except Exception as err:
            error = f"An unexpected error occurred during calculation: {err}"
            print(f"Unexpected Error in Q2: {err}")

    return render_template("Question2.html", inputs=inputs, results=results, explanation=explanation, error=error)


# Route: Question 3  – Link Budget

@app.route("/question3", methods=["GET", "POST"])
def q3():
    inputs = results = explanation = error = None

    if request.method == "POST":
        try:
            # read inputs
            path_loss = float(request.form["L_p"])
            transmit_gain = float(request.form["G_t"])
            receive_gain = float(request.form["G_r"])
            data_rate = float(request.form["R"])
            rate_unit = request.form.get("R_unit", "bps")
            if rate_unit == "kbps":
                data_rate *= 1000

            line_loss = float(request.form["L_o"])
            other_losses = float(request.form["L_f"])
            fade_margin = float(request.form["F_margin"])
            tx_amp_gain = float(request.form["A_t"])
            rx_amp_gain = float(request.form["A_r"])
            noise_figure = float(request.form["N_f"])
            temperature_kelvin = float(request.form["T"])
            eb_n0 = float(request.form["SNR_per_bit"])
            link_margin = float(request.form["link_margin"])

            # calculations
            K_db = -228.6

            Pr_db = (
                link_margin +
                10 * math.log10(data_rate) +
                10 * math.log10(temperature_kelvin) +
                noise_figure +
                K_db +
                eb_n0
            )

            Pt_db = (
                Pr_db +
                path_loss +
                line_loss +
                other_losses +
                fade_margin -
                transmit_gain -
                receive_gain -
                tx_amp_gain -
                rx_amp_gain
            )

            
            Pr_watt = 10 ** (Pr_db / 10)
            Pt_watt = 10 ** (Pt_db / 10)

            Pr_watt_str = f"{Pr_watt:.5f}"
            Pt_watt_str = f"{Pt_watt:.5f}"

            #  Inputs and Results
            inputs = {
                "Path_loss_db":               f"{path_loss:.2f} dB",
                "Tx_antenna_gain_db":         f"{transmit_gain:.2f} dB",
                "Rx_antenna_gain_db":         f"{receive_gain:.2f} dB",
                "Data_rate_bps":              f"{data_rate:.2f} bps",
                "Other_losses_db":            f"{line_loss:.2f} dB",
                "Feed_line_loss_db":          f"{other_losses:.2f} dB",
                "Fade_margin_db":             f"{fade_margin:.2f} dB",
                "Tx_amplifier_gain_db":       f"{tx_amp_gain:.2f} dB",
                "Rx_amplifier_gain_db":       f"{rx_amp_gain:.2f} dB",
                "Noise_figure_db":            f"{noise_figure:.2f} dB",
                "Noise_temperature_k":        f"{temperature_kelvin:.2f} K",
                "Eb_n0_db":                   f"{eb_n0:.2f} dB",
                "Link_margin_db":             f"{link_margin:.2f} dB",
            }

            results = {
                "Received_power_db":       f"{Pr_db:.5f} dB",
                "Received_power_watt":     f"{Pr_watt_str} W",
                "Transmitted_power_db":    f"{Pt_db:.5f} dB",
                "Transmitted_power_watt":  f"{Pt_watt_str} W",
            }
            story_prompt = (
                "A complete link budget accounts for all gains and losses along the communication path. "
                f"The analysis begins with a path loss of {inputs['Path_loss_db']} dB and a required data rate of {inputs['Data_rate_bps']}. "
                f"To maintain link reliability, the design includes an Eb/N0 of {inputs['Eb_n0_db']} dB, a link margin of {inputs['Link_margin_db']} dB, "
                f"a receiver noise figure of {inputs['Noise_figure_db']} dB, and a system noise temperature of {inputs['Noise_temperature_k']} K.\n\n"

                f"On the hardware side, the transmit and receive antennas provide gains of {inputs['Tx_antenna_gain_db']} dB and "
                f"{inputs['Rx_antenna_gain_db']} dB respectively. Amplifiers contribute {inputs['Tx_amplifier_gain_db']} dB on transmit and "
                f"{inputs['Rx_amplifier_gain_db']} dB on receive. Fixed losses include {inputs['Feed_line_loss_db']} dB from feed lines and "
                f"{inputs['Other_losses_db']} dB from other sources. A fade margin of {inputs['Fade_margin_db']} dB ensures protection against deep fades.\n\n"

                f"After applying all these factors, the system yields a minimum received power of {results['Received_power_db']} dB "
                f"({results['Received_power_watt']}), requiring a transmit power of {results['Transmitted_power_db']} dB "
                f"({results['Transmitted_power_watt']}). These values confirm the link's ability to meet performance targets under worst-case conditions.\n\n"

                                "Write this as a clear ~160-word technical narrative that walks the reader "
                "from the input assumptions to the final carrier count."
              
            )


            # ---------- Ask Gemini for the explanation ----------
            explanation = None
            if gemini_model:                       # model is configured
                try:
                    gemini_response = gemini_model.generate_content(story_prompt)
                    explanation = gemini_response.text
                except Exception as e:             # any API problem
                    error = f"Gemini API error: {e}"
                    explanation = "AI explanation could not be generated."
            else:
                explanation = "AI explanation feature is not configured."


        except ValueError as ve:
            error = f"Input validation error: {ve}"
            print(f"Validation Error in Q3: {ve}")
        except Exception as err:
            error = f"An unexpected error occurred during calculation: {err}"
            print(f"Unexpected Error in Q3: {err}")


    return render_template("question3.html", inputs=inputs, results=results, explanation=explanation, error=error)


# Route: Question 4  – Cellular System Design

@app.route("/question4", methods=["GET", "POST"])
def q4():
    inputs = results = explanation = error = None

    if request.method == "POST":
        try:
            # inputs
            area_units = float(request.form["total_area"])
            users = int(request.form["max_num_users"])

            avg_call_duration = float(request.form["avg_call_duration"])
            dur_unit = request.form["avg_call_duration_unit"]
            avg_call_sec = convert_time_to_second(avg_call_duration, dur_unit)

            avg_call_rate = float(request.form["avg_call_rate_per_user"])
            GOS = float(request.form["GOS"])

            SIR = float(request.form["SIR"])
            SIR_unit = request.form["SIR_unit"]
            if SIR_unit == "dB":
                SIR = 10 ** (SIR / 10)

            P0 = float(request.form["P0"])
            P0_unit = request.form["P0_unit"]
            if P0_unit == "dB":
                P0 = convert_db_to_watt(P0)

            receiver_sens = float(request.form["receiver_sensitivity"])
            rs_unit = request.form["receiver_sensitivity_unit"]
            if rs_unit == "dB":
                receiver_sens = convert_db_to_watt(receiver_sens)

            d0 = float(request.form["d0"])
            d0_unit = request.form["d0_unit"]
            d0_m = convert_distance_to_meters_from_unit(d0, d0_unit)

            path_loss_exp = float(request.form["path_loss_exponent"])
            time_slots = int(request.form["time_slots_per_carrier"])

            # calculations
            max_distance = compute_max_communication_distance(P0, receiver_sens, d0_m, path_loss_exp)
            max_cell_size = compute_max_cell_area(max_distance)
            total_cells = compute_required_cell_count(area_units, max_cell_size)
            traffic_per_user = compute_user_traffic_erlang(avg_call_sec, avg_call_rate)
            total_traffic = traffic_per_user * users
            traffic_per_cell = total_traffic / total_cells
            cluster_size = compute_cluster_size_from_sir(SIR, path_loss_exp)
            channels_req = compute_channels_for_gos(traffic_per_cell, GOS)
            carriers_per_cell = compute_carriers_per_cell(channels_req, time_slots)
            carriers_in_system = compute_total_carriers_in_network(carriers_per_cell, cluster_size)

            #  Inputs and Results
            inputs = {
                "Total_area_sq_units":         f"{area_units:.1f} sq units",
                "Num_users":                   users,
                "Avg_call_duration":           f"{avg_call_duration:.1f} {dur_unit}",
                "Avg_call_rate":               f"{avg_call_rate:.1f}",
                "Grade_of_service":            f"{GOS:.2f}",
                "Sir_linear":                  f"{SIR:.3f}",
                "Reference_power_p0_watt":     f"{P0:.5f} W",
                "Receiver_sensitivity_watt":   f"{receiver_sens:.5f} W",
                "Reference_distance":          f"{d0:.1f} {d0_unit}",
                "Path_loss_exponent":          f"{path_loss_exp:.1f}",
                "Time_slots_per_carrier":      time_slots,
            }

            results = {
                "Max_reliable_distance_m":        f"{max_distance:.5f} meters",
                "Mx_cell_area_sq_units":         f"{max_cell_size:.5f} sq units",
                "Total_num_cells":                int(total_cells),
                "Traffic_per_user_cps":           f"{traffic_per_user:.5f} calls/second",
                "Total_traffic_cps":              f"{total_traffic:.5f} calls/second",
                "Traffic_per_cell_cps":           f"{traffic_per_cell:.5f} calls/second",
                "Cluster_size_n":                 cluster_size,
                "Num_channels_required":          channels_req,
                "Carriers_per_cell":              carriers_per_cell,
                "Total_carriers_in_system":       carriers_in_system,
            }
            # Natural language story-style prompt 
            story_prompt = (
                "To design a new cellular network covering a total area of "
                f"{inputs['Total_area_sq_units']}, engineers planned for {inputs['Num_users']} "
                "potential users. User behaviour assumed an average call duration of "
                f"{inputs['Avg_call_duration']} and an average call rate of "
                f"{inputs['Avg_call_rate']}. The target Grade of Service (GOS) was "
                f"{inputs['Grade_of_service']}.\n\n"

                "Radio propagation was characterised by a path-loss exponent of "
                f"{inputs['Path_loss_exponent']} from a reference distance of "
                f"{inputs['Reference_distance']}. The link budget used a reference transmit "
                f"power of {inputs['Reference_power_p0_watt']} and a receiver sensitivity of "
                f"{inputs['Receiver_sensitivity_watt']}. To control interference, a minimum "
                f"SIR of {inputs['Sir_linear']} (linear) was required. Each carrier supports "
                f"{inputs['Time_slots_per_carrier']} time-slots.\n\n"

                "With those parameters, the maximum reliable cell radius gave a service "
                f"distance of {results['Max_reliable_distance_m']} and a cell area of "
                f"{results['Mx_cell_area_sq_units']}. Covering the full region therefore "
                f"demands {results['Total_num_cells']} cells. Traffic engineering shows each "
                f"user generates {results['Traffic_per_user_cps']} Erlangs, totalling "
                f"{results['Total_traffic_cps']} Erlangs system-wide, or "
                f"{results['Traffic_per_cell_cps']} Erlangs per cell. Meeting the GOS requires "
                f"{results['Num_channels_required']} channels per cell. Given the required "
                f"SIR, the optimal reuse cluster size is N = {results['Cluster_size_n']}, so "
                f"each cell needs {results['Carriers_per_cell']} carriers and the whole "
                f"network requires {results['Total_carriers_in_system']} carriers.\n\n"

                "Write this as a clear ~160-word technical narrative that walks the reader "
                "from the input assumptions to the final carrier count."
               
            )

            # ---------- Ask Gemini for the explanation ----------
            explanation = None
            if gemini_model:                       # model is configured
                try:
                    gemini_response = gemini_model.generate_content(story_prompt)
                    explanation = gemini_response.text
                except Exception as e:             # any API problem
                    error = f"Gemini API error: {e}"
                    explanation = "AI explanation could not be generated."
            else:
                explanation = "AI explanation feature is not configured."


        except ValueError as ve:
            error = f"Input validation error: {ve}"
            print(f"Validation Error in Q4: {ve}")
        except Exception as err:
            error = f"An unexpected error occurred during calculation: {err}"
            print(f"Unexpected Error in Q4: {err}")


    return render_template("Question4.html", inputs=inputs, results=results, explanation=explanation, error=error)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Default to 10000 if PORT not set
    app.run(host="0.0.0.0", port=port, debug=True)

