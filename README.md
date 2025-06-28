# 📡 Wireless Communication Design Assistant

> An interactive AI-powered web app that explains and computes core wireless system parameters.  
> Built with **Flask**, **Gemini Pro**, and Python-based calculators.

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Framework](https://img.shields.io/badge/framework-Flask-lightgreen)

---

## 🧠 What This Project Does

This tool helps students and engineers compute and understand various wireless communication scenarios by entering real-world parameters and receiving:

- Accurate calculations
- AI-generated technical explanations (via **Google Gemini 1.5**)
- Instant visual feedback

---

## 📚 Supported Scenarios

| Module No. | Topic                        | Description                                                           |
|------------|-----------------------------|-----------------------------------------------------------------------|
| 1️⃣         | **Voice-band Link**         | Calculates bitrate stages in a digitized voice chain                  |
| 2️⃣         | **OFDM System Design**      | Computes bits per symbol, RB, and spectral efficiency                 |
| 3️⃣         | **Link Budget Analysis**    | Calculates required transmitted and received power                    |
| 4️⃣         | **Cellular Planning**       | Estimates number of cells, cluster size, channels, and traffic Erlang |

Each module includes a friendly form + result breakdown + Gemini-generated explanation.

---

## 🚀 Demo Instructions

### 🧰 Requirements

- Python 3.10+
- Flask
- Google Generative AI API key

### ▶️ Local Setup

```bash
git clone https://github.com/your-username/WirelessProject.git
cd WirelessProject

python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

pip install -r requirements.txt

cp .env.example .env   # then insert your GEMINI_API_KEY
python main.py
