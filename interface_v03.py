"""
╔══════════════════════════════════════════════════════════════════╗
║           BCI-IM Data Collection Suite v2.0                      ║
║           OpenBCI Cyton · Motor Imagery & Movement               ║
║                                                                  ║
║   Protocols: Graz B (Motor Imagery) · Movement Protocol (3 Class)║
║   Board: OpenBCI Cyton 8-ch (BrainFlow ID 0)                     ║
║   Channels: C3, Cz, C4, P3, P4, F3, F4, Pz                       ║
╚══════════════════════════════════════════════════════════════════╝

Dependecies:
    pip install brainflow numpy pandas

"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import os
import json
import math
import queue
from datetime import datetime

import numpy as np
import pandas as pd
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes
from scipy.signal import butter, sosfiltfilt

# Windows Only
try:
    import winsound
    HAS_SOUND = True
except ImportError:
    HAS_SOUND = False

# Configs

VERSION = "2.0"

# Mapping
CHANNEL_NAMES = ["C3", "Cz", "C4", "P3", "P4", "F3", "F4", "Pz"]
CHANNEL_COLORS = [
    "#4FC3F7", "#81C784", "#FF8A65", "#BA68C8",
    "#FFD54F", "#4DD0E1", "#F48FB1", "#A1887F"
]

# Event Markers
MARKERS = {
    "session_start":    100,
    "session_end":      200,
    "block_start":       90,
    "block_end":         91,
    "fixation":           1,
    "beep":               2,
    "rest_start":        50,
    "rest_end":          51,
    "pause_start":       99,
    "pause_end":         98,
    # Graz B - Motor Imagery
    "mi_left":           10,
    "mi_right":          11,
    # Movement Protocol
    "mov_hand":          20,
    "mov_elbow":         21,
    "mov_shoulder":      22,
}

# Protocol: Graz B
GRAZ_B_DEFAULTS = {
    "fixation_s":   2.0,
    "beep_s":       1.0,
    "imagery_s":    5.0,
    "rest_s":       3.0,
    "pause_s":      5.0,
    "trials_per_block": 10,
    "blocks":       2,   # 2 blocos (1 esquerda + 1 direita = 1 run)
}

# Protocol: 3 class MI
MOVEMENT_DEFAULTS = {
    "fixation_s":   2.0,
    "beep_s":       1.0,
    "movement_s":  10.0,
    "rest_s":       3.0,
    "pause_s":      5.0,
    "trials_per_block": 10,
}


# Frequencies
FREQ_BANDS = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta":  (13, 30),
    "Gamma": (30, 45),
}

BAND_COLORS = {
    "Delta": "#7E57C2",
    "Theta": "#26A69A",
    "Alpha": "#42A5F5",
    "Beta":  "#FFA726",
    "Gamma": "#EF5350",
}

# Open BCI conversion factor
# Cyton ADS1299: Vref=4.5V, 24-bit ADC,
CYTON_SCALE_UV = (4.5 / (2**23 - 1) / 24) * 1e6  # ≈ 0.02235 µV/count


# Sounds

def play_beep(freq=440, duration_ms=500):
    if HAS_SOUND:
        threading.Thread(target=winsound.Beep, args=(freq, duration_ms), daemon=True).start()

def play_success():
    if HAS_SOUND:
        def _play():
            winsound.Beep(523, 150)
            winsound.Beep(659, 150)
            winsound.Beep(784, 200)
        threading.Thread(target=_play, daemon=True).start()

def play_error():
    if HAS_SOUND:
        threading.Thread(target=lambda: winsound.Beep(200, 400), daemon=True).start()



THEME = {
    "bg":           "#0F1318",
    "surface":      "#1A1F28",
    "surface2":     "#242A36",
    "border":       "#2E3648",
    "text":         "#D4DAE6",
    "text_muted":   "#6B7A94",
    "accent":       "#4FC3F7",
    "accent_dim":   "#1A3A52",
    "green":        "#66BB6A",
    "green_dim":    "#1A3D1C",
    "red":          "#EF5350",
    "red_dim":      "#3D1A1A",
    "yellow":       "#FFD54F",
    "orange":       "#FFA726",
    "purple":       "#B39DDB",
}


# BCI APP

class BCIApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(f"BCI-IM Collection Suite v{VERSION}")
        self.root.configure(bg=THEME["bg"])
        self.root.geometry("820x780")
        self.root.minsize(780, 700)
        self.root.resizable(True, True)

        # Estado
        self.board = None
        self.is_connected = False
        self.is_collecting = False
        self.collection_thread = None
        self.stop_event = threading.Event()
        self.data_queue = queue.Queue()
        self.all_data = []

        # Variáveis de controle
        self.var_port = tk.StringVar(value="COM8")
        self.var_name = tk.StringVar(value="")
        self.var_session = tk.StringVar(value="1")
        self.var_save_path = tk.StringVar(value=DEFAULT_SAVE_PATH)
        self.var_protocol = tk.StringVar(value="graz_b")
        self.var_synthetic = tk.BooleanVar(value=False)
        self.var_show_power = tk.BooleanVar(value=True)
        self.var_show_concentration = tk.BooleanVar(value=True)
        self.var_show_signal_quality = tk.BooleanVar(value=True)
        # Graz B params
        self.var_graz_trials = tk.IntVar(value=GRAZ_B_DEFAULTS["trials_per_block"])
        self.var_graz_imagery_s = tk.DoubleVar(value=GRAZ_B_DEFAULTS["imagery_s"])
        self.var_graz_rest_s = tk.DoubleVar(value=GRAZ_B_DEFAULTS["rest_s"])
        self.var_graz_pause_s = tk.DoubleVar(value=GRAZ_B_DEFAULTS["pause_s"])
        # Movement params
        self.var_mov_trials = tk.IntVar(value=MOVEMENT_DEFAULTS["trials_per_block"])
        self.var_mov_duration_s = tk.DoubleVar(value=MOVEMENT_DEFAULTS["movement_s"])
        self.var_mov_rest_s = tk.DoubleVar(value=MOVEMENT_DEFAULTS["rest_s"])

        self._build_gui()

    # GUI BUILDER

    def _build_gui(self):

        # Scrollable frame
        main_canvas = tk.Canvas(self.root, bg=THEME["bg"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=main_canvas.yview)
        self.main_frame = tk.Frame(main_canvas, bg=THEME["bg"])

        self.main_frame.bind("<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all")))
        main_canvas.create_window((0, 0), window=self.main_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=scrollbar.set)

        main_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Bind mousewheel
        def _on_mousewheel(event):
            main_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        main_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Padding container
        container = tk.Frame(self.main_frame, bg=THEME["bg"])
        container.pack(fill="x", padx=20, pady=10)

        # ── Header ──
        self._build_header(container)
        # ── Hardware ──
        self._build_hardware_section(container)
        # ── Subject Info ──
        self._build_subject_section(container)
        # ── Protocol Selection ──
        self._build_protocol_section(container)
        # ── Visualization Options ──
        self._build_viz_options(container)
        # ── Action Buttons ──
        self._build_actions(container)
        # ── Status Bar ──
        self._build_status_bar()

    def _make_section(self, parent, title):
        frame = tk.Frame(parent, bg=THEME["surface"], highlightbackground=THEME["border"],
                         highlightthickness=1, padx=16, pady=12)
        frame.pack(fill="x", pady=(0, 10))

        lbl = tk.Label(frame, text=title, font=("Segoe UI", 10, "bold"),
                       fg=THEME["accent"], bg=THEME["surface"])
        lbl.pack(anchor="w", pady=(0, 8))
        return frame

    def _make_label(self, parent, text, **kw):
        defaults = {"font": ("Segoe UI", 9), "fg": THEME["text"], "bg": THEME["surface"]}
        defaults.update(kw)
        return tk.Label(parent, text=text, **defaults)

    def _make_entry(self, parent, textvariable, width=20):
        e = tk.Entry(parent, textvariable=textvariable, width=width,
                     font=("Segoe UI", 9), bg=THEME["surface2"],
                     fg=THEME["text"], insertbackground=THEME["text"],
                     relief="flat", highlightthickness=1,
                     highlightbackground=THEME["border"],
                     highlightcolor=THEME["accent"])
        return e

    def _make_button(self, parent, text, command, style="normal"):
        colors = {
            "normal":  (THEME["surface2"], THEME["text"], THEME["accent"]),
            "accent":  (THEME["accent_dim"], THEME["accent"], THEME["accent"]),
            "green":   (THEME["green_dim"], THEME["green"], THEME["green"]),
            "red":     (THEME["red_dim"], THEME["red"], THEME["red"]),
        }
        bg, fg, border_c = colors.get(style, colors["normal"])
        btn = tk.Button(parent, text=text, command=command,
                        font=("Segoe UI", 9, "bold"), bg=bg, fg=fg,
                        activebackground=border_c, activeforeground="#FFFFFF",
                        relief="flat", cursor="hand2", padx=14, pady=6,
                        highlightthickness=1, highlightbackground=border_c)
        return btn

    def _build_header(self, parent):
        hdr = tk.Frame(parent, bg=THEME["bg"])
        hdr.pack(fill="x", pady=(5, 12))

        tk.Label(hdr, text="⬡", font=("Segoe UI", 22), fg=THEME["accent"],
                 bg=THEME["bg"]).pack(side="left", padx=(0, 10))

        title_frame = tk.Frame(hdr, bg=THEME["bg"])
        title_frame.pack(side="left")
        tk.Label(title_frame, text="BCI-IM Collection Suite",
                 font=("Segoe UI", 16, "bold"), fg=THEME["text"],
                 bg=THEME["bg"]).pack(anchor="w")
        tk.Label(title_frame, text=f"OpenBCI Cyton · Motor Imagery & Movement · v{VERSION}",
                 font=("Segoe UI", 8), fg=THEME["text_muted"],
                 bg=THEME["bg"]).pack(anchor="w")

    def _build_hardware_section(self, parent):
        sec = self._make_section(parent, "⚙  HARDWARE")

        row1 = tk.Frame(sec, bg=THEME["surface"])
        row1.pack(fill="x", pady=(0, 6))

        self._make_label(row1, "Porta Serial:").pack(side="left")
        self._make_entry(row1, self.var_port, width=10).pack(side="left", padx=(6, 12))

        self.chk_synthetic = tk.Checkbutton(
            row1, text="Modo Sintético (sem placa)", variable=self.var_synthetic,
            font=("Segoe UI", 8), fg=THEME["text_muted"], bg=THEME["surface"],
            selectcolor=THEME["surface2"], activebackground=THEME["surface"],
            activeforeground=THEME["text_muted"])
        self.chk_synthetic.pack(side="left", padx=(0, 12))

        self.btn_connect = self._make_button(row1, "Conectar", self._toggle_connection, "accent")
        self.btn_connect.pack(side="left", padx=(0, 6))

        self.btn_sanity = self._make_button(row1, "Sanity Check", self._sanity_check, "normal")
        self.btn_sanity.pack(side="left")

        # Status indicator
        row2 = tk.Frame(sec, bg=THEME["surface"])
        row2.pack(fill="x")

        self.hw_status_dot = tk.Canvas(row2, width=10, height=10,
                                        bg=THEME["surface"], highlightthickness=0)
        self.hw_status_dot.pack(side="left", padx=(0, 6))
        self.hw_status_dot.create_oval(1, 1, 9, 9, fill=THEME["red"], outline="")

        self.hw_status_label = self._make_label(row2, "Desconectado",
                                                 fg=THEME["text_muted"])
        self.hw_status_label.pack(side="left")

    def _build_subject_section(self, parent):
        sec = self._make_section(parent, "SUJEITO")

        row1 = tk.Frame(sec, bg=THEME["surface"])
        row1.pack(fill="x", pady=(0, 6))

        self._make_label(row1, "Nome:").pack(side="left")
        self._make_entry(row1, self.var_name, width=22).pack(side="left", padx=(6, 16))
        self._make_label(row1, "Sessão:").pack(side="left")
        self._make_entry(row1, self.var_session, width=5).pack(side="left", padx=(6, 0))

        row2 = tk.Frame(sec, bg=THEME["surface"])
        row2.pack(fill="x")

        self._make_label(row2, "Salvar em:").pack(side="left")
        self._make_entry(row2, self.var_save_path, width=48).pack(side="left", padx=(6, 6))
        self._make_button(row2, "...", self._browse_path).pack(side="left")

    def _build_protocol_section(self, parent):
        sec = self._make_section(parent, "PROTOCOLO")

        # Radio buttons
        radio_frame = tk.Frame(sec, bg=THEME["surface"])
        radio_frame.pack(fill="x", pady=(0, 8))

        for val, label in [("graz_b", "Graz B — Imaginação Motora (mão esquerda vs direita)"),
                           ("movement", "Movement — 3 tipos de movimento (10s cada)")]:
            rb = tk.Radiobutton(radio_frame, text=label, variable=self.var_protocol,
                                value=val, font=("Segoe UI", 9), fg=THEME["text"],
                                bg=THEME["surface"], selectcolor=THEME["surface2"],
                                activebackground=THEME["surface"],
                                activeforeground=THEME["accent"],
                                command=self._update_protocol_params)
            rb.pack(anchor="w", pady=1)

        # Separator
        tk.Frame(sec, bg=THEME["border"], height=1).pack(fill="x", pady=8)

        # Protocol parameters frame
        self.params_frame = tk.Frame(sec, bg=THEME["surface"])
        self.params_frame.pack(fill="x")

        self._update_protocol_params()

    def _update_protocol_params(self):
        for w in self.params_frame.winfo_children():
            w.destroy()

        protocol = self.var_protocol.get()

        if protocol == "graz_b":
            self._build_graz_params(self.params_frame)
        else:
            self._build_movement_params(self.params_frame)

    def _param_row(self, parent, label, variable, unit=""):
        row = tk.Frame(parent, bg=THEME["surface"])
        row.pack(fill="x", pady=2)
        self._make_label(row, label, font=("Segoe UI", 8)).pack(side="left")
        e = self._make_entry(row, variable, width=6)
        e.pack(side="left", padx=(6, 4))
        if unit:
            self._make_label(row, unit, fg=THEME["text_muted"],
                             font=("Segoe UI", 8)).pack(side="left")

    def _build_graz_params(self, parent):
        tk.Label(parent, text="Parâmetros Graz B", font=("Segoe UI", 9, "bold"),
                 fg=THEME["yellow"], bg=THEME["surface"]).pack(anchor="w", pady=(0, 6))

        grid = tk.Frame(parent, bg=THEME["surface"])
        grid.pack(fill="x")

        left = tk.Frame(grid, bg=THEME["surface"])
        left.pack(side="left", fill="x", expand=True)
        right = tk.Frame(grid, bg=THEME["surface"])
        right.pack(side="left", fill="x", expand=True)

        self._param_row(left, "Trials por bloco:", self.var_graz_trials)
        self._param_row(left, "Imaginação:", self.var_graz_imagery_s, "s")
        self._param_row(right, "Repouso:", self.var_graz_rest_s, "s")
        self._param_row(right, "Pausa entre blocos:", self.var_graz_pause_s, "s")

        # Info
        info = tk.Label(parent, font=("Segoe UI", 8), fg=THEME["text_muted"],
                        bg=THEME["surface"], justify="left",
                        text="Sequência: Fixação (2s) → Beep (1s) → Imaginação → Repouso\n"
                             "Blocos: Esquerda → Pausa → Direita → Pausa\n"
                             "Marcadores e dados de repouso são gravados continuamente.")
        info.pack(anchor="w", pady=(8, 0))

    def _build_movement_params(self, parent):
        tk.Label(parent, text="Parâmetros Movement Protocol", font=("Segoe UI", 9, "bold"),
                 fg=THEME["orange"], bg=THEME["surface"]).pack(anchor="w", pady=(0, 6))

        grid = tk.Frame(parent, bg=THEME["surface"])
        grid.pack(fill="x")

        left = tk.Frame(grid, bg=THEME["surface"])
        left.pack(side="left", fill="x", expand=True)
        right = tk.Frame(grid, bg=THEME["surface"])
        right.pack(side="left", fill="x", expand=True)

        self._param_row(left, "Trials por bloco:", self.var_mov_trials)
        self._param_row(left, "Duração movimento:", self.var_mov_duration_s, "s")
        self._param_row(right, "Repouso:", self.var_mov_rest_s, "s")

        info = tk.Label(parent, font=("Segoe UI", 8), fg=THEME["text_muted"],
                        bg=THEME["surface"], justify="left",
                        text="Mov 1: Abrir e fechar a mão\n"
                             "Mov 2: Esticar o braço (cotovelo)\n"
                             "Mov 3: Levantar/abaixar o braço (ombro)\n"
                             "Cada bloco = 1 tipo de movimento, com pausa entre blocos.")
        info.pack(anchor="w", pady=(8, 0))

    def _build_viz_options(self, parent):
        sec = self._make_section(parent, "VISUALIZAÇÃO EM TEMPO REAL")

        opts_frame = tk.Frame(sec, bg=THEME["surface"])
        opts_frame.pack(fill="x")

        for var, label in [
            (self.var_show_power, "Potência média por canal (bandas)"),
            (self.var_show_concentration, "Barra de concentração (BrainFlow Metrics)"),
            (self.var_show_signal_quality, "Indicador de qualidade de sinal"),
        ]:
            cb = tk.Checkbutton(opts_frame, text=label, variable=var,
                                font=("Segoe UI", 9), fg=THEME["text"],
                                bg=THEME["surface"], selectcolor=THEME["surface2"],
                                activebackground=THEME["surface"])
            cb.pack(anchor="w", pady=1)

    def _build_actions(self, parent):
        frame = tk.Frame(parent, bg=THEME["bg"])
        frame.pack(fill="x", pady=(5, 10))

        self.btn_start = self._make_button(frame, "▶  INICIAR COLETA",
                                            self._start_collection, "green")
        self.btn_start.pack(fill="x", ipady=8)

        row2 = tk.Frame(frame, bg=THEME["bg"])
        row2.pack(fill="x", pady=(8, 0))

        self._make_button(row2, "  Abrir Pasta de Dados",
                          self._open_data_folder).pack(side="left", fill="x", expand=True, padx=(0, 4))
        self._make_button(row2, "  Análise Rápida",
                          self._quick_analysis).pack(side="left", fill="x", expand=True, padx=(4, 0))

    def _build_status_bar(self):
        bar = tk.Frame(self.root, bg=THEME["surface2"], height=28)
        bar.pack(side="bottom", fill="x")
        bar.pack_propagate(False)

        self.status_label = tk.Label(bar, text="Pronto",
                                      font=("Segoe UI", 8), fg=THEME["text_muted"],
                                      bg=THEME["surface2"])
        self.status_label.pack(side="left", padx=12)

        self.time_label = tk.Label(bar, text="",
                                    font=("Consolas", 8), fg=THEME["text_muted"],
                                    bg=THEME["surface2"])
        self.time_label.pack(side="right", padx=12)

    def _set_status(self, text, color=None):
        self.status_label.config(text=text, fg=color or THEME["text_muted"])
        self.root.update_idletasks()

    # HARDWARE 

    def _toggle_connection(self):
        if self.is_connected:
            self._disconnect()
        else:
            self._connect()

    def _connect(self):
        try:
            self._set_status("Conectando...", THEME["yellow"])

            params = BrainFlowInputParams()

            if self.var_synthetic.get():
                board_id = BoardIds.SYNTHETIC_BOARD
                self._set_status("Modo sintético ativado", THEME["yellow"])
            else:
                board_id = BoardIds.CYTON_BOARD
                params.serial_port = self.var_port.get()

            self.board = BoardShim(board_id, params)

            if self.board.is_prepared():
                self.board.release_session()

            self.board.prepare_session()
            self.is_connected = True
            self.board_id = board_id

            # Fator de escala: Cyton retorna ADC counts, sintético já retorna µV
            self.scale_uv = CYTON_SCALE_UV if board_id == BoardIds.CYTON_BOARD else 1.0

            self.hw_status_dot.delete("all")
            self.hw_status_dot.create_oval(1, 1, 9, 9, fill=THEME["green"], outline="")
            self.hw_status_label.config(text="Conectado", fg=THEME["green"])
            self.btn_connect.config(text="Desconectar", bg=THEME["red_dim"],
                                     fg=THEME["red"])

            board_name = "Sintético" if self.var_synthetic.get() else f"Cyton ({self.var_port.get()})"
            self._set_status(f"Conectado: {board_name}", THEME["green"])
            play_success()

        except Exception as e:
            self.is_connected = False
            self.hw_status_dot.delete("all")
            self.hw_status_dot.create_oval(1, 1, 9, 9, fill=THEME["red"], outline="")
            self.hw_status_label.config(text=f"Erro: {str(e)[:60]}", fg=THEME["red"])
            self._set_status(f"Erro de conexão: {str(e)[:80]}", THEME["red"])
            play_error()

    def _disconnect(self):
        try:
            if self.board and self.board.is_prepared():
                try:
                    self.board.stop_stream()
                except:
                    pass
                self.board.release_session()

            self.is_connected = False
            self.board = None

            self.hw_status_dot.delete("all")
            self.hw_status_dot.create_oval(1, 1, 9, 9, fill=THEME["red"], outline="")
            self.hw_status_label.config(text="Desconectado", fg=THEME["text_muted"])
            self.btn_connect.config(text="Conectar", bg=THEME["accent_dim"],
                                     fg=THEME["accent"])
            self._set_status("Desconectado")

        except Exception as e:
            self._set_status(f"Erro ao desconectar: {e}", THEME["red"])

    def _sanity_check(self):
        if not self.is_connected:
            messagebox.showwarning("Sanity Check", "Conecte a placa primeiro.")
            return

        self._set_status("Sanity Check com visualização ao vivo...", THEME["yellow"])

        try:
            self.board.start_stream()
        except Exception as e:
            self._set_status(f"Sanity Check falhou: {e}", THEME["red"])
            messagebox.showerror("Erro", f"Falha ao iniciar stream:\n{e}")
            return

        # ── Janela de Sanity Check com sinais ao vivo ──
        sc_win = tk.Toplevel(self.root)
        sc_win.title("Sanity Check — Visualização ao Vivo")
        sc_win.geometry("1100x700")
        sc_win.configure(bg=THEME["bg"])

        eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        sr = BoardShim.get_sampling_rate(self.board_id)
        n_ch = min(len(eeg_channels), 8)
        buf_sec = 4  # Segundos visíveis
        buf_len = int(sr * buf_sec)

        # Buffer circular para cada canal
        signal_bufs = [np.zeros(buf_len) for _ in range(n_ch)]

        # ── Layout: sinais à esquerda, relatório à direita ──
        content = tk.Frame(sc_win, bg=THEME["bg"])
        content.pack(fill="both", expand=True, padx=10, pady=10)

        # Canvas para sinais (osciloscópio)
        sig_frame = tk.Frame(content, bg=THEME["surface"], highlightbackground=THEME["border"],
                             highlightthickness=1)
        sig_frame.pack(side="left", fill="both", expand=True, padx=(0, 8))

        tk.Label(sig_frame, text="📡  SINAIS AO VIVO", font=("Segoe UI", 10, "bold"),
                 fg=THEME["accent"], bg=THEME["surface"]).pack(anchor="w", padx=12, pady=(8, 4))

        sig_canvas = tk.Canvas(sig_frame, bg=THEME["surface2"], highlightthickness=0)
        sig_canvas.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        # Painel de qualidade à direita
        right_panel = tk.Frame(content, bg=THEME["surface"], width=300,
                               highlightbackground=THEME["border"], highlightthickness=1)
        right_panel.pack(side="right", fill="y")
        right_panel.pack_propagate(False)

        tk.Label(right_panel, text="QUALIDADE DOS CANAIS", font=("Segoe UI", 10, "bold"),
                 fg=THEME["accent"], bg=THEME["surface"]).pack(anchor="w", padx=12, pady=(8, 8))

        # Labels de status por canal
        ch_status_frames = []
        for i in range(n_ch):
            name = CHANNEL_NAMES[i] if i < len(CHANNEL_NAMES) else f"CH{i+1}"
            color = CHANNEL_COLORS[i % len(CHANNEL_COLORS)]

            row = tk.Frame(right_panel, bg=THEME["surface"])
            row.pack(fill="x", padx=12, pady=2)

            dot = tk.Canvas(row, width=12, height=12, bg=THEME["surface"], highlightthickness=0)
            dot.pack(side="left", padx=(0, 6))
            dot.create_oval(2, 2, 10, 10, fill=THEME["text_muted"], outline="")

            tk.Label(row, text=name, font=("Consolas", 10, "bold"), fg=color,
                     bg=THEME["surface"], width=4).pack(side="left")

            info = tk.Label(row, text="...", font=("Consolas", 9), fg=THEME["text_muted"],
                            bg=THEME["surface"])
            info.pack(side="left", padx=8)

            status_lbl = tk.Label(row, text="", font=("Segoe UI", 9, "bold"),
                                  fg=THEME["text_muted"], bg=THEME["surface"])
            status_lbl.pack(side="right")

            ch_status_frames.append({'dot': dot, 'info': info, 'status': status_lbl})

        # Resultado geral
        tk.Frame(right_panel, bg=THEME["border"], height=1).pack(fill="x", padx=12, pady=10)
        result_label = tk.Label(right_panel, text="Analisando...", font=("Segoe UI", 11, "bold"),
                                fg=THEME["yellow"], bg=THEME["surface"])
        result_label.pack(padx=12)

        info_label = tk.Label(right_panel, text=f"SR: {sr} Hz | Buffer: {buf_sec}s",
                              font=("Segoe UI", 8), fg=THEME["text_muted"], bg=THEME["surface"])
        info_label.pack(padx=12, pady=(4, 8))

        # Botões
        btn_frame = tk.Frame(right_panel, bg=THEME["surface"])
        btn_frame.pack(fill="x", padx=12, pady=(8, 12), side="bottom")

        sc_running = [True]  # Mutable flag

        def _stop_sc():
            sc_running[0] = False
            try:
                self.board.stop_stream()
            except:
                pass
            self._set_status("Sanity Check finalizado", THEME["green"])
            stop_btn.config(state="disabled", text="Parado")

        stop_btn = self._make_button(btn_frame, "Parar & Fechar", lambda: (_stop_sc(), sc_win.after(300, sc_win.destroy)), "red")
        stop_btn.pack(fill="x")

        def _on_close():
            if sc_running[0]:
                _stop_sc()
            sc_win.destroy()

        sc_win.protocol("WM_DELETE_WINDOW", _on_close)

        # ── Loop de atualização em tempo real ──
        def _update_live():
            if not sc_running[0]:
                return

            try:
                data = self.board.get_current_board_data(int(sr * 0.15))  # ~150ms de dados novos
                n_new = data.shape[1]

                if n_new > 0:
                    for i in range(n_ch):
                        ch = eeg_channels[i]
                        new_samples = data[ch][:n_new] * self.scale_uv
                        signal_bufs[i] = np.roll(signal_bufs[i], -n_new)
                        signal_bufs[i][-n_new:] = new_samples

                # ── Desenhar sinais no canvas ──
                sig_canvas.delete("all")
                cw = sig_canvas.winfo_width()
                ch_h = sig_canvas.winfo_height()
                if cw < 50 or ch_h < 50:
                    cw, ch_h = 700, 500

                track_h = ch_h / n_ch
                t_axis = np.linspace(0, buf_sec, buf_len)

                for i in range(n_ch):
                    y_center = track_h * i + track_h / 2
                    color = CHANNEL_COLORS[i % len(CHANNEL_COLORS)]
                    name = CHANNEL_NAMES[i] if i < len(CHANNEL_NAMES) else f"CH{i+1}"

                    # Separador
                    if i > 0:
                        sig_canvas.create_line(0, track_h * i, cw, track_h * i,
                                               fill=THEME["border"], width=1)

                    # Label do canal
                    sig_canvas.create_text(6, y_center - track_h * 0.35, text=name,
                                           fill=color, font=("Consolas", 8, "bold"), anchor="w")

                    # Sinal — remover DC offset para visualização
                    sig = signal_bufs[i]
                    sig_ac = sig - np.mean(sig)  # Remove DC offset
                    std = np.std(sig_ac)
                    if std < 0.01:
                        std = 1.0

                    # Escalar sinal para caber na track (ajustado para µV)
                    scale = (track_h * 0.4) / (std * 2.5 + 1e-6)
                    points = []
                    step = max(1, buf_len // cw)
                    for j in range(0, buf_len, step):
                        x = (j / buf_len) * cw
                        y = y_center - sig_ac[j] * scale
                        y = max(track_h * i + 4, min(track_h * (i + 1) - 4, y))
                        points.extend([x, y])

                    if len(points) >= 4:
                        sig_canvas.create_line(points, fill=color, width=1, smooth=False)

                    # ── Atualizar qualidade (valores já em µV) ──
                    mean_v = np.mean(sig[-int(sr):]) if len(sig) >= sr else np.mean(sig)
                    std_v = np.std(sig[-int(sr):]) if len(sig) >= sr else std
                    unique_ratio = len(np.unique(np.round(sig[-int(sr):], 2))) / max(int(sr), 1)

                    is_railed = unique_ratio < 0.05
                    is_flat = std_v < 0.5  # < 0.5 µV = flat

                    if is_railed or is_flat:
                        status, s_color = "RAILED", THEME["red"]
                    elif std_v > 150:  # > 150 µV = muito ruído
                        status, s_color = "RUÍDO", THEME["yellow"]
                    else:
                        status, s_color = "OK", THEME["green"]

                    ch_status_frames[i]['dot'].delete("all")
                    ch_status_frames[i]['dot'].create_oval(2, 2, 10, 10, fill=s_color, outline="")
                    ch_status_frames[i]['info'].config(text=f"µ={mean_v:.1f} σ={std_v:.1f}")
                    ch_status_frames[i]['status'].config(text=status, fg=s_color)

                # Checar resultado geral
                all_ok = all(ch_status_frames[i]['status'].cget("text") == "OK" for i in range(n_ch))
                if all_ok:
                    result_label.config(text="✓ TODOS OK", fg=THEME["green"])
                else:
                    result_label.config(text="⚠ VERIFICAR CANAIS", fg=THEME["yellow"])

            except Exception:
                pass

            if sc_running[0]:
                sc_win.after(80, _update_live)  # ~12 FPS

        # Início do loop
        sc_win.after(500, _update_live)

    # Data aquisition

    def _validate_before_start(self):
        if not self.is_connected:
            messagebox.showwarning("Aviso", "Conecte a placa primeiro.")
            return False
        if not self.var_name.get().strip():
            messagebox.showwarning("Aviso", "Digite o nome do sujeito.")
            return False
        if not os.path.isdir(self.var_save_path.get()):
            messagebox.showwarning("Aviso", "O caminho de salvamento não existe.")
            return False
        return True

    def _start_collection(self):
        if not self._validate_before_start():
            return

        if self.is_collecting:
            self._abort_collection()
            return

        self.is_collecting = True
        self.stop_event.clear()
        self.all_data = []

        self.btn_start.config(text="PARAR COLETA", bg=THEME["red_dim"], fg=THEME["red"])

        protocol = self.var_protocol.get()

        # Abrir janela de coleta
        self.collection_window = CollectionWindow(
            self.root, self, protocol,
            show_power=self.var_show_power.get(),
            show_concentration=self.var_show_concentration.get(),
            show_signal_quality=self.var_show_signal_quality.get(),
        )

    def _abort_collection(self):
        self.stop_event.set()
        self.is_collecting = False
        self.btn_start.config(text="▶  INICIAR COLETA", bg=THEME["green_dim"], fg=THEME["green"])
        self._set_status("Coleta abortada", THEME["yellow"])

    def _finish_collection(self):
        self.is_collecting = False
        self.btn_start.config(text="▶  INICIAR COLETA", bg=THEME["green_dim"], fg=THEME["green"])

    def _save_data(self, all_data):
        if all_data is None or len(all_data) == 0:
            self._set_status("Nenhum dado para salvar", THEME["yellow"])
            return None

        try:
            data_matrix = np.hstack(all_data) if len(all_data) > 1 else all_data[0]
        except Exception:
            self._set_status("Erro ao concatenar dados", THEME["red"])
            return None

        eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        timestamp_ch = BoardShim.get_timestamp_channel(self.board_id)
        marker_ch = BoardShim.get_marker_channel(self.board_id)
        accel_channels = BoardShim.get_accel_channels(self.board_id)

        # Montar DataFrame
        columns = {}

        # Timestamp
        columns["Timestamp"] = data_matrix[timestamp_ch]

        # Canais EEG
        for idx, ch in enumerate(eeg_channels):
            name = CHANNEL_NAMES[idx] if idx < len(CHANNEL_NAMES) else f"EEG_CH{idx+1}"
            columns[name] = data_matrix[ch]

        # Acelerometro
        for idx, ch in enumerate(accel_channels):
            columns[f"Accel_{'XYZ'[idx] if idx < 3 else str(idx)}"] = data_matrix[ch]

        # Marcador
        columns["Marker"] = data_matrix[marker_ch]

        df = pd.DataFrame(columns)

        # Nome do arquivo
        subject = self.var_name.get().strip().replace(" ", "_")
        session = self.var_session.get().strip()
        protocol = self.var_protocol.get()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{subject}_S{session}_{protocol}_{ts}.csv"

        filepath = os.path.join(self.var_save_path.get(), filename)

        # Adicionar header com metadados
        meta_lines = [
            f"# BCI-IM Collection Suite v{VERSION}",
            f"# Subject: {self.var_name.get().strip()}",
            f"# Session: {session}",
            f"# Protocol: {protocol}",
            f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Board: {'Synthetic' if self.var_synthetic.get() else 'Cyton'}",
            f"# Sampling Rate: {BoardShim.get_sampling_rate(self.board_id)} Hz",
            f"# Channels: {', '.join(CHANNEL_NAMES[:len(eeg_channels)])}",
            f"# Markers: {json.dumps(MARKERS)}",
            f"#",
        ]

        with open(filepath, "w", encoding="utf-8") as f:
            for line in meta_lines:
                f.write(line + "\n")

        df.to_csv(filepath, mode="a", index=False)

        self._set_status(f"Dados salvos: {filename}", THEME["green"])
        play_success()
        return filepath

    # Utilities

    def _browse_path(self):
        path = filedialog.askdirectory(initialdir=self.var_save_path.get())
        if path:
            self.var_save_path.set(path)

    def _open_data_folder(self):
        path = self.var_save_path.get()
        if os.path.isdir(path):
            os.startfile(path)
        else:
            messagebox.showwarning("Aviso", "Pasta não encontrada.")

    def _quick_analysis(self):
        filepath = filedialog.askopenfilename(
            initialdir=self.var_save_path.get(),
            filetypes=[("CSV", "*.csv")],
            title="Selecione um arquivo de coleta"
        )
        if not filepath:
            return

        try:
            # Pular linhas de metadados (começam com #)
            df = pd.read_csv(filepath, comment="#")

            win = tk.Toplevel(self.root)
            win.title(f"Análise Rápida — {os.path.basename(filepath)}")
            win.geometry("700x520")
            win.configure(bg=THEME["bg"])

            canvas = tk.Canvas(win, bg=THEME["surface"], highlightthickness=0)
            canvas.pack(fill="both", expand=True, padx=10, pady=10)

            eeg_cols = [c for c in df.columns if c in CHANNEL_NAMES]
            marker_col = "Marker" if "Marker" in df.columns else None

            # Info geral
            sr = 250  # Assume Cyton
            n_samples = len(df)
            duration = n_samples / sr
            y_offset = 20

            canvas.create_text(20, y_offset, anchor="nw",
                text=f"Arquivo: {os.path.basename(filepath)}",
                fill=THEME["accent"], font=("Consolas", 10, "bold"))
            y_offset += 22
            canvas.create_text(20, y_offset, anchor="nw",
                text=f"Duração: {duration:.1f}s  |  Amostras: {n_samples}  |  Canais: {len(eeg_cols)}",
                fill=THEME["text"], font=("Consolas", 9))
            y_offset += 30

            # Estatísticas por canal
            canvas.create_text(20, y_offset, anchor="nw",
                text=f"{'Canal':<8}  {'Média (µV)':<14}  {'Std (µV)':<14}  {'Min':<12}  {'Max':<12}",
                fill=THEME["text_muted"], font=("Consolas", 9, "bold"))
            y_offset += 18

            for i, col in enumerate(eeg_cols):
                vals = df[col].values
                color = CHANNEL_COLORS[i % len(CHANNEL_COLORS)]
                canvas.create_text(20, y_offset, anchor="nw",
                    text=f"{col:<8}  {np.mean(vals):>10.2f}    {np.std(vals):>10.2f}    "
                         f"{np.min(vals):>8.1f}    {np.max(vals):>8.1f}",
                    fill=color, font=("Consolas", 9))
                y_offset += 17

            # Contagem de marcadores
            if marker_col:
                y_offset += 15
                canvas.create_text(20, y_offset, anchor="nw",
                    text="Marcadores encontrados:",
                    fill=THEME["yellow"], font=("Consolas", 10, "bold"))
                y_offset += 20

                markers_found = df[marker_col].value_counts()
                markers_found = markers_found[markers_found.index != 0].sort_index()

                marker_name_map = {v: k for k, v in MARKERS.items()}
                for mk_val, mk_count in markers_found.items():
                    name = marker_name_map.get(int(mk_val), f"unknown_{int(mk_val)}")
                    canvas.create_text(40, y_offset, anchor="nw",
                        text=f"[{int(mk_val):>3}] {name:<20} × {int(mk_count)}",
                        fill=THEME["text"], font=("Consolas", 9))
                    y_offset += 16

            self._make_button(win, "Fechar", win.destroy).pack(pady=8)

        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao analisar:\n{e}")

    def run(self):
        self.root.mainloop()


# Data Aquisition GUI

class CollectionWindow:
    def __init__(self, parent, app, protocol, show_power=True,
                 show_concentration=True, show_signal_quality=True):
        self.app = app
        self.protocol = protocol
        self.show_power = show_power
        self.show_concentration = show_concentration
        self.show_signal_quality = show_signal_quality

        self.win = tk.Toplevel(parent)
        self.win.title("Coleta em Andamento")
        self.win.configure(bg=THEME["bg"])
        self.win.attributes("-fullscreen", True)
        self.win.bind("<Escape>", lambda e: self._abort())
        self.win.bind("<F11>", lambda e: self.win.attributes("-fullscreen",
                        not self.win.attributes("-fullscreen")))
        self.win.protocol("WM_DELETE_WINDOW", self._abort)  # Closing window = safe abort

        self.board = app.board
        self.board_id = app.board_id
        self.all_data = []
        self.is_streaming = False
        self.viz_after_id = None

        self._build_ui()
        self._start_protocol()

    def _build_ui(self):
        # Layout: sidebar (optional) + center cue area + bottom progress
        self.main_container = tk.Frame(self.win, bg=THEME["bg"])
        self.main_container.pack(fill="both", expand=True)

        if self.show_power or self.show_concentration or self.show_signal_quality:
            self.side_panel = tk.Frame(self.main_container, bg=THEME["surface"],
                                       width=320)
            self.side_panel.pack(side="right", fill="y", padx=0)
            self.side_panel.pack_propagate(False)
            self._build_side_panel()

        # Center cue area
        self.center = tk.Frame(self.main_container, bg=THEME["bg"])
        self.center.pack(side="left", fill="both", expand=True)

        # Trial counter
        self.counter_label = tk.Label(self.center, text="",
                                       font=("Segoe UI", 12), fg=THEME["text_muted"],
                                       bg=THEME["bg"])
        self.counter_label.pack(pady=(40, 10))

        # Phase label
        self.phase_label = tk.Label(self.center, text="",
                                     font=("Segoe UI", 14), fg=THEME["text_muted"],
                                     bg=THEME["bg"])
        self.phase_label.pack(pady=(0, 20))

        # Main cue canvas
        self.cue_canvas = tk.Canvas(self.center, width=500, height=350,
                                     bg=THEME["bg"], highlightthickness=0)
        self.cue_canvas.pack(expand=True)

        # Timer display
        self.timer_label = tk.Label(self.center, text="",
                                     font=("Consolas", 28, "bold"),
                                     fg=THEME["accent"], bg=THEME["bg"])
        self.timer_label.pack(pady=(10, 5))

        # Progress bar (canvas-based)
        self.progress_frame = tk.Frame(self.center, bg=THEME["bg"])
        self.progress_frame.pack(fill="x", padx=60, pady=(5, 20))

        self.progress_canvas = tk.Canvas(self.progress_frame, height=8,
                                          bg=THEME["surface2"], highlightthickness=0)
        self.progress_canvas.pack(fill="x")

        # Bottom info bar
        bottom = tk.Frame(self.center, bg=THEME["surface2"], height=36)
        bottom.pack(fill="x", side="bottom")
        bottom.pack_propagate(False)

        self.info_label = tk.Label(bottom, text="Pressione ESC para abortar | F11 para tela cheia",
                                    font=("Segoe UI", 8), fg=THEME["text_muted"],
                                    bg=THEME["surface2"])
        self.info_label.pack(side="left", padx=12)

        self.rec_label = tk.Label(bottom, text="● REC",
                                   font=("Consolas", 9, "bold"), fg=THEME["red"],
                                   bg=THEME["surface2"])
        self.rec_label.pack(side="right", padx=12)

    def _build_side_panel(self):
        tk.Label(self.side_panel, text="MONITORAMENTO",
                 font=("Segoe UI", 8, "bold"), fg=THEME["accent"],
                 bg=THEME["surface"]).pack(anchor="w", padx=12, pady=(12, 8))

        if self.show_signal_quality:
            tk.Label(self.side_panel, text="Qualidade do Sinal",
                     font=("Segoe UI", 8), fg=THEME["text_muted"],
                     bg=THEME["surface"]).pack(anchor="w", padx=12)
            self.quality_canvas = tk.Canvas(self.side_panel, height=40,
                                             bg=THEME["surface"], highlightthickness=0)
            self.quality_canvas.pack(fill="x", padx=12, pady=(2, 10))

        # Sinais
        tk.Label(self.side_panel, text="Sinais EEG ao Vivo",
                 font=("Segoe UI", 8), fg=THEME["text_muted"],
                 bg=THEME["surface"]).pack(anchor="w", padx=12)
        self.live_sig_canvas = tk.Canvas(self.side_panel, height=220,
                                          bg=THEME["surface2"], highlightthickness=0)
        self.live_sig_canvas.pack(fill="x", padx=12, pady=(2, 10))
        self.sig_bufs = None  # Initialized on first update

        if self.show_power:
            tk.Label(self.side_panel, text="Potência por Banda (µV²)",
                     font=("Segoe UI", 8), fg=THEME["text_muted"],
                     bg=THEME["surface"]).pack(anchor="w", padx=12)
            self.power_canvas = tk.Canvas(self.side_panel, height=200,
                                           bg=THEME["surface"], highlightthickness=0)
            self.power_canvas.pack(fill="x", padx=12, pady=(2, 10))

        if self.show_concentration:
            tk.Label(self.side_panel, text="Nível de Concentração",
                     font=("Segoe UI", 8), fg=THEME["text_muted"],
                     bg=THEME["surface"]).pack(anchor="w", padx=12)
            self.conc_canvas = tk.Canvas(self.side_panel, height=35,
                                          bg=THEME["surface"], highlightthickness=0)
            self.conc_canvas.pack(fill="x", padx=12, pady=(2, 10))

    # VISUAL CUES 

    def _show_cross(self):
        self.cue_canvas.delete("all")
        cx, cy = 250, 175
        self.cue_canvas.create_line(cx, cy - 50, cx, cy + 50,
                                     fill=THEME["text"], width=4)
        self.cue_canvas.create_line(cx - 50, cy, cx + 50, cy,
                                     fill=THEME["text"], width=4)
        self.phase_label.config(text="Fixação", fg=THEME["text_muted"])

    def _show_arrow(self, direction):
        self.cue_canvas.delete("all")
        cx, cy = 250, 175

        if direction == "left":
            pts = [cx - 80, cy, cx + 40, cy - 50, cx + 40, cy - 20,
                   cx + 80, cy - 20, cx + 80, cy + 20, cx + 40, cy + 20,
                   cx + 40, cy + 50]
            self.cue_canvas.create_polygon(pts, fill=THEME["red"], outline="")
            self.phase_label.config(text="← MÃO ESQUERDA", fg=THEME["red"])
        else:
            pts = [cx + 80, cy, cx - 40, cy - 50, cx - 40, cy - 20,
                   cx - 80, cy - 20, cx - 80, cy + 20, cx - 40, cy + 20,
                   cx - 40, cy + 50]
            self.cue_canvas.create_polygon(pts, fill=THEME["accent"], outline="")
            self.phase_label.config(text="MÃO DIREITA →", fg=THEME["accent"])

    def _show_movement_cue(self, movement_type):
        self.cue_canvas.delete("all")
        cx, cy = 250, 175

        if movement_type == "hand":
            self.cue_canvas.create_text(cx, cy - 30, text="✊ ↔ ✋",
                font=("Segoe UI", 48), fill=THEME["orange"])
            self.cue_canvas.create_text(cx, cy + 50,
                text="ABRIR E FECHAR A MÃO",
                font=("Segoe UI", 14, "bold"), fill=THEME["orange"])

        elif movement_type == "elbow":
            self.cue_canvas.create_text(cx, cy - 30, text="💪",
                font=("Segoe UI", 48), fill=THEME["purple"])
            self.cue_canvas.create_text(cx, cy + 50,
                text="ESTICAR O BRAÇO (COTOVELO)",
                font=("Segoe UI", 14, "bold"), fill=THEME["purple"])

        elif movement_type == "shoulder":
            self.cue_canvas.create_text(cx, cy - 30, text="🦾",
                font=("Segoe UI", 48), fill=THEME["green"])
            self.cue_canvas.create_text(cx, cy + 50,
                text="LEVANTAR/ABAIXAR O BRAÇO (OMBRO)",
                font=("Segoe UI", 14, "bold"), fill=THEME["green"])

    def _show_rest(self):
        self.cue_canvas.delete("all")
        self.cue_canvas.create_text(250, 175, text="Repouso",
            font=("Segoe UI", 24), fill=THEME["text_muted"])
        self.phase_label.config(text="Período de repouso", fg=THEME["text_muted"])

    def _show_pause(self):
        self.cue_canvas.delete("all")
        self.cue_canvas.create_text(250, 155, text="PAUSA",
            font=("Segoe UI", 28, "bold"), fill=THEME["yellow"])
        self.cue_canvas.create_text(250, 200, text="Descanse e aguarde...",
            font=("Segoe UI", 12), fill=THEME["text_muted"])
        self.phase_label.config(text="Pausa entre blocos", fg=THEME["yellow"])

    def _show_done(self):
        self.cue_canvas.delete("all")
        self.cue_canvas.create_text(250, 155, text="✓",
            font=("Segoe UI", 56), fill=THEME["green"])
        self.cue_canvas.create_text(250, 220, text="Coleta Concluída!",
            font=("Segoe UI", 18, "bold"), fill=THEME["green"])
        self.phase_label.config(text="")
        self.timer_label.config(text="")
        self.counter_label.config(text="")

    def _update_timer(self, remaining):
        self.timer_label.config(text=f"{remaining:.1f}s")

    def _update_progress(self, current, total):
        self.progress_canvas.delete("all")
        w = self.progress_canvas.winfo_width()
        if w < 10:
            w = 600
        frac = current / max(total, 1)
        self.progress_canvas.create_rectangle(0, 0, w * frac, 8,
            fill=THEME["accent"], outline="")

    # LIVE PLOTS

    def _start_viz_loop(self):
        self._update_visualizations()

    def _update_visualizations(self):
        if not self.is_streaming or self.app.stop_event.is_set():
            return

        try:
            sr = BoardShim.get_sampling_rate(self.board_id)
            eeg_channels = BoardShim.get_eeg_channels(self.board_id)
            n_ch = min(len(eeg_channels), 8)

            # Use get_current_board_data — reads WITHOUT clearing the ring buffer
            # This is safe to call concurrently with get_board_data() in the protocol thread
            data = self.board.get_current_board_data(int(sr * 1.5))

            if data is not None and data.ndim == 2 and data.shape[1] > 10:
                if self.show_signal_quality and hasattr(self, 'quality_canvas'):
                    self._draw_signal_quality(data, eeg_channels)

                if hasattr(self, 'live_sig_canvas'):
                    self._draw_live_signals(data, eeg_channels, sr)

                if self.show_power and hasattr(self, 'power_canvas'):
                    self._draw_power_bars(data, eeg_channels)

                if self.show_concentration and hasattr(self, 'conc_canvas'):
                    self._draw_concentration(data, eeg_channels)

        except tk.TclError:
            # Window was destroyed — stop the loop
            return
        except Exception as e:
            # Log but don't crash — the viz loop must keep running
            print(f"[VIZ] Erro na visualização (continuando): {type(e).__name__}: {e}")

        # Always reschedule — the visualization must never silently die
        try:
            self.viz_after_id = self.win.after(150, self._update_visualizations)
        except tk.TclError:
            pass  # Window already destroyed

    def _draw_live_signals(self, data, eeg_channels, sr):
        n_ch = min(len(eeg_channels), 8)
        buf_len = int(sr * 1.5)
        scale_uv = self.app.scale_uv if hasattr(self.app, 'scale_uv') else 1.0

        # Inicializar buffers na primeira chamada
        if self.sig_bufs is None:
            self.sig_bufs = [np.zeros(buf_len) for _ in range(n_ch)]

        # Atualizar buffers com dados novos (convertidos para µV)
        n_new = min(data.shape[1], buf_len)
        for i in range(n_ch):
            ch = eeg_channels[i]
            new_samples = data[ch][-n_new:] * scale_uv
            self.sig_bufs[i] = np.roll(self.sig_bufs[i], -n_new)
            self.sig_bufs[i][-n_new:] = new_samples

        self.live_sig_canvas.delete("all")
        cw = self.live_sig_canvas.winfo_width()
        ch_total = self.live_sig_canvas.winfo_height()
        if cw < 30 or ch_total < 30:
            cw, ch_total = 230, 220

        track_h = ch_total / n_ch

        for i in range(n_ch):
            y_center = track_h * i + track_h / 2
            color = CHANNEL_COLORS[i % len(CHANNEL_COLORS)]
            name = CHANNEL_NAMES[i] if i < len(CHANNEL_NAMES) else f"C{i+1}"

            # Separador
            if i > 0:
                self.live_sig_canvas.create_line(0, int(track_h * i), cw, int(track_h * i),
                                                  fill=THEME["border"], width=1)

            # Label
            self.live_sig_canvas.create_text(3, int(y_center - track_h * 0.35), text=name,
                                              fill=color, font=("Consolas", 6, "bold"), anchor="w")

            # Sinal — remover DC offset para visualização
            sig = self.sig_bufs[i]
            sig_ac = sig - np.mean(sig)
            std = np.std(sig_ac)
            if std < 0.01:
                std = 1.0
            scale = (track_h * 0.4) / (std * 2.5 + 1e-6)

            points = []
            step = max(1, buf_len // (cw - 5))
            for j in range(0, buf_len, step):
                x = (j / buf_len) * cw
                y = y_center - sig_ac[j] * scale
                y = max(track_h * i + 2, min(track_h * (i + 1) - 2, y))
                points.extend([x, y])

            if len(points) >= 4:
                self.live_sig_canvas.create_line(points, fill=color, width=1, smooth=False)

    def _draw_signal_quality(self, data, eeg_channels):
        self.quality_canvas.delete("all")
        w = self.quality_canvas.winfo_width()
        if w < 10:
            w = 230
        n = len(eeg_channels)
        bar_w = max(w // n - 4, 8)
        scale_uv = self.app.scale_uv if hasattr(self.app, 'scale_uv') else 1.0

        for i, ch in enumerate(eeg_channels):
            std = np.std(data[ch] * scale_uv)
            # Classificação em µV
            if std < 0.5:
                color = THEME["red"]  # flat/railed
            elif std > 150:
                color = THEME["yellow"]  # noisy
            else:
                color = THEME["green"]  # ok

            x = i * (bar_w + 3) + 4
            self.quality_canvas.create_rectangle(x, 5, x + bar_w, 25,
                fill=color, outline="")
            name = CHANNEL_NAMES[i] if i < len(CHANNEL_NAMES) else str(i+1)
            self.quality_canvas.create_text(x + bar_w // 2, 33, text=name,
                fill=THEME["text_muted"], font=("Consolas", 6), anchor="center")

    def _draw_power_bars(self, data, eeg_channels):
        self.power_canvas.delete("all")
        sr = BoardShim.get_sampling_rate(self.board_id)
        w = self.power_canvas.winfo_width()
        if w < 10:
            w = 230

        n_ch = min(len(eeg_channels), 8)
        row_h = 22
        bar_max_w = w - 60

        for i in range(n_ch):
            ch = eeg_channels[i]
            ch_data = data[ch].copy() * (self.app.scale_uv if hasattr(self.app, 'scale_uv') else 1.0)
            if len(ch_data) < 50:
                continue

            # Calcular potência por banda
            y_base = i * row_h + 4
            name = CHANNEL_NAMES[i] if i < len(CHANNEL_NAMES) else f"C{i+1}"
            self.power_canvas.create_text(4, y_base + row_h // 2, text=name,
                fill=CHANNEL_COLORS[i % len(CHANNEL_COLORS)],
                font=("Consolas", 7), anchor="w")

            # Potência total (simplificado)
            total_power = max(np.std(ch_data), 0.1)
            x_offset = 30

            for band_name, (f_low, f_high) in FREQ_BANDS.items():
                try:
                    if len(ch_data) > 30 and f_high < sr / 2:
                        sos = butter(2, [f_low, f_high], btype='band', fs=sr, output='sos')
                        filtered = sosfiltfilt(sos, ch_data)
                        band_power = np.std(filtered)
                    else:
                        band_power = 0
                except Exception:
                    band_power = 0

                bar_len = min(band_power / max(total_power, 1) * bar_max_w * 0.3, bar_max_w * 0.18)
                color = BAND_COLORS.get(band_name, THEME["text_muted"])

                self.power_canvas.create_rectangle(
                    x_offset, y_base + 3, x_offset + max(bar_len, 1), y_base + row_h - 3,
                    fill=color, outline="")
                x_offset += bar_len + 1

    def _draw_concentration(self, data, eeg_channels):
        self.conc_canvas.delete("all")
        w = self.conc_canvas.winfo_width()
        if w < 10:
            w = 230

        # Estimativa simplificada: razão beta/alpha nos canais frontais
        sr = BoardShim.get_sampling_rate(self.board_id)
        scale_uv = self.app.scale_uv if hasattr(self.app, 'scale_uv') else 1.0
        alpha_power = 0
        beta_power = 0

        for ch in eeg_channels[:4]:  # Primeiros 4 canais
            ch_data = data[ch].copy() * scale_uv
            if len(ch_data) < 50:
                continue
            try:
                if len(ch_data) > 30:
                    sos_a = butter(2, [8, 13], btype='band', fs=sr, output='sos')
                    alpha_power += np.var(sosfiltfilt(sos_a, ch_data))

                    sos_b = butter(2, [13, 30], btype='band', fs=sr, output='sos')
                    beta_power += np.var(sosfiltfilt(sos_b, ch_data))
            except Exception:
                pass

        ratio = beta_power / max(alpha_power, 0.001)
        level = min(ratio / 3.0, 1.0)  # Normalizar 0-1

        # Barra de concentração
        bar_w = int(level * (w - 20))
        color = THEME["green"] if level > 0.5 else THEME["yellow"] if level > 0.25 else THEME["red"]

        self.conc_canvas.create_rectangle(5, 8, w - 5, 27,
            fill=THEME["surface2"], outline=THEME["border"])
        self.conc_canvas.create_rectangle(5, 8, 5 + max(bar_w, 1), 27,
            fill=color, outline="")
        self.conc_canvas.create_text(w // 2, 17, text=f"{level*100:.0f}%",
            fill="#FFFFFF", font=("Consolas", 8, "bold"))

    # Protocol

    def _start_protocol(self):
        try:
            self.board.start_stream()
            self.is_streaming = True

            # Iniciar visualização
            if self.show_power or self.show_concentration or self.show_signal_quality:
                self.win.after(500, self._start_viz_loop)

            # Iniciar protocolo em thread
            if self.protocol == "graz_b":
                self.collection_thread = threading.Thread(
                    target=self._run_graz_b, daemon=True)
            else:
                self.collection_thread = threading.Thread(
                    target=self._run_movement, daemon=True)

            self.collection_thread.start()

        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao iniciar streaming:\n{e}")
            self._close()

    def _wait(self, seconds, show_countdown=True):
        steps = int(seconds * 10)
        for i in range(steps):
            if self.app.stop_event.is_set():
                return False
            if show_countdown:
                remaining = seconds - i * 0.1
                self.win.after(0, lambda r=remaining: self._update_timer(r))
            time.sleep(0.1)
        return True

    def _collect_chunk(self):
        try:
            data = self.board.get_board_data()
            if data.shape[1] > 0:
                self.all_data.append(data)
        except Exception:
            pass

    def _insert_marker(self, marker_value):
        try:
            self.board.insert_marker(float(marker_value))
        except Exception:
            pass

    def _marker_then_collect(self, marker_value):

        self._insert_marker(marker_value)
        time.sleep(0.02)
        self._collect_chunk()

    # GRAZ B PROTOCOL 

    def _run_graz_b(self):
        imagery_s = self.app.var_graz_imagery_s.get()
        rest_s = self.app.var_graz_rest_s.get()
        pause_s = self.app.var_graz_pause_s.get()
        n_trials = self.app.var_graz_trials.get()

        sides = ["esquerda", "direita"]
        total_trials = n_trials * len(sides)
        trial_global = 0
        aborted = False

        self._insert_marker(MARKERS["session_start"])

        for block_idx, side in enumerate(sides):
            if self.app.stop_event.is_set():
                aborted = True
                break

            self._insert_marker(MARKERS["block_start"])
            marker_cue = MARKERS["mi_left"] if side == "esquerda" else MARKERS["mi_right"]

            for trial in range(1, n_trials + 1):
                if self.app.stop_event.is_set():
                    aborted = True
                    break

                trial_global += 1
                self.win.after(0, lambda t=trial, n=n_trials, s=side, tg=trial_global, tt=total_trials:
                    (self.counter_label.config(
                        text=f"Trial {t}/{n} — Bloco {s.upper()}  |  "
                             f"Total: {tg}/{tt}"),
                     self._update_progress(tg, tt)))

                # 1. Fixação
                self.win.after(0, self._show_cross)
                self._insert_marker(MARKERS["fixation"])
                if not self._wait(2.0):
                    aborted = True; break
                self._collect_chunk()

                # 2. Beep
                self._insert_marker(MARKERS["beep"])
                play_beep(440, 500)
                if not self._wait(1.0):
                    aborted = True; break
                self._collect_chunk()

                # 3. Imaginação motora
                self._insert_marker(marker_cue)
                direction = "left" if side == "esquerda" else "right"
                self.win.after(0, lambda d=direction: self._show_arrow(d))
                if not self._wait(imagery_s):
                    aborted = True; break
                self._collect_chunk()

                # 4. Repouso (GRAVADO com marcador!)
                self._insert_marker(MARKERS["rest_start"])
                self.win.after(0, self._show_rest)
                if not self._wait(rest_s):
                    aborted = True; break
                self._marker_then_collect(MARKERS["rest_end"])

            # Always close the block, even after abort
            self._insert_marker(MARKERS["block_end"])

            if aborted:
                break

            # Pausa entre blocos (exceto após o último)
            if block_idx < len(sides) - 1:
                self._insert_marker(MARKERS["pause_start"])
                self.win.after(0, self._show_pause)
                play_beep(330, 300)
                if not self._wait(pause_s):
                    aborted = True
                    self._insert_marker(MARKERS["pause_end"])
                    break
                self._marker_then_collect(MARKERS["pause_end"])

        # Coleta final — always insert session_end
        self._insert_marker(MARKERS["session_end"])
        # Wait for Cyton to deliver samples carrying the markers
        time.sleep(1.0)
        self._collect_chunk()

        # Finalizar
        self.win.after(0, self._finish)

    # 3 CLASS MOVEMENT PROTOCOL 

    def _run_movement(self):
        movement_s = self.app.var_mov_duration_s.get()
        rest_s = self.app.var_mov_rest_s.get()
        pause_s = self.app.var_graz_pause_s.get()
        n_trials = self.app.var_mov_trials.get()

        movements = [
            ("hand",     "Abrir/Fechar Mão",       MARKERS["mov_hand"]),
            ("elbow",    "Extensão Cotovelo",       MARKERS["mov_elbow"]),
            ("shoulder", "Levantar/Abaixar Ombro",  MARKERS["mov_shoulder"]),
        ]

        total_trials = n_trials * len(movements)
        trial_global = 0
        aborted = False

        self._insert_marker(MARKERS["session_start"])

        for block_idx, (mov_key, mov_name, mov_marker) in enumerate(movements):
            if self.app.stop_event.is_set():
                aborted = True
                break

            self._insert_marker(MARKERS["block_start"])

            for trial in range(1, n_trials + 1):
                if self.app.stop_event.is_set():
                    aborted = True
                    break

                trial_global += 1
                self.win.after(0, lambda t=trial, n=n_trials, mn=mov_name, tg=trial_global, tt=total_trials:
                    (self.counter_label.config(
                        text=f"Trial {t}/{n} — {mn}  |  "
                             f"Total: {tg}/{tt}"),
                     self._update_progress(tg, tt)))

                # 1. Fixação
                self.win.after(0, self._show_cross)
                self._insert_marker(MARKERS["fixation"])
                if not self._wait(2.0):
                    aborted = True; break
                self._collect_chunk()

                # 2. Beep
                self._insert_marker(MARKERS["beep"])
                play_beep(440, 500)
                if not self._wait(1.0):
                    aborted = True; break
                self._collect_chunk()

                # 3. Movimento
                self._insert_marker(mov_marker)
                self.win.after(0, lambda mk=mov_key: self._show_movement_cue(mk))
                if not self._wait(movement_s):
                    aborted = True; break
                self._collect_chunk()

                # 4. Repouso
                self._insert_marker(MARKERS["rest_start"])
                self.win.after(0, self._show_rest)
                if not self._wait(rest_s):
                    aborted = True; break
                self._marker_then_collect(MARKERS["rest_end"])

            # Always close the block, even after abort
            self._insert_marker(MARKERS["block_end"])

            if aborted:
                break

            # Pausa entre blocos
            if block_idx < len(movements) - 1:
                self._insert_marker(MARKERS["pause_start"])
                self.win.after(0, self._show_pause)
                play_beep(330, 300)
                if not self._wait(pause_s):
                    aborted = True
                    self._insert_marker(MARKERS["pause_end"])
                    break
                self._marker_then_collect(MARKERS["pause_end"])

        # Always insert session_end, even on abort
        self._insert_marker(MARKERS["session_end"])
        # Wait for Cyton to deliver at least 2 more samples so markers are stamped
        # Cyton at 250Hz = 4ms/sample; 1.0s guarantees multiple packets arrive
        time.sleep(1.0)
        self._collect_chunk()

        self.win.after(0, self._finish)

    # FINALIZATION

    def _finish(self):
        # Final safety collect — grab anything still in the ring buffer
        # BEFORE stopping the stream (stop_stream clears the buffer)
        try:
            data = self.board.get_board_data()
            if data.shape[1] > 0:
                self.all_data.append(data)
        except Exception:
            pass

        try:
            self.board.stop_stream()
        except Exception:
            pass

        self.is_streaming = False

        if self.viz_after_id:
            self.win.after_cancel(self.viz_after_id)

        self.win.after(0, self._show_done)
        play_success()

        # Salvar dados
        filepath = self.app._save_data(self.all_data)

        if filepath:
            self.win.after(0, lambda: self.info_label.config(
                text=f"Salvo: {os.path.basename(filepath)}   |   Clique para fechar",
                fg=THEME["green"]))

        # Botão de fechar
        self.win.after(1500, lambda: self._add_close_button())

        self.app._finish_collection()

    def _add_close_button(self):
        btn = tk.Button(self.center, text="FECHAR E VOLTAR",
                        command=self._close,
                        font=("Segoe UI", 12, "bold"),
                        bg=THEME["accent_dim"], fg=THEME["accent"],
                        activebackground=THEME["accent"], activeforeground="#FFFFFF",
                        relief="flat", cursor="hand2", padx=24, pady=10)
        btn.pack(pady=20)

    def _abort(self):
        self.app.stop_event.set()
        time.sleep(0.3)

        # Insert safety markers so analysis pipeline can detect partial sessions
        try:
            self._insert_marker(MARKERS["block_end"])
            self._insert_marker(MARKERS["session_end"])
        except Exception:
            pass

        # Collect any remaining data in the buffer
        try:
            self._collect_chunk()
        except Exception:
            pass

        try:
            self.board.stop_stream()
        except Exception:
            pass

        self.is_streaming = False

        if self.viz_after_id:
            try:
                self.win.after_cancel(self.viz_after_id)
            except Exception:
                pass

        # Save whatever data was collected
        filepath = self.app._save_data(self.all_data)
        if filepath:
            print(f"[ABORT] Dados parciais salvos: {filepath}")

        self.app._finish_collection()
        self.win.destroy()

    def _close(self):
        self.win.destroy()


# ENTRY POINT

if __name__ == "__main__":
    BoardShim.enable_dev_board_logger()
    app = BCIApp()
    app.run()