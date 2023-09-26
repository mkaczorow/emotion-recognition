import os
import pickle
import threading
import wave
from tkinter import *
from tkinter import messagebox
from tkinter import ttk, filedialog
from tkinter.filedialog import askopenfile
from tkinter.filedialog import asksaveasfilename
from matplotlib.figure import Figure
import numpy as np
import pyaudio
import librosa
import matplotlib.pyplot as plt
from numpy import ndarray
from sklearn.mixture import GaussianMixture
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)


class Audio:
    def __init__(self):
        self.rozpocznij_nagrywanie = False
        self.frames = []
        self.frames_save = []
        self.format = pyaudio.paInt16
        self.channels = 1
        self.fs = 16000 
        self.frames_per_buffer = 1024
        self.filename = ''
        self.filetypes = ['*.wav', "*.mp3", "*.flac", "*.m4a"]
        
    def nagraj_audio(self):
        self.rozpocznij_nagrywanie = True
        t = threading.Thread(target=self.nagrywanie)
        t.start()

    def nagrywanie(self):
        self.frames = []
        stream = pyaudio.PyAudio().open(format=self.format, channels=self.channels, rate=self.fs, input=True,
                                        frames_per_buffer=self.frames_per_buffer)
        while self.rozpocznij_nagrywanie:
            data = stream.read(self.frames_per_buffer)
            numpydata = np.frombuffer(data, dtype=np.int16)
            self.frames.append(numpydata)

        stream.stop_stream()
        stream.close()
        pyaudio.PyAudio().terminate()
        self.frames_save = self.frames
        self.frames = np.hstack(self.frames)
        self.frames = np.array(self.frames).astype(np.float32)

    def zakoncz_audio(self):
        self.rozpocznij_nagrywanie = False

    def znajdz_audio(self):
        self.frames = np.array([])
        print(type(self.frames))
        file = filedialog.askopenfilename(filetypes=[("Audio files", self.filetypes)])
        if file:
            if os.path.exists(file):
                self.frames, fs = librosa.load(file, sr=self.fs)
                print(self.frames)
        # self.frames = np.array(self.frames).astype(np.float32)

    def zapisz_audio(self):
        self.filename = asksaveasfilename(title="Save as",
                                          filetypes=[("wav", '*.wav'), ("mp3", '*.mp3'), ("m4a", '*.m4a'), ("flac", '*.flac')],
                                          defaultextension=".wav")
        obj = wave.open(self.filename, 'wb')
        obj.setnchannels(self.channels)
        obj.setsampwidth(pyaudio.PyAudio().get_sample_size(self.format))
        obj.setframerate(self.fs)
        obj.writeframes(b''.join(self.frames_save))
        obj.close()


class F0:
    def __init__(self):
        self.fs = 16000
        self.frame_length = 100
        self.f0_cale = np.empty([0])
        self.frame_length_in_samples = 0
        self.f0 = []
        self.times = []

    def detect_f0(self, frames):

        self.frame_length_in_samples =self.fs * self.frame_length / 1000
        hop_length = self.frame_length_in_samples / 4
        f0, voiced_flag, voiced_probs = librosa.pyin(
            frames,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C6"),
            sr=self.fs,
            frame_length=int(self.frame_length_in_samples),
            win_length=None,
            hop_length=int(hop_length),
            n_thresholds=100,
            beta_parameters=(2, 18),
            boltzmann_parameter=2,
            resolution=0.1,
            max_transition_rate=35.92,
            switch_prob=0.01,
            no_trough_prob=0.01,
            fill_na=np.nan,
            center=True,
            pad_mode="constant",
        )
        self.f0_cale = f0
        return f0, voiced_flag

    def wyswietl_f0(self, frames):
        self.f0, voiced_flag = self.detect_f0(frames)
        self.times = librosa.times_like(self.f0, sr=self.fs, hop_length=(self.frame_length_in_samples / 4))
        return self.times, self.f0

    def zapisz_f0(self):
        filename = asksaveasfilename(initialfile='Untitled.png', defaultextension=".png",
                                     filetypes=[("png", "*.png"), ("jpg", "*.jpg")])
        plt.plot(self.times, self.f0)
        plt.ylabel("Częstotliwość [Hz]")
        plt.xlabel("Czas [s]")
        plt.title("F0")
        plt.savefig(filename)


class Klasyfikacja(F0):

    def __init__(self):
        super().__init__()

    def extract_mfcc(self, file_path: str or ndarray, fs=16000, frame_length=100) -> ndarray:

        if os.path.exists(file_path):
            y, sr = librosa.load(file_path, sr=fs)
        elif type(file_path) == ndarray and type(fs) == int:
            y = file_path
            # y = librosa.resample(file_path, orig_sr=fs, target_sr=4000)

        frame_length_in_samples = fs * frame_length / 1000
        hop_length = frame_length_in_samples / 4
        MFCC = librosa.feature.mfcc(
            y=y, n_fft=int(frame_length_in_samples), hop_length=int(hop_length),
        )
        return MFCC

    def feature(self, file_path: str, fs=16000) -> ndarray:
        mfcc = self.extract_mfcc(file_path, fs, frame_length=100)
        delta_1 = librosa.feature.delta(mfcc)
        delta_2 = librosa.feature.delta(mfcc, order=2)
        features = np.hstack((mfcc, delta_1, delta_2)).T
        return features

    def classify(self, filename_path: str or ndarray, gmm_models: list, fs=16000):

        feature_proba = self.feature(filename_path, fs)
        class_name = ["kobieta", "mężczyzna"]

        audio_scores_gender = []
        for i in range(0, len(gmm_models)):
            audio_scores_gender.append(gmm_models[i].score(feature_proba))
        # print(audio_scores_gender)
        i_max = np.argmax(audio_scores_gender)
        class_prediction = class_name[i_max]

        # print(f' predykcja = {class_name[i_max]}')
        return audio_scores_gender, class_prediction


    def classify_intonation(
            self, filename_path: str or ndarray, gmms_model: list, fs=4000
    ) -> tuple[list, int]:

        f0, voiced_flag= self.detect_f0(filename_path)

        indx = [i for i, vf in enumerate(voiced_flag) if vf]
        f0 = f0[indx]
        f0 = f0 - np.mean(f0)
        new_f0_test = f0.reshape(-1, 1)

        if len(f0) == 0:
            class_prediction = -1
            audio_scores = []
        else:
            # class_name = ["płaska intonacja", "duża intonacja", "normalna intonacja"]
            class_name = ["STAN NEUTRALNY", "RADOŚĆ", "SMUTEK", "ST", "ZDZIWIENIE"]
            audio_scores = []
            for i in range(0, len(gmms_model)):
                audio_scores.append(gmms_model[i].score(new_f0_test))
            # print(audio_scores)
            i_max = np.argmax(audio_scores)
            class_prediction = class_name[i_max]
            # print(f'predykcja = {class_name[i_max]}')
        return audio_scores, class_prediction

    def recognize_intonation(self, wav: str or ndarray, gender_gmm: list[GaussianMixture],
                             gmm_female: list[GaussianMixture], gmm_male: list[GaussianMixture],
                             fs=4000, fs2=16000):
        audio_scores_gender, class_prediction_gender = self.classify(
            wav, gender_gmm, fs2
        )
        if class_prediction_gender == "kobieta":
            audio_scores, class_prediction_intonation = self.classify_intonation(
                wav, gmm_female, fs
            )
        else:  # class_prediction_gender == "m"
            audio_scores, class_prediction_intonation = self.classify_intonation(
                wav, gmm_male, fs
            )

        print(audio_scores_gender, class_prediction_gender)
        print(audio_scores, class_prediction_intonation)
        return (
            audio_scores_gender,
            class_prediction_gender,
            audio_scores,
            class_prediction_intonation,
        )


class GUI:
    def open_file(self, filename):
        with open(filename, "rb") as handle:
            return pickle.load(handle)

    def __init__(self):
        self.audio = Audio()
        self.f0 = F0()
        self.klasyfikacja = Klasyfikacja()
        self.gmms = self.open_file("modele/gmms")  # model do płci
        self.gmms_female = self.open_file("modele/gmms_F_cms")   #model emocje kobiety
        self.gmms_male = self.open_file("modele/gmms_M_cms")  #mmodel emocje mezczyzn
        self.window = Tk()
        self.window.title("Aplikacja do rozpoznawania emocji")
        self.window.geometry("900x500")
        self.gender = ""
        self.intonation = ""
        self.intonation_label = Label(self.window,
                          text=self.intonation)
        self.gender_label = Label(self.window,
                                      text=self.gender)
        self.fig = Figure(figsize=(5, 3.7),
                     dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.get_tk_widget().configure(bg='#C0CFB8')
        self.window['background'] = '#C0CFB8'
    def show_msg(self):
        messagebox.showinfo("Message", "Hey There! I hope you are doing well.")  # do wywalenia

    def wyswietl_emocje_text(self):
        audio_scores_gender, self.gender, audio_scores, self.intonation = self.klasyfikacja.recognize_intonation(
            wav=self.audio.frames, gender_gmm=self.gmms,
            gmm_female=self.gmms_female, gmm_male=self.gmms_male,
            fs=4000, fs2=16000)
        self.gender_label.config(text = self.gender)
        self.intonation_label.config(text=self.intonation)

    def plot_f0(self):
        self.fig.clear()
        times, f0 = self.f0.wyswietl_f0(self.audio.frames)
        plot1 = self.fig.add_subplot(111)
        plot1.clear()
        self.canvas.draw()
        plot1.plot(times, f0)
        # self.fig.subplots_adjust(top=-1)
        plot1.set_position([0.15, 0.15, 0.8, 0.75])
        plot1.set_ylabel("Częstotliwość [Hz]")
        plot1.set_xlabel("Czas [s]")
        plot1.set_title("F0")
        self.canvas.get_tk_widget().pack(side=BOTTOM)
        self.canvas.draw()

        # toolbar = NavigationToolbar2Tk(self.canvas,
        #                                self.window)
        # toolbar.update()
        self.canvas.draw_idle()

        # canvas.get_tk_widget().pack()

    def buttons(self):
        wprowadz_audio = Button(text='wprowadź audio', command=self.audio.znajdz_audio)
        wprowadz_audio.place(x=100, y=60, anchor=CENTER)
        rozpocznij_nagrywanie = Button(text='rozpocznij nagrywanie', command=self.audio.nagraj_audio)
        rozpocznij_nagrywanie.place(x=100, y=120, anchor=CENTER)
        zakoncz_nagrywanie = Button(text='zakończ nagrywanie', command=self.audio.zakoncz_audio)
        zakoncz_nagrywanie.place(x=100, y=150, anchor=CENTER)
        zapisz_audio = Button(text='zapisz audio', command=self.audio.zapisz_audio)
        zapisz_audio.place(x=100, y=200, anchor=CENTER)
        wyswietl_f0 = Button(text='wyświetl kontur F0', command=self.plot_f0)
        wyswietl_f0.place(x=100, y=260, anchor=CENTER)
        zapisz_f0 = Button(text='zapisz kontur F0', command=self.f0.zapisz_f0)
        zapisz_f0.place(x=100, y=290, anchor=CENTER)
        wyswietl_emocje = Button(text='wyświetl emocje', command=self.wyswietl_emocje_text)
        wyswietl_emocje.place(x=100, y=350, anchor=CENTER)

        self.gender_label = Label(self.window,
                          text=self.gender, width=20, height=1, bg='#b8c0cf')
        self.gender_label.place(x=480,y=60, anchor=CENTER)
        self.intonation_label = Label(self.window,
                          text=self.intonation, width=20, height=1, bg='#b8c0cf')
        self.intonation_label.place(x=480,y=90, anchor=CENTER)
        text_box = Label(self.window,text = 'Płeć: ', width=7, height=1, bg='#b8c0cf')
        text_box.place(x = 400, y = 60, anchor = CENTER)
        text_box1 = Label(self.window, text='Emocja: ', width=7, height=1, bg='#b8c0cf')
        text_box1.place(x=400, y=90, anchor=CENTER)
        self.window.mainloop()

gui = GUI()
gui.buttons()
