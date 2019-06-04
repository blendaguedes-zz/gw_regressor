# Python script for simulation the gravitational waves.
# Written by Professor Antonio de Padua Santos, Octocber 2018;
# Written by student Blenda Guedes, Octuber 2018;
import math
import numpy as np


class GWGeneratorWiki(object):

    def calculate_mass_ratio(self, m1, m2):
        return (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5)

    def wave_form(self, M):

        c = 3.0e8  # Light velocity in m/s
        G = 6.674e-11  # Gravity constant in m^3 / kg / s^(-2)
        P0 = 0.5  # Initial period
        fi = 0.0  # Initial angle
        dt = 0.1  # time increment
        t = 0
        h1 = 0.0
        Amp = 0.0
        phi = 0.0
        k = (96.0 / 5.0) * (2.0 * math.pi) ** (8.0 / 3.0) * (G * M / c ** 3.0) ** (5.0 / 3.0)
        r = 410000  # in kpc  # Luminosity distance
        x = 2.0 * (G * M) ** (5 / 3) / (r * c ** 4)
        ampl = []
        time = []
        strain = []
        print('the start')
        print(3 * P0 ** (8 / 3) / (8 * k))
        while t < (3 * P0 ** (8 / 3) / (8 * k)):
            ampl.append(Amp)
            strain.append(h1 / 1e23)
            time.append(t / 35)

            pgw = 0.5 * (P0 ** (8 / 3) - (8 / 3) * k * t) ** (3 / 8)
            Amp = x * (math.pi / pgw) ** (2 / 3)

            phi = fi - (4.0 / 5.0) * math.pi * (3.0 * P0 ** (8 / 3) - 8.0 * k * t) / (
                    k * (P0 ** (3 / 8) - (8 / 3) * k * t) ** (3 / 8))
            h1 = h1 + Amp * math.cos(phi)
            t = t + dt
        print('the end')
        return strain

    #normaliza uma lista entre 0 e 1
    def normalizer(self, data):
        data_result = []
        max_data = max(data)
        min_data = min(data)
        for i in data:
            data_result.append((i-min_data)/(max_data-min_data))
        return data_result


    def normalizer_list_of_list(self, data):
        data_result = []
        for d in data:

            max_data = max(d)
            min_data = min(d)
            aux_list = []
            for i in d:
                aux_list.append((i-min_data)/(max_data-min_data))
            data_result.append(aux_list)

        return data_result

    #Metodo usado para a criacao das relacoes de massa
    def masses(self):
        result_list = []
        masses = []
        for i in np.arange(1, 10, 0.05):
            for j in np.arange(i, 10, 0.05):
                calc = self.calculate_mass_ratio(j*1.989e30,i*1.989e30)
                if calc not in result_list:
                    result_list.append(calc)
                    masses.append([j,i])

        return masses


    def normalize_masses(self, masses):
        max_m = max(max(masses))
        min_m = min(min(masses))
        data_result = []

        for m in masses:
            first_m = (m[0]-min_m)/(max_m-min_m)
            next_m = (m[1]-min_m )/ (max_m-min_m)
            data_result.append([first_m, next_m])
        return data_result


    def write_masses_file(self, masses):
        waves = open('masses', 'w')
        for w in range(len(masses)):
            waves.writelines(["%s " % str(item) for item in masses[w]])
            waves.write('\n')


    def relations(self):
        result_list = []
        for i in np.arange(1, 10, 0.05):
            for j in np.arange(i, 10, 0.05):
                calc = self.calculate_mass_ratio(j*1.989e30,i*1.989e30)
                if calc not in result_list:
                    result_list.append(calc)

        return result_list

    def make_wave_form_data(self, dado):
        list_wave_form = []
        for d in dado:
            list_wave_form.append(self.wave_form(d))
        return list_wave_form

    def store_wave_form(self, relations):
        list_wave_form = []
        for d in relations:
            list_wave_form.append(self.wave_form(d))
        return list_wave_form

    def write_wave_form_data(self, wave_data):
        waves = open('waves_100_histograma', 'w')
        for w in range(len(wave_data)):
            waves.writelines(["%s " % str(item) for item in wave_data[w]])
            waves.write('\n')


    def read_data(self, file):
        data = open(file).readlines()
        data_str_list = list(data)
        wave_froms = []
        for dl in data_str_list[0:-1]:
            aux_list = dl.split(' ')
            wave_froms.append([float(i) for i in aux_list[0:-1]])

        return wave_froms

    def read_data_v2(self, file):

        with open(file, 'r') as f:
            wave_froms = []
            for line in f:
                data_str_list = line.split(' ')
                aux_list = data_str_list[:-1]
                wave_froms.append([float(i) for i in aux_list[0:-1]])

        return wave_froms

    def read_mass(self):
        data = open('masses').readlines()
        data_str_list = list(data)
        masses_list = []
        for dl in data_str_list[0:-1]:
            aux_list = dl.split(' ')
            masses_list.append([float(i) for i in aux_list[0:-1]])

        return masses_list


    def make_histogram(self, waves):
        import numpy as np
        waves_normalized = []
        for w in waves:
            n, bins = np.histogram(w, bins=300, range=None, normed=None, weights=None, density=None)
            waves_normalized.append(n)

        return waves_normalized
    def make_histogram2(self, wave):
        import numpy as np

        hist, bins = np.histogram(wave, bins=300, range=None, normed=None, weights=None, density=None)

        return hist
