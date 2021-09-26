from matplotlib import pylab as plt
import numpy as np
import joblib

plt.rcParams.update({'font.size': 19})


class BenchDrawer:
    def __init__(self, method, several_dots=False):
        if callable(method):
            self._method_name = method.__name__
        else:
            self._method_name = method

        self.__load()
        self.__several_dots = several_dots

    def __load(self):
        self.__data = joblib.load(f"./{self._method_name}/{self._method_name}.p")

    def __get_dof(self, mol_name):
        with open(f"./{self._method_name}/dump_{self._method_name}.log", "r") as f:
            for line in f.readlines():
                if line.startswith(f"calc molecule: {mol_name}"):
                    return int(line.split(", dof = ")[1])

    def get_molecule_data(self, molecule_num, normalized=False):
        data = self.__data[molecule_num]["iterations"]
        mol_name = self.__data[molecule_num]["molecule_name"]
        if normalized:
            dof = self.__get_dof(mol_name)

        x = np.arange(len(data[0]))

        y_rmsd, y_de, time_data, xtbcalls_ = data

        if self.__several_dots:
            if normalized:
                y_rmsd = np.asarray([item / dof for item in y_rmsd])
                y_de = np.asarray([item / dof for item in y_de])

            def get_min_loss(loss_list):
                losses = []
                for k in range(1, len(loss_list) + 1):
                    losses.append(np.min(loss_list[:k]))
                return losses

            def get_mean_loss(loss_list):
                losses = []
                for k in range(1, len(loss_list) + 1):
                    losses.append(np.mean(loss_list[:k]))
                return losses

            y_min_rmsd = get_min_loss(y_rmsd)
            y_min_de = get_min_loss(y_de)
            y_mean_rmsd = get_mean_loss(y_rmsd)
            y_mean_de = get_mean_loss(y_de)

            min_idx_rmsd = np.argmin(y_min_rmsd)
            min_idx_de = np.argmin(y_min_de)


        else:
            if normalized:
                y_rmsd /= dof
                y_de /= dof
            min_idx_rmsd = np.argmin(y_rmsd)
            min_idx_de = np.argmin(y_de)

        rmsd_time_ = 0
        for item in time_data[:min_idx_rmsd]:
            rmsd_time_ += item

        de_time_ = 0
        for item in time_data[:min_idx_de]:
            de_time_ += item

        rmsd_xtbcalls = xtbcalls_[min_idx_rmsd]
        de_xtbcalls = xtbcalls_[min_idx_de]

        if self.__several_dots:
            return mol_name, x, y_rmsd, y_de, y_min_rmsd, y_min_de, y_mean_rmsd, y_mean_de, \
                   min_idx_rmsd, min_idx_de, \
                   rmsd_time_, de_time_, \
                   rmsd_xtbcalls, de_xtbcalls
        else:
            return mol_name, x, y_rmsd, y_de, \
                   min_idx_rmsd, min_idx_de, \
                   rmsd_time_, de_time_, \
                   rmsd_xtbcalls, de_xtbcalls

    def draw_molecule(self, molecule_num, save=False, normalized=False):
        if self.__several_dots:
            mol_name, x, y1, y2, miy1, miy2, mey1, mey2, idx1, idx2, t1, t2, xc1, xc2 = self.get_molecule_data(
                molecule_num, normalized)
        else:
            mol_name, x, y1, y2, idx1, idx2, t1, t2, xc1, xc2 = self.get_molecule_data(molecule_num, normalized)

        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, tight_layout=True)
        fig.set_size_inches(16, 10)
        fig.subplots_adjust(hspace=.2)

        ax1.grid()
        ax2.grid()

        if self.__several_dots:
            for y_group1, y_group2 in zip(y1.T, y2.T):
                ax1.scatter(x, y_group1, c='r')
                ax2.scatter(x, y_group2, c='r')

            ax1.plot(x, miy1, color='b', label='min value')
            ax2.plot(x, miy2, color='b', label='min value')
            ax1.plot(x, mey1, color='g', label='mean value')
            ax2.plot(x, mey2, color='g', label='mean value')

            ax1.annotate(f'{idx1 + 1} iters, {t1:.2f} s, {xc1} calls',
                         xy=(x[idx1], miy1[idx1]), xycoords='data',
                         xytext=(-15, -50), textcoords='offset points',
                         arrowprops=dict(facecolor='black', shrink=0.05),
                         horizontalalignment='right', verticalalignment='bottom')
            ax2.annotate(f'{idx2 + 1} iters, {t2:.2f} s, {xc2} calls',
                         xy=(x[idx2], miy2[idx2]), xycoords='data',
                         xytext=(-15, -100), textcoords='offset points',
                         arrowprops=dict(facecolor='black', shrink=0.05),
                         horizontalalignment='right', verticalalignment='bottom')

        else:
            ax1.scatter(x, y1, c='r')
            ax2.scatter(x, y2, c='r')

            ax1.annotate(f'{idx1 + 1} iters, {t1:.2f} s, {xc1} calls',
                         xy=(x[idx1], y1[idx1]), xycoords='data',
                         xytext=(-15, -50), textcoords='offset points',
                         arrowprops=dict(facecolor='black', shrink=0.05),
                         horizontalalignment='right', verticalalignment='bottom')
            ax2.annotate(f'{idx2 + 1} iters, {t2:.2f} s, {xc2} calls',
                         xy=(x[idx2], y2[idx2]), xycoords='data',
                         xytext=(-15, -100), textcoords='offset points',
                         arrowprops=dict(facecolor='black', shrink=0.05),
                         horizontalalignment='right', verticalalignment='bottom')

        plt.suptitle(f"{mol_name}")

        if self.__several_dots:
            plt.legend()

        if normalized:
            prefix = "norm_"
            ax1.set(ylabel="RMSD per DoF, $\AA$")
            ax2.set(ylabel="dE per DoF, kJ/mol")
        else:
            prefix = ""
            ax1.set(ylabel="RMSD, $\AA$")
            ax2.set(ylabel="dE, kJ/mol")

        plt.xlabel("# iteration")

        if save:
            fig.savefig(f'./{self._method_name}/{prefix}{molecule_num}.png', dpi=300)
            plt.close()
        else:
            fig.show()

    def draw_all_molecules(self, save=False, normalized=False):
        for i in range(len(self.__data)):
            self.draw_molecule(i, save, normalized)

    # ===========================================

    def get_summary_data(self, normalized=False, with_idxs=False):
        name_list = []
        best_rmsd = []
        best_E = []
        best_time = []
        best_xtbc = []
        if with_idxs:
            idxs1 = []
            idxs2 = []

        for i in range(len(self.__data)):
            if self.__several_dots:
                mol_name, x, _, _, miy1, miy2, _, _, idx1, idx2, t1, t2, xc1, xc2 = self.get_molecule_data(i,
                                                                                                           normalized)
                y1 = miy1
                y2 = miy2
            else:
                mol_name, x, y1, y2, idx1, idx2, t1, t2, xc1, xc2 = self.get_molecule_data(i, normalized)

            name_list.append(mol_name)
            best_rmsd.append(y1[idx1])
            best_E.append(y2[idx2])
            best_time.append((t1, t2))
            best_xtbc.append((xc1, xc2))
            if with_idxs:
                idxs1.append(idx1)
                idxs2.append(idx2)

        if with_idxs:
            return name_list, best_rmsd, best_E, best_time, best_xtbc, idxs1, idxs2
        else:
            return name_list, best_rmsd, best_E, best_time, best_xtbc

    def draw_summary(self, save=False, normalized=False):
        name_list, best_rmsd, best_E, best_time, best_xtbc = self.get_summary_data(normalized=normalized,
                                                                                   with_idxs=False)
        fig, axs = plt.subplots(nrows=4, sharex=True, tight_layout=True)
        fig.set_size_inches(20, 20)
        fig.subplots_adjust(hspace=.3)

        for ax in axs:
            ax.grid()

        x = range(len(name_list))

        axs[0].scatter(x, best_rmsd, c='r')
        axs[1].scatter(x, best_E, c='b')
        axs[2].scatter(x, [t[0] for t in best_time], c='r', label="by RMSD")
        axs[2].scatter(x, [t[1] for t in best_time], c='b', label="by dE")
        axs[2].legend()
        axs[3].scatter(x, [cc[0] for cc in best_xtbc], c='r', label="by RMSD")
        axs[3].scatter(x, [cc[1] for cc in best_xtbc], c='b', label="by dE")

        if normalized:
            prefix = "NORM_"
            axs[0].set(ylabel="best RMSD per DoF, $\AA$")
            axs[1].set(ylabel="best dE per DoF, kJ/mol")
        else:
            prefix = ""
            axs[0].set(ylabel="best RMSD, $\AA$")
            axs[1].set(ylabel="best dE, kJ/mol")
        axs[2].set(ylabel="best time, s")
        axs[3].set(ylabel="best XTB call count")

        plt.xlabel("# molecule")

        if save:
            fig.savefig(f'./{self._method_name}/{prefix}SUMMARY.png', dpi=300)
            plt.close()
        else:
            fig.show()

    # ===========================================

    def get_method_statistics(self, normalized=False, with_idxs=False):
        if with_idxs:
            _, best_rmsd, best_E, best_time, best_xtbc, idxs1, idxs2 = self.get_summary_data(normalized=normalized,
                                                                                             with_idxs=with_idxs)
            idxs_data = {"i_RMSD": (np.mean(idxs1), np.std(idxs1), np.min(idxs1), np.max(idxs1)),
                         "i_dE": (np.mean(idxs2), np.std(idxs2), np.min(idxs2), np.max(idxs2))}
        else:
            _, best_rmsd, best_E, best_time, best_xtbc = self.get_summary_data(normalized=normalized,
                                                                               with_idxs=with_idxs)

        ret_data = {"RMSD": (np.mean(best_rmsd), np.std(best_rmsd), np.min(best_rmsd), np.max(best_rmsd)),
                    "dE": (np.mean(best_E), np.std(best_E), np.min(best_E), np.max(best_E)),
                    "time": (np.mean(best_time), np.std(best_time), np.min(best_time), np.max(best_time)),
                    "xtbcc": (np.mean(best_xtbc), np.std(best_xtbc), np.min(best_xtbc), np.max(best_xtbc))}

        if with_idxs:
            ret_data.update(idxs_data)

        return ret_data

    # ===========================================
