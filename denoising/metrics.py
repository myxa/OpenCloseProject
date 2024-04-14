import numpy as np
from typing import List
from numpy.typing import NDArray


class ICC:
    def __init__(self, data: List[NDArray], mask_file=None):
        """
        Class to calculate ICC and BSS.

        Parameters
        ----------
        data: list
            list of data for every session to compute metrics
            one session - np.array, vector of functional connectivity
            Note! for BSS calculation only two sessions should be passed
        mask_file: None or np.array, optional
            mask to be used to exclude zero elements (for simulations)

        """
        self.n_ses = len(data)
        self.n_sub = len(data[0])

        self.thrsh = 0.2

        if mask_file is None:
            mask = [d > self.thrsh for d in data]
            self.data = [data[i][mask[i]] for i in range(self.n_ses)]
        else:
            self.mask = [np.array([mask_file for _ in range(self.n_sub)]) for _ in range(self.n_ses)]
            self.data = [data[i][mask[i]] for i in range(self.n_ses)]


    @property
    def _avg_vec(self):
        return np.mean(
            np.concatenate(self.data), axis=0)
    

    def _bms(self) -> float:
        """
        Calculates the sum of squared between-subj variance,
        the average subject value subtracted from overall avg of values

        """
        # group vecs by subject
        sub_vec = [[self.data[ses][sub] for ses in range(self.n_ses)] for sub in range(self.n_sub)]
        
        return np.sum([(self._avg_vec - 
                        np.mean(sub_vec[sub], axis=0)) **2 for sub in range(self.n_sub)]) * self.n_ses


    def _wms(self) -> float:
        """
        calculates the sum of squared Intra-subj variance,
        the average session value subtracted from overall avg of values

        """
        return np.sum([(self._avg_vec - 
                        np.mean(self.data[ses], axis=0)) **2 for ses in range(self.n_ses)]) * self.n_sub


    def icc(self) -> float:
        """
        icc metric

        """
        return ((self._bms() - self._wms()) / 
                (self._bms() + (self.n_ses - 1) * self._wms()))
    


def bss(a: List) -> float:
    """
    BSS calculation

    parameters
    ----------
    a: list
        data of two sessions
        
    returns
    -------
    array of BSS for every subject
    """
    ans = np.zeros(len(a[0]))
    for i in range(len(a[0])):
        ans[i] = np.corrcoef(a[0][i], a[1][i])[0, 1]
    return ans
