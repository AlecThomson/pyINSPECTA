from __future__ import annotations

__author__ = ["Danny Price", "Alec Thomson", "Lawrence Toomey"]

# Ignore astropy warnings
import warnings

import h5py
import pandas as pd

warnings.filterwarnings("ignore", category=Warning, append=True)


class SDHDFTable:
    def __init__(self, sdhdf_dataset: h5py.Dataset, version):
        """Read an SDHDF table

        Args:
            sdhdf_dataset (h5py.Dataset): SDHDF table dataset
        """
        self.attrs = dict(sdhdf_dataset.attrs)
        if float(version) < 4.0:
            compound_attr = False
        elif "frequency" in sdhdf_dataset.name:
            compound_attr = True
        else:
            compound_attr = False
        if compound_attr:
            self.table = self._decode_compound_attr(sdhdf_dataset)
        elif "_data" in sdhdf_dataset.name:
            self.table = pd.DataFrame((self.attrs.keys(), self.attrs.values()))
        else:
            self.table = self._decode_df(pd.DataFrame(sdhdf_dataset[:]))

    def __repr__(self):
        return self.table.__repr__()

    def __str__(self):
        return self.table.__str__()

    def _repr_html_(self):
        return self.table._repr_html_()

    def __getitem__(self, key):
        return self.table.__getitem__(key)

    def __setitem__(self, key, value):
        return self.table.__setitem__(key, value)

    def __len__(self):
        return self.table.__len__()

    def __iter__(self):
        return self.table.__iter__()

    def __contains__(self, key):
        return self.table.__contains__(key)

    @staticmethod
    def _decode_df(df: pd.DataFrame) -> pd.DataFrame:
        """Decode a pandas dataframe to a string"""
        str_df = df.select_dtypes([object])
        str_df = str_df.stack().str.decode("utf-8").unstack()  # noqa: PD010, PD013
        for col in str_df:
            df[col] = str_df[col]
        return df

    @staticmethod
    def _decode_compound_attr(dset_obj):
        """Decode compound attributes"""
        keys = dset_obj.attrs.keys()

        key_list = []
        val_list = []

        for key in keys:
            attr = dset_obj.attrs[key]
            if len(dset_obj[:]) == 1 and "frequency" not in dset_obj.name:
                if key in ("SDHDF_CLASS", "SDHDF_DESCRIPTION"):
                    value = attr[0][2].decode()
                elif key in dset_obj.dtype.fields:
                    value = dset_obj[key][0]
                else:
                    value = attr[0][2].decode()
            elif len(dset_obj[:].dtype) == 0:
                if key not in dset_obj[:]:
                    if key in ("DIMENSION_LABELS", "REFERENCE_LIST", "DIMENSION_LIST"):
                        value = attr
                    elif key == "CLASS":
                        value = attr.decode()
                    else:
                        value = attr[0][2].decode()
            elif key in dset_obj.dtype.fields:
                value = dset_obj[key][0]
            else:
                value = attr[0][2].decode()

            key_list.append(key)
            val_list.append(value)

        return pd.DataFrame([val_list], columns=[key_list])
