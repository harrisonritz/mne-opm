"""Basic class to load Cerca cMEG data into a MNE Raw array

Lukas Rier (lukas.rier{at}nottingham.ac.uk) 2023
"""

import os
import json

import pandas
import numpy as np
import mne
from mne.io.constants import FIFF


class cMEGRaw(mne.io.RawArray):
    def __init__(self, fname):
        # split into folder name and file name
        file_path_split = os.path.split(fname)
        filename = file_path_split[1]
        foldername = file_path_split[0] + "/"

        filename = self._cmeg_concatenate(foldername, filename)

        cmeg_data = self._read_cMEG_data(foldername + filename)

        # time = cmeg_data[0, :]
        channel_data = cmeg_data[1:, :]
        del cmeg_data

        fname_pre = filename.split("_meg.cMEG")[0]
        with open(foldername + fname_pre + "_meg.json") as f:
            json_info = json.load(f)
        sfreq = json_info["SamplingFrequency"]

        tsv_file_dataframes = {
            "channels": pandas.read_csv(
                foldername + fname_pre + "_channels.tsv", sep="\t"
            ),
            "HelmConfig": pandas.read_csv(
                foldername + fname_pre + "_HelmConfig.tsv", sep="\t"
            ),
            "SensorTransform": pandas.read_csv(
                foldername + fname_pre + "_SensorTransform.tsv",
                header=None,
                sep="\t",
            ),
        }
        # Extend channel TSV with helmet information:
        #    remove square brackets first
        tsv_file_dataframes["channels"]["name"] = tsv_file_dataframes["channels"][
            "name"
        ].str.replace(r"\[X\]$", "X", regex=True)
        tsv_file_dataframes["channels"]["name"] = tsv_file_dataframes["channels"][
            "name"
        ].str.replace(r"\[Y\]$", "Y", regex=True)
        tsv_file_dataframes["channels"]["name"] = tsv_file_dataframes["channels"][
            "name"
        ].str.replace(r"\[Z\]$", "Z", regex=True)

        # remove channel axis from trigger names completely
        is_trigger = tsv_file_dataframes["channels"]["type"] == "TRIG"
        tsv_file_dataframes["channels"].loc[is_trigger, "name"] = (
            tsv_file_dataframes["channels"]
            .loc[is_trigger, "name"]
            .str.replace(r" Z$", "", regex=True)
        )
        # remove x y or z from helmet location names
        # Deals with various versions of cMEG helmet config tsv headers and sensor name formats
        tsv_file_dataframes["HelmConfig"]["Name"] = tsv_file_dataframes["HelmConfig"][
            "Name"
        ].str.replace(r"[\[\]]", "", regex=True)
        tsv_file_dataframes["HelmConfig"]["Name"] = tsv_file_dataframes["HelmConfig"][
            "Name"
        ].str.replace(r"\s?[XYZ]\s*$", "", regex=True)

        if "nT0x2FV" in tsv_file_dataframes["channels"].columns:
            tsv_file_dataframes["channels"].rename(
                columns={"nT0x2FV": "nT/V"}, inplace=True
            )
        if "V0x2FnT" in tsv_file_dataframes["channels"].columns:
            tsv_file_dataframes["channels"].rename(
                columns={"V0x2FnT": "nT/V"}, inplace=True
            )
        if "nT/V" in tsv_file_dataframes["channels"].columns:
            tsv_file_dataframes["channels"].rename(
                columns={"nT/V": "V/nT"}, inplace=True
            )
        if "V/nT" not in tsv_file_dataframes["channels"].columns:
            raise ValueError("No V/nT column in channel tsv file detected!")

        # merge helmet info into channel dataframe
        channel_info_df = tsv_file_dataframes["channels"]
        helmet_info_df = tsv_file_dataframes["HelmConfig"]
        helmet_info_df.rename(
            columns={"Name": "Helmet_location", "Sensor": "name"}, inplace=True
        )

        channel_info_df = pandas.merge(
            channel_info_df, helmet_info_df, on="name", how="left"
        )

        is_not_trigger = channel_info_df["type"] != "TRIG"
        channel_info_df["Helmet_location"] = channel_info_df["Helmet_location"].fillna(
            "NoSlot"
        )
        channel_info_df["raw_info_names"] = np.nan
        channel_info_df.loc[is_not_trigger, "raw_info_names"] = (
            channel_info_df.loc[is_not_trigger, "Helmet_location"]
            + " "
            + channel_info_df.loc[is_not_trigger, "name"]
        )
        is_not_megmag = channel_info_df["type"] != "MEGMAG"
        channel_info_df.loc[is_not_megmag, "raw_info_names"] = channel_info_df.loc[
            is_not_megmag, "name"
        ]
        # %% Sensor information
        print("Sorting Sensor Information...\n")

        ch_scale = pandas.Series.to_list(channel_info_df["V/nT"])

        # Deal with countries having 'comma' decimal points
        ch_scale = [
            float(x.replace(",", ".")) if isinstance(x, str) else x for x in ch_scale
        ]

        ch_names = pandas.Series.tolist(channel_info_df["raw_info_names"])
        ch_types = [None] * len(ch_names)

        print("Scaling data to SI Units...")

        data = np.empty((len(ch_names), channel_data.shape[1]))

        for count, ch_type in enumerate(pandas.Series.tolist(channel_info_df["type"])):
            if ch_type.replace(" ", "") == "MEGMAG":
                ch_types[count] = "mag"
                # convert mag channels to T
                data[count, :] = 1e-9 * channel_data[count, :] / ch_scale[count]
            elif ch_type.replace(" ", "") == "TRIG":
                ch_types[count] = "stim"
                # Trigger channels stay as Volts
                data[count, :] = channel_data[count, :]
            elif ch_type.replace(" ", "") == "MISC":
                ch_types[count] = "stim"
                data[count, :] = channel_data[count, :]  # BNC channels stay as Volts

        # %% Create MNE info object
        print("\nCreating MNE Info\n")
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        info["line_freq"] = json_info["PowerLineFrequency"]
        info["device_info"] = {}
        info["device_info"]["type"] = "Cerca"
        info["device_info"]["model"] = "cMEG"

        # %% Sort sensor locations
        print("Getting sensor Location Information\n")
        # Get position and orientation vectors into a new column
        channel_info_df["pos"] = channel_info_df.apply(
            lambda row: [float(row["Px"]), float(row["Py"]), float(row["Pz"])], axis=1
        )
        channel_info_df["ori"] = channel_info_df.apply(
            lambda row: [float(row["Ox"]), float(row["Oy"]), float(row["Oz"])], axis=1
        )

        nmeg = nstim = nref = 0
        channel_info_df.reset_index()  # ensures index pairs with number of rows

        # Non unit length orientation vectors mean secondary sensor calibration.
        # Gain is multiplied by length of orientation vector to adjust gain, then orientations are normalised
        gain_correction_factor = np.ones((data.shape[0], 1))

        for index, channel_info in channel_info_df.iterrows():
            pos = channel_info["pos"]
            ori = channel_info["ori"]

            # create the channel information
            if sum(np.isnan(pos)) == 0:
                gain_correction_factor[index] = np.linalg.norm(ori)

                r0 = pos.copy()
                ez = ori.copy()
                ez = ez / np.linalg.norm(ez)
                ex, ey = calc_tangent(ez)
                channel_info["loc"] = np.concatenate([r0, ex, ey, ez])

            match channel_info["type"].replace(" ", ""):
                case "TRIG":  # its a trigger!
                    nstim += 1
                    update_kwargs = dict(
                        logno=nstim,
                        coord_frame=FIFF.FIFFV_COORD_UNKNOWN,
                        kind=FIFF.FIFFV_STIM_CH,
                        unit=FIFF.FIFF_UNIT_V,
                        cal=1.0,
                    )

                case "MISC":  # its a BNC channel
                    nref += 1
                    update_kwargs = dict(
                        logno=nref,
                        coord_frame=FIFF.FIFFV_COORD_UNKNOWN,
                        kind=FIFF.FIFFV_STIM_CH,
                        unit=FIFF.FIFF_UNIT_V,
                        cal=1.0,
                    )
                case "MEGMAG":
                    if sum(np.isnan(pos)) == 3:  # its a sensor with no location info
                        nref += 1
                        update_kwargs = dict(
                            logno=nref,
                            coord_frame=FIFF.FIFFV_COORD_UNKNOWN,
                            kind=FIFF.FIFFV_REF_MEG_CH,
                            unit=FIFF.FIFF_UNIT_T,
                            coil_type=FIFF.FIFFV_COIL_QUSPIN_ZFOPM_MAG2,
                            cal=1.0,
                        )
                    else:  # its a proper sensor!
                        nmeg += 1
                        update_kwargs = dict(
                            logno=nmeg,
                            # set coordinate system of fif to device
                            coord_frame=FIFF.FIFFV_COORD_DEVICE,
                            kind=FIFF.FIFFV_MEG_CH,
                            unit=FIFF.FIFF_UNIT_T,
                            coil_type=FIFF.FIFFV_COIL_QUSPIN_ZFOPM_MAG2,
                            loc=channel_info["loc"],
                            cal=1.0,
                        )

            info["chs"][index].update(**update_kwargs)

        # add device to head transform
        dev_to_dig_transform = pandas.DataFrame(
            tsv_file_dataframes["SensorTransform"]
        ).to_numpy()
        info["temp"] = {"dev_digitisation_transform": dev_to_dig_transform}
        # finally build raw object
        print("Creating raw object\n")

        # apply gain correction
        data = data / gain_correction_factor

        super().__init__(data, info)

        self._add_montage(foldername)

    def _read_cMEG_data(self, filepath):
        """Read data from single cMEG file

        Args:
            filepath (str): full path to .cMEG file

        Returns:
            NDArray: numpy array containing channel data
        """
        size = os.path.getsize(filepath)  # Find its byte size
        array_conv = np.array([2**32, 2**16, 2**8, 1])  # header conversion table
        arrays = []
        with open(filepath, "rb") as fid:
            while fid.tell() < size:
                # Read the header of the array which gives its dimensions
                Nch = np.fromfile(
                    fid, ">u1", sep="", count=4
                )  # 4 unsigned 8-bit integers
                N_samples = np.fromfile(
                    fid, ">u1", sep="", count=4
                )  # 4 unsigned 8-bit integers
                # Multiply by convertion array
                dims = np.array(
                    [np.dot(array_conv, Nch), np.dot(array_conv, N_samples)]
                )
                # Read the array and shape it to dimensions given by header
                array = np.fromfile(fid, ">f8", sep="", count=dims.prod())
                arrays.append(array.reshape(dims))

            cmeg_data = np.concatenate(arrays, axis=1)
        return cmeg_data

    def _cmeg_concatenate(self, foldername, fname):
        """Concatenate cMEG files if needed.

        Args:
            foldername (str): path to folder containin cMEG files
            fname (str): user selected cMEG file

        Returns:
            str: filename of concatenated cMEG file
        """
        all_files = os.listdir(foldername)
        if "_meg.cMEG" not in fname:
            fname_new = fname[0 : fname.find("_meg_")] + "_meg.cMEG"
        else:
            fname_new = fname

        if fname_new in all_files:
            print("Data already concatenated\n")

        else:
            open(foldername + fname_new, "a").close()
            for nfile in all_files:
                print(nfile)
                if ".cMEG" in nfile:
                    print(f"Writing {nfile}")
                    with (
                        open(foldername + nfile, "rb") as myfile1,
                        open(foldername + fname_new, "ab") as myfilenew,
                    ):
                        myfilenew.write(myfile1.read())

        return fname_new

    def _add_montage(self, cmeg_folder):
        xyz_files = []
        digitisation_path = ""
        for filename in os.listdir(cmeg_folder):
            if filename.endswith(".xyz"):
                xyz_files.append(filename)
        if len(xyz_files) > 1:
            print(
                "Too many .xyz files found. Only one can be considered!\n",
            )
            raise ValueError("Too many .xyz files found. Only one can be considered")

        if len(xyz_files) == 1:
            digitisation_path = cmeg_folder + xyz_files[0]

        if os.path.isfile(digitisation_path):
            print("Adding digitisation info...\n")
            ch_pos = dict()
            for ch in self.info["chs"]:
                pos1 = ch["loc"][0:3]
                if sum(np.isnan(pos1)) == 0:
                    ch_pos[ch["ch_name"]] = pos1

            # Load coreg points
            hsp = pandas.DataFrame(
                pandas.read_table(
                    digitisation_path,
                    skiprows=2,
                    delim_whitespace=True,
                    names=["x", "y", "z"],
                )
            ).to_numpy()
            fids = np.array((hsp[-3, :], hsp[-2, :], hsp[-1, :]))

            mtg = mne.channels.make_dig_montage(
                # ch_pos=ch_pos,
                nasion=fids[0, :],
                lpa=fids[1, :],
                rpa=fids[2, :],
                hsp=hsp,
                hpi=None,
                coord_frame="unknown",
            )

            dev_to_dig_transform = self.info["temp"]["dev_digitisation_transform"]
            digitisation_to_head_transform = mne.transforms.get_ras_to_neuromag_trans(
                nasion=fids[0, :], lpa=fids[1, :], rpa=fids[2, :]
            )
            self.info["dev_head_t"] = mne.transforms.Transform(
                "meg", "head", digitisation_to_head_transform @ dev_to_dig_transform
            )

            self.set_montage(mtg)
            mne.viz.plot_alignment(self.info, dig=True, coord_frame="meg")

    def set_events_annotations(self):
        # Create events
        stm_misc_chans = mne.pick_types(self.info, stim=True, misc=True)
        data = self.get_data(stm_misc_chans)
        trig_data = 1 * np.array(data > 0.5)
        trig_ID, on_inds = np.where(np.diff(trig_data, axis=1) == 1)
        if len(trig_ID) > 0:
            events = np.concatenate(
                [
                    np.expand_dims(on_inds, axis=1) + 1,
                    np.expand_dims(np.zeros(np.shape(on_inds)), axis=1),
                    np.expand_dims(trig_ID, axis=1) + 1,
                ],
                axis=1,
            ).astype(np.int64)

            trig_ch_names = np.array(self.info["ch_names"], dtype=object)[
                stm_misc_chans
            ]
            descriptions = trig_ch_names[trig_ID]
            durations = events[:, 1]
            annotations = mne.Annotations(
                onset=on_inds / self.info["sfreq"],
                duration=durations,
                description=descriptions,
            )

            self.set_annotations(annotations)
            return events, self


def calc_tangent(dipole_pos_vector):
    """Calculates tangent vectors for a position vector.

    Args:
        dipole_pos_vector (arraylike of length 3): 3D position vector (x,y,z coordinates).

    Returns:
        tuple[NDArray[float64], NDArray[float64]]: two tangential 3D vectors
    """
    x = dipole_pos_vector[0]
    y = dipole_pos_vector[1]
    z = dipole_pos_vector[2]
    r = np.sqrt(x * x + y * y + z * z)
    tanu = np.zeros(3)
    tanv = np.zeros(3)
    if x == 0 and y == 0:
        tanu[0] = 1.0
        tanu[1] = 0
        tanu[2] = 0
        tanv[0] = 0
        tanv[1] = 1.0
        tanv[2] = 0
    else:
        RZXY = -(r - z) * x * y
        X2Y2 = 1 / (x * x + y * y)

        tanu[0] = (z * x * x + r * y * y) * X2Y2 / r
        tanu[1] = RZXY * X2Y2 / r
        tanu[2] = -x / r

        tanv[0] = RZXY * X2Y2 / r
        tanv[1] = (z * y * y + r * x * x) * X2Y2 / r
        tanv[2] = -y / r

    return tanu, tanv
