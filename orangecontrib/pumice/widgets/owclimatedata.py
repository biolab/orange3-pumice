import os.path
import pickle
import gzip

import numpy as np

from AnyQt.QtWidgets import QSizePolicy

from Orange.data import (
    Table, Domain, ContinuousVariable, StringVariable, DiscreteVariable)
from Orange.widgets import gui
from Orange.widgets.widget import OWWidget
from orangewidget.settings import Setting
from orangewidget.utils.signals import Input, Output
from orangewidget.utils.widgetpreview import WidgetPreview
from orangewidget.widget import Msg

Months = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]

MonthTempAttrs = [f"T-{month[:3]}" for month in Months]
MonthPrecAttrs = [f"P-{month[:3]}" for month in Months]

DATA_DIR = os.path.join(os.path.dirname(__file__), "../datasets/weather")

def dopen(s, mode="r"):
    return gzip.open(os.path.join(DATA_DIR, s) + ".gz", mode)

StationData = Table(os.path.join(DATA_DIR, "station-data.pkl.gz"))
Countries = sorted(set(StationData.get_column("Country")) - {""})
cont_values = StationData.domain["Continent"].values
Continents = sorted(set(cont_values) - {"", "?"})
Stations = sorted(set(StationData.get_column("Station")) - {""})


try:
    _daily_mask = pickle.load(dopen("S-Y-mask.pkl", "rb"))
    DailyStations = sorted(set(StationData.get_column("Station")[_daily_mask]) - {""})
except:
    # If the file is not available, we assume that daily values are not included
    INCLUDE_DAILY_VALUES = False
    DailyStations = []
else:
    INCLUDE_DAILY_VALUES = True

CountriesContinents = pickle.load(dopen("countries.pkl", "rb"))

DefaultContinent = "Europe"
DefaultCountry = "Slovenia"
DefaultStation = "LJUBLJANA BEZIGRAD, SI"

class OWClimateData(OWWidget):
    name = "Climate Data"
    description = "Climate data"
    icon = "icons/climatedata.svg"
    priority = 10

    class Error(OWWidget.Error):
        select_single = Msg("Select a single station")
        invalid_in_selection = Msg("Input data does not have a column 'Station'")

    class Warning(OWWidget.Warning):
        missing_stations = Msg("Some selected stations are missing in the data set")

    class Inputs:
        stations = Input("Weather Stations", Table)

    class Outputs:
        data = Output("Climate Data", Table)

    _AllGeo = Countries, CountriesOnContinent, Country, SingleStation = range(4)
    TotalMonthly, MonthlyByDecades, MonthMeanByDecades, DailyValues = range(4)

    Allowed = {
        TotalMonthly: _AllGeo,
        MonthMeanByDecades: _AllGeo,
        MonthlyByDecades: (SingleStation, ),
        DailyValues: (Country, SingleStation)
    }

    Avg, Min, Max, Span = range(4)

    geo_selection = Setting(Countries)
    continent = Setting(DefaultContinent)
    country = Setting(DefaultCountry)
    station = Setting(DefaultStation)

    time_selection = Setting(TotalMonthly)
    month_index = Setting(0)

    get_temperature = Setting(True)
    temperature_value = Setting(0)
    get_precipitation = Setting(False)

    want_main_area = False
    resizing_enabled = False

    def __init__(self):
        super().__init__()
        self.selected_stations = None

        tf = gui.radioButtonsInBox(
            self.controlArea, self, "time_selection", box="Time Frame",
            callback=self.time_selection_changed)
        tf.layout().setSpacing(1)
        gui.appendRadioButton(
            tf, "All-time monthly means", insertInto=gui.hBox(tf))
        tf.layout().addSpacing(6)
        gui.appendRadioButton(
            tf, "Monthly means by decades", insertInto=gui.hBox(tf))
        b = gui.hBox(tf)
        gui.appendRadioButton(
            tf, "Means for chosen month by decades", insertInto=b)
        gui.comboBox(
            b, self, "month_index",
            items=Months,
            sendSelectedValue=False,
            callback=self.month_changed,
            sizePolicy=(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        )
        if INCLUDE_DAILY_VALUES:
            gui.appendRadioButton(
                tf, "Daily values", insertInto=gui.hBox(tf))

        self.station_selector = ss = gui.radioButtonsInBox(
            self.controlArea, self, "geo_selection", box="Weather Stations",
            callback=self.geo_selection_changed)
        self.station_boxes = [gui.hBox(ss) for _ in range(4)]
        ss.layout().insertSpacing(1, 3)
        boxi = iter(self.station_boxes)
        self.station_selector.layout().setSpacing(1)
        gui.appendRadioButton(
            ss, "All stations (averages by country)", insertInto=next(boxi))
        b = next(boxi)
        gui.appendRadioButton(
            ss, "Countries on continent:", insertInto=b)
        gui.comboBox(
            b, self, "continent",
            items=Continents,
            sendSelectedValue=True,
            callback=self.continent_changed,
            sizePolicy=(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        )

        b = next(boxi)
        gui.appendRadioButton(ss, "Country: ", insertInto=b)
        gui.comboBox(
            b, self, "country",
            items=Countries,
            sendSelectedValue=True,
            callback=self.country_changed,
            sizePolicy=(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        )

        b = next(boxi)
        gui.appendRadioButton(
            ss, "Single station: ", insertInto=b)
        gui.comboBox(
            b, self, "station",
            items=Stations,
            sendSelectedValue=True,
            callback=self.single_station_changed,
            sizePolicy=(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        )

        vb = gui.vBox(self.controlArea, "Values")
        b = gui.hBox(vb)
        gui.checkBox(
            b, self, "get_temperature", "Temperature: ",
            callback=self.value_selection_changed
        )
        gui.comboBox(
            b, self, "temperature_value",
            items=["Average", "Minimum", "Maximum", "Span"],
            sendSelectedValue=False,
            callback=self.value_selection_changed,
        )
        gui.rubber(b)

        gui.checkBox(
            gui.hBox(vb), self, "get_precipitation", "Precipitation",
            callback=self.value_selection_changed
        )

        self._update_time_selection()
        self.update_data()

    @Inputs.stations
    def set_stations(self, data):
        self.selected_stations = data
        self.station_selector.setDisabled(data is not None)
        self.update_data()

    def continent_changed(self):
        self.geo_selection = self.CountriesOnContinent
        self.geo_selection_changed()

    def country_changed(self):
        self.geo_selection = self.Country
        self.geo_selection_changed()

    def single_station_changed(self):
        self.geo_selection = self.SingleStation
        self.geo_selection_changed()

    def geo_selection_changed(self):
        self.update_data()

    def month_changed(self):
        self.time_selection = self.MonthMeanByDecades
        self.time_selection_changed()

    def _update_time_selection(self):
        allowed = self.Allowed[self.time_selection]
        for i, button in enumerate(self.station_boxes):
            button.setDisabled(i not in allowed)
        if self.geo_selection not in allowed:
            self.geo_selection = allowed[0]

        stations = DailyStations if self.time_selection == self.DailyValues \
            else Stations
        stat_combo = self.controls.station
        if stat_combo.count() != len(stations):
            prev_station = self.station
            stat_combo.clear()
            stat_combo.addItems(stations)
            if prev_station in stations:
                self.station = prev_station
            else:
                assert DefaultStation in stations
                self.station = DefaultStation

    def time_selection_changed(self):
        self._update_time_selection()
        self.update_data()

    def value_selection_changed(self):
        self.update_data()

    def update_data(self):
        self.Error.select_single.clear()
        self.Error.invalid_in_selection.clear()
        self.Warning.missing_stations.clear()

        if not (self.get_precipitation or self.get_temperature):
            self.Outputs.data.send(None)
            return

        tdata, pdata, attrs, meta, meta_attrs = self.Getters[self.time_selection](self)
        if tdata is None and pdata is None:
            self.Outputs.data.send(None)
            return
        data = tdata if not self.get_precipitation \
            else pdata if not self.get_temperature \
            else np.hstack((tdata, pdata))
        domain = Domain(
            [ContinuousVariable(attr) for attr in attrs],
            None,
            meta_attrs
        )
        self.Outputs.data.send(Table.from_numpy(domain, data, None, meta))

    def _load_tdata(self, prefix):
        """
        Load temperature data for the given prefix, e.g. "C-MT-" or "S-M2024-".

        The methods loads avg, min, max or max - min, depending, as set in
        self.temperature_value.

        The data is loaded from the corresponding pickle.

        :param prefix: Prefix for the data file.
        :return:
        """
        if self.temperature_value == self.Avg:
            return pickle.load(dopen(prefix + "tavg.pkl", "rb"))
        elif self.temperature_value == self.Min:
            return pickle.load(dopen(prefix + "tmin.pkl", "rb"))
        elif self.temperature_value == self.Max:
            return pickle.load(dopen(prefix + "tmax.pkl", "rb"))
        else:
            return (pickle.load(dopen(prefix + "tmax.pkl", "rb"))
                    - pickle.load(dopen(prefix + "tmin.pkl", "rb")))

    def _country_indices(self):
        """
        Get the indices of the countries in the data set. These are either
        all countries or those on the selected continent.

        :return: a tuple of
          - indices (... or np.array of bool):
              Indices of countries within country tables for the given
              continent or all countries.
          - meta (np.array of object):
              metadata with country names, and a continent if it is not selected
          - meta_attrs (list[str]):
              the names of meta attributes for the above.
        """
        meta_attrs = [StationData.domain["Country"]]
        if self.geo_selection == self.Countries:
            indices = ...
            meta_attrs.append(StationData.domain["Continent"])
            meta = CountriesContinents
        else:
            indices = CountriesContinents[:, 1] == self.continent
            meta = CountriesContinents[indices, :1]
        return indices, meta, meta_attrs

    def _station_indices(self):
        """
        Get the indices of the statiions: either a single station or all
        stations in the selected country.

        :return: a tuple of
          - indices (... or np.array of bool):
              Index(-ices) of selected station(s)
          - meta (None or np.array of object):
              Station names, or None if a station is selected
          - meta_attrs (list[str]):
              the names of meta attributes for the above.
        """

        if self.geo_selection == self.SingleStation:
            indices = np.flatnonzero(
                StationData.get_column("Station") == self.station)
            meta_attrs = []
            meta = None
        else:
            indices = np.flatnonzero(
                StationData.get_column("Country") == self.country)
            meta_attrs = [StringVariable("Station")]
            meta = StationData.get_column("Station")[indices][:, None]
        return indices, meta, meta_attrs

    def _selection_indices(self):
        n = len(self.selected_stations)
        nothing = (None, ) * 3
        if n == 0:
            return nothing
        if n > 1 and self.time_selection == self.MonthlyByDecades:
            self.Error.select_single()
            return nothing
        if "Station" not in self.selected_stations.domain:
            self.Error.invalid_in_selection()
            return nothing

        indices = np.isin(StationData.get_column("Station"),
                          self.selected_stations.get_column("Station"))
        sel = np.sum(indices)
        if sel == 0:
            return nothing
        if sel != n:
            self.Warning.missing_stations()

        if n == 1:
            meta_attrs = []
            meta = None
        else:
            if self.time_selection == self.MonthlyByDecades:
                self.Error.select_single()
                return nothing
            meta_attrs = [StationData.domain["Station"],
                          StationData.domain["Country"]]
            meta = np.vstack((
                StationData.get_column("Country"),
                StationData.get_column("Station"))).T
        return indices, meta, meta_attrs


    def _month_attrs(self):
        """
        Return names of attributes for monthly data.
        """
        return (MonthTempAttrs if self.get_temperature else []) \
            + (MonthPrecAttrs if self.get_precipitation else [])

    def _get_data(self, infix):
        if self.selected_stations is not None:
            indices, meta, meta_attrs = self._selection_indices()
            prefix = "S"
        elif self.geo_selection in (self.Countries, self.CountriesOnContinent):
            indices, meta, meta_attrs = self._country_indices()
            prefix = "C"
        else:
            indices, meta, meta_attrs = self._station_indices()
            prefix = "S"
        assert indices is not None, \
            "Is this monthly mean going through _selection_indices via _get_data?"
        tdata = (self.get_temperature and
                 self._load_tdata(f"{prefix}-{infix}-")[indices])
        pdata = (self.get_precipitation and
                 pickle.load(dopen(f"{prefix}-{infix}-prcp.pkl", "rb"))[indices])
        return tdata, pdata, meta, meta_attrs

    def _total_monthly(self):
        tdata, pdata, meta, meta_attrs = self._get_data("MT")
        return tdata, pdata, self._month_attrs(), meta, meta_attrs

    def _decades_monthly(self):
        if self.selected_stations is None:
            assert self.geo_selection == self.SingleStation
            stationIdx = np.flatnonzero(
                StationData.get_column("Station") == self.station)[0]
        else:
            indices, *_ = self._selection_indices()
            if indices is None:
                return None, None, [], None, []
            stationIdx = np.flatnonzero(indices)[0]
        if self.get_temperature:
            tdata = self._load_tdata("S-MD-")[stationIdx].T
        else:
            tdata = np.array((13, 0))
        if self.get_precipitation:
            pdata = pickle.load(dopen("S-MD-prcp.pkl", "rb"))[stationIdx].T
        else:
            pdata = np.array((13, 0))
        decades = tuple(f"{decade}-{decade % 100 + 9:02}"
                        for decade in range(1900, 2030, 10))
        meta_attrs = [DiscreteVariable("Decade", values=decades)]
        meta = np.arange(len(decades))[:, None]
        return tdata, pdata, self._month_attrs(), meta, meta_attrs

    def _month_by_decades(self):
        tdata, pdata, meta, meta_attrs = self._get_data("MD")
        attrs = []
        if self.get_temperature:
            tdata = tdata[:, self.month_index]
            attrs += [f"T-{decade}-{decade % 100 + 9:02}"
                     for decade in range(1900, 2030, 10)]
        if self.get_precipitation:
            pdata = pdata[:, self.month_index]
            attrs += [f"P-{decade}-{decade % 100 + 9:02}"
                      for decade in range(1900, 2030, 10)]
        return tdata, pdata, attrs, meta, meta_attrs

    def _daily_values(self):
        tdata, pdata, meta, meta_attrs = self._get_data("Y")
        ndays = [31, 28 + (tdata.shape[1] == 366), 31, 30, 31, 30,
                31, 31, 30, 31, 30, 31]
        daynames = [f"{month[:3]} {day + 1}"
                    for month, days in zip(Months, ndays)
                    for day in range(days)]
        attrs = []
        if self.get_temperature:
            attrs += [f"T-{d}" for d in daynames]
            tdata /= 10
        if self.get_precipitation:
            attrs += [f"P-{d}" for d in daynames]
        return tdata, pdata, attrs, meta, meta_attrs


    Getters = {
      TotalMonthly: _total_monthly,
      MonthlyByDecades: _decades_monthly,
      MonthMeanByDecades: _month_by_decades,
      DailyValues: _daily_values
    }

if __name__ == "__main__":
    WidgetPreview(OWClimateData).run()
    # WidgetPreview(OWClimateData).run(set_stations=StationData[:1])
    # WidgetPreview(OWClimateData).run(set_stations=StationData[:10])
