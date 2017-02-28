from TableReader import TableReader


class Lookup:
    """
    Lookup class to pull (or reverse pull) city, region and tag info
    """

    ID="id"
    CITYNAME="cityname"
    REGIONNAME = "regionname"
    TAGNAME = "tagname"

    def __init__(self):
        ## Load in the lookup csvs upon loading
        self._citiesTable=TableReader("data.lookup/city.en.txt", header=0, delimiter="\t").getDataFrame()
        self._regionsTable=TableReader("data.lookup/region.en.txt", header=0, delimiter="\t").getDataFrame()
        self._tagsTable=TableReader("data.lookup/user.profile.tags.en.txt", header=0, delimiter="\t").getDataFrame()


        ## Setting the column names
        self._citiesTable.columns = [self.ID, self.CITYNAME]
        self._regionsTable.columns = [self.ID, self.REGIONNAME]
        self._tagsTable.columns = [self.ID, self.TAGNAME]


    def lookupCity(self, _id):
        """
        lookup city name by its id
        :param _id:
        :return: Empty string if not found
        """
        cityname=""
        cityresult=self._citiesTable.loc[self._citiesTable[self.ID]==_id]

        if(len(cityresult)==1):
            cityname=cityresult[self.CITYNAME].iloc[0]

        return(cityname)


    def lookupCityReverse(self, _cityname):
        """
        Lookup city ID by its name. Cannot support partial string.
        :param _cityname:
        :return: -1 if not found.
        """
        cityid = -1
        cityresult = self._citiesTable.loc[self._citiesTable[self.CITYNAME] == str.lower(_cityname)]

        if (len(cityresult) == 1):
            cityid = cityresult[self.ID].iloc[0]

        return (cityid)

    def lookupRegion(self, _id):
        """
        lookup region name by its id
        :param _id:
        :return: Empty string if not found
        """
        regionname=""
        regionresult=self._regionsTable.loc[self._regionsTable[self.ID]==_id]

        if(len(regionresult)==1):
            regionname=regionresult[self.REGIONNAME].iloc[0]

        return(regionname)


    def lookupRegionReverse(self, _regionname):
        """
        Lookup region ID by its name. Cannot support partial string.
        :param _regionname:
        :return: -1 if not found
        """
        regionid = -1
        regionresult = self._regionsTable.loc[self._regionsTable[self.REGIONNAME] == str.lower(_regionname)]

        if (len(regionresult) == 1):
            regionid = regionresult[self.ID].iloc[0]

        return (regionid)


    def lookupTag(self, _id):
        """
        lookup tag name by its id
        :param _id:
        :return: Empty string if not found
        """
        tagname=""
        tagresult=self._tagsTable.loc[self._tagsTable[self.ID]==_id]

        if(len(tagresult)==1):
            tagname=tagresult[self.TAGNAME].iloc[0]

        return(tagname)


    def lookupTagReverse(self, _tagname):
        """
        Lookup tag ID by its name. Cannot support partial string.
        :param _tagname:
        :return: -1 if not found
        """
        tagid = -1
        tagresult = self._tagsTable.loc[self._tagsTable[self.TAGNAME] == (_tagname)]

        if (len(tagresult) == 1):
            tagid = tagresult[self.ID].iloc[0]

        return (tagid)


# lookupme=Lookup()
# print(lookupme.lookupCity(4))
# print(lookupme.lookupCityReverse("tangshan"))
# print(lookupme.lookupRegionReverse("shandong"))
# print(lookupme.lookupRegion(325))
# print(lookupme.lookupTagReverse("In-market/3c product"))
# print(lookupme.lookupTag(10684))
