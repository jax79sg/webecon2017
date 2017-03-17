
import datetime

class Utility():

    starttime=None
    def startTimeTrack(self):
        """
        This must be the first method to call before calling stopTimeTrack or checkpointTimeTrack
        Will start the recording of time.
        :return:
        """
        self.starttime=datetime.datetime.now()
        pass

    def stopTimeTrack(self):
        """
        This is called in pair with startTimeTrack everytime.
        It will print time lapse after startTimeTrack
        :return:
        """
        endtime=datetime.datetime.now()
        duration=endtime-self.starttime
        result="Time taken: ",duration.seconds," secs"
        print(result)
        return result
        pass


    def checkpointTimeTrack(self):
        """
        This can be called consecutively for as many times as long as startTimeTrack has been first called.
        It will print the time lapse from last check point
        E.g.
        Utility().startTimeTrack)_
        Utility().checkpointTimeTrack()
        Utility().checkpointTimeTrack()
        Utility().checkpointTimeTrack()
        :return:
        """
        endtime = datetime.datetime.now()
        duration = endtime - self.starttime
        result = "Time taken: ", duration.seconds, " secs"
        self.starttime=endtime
        print(result)
        return result
        pass