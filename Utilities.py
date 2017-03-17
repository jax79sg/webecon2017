
import datetime

class Utility():

    starttime=None
    def startTimeTrack(self):
        self.starttime=datetime.datetime.now()
        pass

    def stopTimeTrack(self):
        endtime=datetime.datetime.now()
        duration=endtime-self.starttime
        result="Time taken: ",duration.seconds," secs"
        print(result)
        return result
        pass


    def checkpointTimeTrack(self):
        endtime = datetime.datetime.now()
        duration = endtime - self.starttime
        result = "Time taken: ", duration.seconds, " secs"
        self.starttime=endtime
        print(result)
        return result
        pass